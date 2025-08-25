# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.vjepa2.configuration_vjepa2 import VJEPA2Config
from transformers.models.vjepa2.modeling_vjepa2 import (
    VJEPA2MLP,
    VJEPA2Embeddings,
    VJEPA2DropPath,
    eager_attention_forward,
    rotate_queries_or_keys,
)
from transformers.utils import logging
from transformers.utils.generic import can_return_tuple

logger = logging.get_logger(__name__)


class ActionEmbeddings(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int = 1024):
        self.embed = nn.Sequential(
            nn.Linear(action_dim, embed_dim // 2),
            nn.Mish(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.unembed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Mish(),
            nn.Linear(embed_dim // 2, action_dim),
        )


class VJEPA2JointDiffuserRopeAttention(nn.Module):
    def __init__(
        self,
        config: VJEPA2Config,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {(hidden_size,)} is not a multiple of the number of attention "
                f"heads {num_attention_heads}."
            )

        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.query_action = nn.Linear(
            hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key_action = nn.Linear(
            hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.value_action = nn.Linear(
            hidden_size, self.all_head_size, bias=config.qkv_bias
        )

        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_action = nn.Linear(hidden_size, hidden_size)
        self.dropout_prob = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)

        self.grid_size = self.config.crop_size // self.config.patch_size
        self.grid_depth = self.config.frames_per_clip // self.config.tubelet_size

        self.d_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.h_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.w_dim = int(2 * ((self.attention_head_size // 3) // 2))

        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

    def _get_frame_pos(self, ids):
        tokens_per_frame = int(self.grid_size * self.grid_size)
        return ids // tokens_per_frame

    def _get_height_pos(self, ids):
        # Remove frame component from ids
        tokens_per_frame = int(self.grid_size * self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        # --
        tokens_per_row = self.grid_size
        return ids // tokens_per_row

    def get_video_position_ids(self, x, masks=None):
        device = x.device
        token_size = x.size(1)

        # Note: when masks is none, we use a 1d id instead of Bxnum_attention_heads mask,
        # as 1d vector is broadcasted to the correct shapes.
        if masks is not None:
            ids = masks.unsqueeze(1).repeat(1, self.num_attention_heads, 1)
        else:
            ids = torch.arange(token_size, device=device)
        # change to allow for extrapolation
        tokens_per_frame = int(self.grid_size * self.grid_size)
        frame_ids = self._get_frame_pos(ids)
        # --
        tokens_per_row = self.grid_size
        height_ids = self._get_height_pos(ids)
        # --
        # Remove frame component from ids (1st term) and height component (2nd term)
        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids
        return frame_ids, height_ids, width_ids

    # to do 1d rotary embeddings, use the d_mask == h_mask == w_mask
    def apply_rotary_embeddings(self, qk, pos_ids):
        d_mask, h_mask, w_mask = pos_ids
        s = 0
        qkd = rotate_queries_or_keys(qk[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qkh = rotate_queries_or_keys(qk[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qkw = rotate_queries_or_keys(qk[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim
        # Combine rotated dimension
        if s < self.attention_head_size:
            qkr = qk[..., s:]
            qk = torch.cat([qkd, qkh, qkw, qkr], dim=-1)
        else:
            qk = torch.cat([qkd, qkh, qkw], dim=-1)
        return qk

    def forward(
        self,
        video_hidden_states,
        action_hidden_states,
        position_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        head_mask: Optional[torch.Tensor] = None,
    ) -> Union[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        batch_size, _, _ = video_hidden_states.shape

        def get_proj(layer, states):
            return (
                layer(states)
                .view(
                    batch_size, -1, self.num_attention_heads, self.attention_head_size
                )
                .transpose(1, 2)
            )

        # video qkv
        query_layer_video = get_proj(self.query, video_hidden_states)
        key_layer_video = get_proj(self.key, video_hidden_states)
        value_layer_video = get_proj(self.value, video_hidden_states)

        pos_ids = self.get_video_position_ids(video_hidden_states, masks=position_mask)
        key_layer_video = self.apply_rotary_embeddings(key_layer_video, pos_ids)
        query_layer_video = self.apply_rotary_embeddings(query_layer_video, pos_ids)

        """
        action qkv.
        assumptions:
        - first action = first video frame, last action = last video frame
        - actions evenly-spaced
        - (num_actions - 1) / num_video_frames = k, integer, "subsampling amount"
        """
        query_layer_action = get_proj(self.query_action, action_hidden_states)
        key_layer_action = get_proj(self.key_action, action_hidden_states)
        value_layer_action = get_proj(self.value_action, action_hidden_states)
        d_mask = pos_ids[0]
        d_mask_actions = torch.linspace(
            d_mask[0], d_mask[1], action_hidden_states.shape[1]
        )
        key_layer_action = self.apply_rotary_embeddings(
            key_layer_action, (d_mask_actions, d_mask_actions, d_mask_actions)
        )
        query_layer_action = self.apply_rotary_embeddings(
            query_layer_action, (d_mask_actions, d_mask_actions, d_mask_actions)
        )

        query_layer = torch.cat((query_layer_video, query_layer_action), dim=1)
        key_layer = torch.cat((key_layer_video, key_layer_action), dim=1)
        value_layer = torch.cat((value_layer_video, value_layer_action), dim=1)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        context_layer_video, context_layer_action = torch.split(
            context_layer,
            [query_layer_video.shape[1], query_layer_action.shape[1]],
            dim=1,
        )
        context_layer_video = self.proj(context_layer_video)
        context_layer_action = self.proj_action(context_layer_action)

        outputs = (
            (context_layer_video, context_layer_action, attention_probs)
            if output_attentions
            else (context_layer, context_layer_action)
        )

        return outputs


class VJEPA2JointDiffuserLayer(GradientCheckpointingLayer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(
        self,
        config: VJEPA2Config,
        drop_path_rate: float = 0.0,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = VJEPA2JointDiffuserRopeAttention(
            config, hidden_size, num_attention_heads
        )
        self.drop_path = (
            VJEPA2DropPath(drop_path_rate)
            if config.drop_path_rate > 0.0
            else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA2MLP(config, hidden_size=hidden_size, mlp_ratio=mlp_ratio)
        self.mlp_actions = VJEPA2MLP(
            config, hidden_size=hidden_size, mlp_ratio=mlp_ratio
        )

    def forward(
        self,
        video_hidden_states: torch.Tensor,
        action_hidden_states: torch.Tensor,
        video_modulation: torch.Tensor,
        action_modulation: torch.Tensor,
        position_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        # Self-Attention
        video_residual = video_hidden_states
        action_residual = action_hidden_states
        video_hidden_states = self.norm1(video_hidden_states)

        video_shift, video_scale, video_residual_scale = torch.chunk(
            video_modulation, 3, dim=-1
        )
        action_shift, action_scale, action_residual_scale = torch.chunk(
            action_modulation, 3, dim=-1
        )

        self_attention_outputs = self.attention(
            video_hidden_states * (1 + video_scale) + video_shift,
            action_hidden_states * (1 + action_scale) + action_shift,
            position_mask=position_mask,  # position mask for context/target selection
            head_mask=head_mask,  # head mask is applied at F.scaled_dot_product_attention
            output_attentions=output_attentions,
        )
        video_attention_output = self_attention_outputs[0]
        action_attention_output = self_attention_outputs[-1]
        video_hidden_states = (
            self.drop_path(video_attention_output)
            + video_residual * video_residual_scale
        )
        action_hidden_states = (
            self.drop_path(action_attention_output)
            + action_residual * action_residual_scale
        )

        # MLP
        residual = video_hidden_states
        video_hidden_states = self.norm2(video_hidden_states)
        video_hidden_states = self.mlp(video_hidden_states)
        video_hidden_states = self.drop_path(video_hidden_states) + residual

        residual = action_hidden_states
        action_hidden_states = self.norm2(action_hidden_states)
        action_hidden_states = self.mlp_actions(action_hidden_states)
        action_hidden_states = self.drop_path(action_hidden_states) + residual

        # Add self attentions if we output attention weights
        outputs = (video_hidden_states, action_hidden_states) + self_attention_outputs[
            -1:
        ]

        return outputs


class SinusoidalPosEmbed(nn.Module):
    # by Michael
    def __init__(self, size: int, max_length: float, dtype=torch.float32):
        super().__init__()

        self.size = size
        self.max_length = max_length
        self.dtype = dtype

    def forward(self, position: torch.Tensor):
        angle = torch.exp(
            torch.log(position.unsqueeze(-1) / self.max_length)
            * (torch.arange(0, self.size, 2) / float(self.size))
        )
        embed = torch.zeros(
            (*position.shape, self.size), device=position.device, dtype=self.dtype
        )
        embed[:, 0::2] = torch.sin(angle)
        embed[:, 1::2] = torch.cos(angle)
        return embed


class VJEPA2JointDiffuser(nn.Module):
    def __init__(self, config: VJEPA2Config, action_dim=2):
        super().__init__()
        self.config = config

        self.embeddings = VJEPA2Embeddings(config, hidden_size=config.hidden_size)
        self.action_embeddings = ActionEmbeddings(
            action_dim=action_dim, embed_dim=config.hidden_size
        )
        drop_path_rates = [
            (
                config.drop_path_rate * i / (config.num_hidden_layers - 1)
                if config.num_hidden_layers > 1
                else 0.0
            )
            for i in range(config.num_hidden_layers)
        ]
        self.layer = nn.ModuleList(
            [
                VJEPA2JointDiffuserLayer(
                    config,
                    drop_path_rate=drop_path_rates[i],
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

        # flow-matching; timesteps are in [0.0, 1.0]
        timestep_embed_dim = 128
        self.video_modulation = nn.Sequential(
            SinusoidalPosEmbed(timestep_embed_dim, 1.0),
            nn.Linear(timestep_embed_dim, config.hidden_size // 2),
            nn.Mish(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
        )
        self.action_modulation = nn.Sequential(
            SinusoidalPosEmbed(timestep_embed_dim, 1.0),
            nn.Linear(timestep_embed_dim, config.hidden_size // 2),
            nn.Mish(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
        )

    @can_return_tuple
    def forward(
        self,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_t: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        actions_t: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        video_hidden_states = self.embeddings(pixel_values_videos)
        action_hidden_states = self.action_embeddings.embed(actions)
        video_modulation = self.video_modulation(video_t)
        action_modulation = self.action_modulation(actions_t)

        for i, layer_module in enumerate(self.layer):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                video_hidden_states,
                action_hidden_states,
                video_modulation,
                action_modulation,
                None,
                layer_head_mask,
                output_attentions,
            )
            video_hidden_states = layer_outputs[0]
            action_hidden_states = layer_outputs[1]

        video_hidden_states = self.layernorm(video_hidden_states)

        return BaseModelOutput(
            last_hidden_state=video_hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
