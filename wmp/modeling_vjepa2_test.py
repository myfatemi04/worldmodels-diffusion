from typing import cast

import torch
from transformers.models.vjepa2.configuration_vjepa2 import VJEPA2Config

from wmp.modeling_vjepa2 import (
    VJEPA2JointDiffuser,
    VJEPA2JointDiffuserLayer,
    VJEPA2JointDiffuserRopeAttention,
)

config = VJEPA2Config.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
config._attn_implementation = "eager"
diffuser = VJEPA2JointDiffuser(config, action_dim=2)
attn = VJEPA2JointDiffuserRopeAttention(
    config,
    hidden_size=1024,
    num_attention_heads=16,
)
num_frames = 8
num_actions_per_frame = 4
num_actions = (num_frames - 2) * num_actions_per_frame + 1
hidden_size = config.hidden_size
tokens_per_rastor = 256
num_video_tokens = tokens_per_rastor * (num_frames // config.tubelet_size)
attn_output = attn.forward(
    video_hidden_states=torch.randn(1, num_video_tokens, hidden_size),
    action_hidden_states=torch.randn(1, num_actions, hidden_size),
    video_fps=4.0,
    action_fps=4.0 * num_actions_per_frame,
    num_condition_frames=2,
)
layer = cast(VJEPA2JointDiffuserLayer, diffuser.layer[0])
layer_output = layer.forward(
    video_hidden_states=torch.randn(1, num_video_tokens, hidden_size),
    action_hidden_states=torch.randn(1, num_actions, hidden_size),
    video_modulation=diffuser.video_modulation(torch.rand((1,))),
    action_modulation=diffuser.action_modulation(torch.rand((1,))),
    video_fps=4.0,
    action_fps=4.0 * num_actions_per_frame,
    num_condition_frames=2,
)
diffuser_output = diffuser.forward(
    video_latents=torch.randn(1, num_video_tokens, hidden_size),
    action=torch.randn(1, num_actions, 2),
    video_t=torch.rand((1,)),
    action_t=torch.rand((1,)),
    video_fps=4.0,
    action_fps=4.0 * num_actions_per_frame,
    num_condition_frames=2,
)
