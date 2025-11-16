"""
This is a diffusion-based transformer.

State observations are:
- Agent x and y; [0, 512]. Normalized to [-1, 1].
- Block x and y; [0, 512]. Normalized to [-1, 1].
- Block angle; [0, 2 * pi]. Represented as cos and sin components to avoid angle wrapping.

Actions are:
- Delta x and delta y; normalized to [-1, 1].

Positional embeddings are represented through RoPE.

Diffusion timesteps are represented through AdaLN:
    (shift, scale, residual) = mlp(spe(timestep))
    y = attention((1 + scale) * x + shift)
    x = x + residual * y

To allow communication between the modalities, we use separate weight matrices for queries, keys, and values for the states and actions.
They undergo full attention together. Then, they get passed through separate MLPs.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_positional_encoding(
    positions: torch.Tensor, embedding_dim: int, max_len: float
):
    position = positions.unsqueeze(-1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=torch.float32, device=positions.device)
        / embedding_dim
        * -math.log(max_len)
    )
    emb = torch.stack(
        [torch.sin(position * div_term), torch.cos(position * div_term)]
    ).reshape(*positions.shape[:-1], embedding_dim)
    return emb


def rope_embedding_1d(data: torch.Tensor, max_len: float):
    # View as complex, perform complex multiplication, and then view as real.
    # Based on https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py.
    seqlen = data.shape[-2]
    dim = data.shape[-1]

    # (seqlen, dim/2) complex tensor
    freqs = torch.outer(
        torch.arange(seqlen),
        torch.exp(-math.log(max_len) * torch.arange(0, dim, 2) / dim),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)

    # View as complex, rotate, and then view as real.
    data = torch.view_as_complex(data.reshape(*data.shape[:-1], -1, 2))
    data = data * freqs
    data = torch.view_as_real(data).reshape(*data.shape[:-2], -1)
    return data


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, nhead: int):
        super().__init__()
        self.qkv_state = nn.Linear(embedding_dim, embedding_dim * 3)
        self.qkv_action = nn.Linear(embedding_dim, embedding_dim * 3)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=nhead, batch_first=True
        )
        self.mlp_state = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.mlp_action = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )

    def forward(self, state_z: torch.Tensor, action_z: torch.Tensor, e: torch.Tensor):
        shift, scale, residual = e.chunk(3, dim=-1)
        q_state, k_state, v_state = self.qkv_state((1 + scale) * state_z + shift).chunk(
            3, dim=-1
        )
        q_action, k_action, v_action = self.qkv_action(
            (1 + scale) * action_z + shift
        ).chunk(3, dim=-1)

        q_state = rope_embedding_1d(q_state, max_len=512)
        k_state = rope_embedding_1d(k_state, max_len=512)
        q_action = rope_embedding_1d(q_action, max_len=512)
        k_action = rope_embedding_1d(k_action, max_len=512)

        q = torch.cat([q_state, q_action], dim=-2)
        k = torch.cat([k_state, k_action], dim=-2)
        v = torch.cat([v_state, v_action], dim=-2)

        y, _ = self.attention(q, k, v)
        y_state, y_action = y.chunk(2, dim=-2)
        state_z = state_z + self.mlp_state(y_state) * residual
        action_z = action_z + self.mlp_action(y_action) * residual

        return state_z, action_z


class StateObservationTransformer(nn.Module):
    def __init__(self, embedding_dim: int, layers: int = 6, heads: int = 8):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.state_proj = nn.Linear(5, embedding_dim)
        self.action_proj = nn.Linear(2, embedding_dim)
        self.state_unproj = nn.Linear(embedding_dim, 5)
        self.action_unproj = nn.Linear(embedding_dim, 2)
        self.diffusion_step_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(approximate="tanh"),
            nn.Linear(128, self.embedding_dim),
        )
        self.layers = nn.ModuleList(
            [Attention(embedding_dim, nhead=heads) for _ in range(layers)]
        )

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, diffusion_step: torch.Tensor
    ):
        state_z = self.state_proj(states)
        action_z = self.action_proj(actions)
        e = self.diffusion_step_mlp(
            sinusoidal_positional_encoding(diffusion_step, 128, 4.0)
        )

        # Apply attention
        for layer in self.layers:
            state_z, action_z = layer(state_z, action_z, e)

        return state_z
