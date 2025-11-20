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

For the modeling approach, we can use diffusion or flow matching. I am inclined to use diffusion because of its connection to the energy-based formulation. For this, I will use EDM preconditioning: https://arxiv.org/pdf/2206.00364

The ODE integration can be done separately, as long as some predictor of the original sample is provided. So, the model only needs to expose a `get_loss` and forward pass function.
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
    shape = data.shape
    data = torch.view_as_complex(data.reshape(*shape[:-1], -1, 2))
    data = data * freqs
    data = torch.view_as_real(data).reshape(*shape[:-1], -1)
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


def edm_coefficients(sigma: torch.Tensor, sigma_data: float):
    c_skip = (sigma_data**2) / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / torch.sqrt(sigma**2 + sigma_data**2)
    c_in = 1 / (sigma**2 + sigma_data**2)
    c_noise = torch.log(sigma) / 4.0
    return c_skip, c_out, c_in, c_noise


def edm_sample_timesteps(N, rho=7, sigma_min=0.002, sigma_max=80):
    """i = 0 corresponds to sigma_max. i = N corresponds to sigma = 0."""
    sigmas = (
        sigma_max ** (1 / rho)
        + torch.arange(N) / (N - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** (1 / rho)
    # add sigma = 0 so that we can use sigmas[N] = 0
    sigmas = torch.cat([sigmas, torch.zeros(1)], dim=0)
    return sigmas


class StateObservationTransformer(nn.Module):
    def __init__(self, embedding_dim, layers=6, heads=8, sigma_data=0.5):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.sigma_data = sigma_data
        self.state_proj = nn.Linear(5, embedding_dim)
        self.action_proj = nn.Linear(2, embedding_dim)
        self.state_unproj = nn.Linear(embedding_dim, 5)
        self.action_unproj = nn.Linear(embedding_dim, 2)
        self.diffusion_step_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(approximate="tanh"),
            nn.Linear(128, self.embedding_dim * 3),
        )
        self.layers = nn.ModuleList(
            [Attention(embedding_dim, nhead=heads) for _ in range(layers)]
        )

    def forward(self, states, actions, sigma):
        """Does EDM preconditioning for us."""

        # coefficients: (B,)
        # states: (B, T, 5)
        # actions: (B, T, 2)
        c_skip, c_out, c_in, c_sigma = edm_coefficients(sigma, self.sigma_data)
        c_skip = c_skip.view(-1, 1, 1)
        c_out = c_out.view(-1, 1, 1)
        c_in = c_in.view(-1, 1, 1)
        c_sigma = c_sigma.view(-1, 1, 1)

        state_z = self.state_proj(c_in * states)
        action_z = self.action_proj(c_in * actions)
        e = self.diffusion_step_mlp(sinusoidal_positional_encoding(c_sigma, 128, 4.0))

        for layer in self.layers:
            state_z, action_z = layer(state_z, action_z, e)

        states_prime = self.state_unproj(state_z)
        actions_prime = self.action_unproj(action_z)

        return (
            states * c_skip + states_prime * c_out,
            actions * c_skip + actions_prime * c_out,
        )

    def get_loss(self, states, actions):
        states_noise = torch.randn_like(states)
        actions_noise = torch.randn_like(actions)

        P_mean = 1.2
        P_std = 1.2
        sigma = torch.exp(
            torch.randn(states.shape[0], device=states.device) * P_std + P_mean
        )  # (B,)
        loss_weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        states_noisy = states + states_noise * sigma.view(-1, 1, 1)
        actions_noisy = actions + actions_noise * sigma.view(-1, 1, 1)

        states_pred, actions_pred = self.forward(states_noisy, actions_noisy, sigma)

        state_loss = ((states_pred - states) ** 2).mean(dim=(1, 2)) * loss_weight
        action_loss = ((actions_pred - actions) ** 2).mean(dim=(1, 2)) * loss_weight

        return {
            "state_loss": state_loss,
            "action_loss": action_loss,
            "total": state_loss.mean() + action_loss.mean(),
            "sigma": sigma,
        }

    def sample(self, horizon, N, deterministic=True):
        device = next(self.parameters()).device
        sigmas = edm_sample_timesteps(N).to(device)

        # Generate initial sample.
        states = torch.randn(1, horizon, 5).to(device)
        actions = torch.randn(1, horizon, 2).to(device)

        for i in range(N):
            # "source_t" is here so that code can be shared between the stochastic and deterministic samplers. If deterministic == False, then it is equivalent to the original t. Otherwise, we add a temporary increased noise level to create "Langevin churn".
            source_t = sigmas[i]
            if not deterministic:
                S_churn = 80
                S_tmin = 0.05
                S_tmax = 50
                gamma = (
                    min(S_churn / N, math.sqrt(2) - 1)
                    if S_tmin <= source_t <= S_tmax
                    else 0
                )
                if gamma != 0:
                    source_t = sigmas[i] * (1 + gamma)
                    states = states + torch.randn_like(states) * math.sqrt(
                        source_t**2 - sigmas[i] ** 2
                    )
                    actions = actions + torch.randn_like(actions) * math.sqrt(
                        source_t**2 - sigmas[i] ** 2
                    )

            # Evaluate dx/dt.
            states_pred, actions_pred = self(states, actions, source_t)
            states_d = 1 / source_t * (states - states_pred)
            actions_d = 1 / source_t * (actions - actions_pred)

            # Euler step.
            states_next = states + states_d * (sigmas[i + 1] - source_t)
            actions_next = actions + actions_d * (sigmas[i + 1] - source_t)

            # Apply second-order correction if needed.
            if sigmas[i + 1] != 0:
                # Evaluate dx/dt at next step.
                states_d_prime = 1 / sigmas[i + 1] * (states_next - states_pred)
                actions_d_prime = 1 / sigmas[i + 1] * (actions_next - actions_pred)
                # Trapezoidal rule.
                states_next = states_next + (sigmas[i + 1] - source_t) * 0.5 * (
                    states_d + states_d_prime
                )
                actions_next = actions_next + (sigmas[i + 1] - source_t) * 0.5 * (
                    actions_d + actions_d_prime
                )

            states = states_next
            actions = actions_next

        return (states, actions)
