# pip install --root-user-action ignore accelerate diffusers av loguru matplotlib gymnasium gym-pusht "pymunk<7"; apt install -y libgl1-mesa-glx htop zip unzip

import pickle

import av
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym_pusht.envs.pusht import PushTEnv
from loguru import logger
import time

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
        torch.arange(seqlen, device=data.device),
        torch.exp(
            -math.log(max_len) * torch.arange(0, dim, 2, device=data.device) / dim
        ),
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

    def sample(self, horizon, N, deterministic=True, initial_noise_seed=None):
        device = next(self.parameters()).device
        sigmas = edm_sample_timesteps(N).to(device)

        # Generate initial sample.
        if initial_noise_seed is None:
            initial_noise_seed = int(torch.randint(0, 1000000, ()).item())
        generator = torch.Generator()
        generator.manual_seed(initial_noise_seed)
        states = torch.randn(1, horizon, 5, generator=generator).to(device)
        actions = torch.randn(1, horizon, 2, generator=generator).to(device)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_demonstrations():
    states = torch.load("data/pusht_noise/train/states.pth")
    with open("data/pusht_noise/train/seq_lengths.pkl", "rb") as f:
        seq_lengths = pickle.load(f)
    abs_actions = torch.load("data/pusht_noise/train/abs_actions.pth")
    rel_actions = torch.load("data/pusht_noise/train/rel_actions.pth")

    return [
        (
            states[i][: seq_lengths[i]],
            abs_actions[i][: seq_lengths[i]],
            rel_actions[i][: seq_lengths[i]],
        )
        for i in range(len(states))
    ]


def compute_dataset_statistics(demos):
    state_sum = torch.zeros(5)
    state_sq_sum = torch.zeros(5)
    state_count = 0
    abs_action_sum = torch.zeros(2)
    abs_action_sq_sum = torch.zeros(2)
    abs_action_count = 0
    rel_action_sum = torch.zeros(2)
    rel_action_sq_sum = torch.zeros(2)
    rel_action_count = 0

    for states, abs_actions, rel_actions in demos:
        state_sum += states.sum(dim=0)
        state_sq_sum += (states**2).sum(dim=0)
        state_count += states.shape[0]

        abs_action_sum += abs_actions.sum(dim=0)
        abs_action_sq_sum += (abs_actions**2).sum(dim=0)
        abs_action_count += abs_actions.shape[0]
        rel_action_sum += rel_actions.sum(dim=0)
        rel_action_sq_sum += (rel_actions**2).sum(dim=0)
        rel_action_count += rel_actions.shape[0]

    state_mean = state_sum / state_count
    state_var = state_sq_sum / state_count - state_mean**2
    state_std = torch.sqrt(state_var)
    abs_action_mean = abs_action_sum / abs_action_count
    abs_action_var = abs_action_sq_sum / abs_action_count - abs_action_mean**2
    abs_action_std = torch.sqrt(abs_action_var)
    rel_action_mean = rel_action_sum / rel_action_count
    rel_action_var = rel_action_sq_sum / rel_action_count - rel_action_mean**2
    rel_action_std = torch.sqrt(rel_action_var)

    return (
        state_mean,
        state_std,
        abs_action_mean,
        abs_action_std,
        rel_action_mean,
        rel_action_std,
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, demos, h, device, obs_mode="state", obs_size=512):
        # h = horizon
        demos = [d for d in demos if d[0].shape[0] >= h]
        self.demos = demos

        (
            state_mean,
            state_std,
            abs_action_mean,
            abs_action_std,
            rel_action_mean,
            rel_action_std,
        ) = compute_dataset_statistics(self.demos)

        self.device = device
        self.state_mean = state_mean.to(self.device)
        self.state_std = state_std.to(self.device)
        self.abs_action_mean = abs_action_mean.to(self.device)
        self.abs_action_std = abs_action_std.to(self.device)
        self.rel_action_mean = rel_action_mean.to(self.device)
        self.rel_action_std = rel_action_std.to(self.device)
        self.h = h
        self.obs_mode = obs_mode
        self.obs_size = obs_size

        if obs_mode == "rgb":
            self.env = gym.make(
                "gym_pusht/PushT-v0",
                render_mode="rgb_array",
                visualization_width=obs_size,
                visualization_height=obs_size,
            )
        else:
            self.env = None

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, idx):
        d = self.demos[idx]
        states, abs_actions, rel_actions = self.demos[idx]
        states = states.to(self.device)
        abs_actions = abs_actions.to(self.device)
        rel_actions = rel_actions.to(self.device)
        start_idx = torch.randint(0, d[0].shape[0] - self.h, (1,)).item()

        states = states[start_idx : start_idx + 32]
        abs_actions = abs_actions[start_idx : start_idx + 32]
        rel_actions = rel_actions[start_idx : start_idx + 32]

        norm_abs_actions = (
            (abs_actions - self.abs_action_mean) / (2 * self.abs_action_std)
        ).float()
        norm_rel_actions = (
            (rel_actions - self.rel_action_mean) / (2 * self.rel_action_std)
        ).float()

        # Compute normalized states.
        if self.obs_mode == "state":
            norm_states = (states - self.state_mean) / (2 * self.state_std)
            norm_states = norm_states.float()

            return norm_states, norm_abs_actions, norm_rel_actions

        # Render images.
        elif self.obs_mode == "rgb":
            images = render_demonstration(self.env.env.env.env, states)

            return images, norm_abs_actions, norm_rel_actions


def render_demonstration(env: PushTEnv, states: torch.Tensor) -> list[np.ndarray]:
    # Creates a video ndarray in [T, H, W, C] int8 format.
    images = []

    env.reset()
    for state in states:
        env._set_state(state.detach().cpu().numpy())
        images.append(env.render())

    return images


def save_video(filename: str, video):
    # Save video.
    container = av.open(filename, mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width = 512
    stream.height = 512
    for frame in video:
        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main():
    demos = load_demonstrations()
    dataset = Dataset(demos, h=32, device=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

    model = StateObservationTransformer(
        embedding_dim=256, layers=6, heads=8, sigma_data=0.5
    ).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    buckets = torch.tensor(
        [
            0,
            0.2,
            0.4,
            1,
            3,
            9,
            27,
            81,
        ]
    )
    state_losses_by_bucket = {}
    state_loss_steps_by_bucket = {}
    action_losses_by_bucket = {}
    action_loss_steps_by_bucket = {}
    seen = 0

    env = gym.make(
        "gym_pusht/PushT-v0",
        render_mode="rgb_array",
        visualization_width=512,
        visualization_height=512,
    )

    model.load_state_dict(torch.load("transformer_model_epoch_50.pth"))

    epoch = 50
    for _ in range(100):
        for norm_states, norm_abs_actions, norm_rel_actions in dataloader:
            seen += 1

            loss = model.get_loss(norm_states, norm_abs_actions)

            optimizer.zero_grad()
            loss["total"].backward()
            optimizer.step()

            for b in range(len(buckets) - 1):
                bucket_min = buckets[b]
                bucket_max = buckets[b + 1]
                mask = (loss["sigma"][:] >= bucket_min) & (
                    loss["sigma"][:] < bucket_max
                )
                if mask.sum() > 0:
                    if b not in state_losses_by_bucket:
                        state_losses_by_bucket[b] = []
                        action_losses_by_bucket[b] = []
                        state_loss_steps_by_bucket[b] = []
                        action_loss_steps_by_bucket[b] = []

                    state_losses_by_bucket[b].append(
                        (loss["state_loss"] * mask).sum().item() / mask.sum().item()
                    )
                    action_losses_by_bucket[b].append(
                        (loss["action_loss"] * mask).sum().item() / mask.sum().item()
                    )
                    state_loss_steps_by_bucket[b].append(seen)
                    action_loss_steps_by_bucket[b].append(seen)

            if seen % 100 == 0:
                logger.info(f"Step {seen}, Loss: {loss['total'].item()}")

        # Every so often, we want to visualize samples from the model.
        # These samples can be rendered as videos and then saved. The
        # rendering can be done directly by the environment.
        for seed in range(5):
            states, actions = model.sample(
                horizon=32, N=50, deterministic=True, initial_noise_seed=seed
            )

            # Denormalize states
            states = states.squeeze(0)
            states = states * (2 * dataset.state_std) + dataset.state_mean

            # pusht environment has several wrappers
            video = render_demonstration(env.env.env.env, states)  # type: ignore
            save_video(f"sample_epoch={epoch}_seed={seed}.mp4", video)

        epoch += 1

    for i in sorted(state_losses_by_bucket):
        plt.title("State loss for bucket")
        plt.plot(
            state_loss_steps_by_bucket[i],
            state_losses_by_bucket[i],
            label=rf"$\sigma \in [{buckets[i]:.2f}, {buckets[i+1]:.2f}]$",
        )
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"state_loss_bucket_{i}.png")
        plt.clf()

    for i in sorted(action_losses_by_bucket):
        plt.title("Action loss for bucket")
        plt.plot(
            action_loss_steps_by_bucket[i],
            action_losses_by_bucket[i],
            label=rf"$\sigma \in [{buckets[i]:.2f}, {buckets[i+1]:.2f}]$",
        )
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"action_loss_bucket_{i}.png")
        plt.clf()

    # Save the model.
    torch.save(model.state_dict(), f"transformer_model_epoch_{epoch}.pth")


def test_tokenizers():
    # Load tokenizers
    from diffusers import AutoencoderKLWan

    wan21_tokenizer = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", low_cpu_mem_usage=True
    ).to(device)
    wan22_tokenizer = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers", subfolder="vae", low_cpu_mem_usage=True
    ).to(device)

    # Get a sample 96x96 video
    demos = load_demonstrations()

    for size in [96, 128, 256, 512]:
        dataset = Dataset(demos, h=32, device=device, obs_mode="rgb", obs_size=size)
        video, abs_act, rel_act = dataset[0]

        save_video(f"sample_{size}x{size}.mp4", video)
        # Convert video to the expected format for Wan: (B, C, T, H, W) normalized to [-1, 1]
        video = (
            torch.from_numpy(np.stack(video, axis=0))
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .float()
            / 127.5
            - 1.0
        )
        video = video.to(device)

        # Create batch size of 32
        # video = video.repeat(32, 1, 1, 1, 1)

        with torch.no_grad():
            t0 = time.time()
            encoded = wan21_tokenizer.encode(video).latent_dist.mode()
            decoded = wan21_tokenizer.decode(encoded).sample
            t1 = time.time()

            print(f"Wan2.1 tokens (T, H, W): {encoded.shape[2:]} (t={t1 - t0:.2f}s)")

        decoded = decoded[:1]

        video_reconstructed = decoded.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() + 1.0
        video_reconstructed = (video_reconstructed * 127.5).astype(np.uint8)
        save_video(f"reconstructed_wan2.1_{size}x{size}.mp4", video_reconstructed)
        with torch.no_grad():
            t0 = time.time()
            encoded = wan22_tokenizer.encode(video).latent_dist.mode()
            decoded = wan22_tokenizer.decode(encoded).sample
            t1 = time.time()

            print(f"Wan2.2 tokens (T, H, W): {encoded.shape[2:]} (t={t1 - t0:.2f}s)")

        decoded = decoded[:1]

        video_reconstructed = decoded.squeeze(0).permute(1, 2, 3, 0).cpu().numpy() + 1.0
        video_reconstructed = (video_reconstructed * 127.5).astype(np.uint8)
        save_video(f"reconstructed_wan2.2_{size}x{size}.mp4", video_reconstructed)


# Train video model.
from diffusers import AutoencoderKLWan, WanTransformer3DModel

wan21_tokenizer = AutoencoderKLWan.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="vae", low_cpu_mem_usage=True
).to(device)
wan21_1_3_b = WanTransformer3DModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer", low_cpu_mem_usage=True
).to(device)

# Sample from the model.
