# pip install av loguru matplotlib gymnasium gym-pusht "pymunk<7"
# apt install libgl1-mesa-glx

import pickle

import matplotlib.pyplot as plt
import torch

from wmp.pusht.state_obs_transformer import StateObservationTransformer
from loguru import logger
import gymnasium as gym
from gym_pusht.envs.pusht import PushTEnv
import numpy as np
import av


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
    def __init__(self, demos, h, device):
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

        norm_states = (states - self.state_mean) / (2 * self.state_std)
        norm_abs_actions = (abs_actions - self.abs_action_mean) / (
            2 * self.abs_action_std
        )
        norm_rel_actions = (rel_actions - self.rel_action_mean) / (
            2 * self.rel_action_std
        )
        norm_states = norm_states.float()
        norm_rel_actions = norm_rel_actions.float()
        norm_abs_actions = norm_abs_actions.float()

        return norm_states, norm_abs_actions, norm_rel_actions


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


if __name__ == "__main__":
    main()
