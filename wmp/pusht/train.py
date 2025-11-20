import pickle

import matplotlib.pyplot as plt
import torch

from wmp.pusht.state_obs_transformer import StateObservationTransformer


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


demos = load_demonstrations()
(
    state_mean,
    state_std,
    abs_action_mean,
    abs_action_std,
    rel_action_mean,
    rel_action_std,
) = compute_dataset_statistics(demos)

# To normalize, we will subtract mean and divide by twice std.

transformer = StateObservationTransformer(
    embedding_dim=256, layers=6, heads=8, sigma_data=0.5
)
optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)

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

for epoch in range(1):
    for i in torch.randperm(len(demos))[:1000]:
        states, abs_actions, rel_actions = demos[i]
        # Choose random 32-step segment
        if states.shape[0] <= 32:
            continue

        seen += 1

        start_idx = torch.randint(0, states.shape[0] - 32, (1,)).item()
        states = states[start_idx : start_idx + 32]
        abs_actions = abs_actions[start_idx : start_idx + 32]
        rel_actions = rel_actions[start_idx : start_idx + 32]

        norm_states = (states - state_mean) / (2 * state_std)
        norm_abs_actions = (abs_actions - abs_action_mean) / (2 * abs_action_std)
        norm_rel_actions = (rel_actions - rel_action_mean) / (2 * rel_action_std)
        norm_states = norm_states.unsqueeze(0).float()
        norm_rel_actions = norm_rel_actions.unsqueeze(0).float()

        loss = transformer.get_loss(norm_states, norm_rel_actions)

        optimizer.zero_grad()
        loss["total"].backward()
        optimizer.step()

        for b in range(len(buckets) - 1):
            bucket_min = buckets[b]
            bucket_max = buckets[b + 1]
            mask = (states[:, 0] >= bucket_min) & (states[:, 0] < bucket_max)
            if mask.sum() > 0:
                if b not in state_losses_by_bucket:
                    state_losses_by_bucket[b] = []
                    action_losses_by_bucket[b] = []
                    state_loss_steps_by_bucket[b] = []
                    action_loss_steps_by_bucket[b] = []

                state_losses_by_bucket[b].append(
                    (loss["state_loss"].item() * mask.sum().item()) / mask.sum().item()
                )
                action_losses_by_bucket[b].append(
                    (loss["action_loss"].item() * mask.sum().item()) / mask.sum().item()
                )
                state_loss_steps_by_bucket[b] = seen
                action_loss_steps_by_bucket[b] = seen

        if seen % 100 == 0:
            print(seen)

for i in state_losses_by_bucket:
    plt.plot(
        state_loss_steps_by_bucket[i],
        state_losses_by_bucket[i],
        label=f"State Bucket {i}: {buckets[i]}-{buckets[i+1]}",
    )
plt.title("State Losses by Bucket")
plt.legend()
plt.show()

for i in action_losses_by_bucket:
    plt.plot(
        action_loss_steps_by_bucket[i],
        action_losses_by_bucket[i],
        label=f"Action Bucket {i}: {buckets[i]}-{buckets[i+1]}",
    )
plt.title("Action Losses by Bucket")
plt.legend()
plt.show()
