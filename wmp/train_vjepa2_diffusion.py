import os
from typing import cast

import av
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from transformers import VJEPA2Model, VJEPA2VideoProcessor

from wmp.modeling_vjepa2 import VJEPA2JointDiffuser


def load_video(path):
    with av.open(path) as container:
        return [frame.to_image() for frame in container.decode(video=0)]  # type: ignore


# https://github.com/Wan-Video/Wan2.1/blob/main/wan/utils/fm_solvers.py
def get_sampling_sigmas(sampling_steps, shift):
    sigma = torch.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)

    return sigma


def logit_normal_distribution(x, mu, sigma):
    logit_x = torch.log(x / (1 - x + 1e-8))
    return (
        1
        / (sigma * ((2 * torch.pi) ** 0.5) * (x * (1 - x) + 1e-8))
        * torch.exp(-0.5 * ((logit_x - mu) / sigma) ** 2)
    )


def main():
    device = "cpu"
    ckpt = "facebook/vjepa2-vitl-fpc64-256"
    original_model = VJEPA2Model.from_pretrained(ckpt).to(device)  # type: ignore
    model = VJEPA2JointDiffuser(config=original_model.config, action_dim=2)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.load_state_dict(torch.load("checkpoints_0/model_600.pth"))

    # missing_keys, unexpected_keys = model.load_state_dict(
    #     original_model.encoder.state_dict(), strict=False
    # )
    # if len(unexpected_keys) > 0:
    #     print(unexpected_keys)
    # else:
    #     print("No unexpected keys")

    processor = cast(VJEPA2VideoProcessor, VJEPA2VideoProcessor.from_pretrained(ckpt))

    # overfit to a single example

    all_abs_actions = torch.load("./data/pusht_noise/train/abs_actions.pth")

    # scale from [-200, 800] to [-1, 1]
    all_abs_actions = (all_abs_actions - (-200)) / (800 - (-200)) * 2 - 1

    # load a video, 16x256x256x3 float32, and the corresponding actions
    obs = load_video("./data/pusht_noise/train/obses/episode_000.mp4")
    obs = obs[:8]

    writer = SummaryWriter()

    num_condition_frames = 2
    action = (
        all_abs_actions[0][num_condition_frames - 1 : len(obs)]
        .to(torch.float32)
        .unsqueeze(0)
    )
    obs = processor(obs, return_tensors="pt").to(device)
    pv = obs["pixel_values_videos"]

    with torch.no_grad():
        video_latents = original_model(
            pixel_values_videos=pv, skip_predictor=True
        ).last_hidden_state

    sigmas = get_sampling_sigmas(sampling_steps=8, shift=1)
    sigma_probs = logit_normal_distribution(sigmas, mu=0, sigma=1)

    for i in range(10000):
        sigma_index = torch.multinomial(sigma_probs, num_samples=1, replacement=False)
        sigma = sigmas[sigma_index]
        alpha = 1 - sigma

        video_latents_noise = torch.randn_like(video_latents)
        video_latents_noised = alpha * video_latents + sigma * video_latents_noise
        action_noise = torch.randn_like(action)
        action_noised = alpha * action + sigma * action_noise
        video_latents_flow = video_latents_noise - video_latents
        action_flow = action - action_noise
        video_flow_pred, actions_flow_pred = model(
            video_latents=video_latents_noised,
            action=action_noised,
            video_t=alpha,
            action_t=alpha,
            video_fps=1.0,
            action_fps=1.0,
            num_condition_frames=num_condition_frames,
        )
        video_flow_error = F.mse_loss(video_latents_flow, video_flow_pred)
        action_flow_error = F.mse_loss(action_flow, actions_flow_pred)
        loss = video_flow_error + action_flow_error
        loss.backward()
        optim.step()
        optim.zero_grad()

        writer.add_scalar("loss/total", loss.item(), i)
        writer.add_scalar("loss/video", video_flow_error.item(), i)
        writer.add_scalar("loss/actions", action_flow_error.item(), i)
        writer.flush()

        logger.info(
            f"Step {i}: Loss =  {video_flow_error.item()} (video) + {action_flow_error.item()} (action) = {loss.item()}"
        )

        if (i + 1) % 500 == 0:
            # save checkpoint
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"./checkpoints/model_{i + 1}.pth")


if __name__ == "__main__":
    main()
