"""
V-JEPA2
"""

import time
from typing import cast

import av
import torch
import torch.nn.functional as F
from transformers import VJEPA2Model, VJEPA2VideoProcessor


class SimultaneousDiffuser:
    pass


def load_video(path):
    with av.open(path) as container:
        return [frame.to_image() for frame in container.decode(video=0)]


def main():
    device = "cpu"
    ckpt = "facebook/vjepa2-vitl-fpc64-256"
    # model = VJEPA2Model.from_pretrained(ckpt).to(device)
    processor = cast(VJEPA2VideoProcessor, VJEPA2VideoProcessor.from_pretrained(ckpt))

    all_abs_actions = torch.load("./data/pusht_noise/train/abs_actions.pth")

    # load a video, 16x256x256x3 float32, and the corresponding actions
    obs = load_video("./data/pusht_noise/train/obses/episode_000.mp4")
    actions = all_abs_actions[0][: len(obs)]
    obs = processor(obs, return_tensors="pt").to(device)
    pv = obs["pixel_values_videos"]

    print(pv.shape, actions.shape)


if __name__ == "__main__":
    main()
