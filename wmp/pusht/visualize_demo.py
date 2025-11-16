"""
This test demonstrates the ability to interpret states from PushT.
"""

import pickle
from typing import cast

import av
import gym_pusht
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym_pusht.envs.pusht import PushTEnv


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


def render_demonstration(env: PushTEnv, states: torch.Tensor) -> list[np.ndarray]:
    # Creates a video ndarray in [T, H, W, C] int8 format.

    images = []

    env.reset()
    for state in states:
        env._set_state(state.numpy())
        images.append(env.render())

    return images


if __name__ == "__main__":
    env = gym.make(
        "gym_pusht/PushT-v0",
        render_mode="rgb_array",
        visualization_width=512,
        visualization_height=512,
    )

    demos = load_demonstrations()
    states, abs_actions, rel_actions = demos[0]

    video = render_demonstration(env.env.env.env, states)  # type: ignore

    # Save video.
    container = av.open("pusht_demo.mp4", mode="w")
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
