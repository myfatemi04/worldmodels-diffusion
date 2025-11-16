import io
import time

import av
import gym_pusht
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image


def mppi(H, sigma, temperature):
    mean = np.zeros((H, 2))
    means = [mean]

    for i in range(100):
        actions = mean + np.random.normal(0, sigma, size=(100, H, 2))
        states = np.cumsum(actions, axis=1)

        desired_start_state = np.array([0.0, 0.0])
        desired_end_state = np.array([1.0, 1.0])
        start_state_cost = ((states[:, 0, :] - desired_start_state) ** 2).sum(axis=-1)
        end_state_cost = ((states[:, -1, :] - desired_end_state) ** 2).sum(axis=-1)
        movement_cost = (actions**2).sum(axis=(1, 2))
        costs = start_state_cost + end_state_cost + 1 * movement_cost
        weights = np.exp(-costs / temperature)
        weights /= np.sum(weights)
        mean = np.sum(weights[:, None, None] * actions, axis=0)
        sigma *= 0.95  # Anneal sigma

        means.append(mean)

    return means


def mppi_pusht(H, sigma, temperature):
    env = gym.vector.SyncVectorEnv(
        [
            lambda: gym.make(
                "gym_pusht/PushT-v0", visualization_width=512, visualization_height=512
            )
            for _ in range(100)
        ]
    )
    mean = np.zeros((H, 2))
    means = [mean]

    for i in range(10):
        actions = mean + np.random.normal(0, sigma, size=(100, H, 2))
        obs, infos = env.reset(seed=[1 for _ in range(100)])

        states = []
        rewards = []

        for t in range(H):
            actions_translated = actions[:, t, :] + obs[:, :2]
            obs, reward, terminations, truncations, infos = env.step(actions_translated)
            states.append(obs)
            rewards.append(reward)

        rewards = np.array(rewards).transpose(1, 0).sum(axis=-1)
        weights = np.exp(rewards / temperature)
        weights /= np.sum(weights)
        mean = np.sum(weights[:, None, None] * actions, axis=0)
        sigma *= 0.95  # Anneal sigma

        means.append(mean)

    env.close()

    return means


def render_optimization_process(actions):
    bio = io.BytesIO()
    container = av.open("mppi_trajectory.mp4", mode="w")
    stream = container.add_stream("libx264", rate=30)

    for i, mean in enumerate(actions):
        states = np.cumsum(mean, axis=0)

        plt.plot(states[:, 0], states[:, 1], marker="o")
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Step {}".format(i + 1))
        plt.grid()
        plt.savefig(bio, format="png")
        plt.clf()

        bio.seek(0)

        image = Image.open(bio)

        stream.width = image.width
        stream.height = image.height

        frame = av.VideoFrame.from_image(image)
        for packet in stream.encode(frame):
            container.mux(packet)
        bio.truncate(0)
        bio.seek(0)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def render_optimization_process_pusht(actions):
    env = gym.make(
        "gym_pusht/PushT-v0",
        render_mode="rgb_array",
        visualization_width=512,
        visualization_height=512,
    )

    container = av.open("mppi_trajectory.mp4", mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width = 512
    stream.height = 512

    for i, mean in enumerate(actions):
        obs, info = env.reset(seed=1)
        images = []

        for action in mean:
            action_translated = action + obs[:2]
            obs, reward, terminated, truncated, info = env.step(action_translated)
            image = env.render()
            images.append(image)

        for img in images:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()


t0 = time.time()
actions = mppi_pusht(H=10, sigma=1.0, temperature=0.1)
render_optimization_process_pusht(actions)
t1 = time.time()

# print(f"MPPI took {t1 - t0:.2f} seconds")

# render_optimization_process(actions)
