from typing import cast

import gym_pusht
import gymnasium as gym
import numpy as np
import pygame

env = gym.make(
    "gym_pusht/PushT-v0",
    render_mode="rgb_array",
    visualization_width=512,
    visualization_height=512,
)


observation, info = env.reset()

image = env.render()

# Display the image using pygame
pygame.init()
screen = pygame.display.set_mode((512, 512))
pygame.display.set_caption("PushT Environment")
running = True

action_size = 2.0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = observation[:2]

    # Handle key presses for actions
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        action[1] -= action_size
    if keys[pygame.K_DOWN]:
        action[1] += action_size
    if keys[pygame.K_LEFT]:
        action[0] -= action_size
    if keys[pygame.K_RIGHT]:
        action[0] += action_size
    if keys[pygame.K_q]:
        running = False

    observation, reward, terminated, truncated, info = env.step(action)
    image = cast(np.ndarray, env.render())
    image_chw = np.transpose(image, (1, 0, 2))

    # Plot image
    surf = pygame.surfarray.make_surface(image_chw)
    screen.blit(surf, (0, 0))
    pygame.display.flip()

    # Display the reward
    pygame.display.set_caption(f"Reward: {reward:.2f}")

env.close()

# agent_x, agent_y, block_x, block_y, block_angle = observation

# block_ray_length = 50  # Example length of the ray

# plt.imshow(image, origin="lower")
# plt.scatter([agent_x], [agent_y], c="blue", label="Agent")
# plt.gca().annotate(
#     "",
#     xy=(
#         block_x + np.cos(block_angle) * block_ray_length,
#         block_y + np.sin(block_angle) * block_ray_length,
#     ),
#     xytext=(block_x, block_y),
#     arrowprops=dict(color="red", arrowstyle="->", lw=2),
# )
# plt.show()
