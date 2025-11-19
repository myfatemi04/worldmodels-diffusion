import io

import matplotlib.pyplot as plt
import numpy as np

try:
    import PIL.Image as Image
    import av
except ImportError:
    av = None
    Image = None
    pass

"""
We will have a moving frame and a rest frame.
During rendering, the contents of the moving frame will be rendered according to the Lorentz transformations.
Depending on the speed of the moving frame, we can determine how far ahead or behind in time we need to render the moving frame's contents.

We can create a set of "events" in the moving frame, representing objects being at a particular place and time. All coordinates, including
the corners of the train car, can be represented in this way. We can track events using world lines. World lines will be kept separate for
the sake of simplicity. The moving frame will be centered such that x' = 0 at the center of the train car, and y' = 0 at the train car's floor.

The Lorentz transformation equations are:

    x' = gamma * (x - vt)
    ct' = gamma * (ct - beta * x)

Where:

    v = the velocity of a fixed point in the moving frame as seen from the rest frame
    beta = v/c
    gamma = 1 / sqrt(1 - beta^2)

Here, we assume (x = 0, t = 0) and (x' = 0, t' = 0) coincide.

To determine what time ranges to render, we note that the minimum t' occurs when t = -t_max and x = x_max (and vice versa):
    t'_min = gamma * (t_min - beta * x_max)
    t'_max = gamma * (t_max - beta * x_min)

Distance is in meters.
Velocity is in units of c.
Time is in units of (1 meter) / c.
Thus, dx = v * dt, where v is in units of c.

To start at the left of the frame, the train car will have moved X_MAX + TRAIN_WIDTH/2 meters to the left (and vice versa).
"""

V = 0.6
BETA = V
GAMMA = 1 / np.sqrt(1 - V**2)

TRAIN_PROPER_WIDTH = 5
TRAIN_OBSERVER_WIDTH = TRAIN_PROPER_WIDTH / GAMMA
TRAIN_HEIGHT = 3
X_MAX = 10
X_MIN = -10

T_MAX = (X_MAX + TRAIN_OBSERVER_WIDTH) / V
T_MIN = -(X_MAX + TRAIN_OBSERVER_WIDTH) / V
T_PRIME_MIN = GAMMA * (T_MIN - BETA * X_MAX)
T_PRIME_MAX = GAMMA * (T_MAX - BETA * X_MIN)


def generate_moving_frame_photon_example_events(photon_movement_style="reflections"):
    # Events expressed in (t', x', y') coordinates, in the moving frame.
    events = {
        "train_topleft_corner": [],
        "train_topright_corner": [],
        "train_bottomleft_corner": [],
        "train_bottomright_corner": [],
        "photon_position": [],
    }

    # t = np.linspace(T_PRIME_MIN, T_PRIME_MAX, 500)
    # wave = (TRAIN_PROPER_WIDTH / 2) * (
    #     2 * np.abs(2 * (((t / (TRAIN_PROPER_WIDTH / 2)) * 0.5 - 0.5) % 1) - 1) - 1
    # )
    # plt.plot(t, wave)
    # plt.show()

    for t in np.linspace(T_PRIME_MIN, T_PRIME_MAX, 500):
        # Train corners are at the same position at all times.
        events["train_topleft_corner"].append(
            (t, -TRAIN_PROPER_WIDTH / 2, TRAIN_HEIGHT)
        )
        events["train_topright_corner"].append(
            (t, TRAIN_PROPER_WIDTH / 2, TRAIN_HEIGHT)
        )
        events["train_bottomleft_corner"].append((t, -TRAIN_PROPER_WIDTH / 2, 0))
        events["train_bottomright_corner"].append((t, TRAIN_PROPER_WIDTH / 2, 0))

        # Photon moves from left to right at speed c in the moving frame.
        wave = (TRAIN_PROPER_WIDTH / 2) * (
            2 * np.abs(2 * (((t / (TRAIN_PROPER_WIDTH)) * 0.5 - 0.5) % 1) - 1) - 1
        )

        events["photon_position"].append(
            (
                t,
                # Use triangle wave.
                (
                    wave
                    if photon_movement_style == "reflections"
                    else np.clip(
                        TRAIN_PROPER_WIDTH - t,
                        -TRAIN_PROPER_WIDTH / 2,
                        TRAIN_PROPER_WIDTH / 2,
                    )
                ),
                TRAIN_HEIGHT / 2,
            )
        )

    events = {k: np.array(v) for k, v in events.items()}

    return events


def generate_falling_balls_example_events():
    events = {
        "left_ball": [],
        "right_ball": [],
        "left_photon": [],
        "right_photon": [],
        "train_topleft_corner": [],
        "train_topright_corner": [],
        "train_bottomleft_corner": [],
        "train_bottomright_corner": [],
    }

    # Initial period: photons come out from center. Takes delta-T = TRAIN_PROPER_WIDTH / 2.
    base_t = 5

    # Add some padding time beforehand.
    for t in np.linspace(-5, 0, 20):
        t = t - base_t
        events["left_ball"].append((t, -TRAIN_PROPER_WIDTH / 2, TRAIN_PROPER_WIDTH / 2))
        events["right_ball"].append((t, TRAIN_PROPER_WIDTH / 2, TRAIN_PROPER_WIDTH / 2))
        events["left_photon"].append((t, -0, TRAIN_PROPER_WIDTH / 2))
        events["right_photon"].append((t, 0, TRAIN_PROPER_WIDTH / 2))

        # Train corners are at the same position at all times.
        events["train_topleft_corner"].append(
            (t, -TRAIN_PROPER_WIDTH / 2, TRAIN_HEIGHT)
        )
        events["train_topright_corner"].append(
            (t, TRAIN_PROPER_WIDTH / 2, TRAIN_HEIGHT)
        )
        events["train_bottomleft_corner"].append((t, -TRAIN_PROPER_WIDTH / 2, 0))
        events["train_bottomright_corner"].append((t, TRAIN_PROPER_WIDTH / 2, 0))

    for t in np.linspace(0, TRAIN_PROPER_WIDTH / 2, 50):
        # Make the end of this event correspond to t = 0.
        photon_t = t
        t = t - base_t
        events["left_ball"].append((t, -TRAIN_PROPER_WIDTH / 2, TRAIN_PROPER_WIDTH / 2))
        events["right_ball"].append((t, TRAIN_PROPER_WIDTH / 2, TRAIN_PROPER_WIDTH / 2))
        events["left_photon"].append((t, -photon_t, TRAIN_PROPER_WIDTH / 2))
        events["right_photon"].append((t, photon_t, TRAIN_PROPER_WIDTH / 2))

        # Train corners are at the same position at all times.
        events["train_topleft_corner"].append(
            (t, -TRAIN_PROPER_WIDTH / 2, TRAIN_HEIGHT)
        )
        events["train_topright_corner"].append(
            (t, TRAIN_PROPER_WIDTH / 2, TRAIN_HEIGHT)
        )
        events["train_bottomleft_corner"].append((t, -TRAIN_PROPER_WIDTH / 2, 0))
        events["train_bottomright_corner"].append((t, TRAIN_PROPER_WIDTH / 2, 0))

    """
    Falling period: balls roll towards the center. Classical mechanics tells us:

    theta'' = -u * sin(theta)
    u = 2/3 * (g / r)

    Where the 2/3 comes from the rotational kinetic energy in the ball's moment of inertia. In practice, I just want to set u directly,
    so that the balls fall quickly enough. The model actually is slightly incorrect because if the balls fall quickly enough, relativistic
    effects should be considered. But I'll just kinda ignore that for now.
    """

    t = TRAIN_PROPER_WIDTH / 2 - base_t
    animation_duration = 15
    dt = 1 / 40
    left_ball_theta = np.pi / 2
    right_ball_theta = -np.pi / 2
    left_ball_theta_dot = 0
    right_ball_theta_dot = 0

    u = 0.5

    for _ in range(int(animation_duration / dt)):
        left_ball_theta_dot -= u * np.sin(left_ball_theta) * dt
        right_ball_theta_dot -= u * np.sin(right_ball_theta) * dt
        left_ball_theta += left_ball_theta_dot * dt
        right_ball_theta += right_ball_theta_dot * dt
        left_ball_theta_dot = left_ball_theta_dot * 0.98
        right_ball_theta_dot = right_ball_theta_dot * 0.98

        left_ball_x = -(TRAIN_PROPER_WIDTH / 2) * np.sin(left_ball_theta)
        right_ball_x = -(TRAIN_PROPER_WIDTH / 2) * np.sin(right_ball_theta)
        left_ball_y = (
            -(TRAIN_PROPER_WIDTH / 2) * np.cos(left_ball_theta) + TRAIN_PROPER_WIDTH / 2
        )
        right_ball_y = (
            -(TRAIN_PROPER_WIDTH / 2) * np.cos(right_ball_theta)
            + TRAIN_PROPER_WIDTH / 2
        )

        events["left_ball"].append((t, left_ball_x, left_ball_y))
        events["right_ball"].append((t, right_ball_x, right_ball_y))
        events["left_photon"].append(
            (t, -TRAIN_PROPER_WIDTH / 2, TRAIN_PROPER_WIDTH / 2)
        )
        events["right_photon"].append(
            (t, TRAIN_PROPER_WIDTH / 2, TRAIN_PROPER_WIDTH / 2)
        )

        events["train_topleft_corner"].append(
            (t, -TRAIN_PROPER_WIDTH / 2, TRAIN_HEIGHT)
        )
        events["train_topright_corner"].append(
            (t, TRAIN_PROPER_WIDTH / 2, TRAIN_HEIGHT)
        )
        events["train_bottomleft_corner"].append((t, -TRAIN_PROPER_WIDTH / 2, 0))
        events["train_bottomright_corner"].append((t, TRAIN_PROPER_WIDTH / 2, 0))

        t += dt

    events = {k: np.array(v) for k, v in events.items()}
    return events


def inverse_lorentz_transform(moving_frame_events, v):
    beta = v
    gamma = 1 / np.sqrt(1 - beta**2)
    transformed_events = {}

    for k, worldline in moving_frame_events.items():
        transformed_events[k] = np.array(
            [
                # An interesting symmetry emerges when t is in units of c...
                [gamma * (t + beta * x), gamma * (x + beta * t), y]
                for t, x, y in worldline
            ]
        )

    return transformed_events


def interpolate_worldlines(events, t):
    """Worldlines and t are assumed to be for the same frame."""

    interpolated_events = {}

    for k, worldline in events.items():
        times = worldline[:, 0]
        xs = worldline[:, 1]
        ys = worldline[:, 2]

        interp_x = np.interp(t, times, xs)
        interp_y = np.interp(t, times, ys)

        interpolated_events[k] = (t, interp_x, interp_y)

    interpolated_events = {k: np.array(v) for k, v in interpolated_events.items()}

    return interpolated_events


def plot_world_lines(events):
    # Plot the worldlines of the events.
    plt.figure(figsize=(10, 10))
    plt.title("Worldlines")
    plt.xlim(X_MIN - 1, X_MAX + 1)
    plt.ylim(T_MIN - 1, T_MAX + 1)

    # Plot light cone
    plt.plot([T_MIN - 1, T_MAX + 1], [T_MIN - 1, T_MAX + 1], "k--", c="gray")
    plt.plot(
        [T_MIN - 1, T_MAX + 1],
        [T_MAX + 1, T_MIN - 1],
        "k--",
        c="gray",
        label="Light cone",
    )

    for k, worldline in events.items():
        plt.plot(worldline[:, 1], worldline[:, 0], label=k)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("Position (m)")
    plt.ylabel("Time (m/c)")

    plt.legend()
    plt.show()


def plot_events(events):
    plt.figure(figsize=(10, 6))
    plt.title("Rest Frame View of Moving Train and Photon")
    plt.xlim(X_MIN - 1, X_MAX + 1)
    plt.ylim(-1, TRAIN_HEIGHT + 1)
    plt.axhline(0, color="black", linewidth=0.5)

    for t in np.linspace(-T_MAX, T_MAX, 50):
        plt.xlim(X_MIN - 1, X_MAX + 1)

        resolved = interpolate_worldlines(events, t)

        # Plot train
        train_xs = [
            resolved["train_bottomleft_corner"][1],
            resolved["train_bottomright_corner"][1],
            resolved["train_topright_corner"][1],
            resolved["train_topleft_corner"][1],
            resolved["train_bottomleft_corner"][1],
        ]
        train_ys = [
            resolved["train_bottomleft_corner"][2],
            resolved["train_bottomright_corner"][2],
            resolved["train_topright_corner"][2],
            resolved["train_topleft_corner"][2],
            resolved["train_bottomleft_corner"][2],
        ]
        plt.plot(train_xs, train_ys, color="blue", linewidth=3)

        # Set axes to be equal
        plt.gca().set_aspect("equal", adjustable="box")

        # Plot photon
        photon_x = resolved["photon_position"][1]
        photon_y = resolved["photon_position"][2]
        plt.plot(photon_x, photon_y, "ro", label="Photon")

        plt.legend()
        plt.xlabel("Position (m)")
        plt.ylabel("Height (m)")
        plt.grid()
        plt.show()


def plot_events_with_ball(events, title):
    # plt.figure(figsize=(10, 10))
    # plt.xlim(X_MIN - 1, X_MAX + 1)
    # plt.ylim(-1, TRAIN_HEIGHT + 1)
    # plt.axhline(0, color="black", linewidth=0.5)

    bio = io.BytesIO()
    imgs = []

    for t in np.linspace(-T_MAX, T_MAX, 50):
        plt.figure(figsize=(10, 10))

        plt.title("Consistency of simultaneity, v={}".format(title))

        plt.xlim(X_MIN - 1, X_MAX + 1)
        plt.ylim(-1, TRAIN_HEIGHT * 3 + 1)

        resolved = interpolate_worldlines(events, t)

        # Plot train
        train_xs = [
            resolved["train_bottomleft_corner"][1],
            resolved["train_bottomright_corner"][1],
            resolved["train_topright_corner"][1],
            resolved["train_topleft_corner"][1],
            resolved["train_bottomleft_corner"][1],
        ]
        train_ys = [
            resolved["train_bottomleft_corner"][2],
            resolved["train_bottomright_corner"][2],
            resolved["train_topright_corner"][2],
            resolved["train_topleft_corner"][2],
            resolved["train_bottomleft_corner"][2],
        ]
        plt.plot(train_xs, train_ys, color="gold", linewidth=3)

        # Draw arc that the balls fall along
        theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
        train_center_x = (
            resolved["train_bottomleft_corner"][1]
            + resolved["train_bottomright_corner"][1]
        ) / 2
        train_center_y = (
            resolved["train_topleft_corner"][2] + resolved["train_bottomleft_corner"][2]
        ) / 2
        train_width = (
            resolved["train_bottomright_corner"][1]
            - resolved["train_bottomleft_corner"][1]
        )
        start_height = resolved["left_photon"][2]

        arc_x = train_center_x - (train_width / 2) * np.sin(theta)
        arc_y = -(start_height) * np.cos(theta) + start_height

        plt.plot(arc_x, arc_y, c="gray")

        # Set axes to be equal
        plt.gca().set_aspect("equal", adjustable="box")

        # Plot left photon and right photon
        left_photon_x = resolved["left_photon"][1]
        left_photon_y = resolved["left_photon"][2]
        right_photon_x = resolved["right_photon"][1]
        right_photon_y = resolved["right_photon"][2]
        plt.scatter(left_photon_x, left_photon_y, label="Left Photon")
        plt.scatter(right_photon_x, right_photon_y, label="Right Photon")

        # Plot left ball and right ball
        left_ball_x = resolved["left_ball"][1]
        left_ball_y = resolved["left_ball"][2]
        right_ball_x = resolved["right_ball"][1]
        right_ball_y = resolved["right_ball"][2]
        plt.scatter(left_ball_x, left_ball_y, label="Left Ball")
        plt.scatter(right_ball_x, right_ball_y, label="Right Ball")

        plt.legend()
        plt.xlabel("Position (m)")
        plt.ylabel("Height (m)")
        plt.grid()

        if Image is not None:
            plt.savefig(bio, format="png")

            bio.seek(0)
            imgs.append(Image.open(bio).copy())

            bio.truncate(0)
            bio.seek(0)
        else:
            plt.show()

    if av is not None and len(imgs) > 0:
        output = av.open(f"falling_balls_simulation_{title}.mp4", mode="w")
        stream = output.add_stream("libx264", rate=10)
        stream.width = imgs[0].width
        stream.height = imgs[0].height
        stream.pix_fmt = "yuv420p"

        for img in imgs:
            frame = av.VideoFrame.from_image(img)
            packet = stream.encode(frame)
            if packet:
                output.mux(packet)

        # Flush stream
        packet = stream.encode(None)
        if packet:
            output.mux(packet)

        output.close()


# train_example = generate_moving_frame_photon_example_events()

# rest_frame_events = inverse_lorentz_transform(train_example, V)

# plot_world_lines(inverse_lorentz_transform(train_example, 0))
# plot_world_lines(inverse_lorentz_transform(train_example, 0.5 * V))
# plot_world_lines(inverse_lorentz_transform(train_example, V))

# plot_events(rest_frame_events)

T_MAX = 10
T_MIN = -10

train_example_with_balls = generate_falling_balls_example_events()

# plot_world_lines(inverse_lorentz_transform(train_example_with_balls, 0))
# plot_world_lines(inverse_lorentz_transform(train_example_with_balls, 0.5 * V))
# plot_world_lines(inverse_lorentz_transform(train_example_with_balls, V))

# plot_events_with_ball(
#     inverse_lorentz_transform(train_example_with_balls, 0), title="normal"
# )
# plot_events_with_ball(
#     inverse_lorentz_transform(train_example_with_balls, 0.5 * V), title="0.3c"
# )
# plot_events_with_ball(
#     inverse_lorentz_transform(train_example_with_balls, V), title="0.6c"
# )


plot_events_with_ball(
    inverse_lorentz_transform(train_example_with_balls, 0.9), title="0.9c"
)
