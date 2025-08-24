"""
a model that predicts future V-JEPA2 states through diffusion. initialized from original V-JEPA2 checkpoint. uses AdaLN-Zero for modulation.
it would be really nice if it were trainable on a macbook pro. why not try?

also, will it actually be difficult to adapt the existing model for diffusion? maybe it depends on the architecture used. we could also use LoRA
or ReLoRA for training.

note that each training step will require *two* forward passes (one through the original model to get labels, and another through the trained model to get predictions).
however, it's possible that running through the original model to get labels can be performed once in a single preprocessing step.

There is lots of variability. Maybe related to scheduler?

When using a batch size of 1,

mps:
    forward pass:
        8 frames (1024 tokens): 1.2s -> 0.87s without predictor
        16 frames (2048 tokens): 3.1s
    forward + backward + optimizer step:
        8 frames (1024 tokens):
            8.181s
            6.180s
            6.971s
            6.148s
            3.260s
            3.336s
            4.476s
            8.290s
            7.187s
            5.951s
        16 frames (2048 tokens):
            11.404s
            12.366s
            11.628s
            13.210s
            11.217s
            10.005s
            27.902s
            18.821s
            16.388s
            15.020s

T4 GPU:
    forward pass:
        8 frames (1024 tokens): 0.3s -> faster without predictor?
        16 frames (2048 tokens): 0.73s
    forward + backward + optimizer step:
        8 frames (1024 tokens): 0.89s
        16 frames (2048 tokens): 2.11s

When using a batch size of 4, there is a very modest speedup beyond linear (but then we OOM):

T4 GPU:
    forward pass:
        8 frames (1024 tokens): 1.2s
    forward + backward + optimizer step:
        8 frames (1024 tokens): 3.075s

torch.compile didn't work on Colab or locally, so not testing for now.

oh wait, also, V-JEPA2 uses the "predictor" by default, which is a second forward pass.

"""

import time
from typing import cast

import av
import torch
import torch.nn.functional as F
from transformers import VJEPA2Model, VJEPA2VideoProcessor


def load_video(path):
    with av.open(path) as container:
        return [frame.to_image() for frame in container.decode(video=0)]


def main():
    device = "cpu"
    ckpt = "facebook/vjepa2-vitl-fpc64-256"
    model = VJEPA2Model.from_pretrained(ckpt).to(device)
    processor = cast(VJEPA2VideoProcessor, VJEPA2VideoProcessor.from_pretrained(ckpt))

    # load a video, 16x256x256x3 float32
    original_video = load_video("./data/pusht_noise/train/obses/episode_000.mp4")

    num_frames = [8, 16]
    do_forward_pass = True
    do_backward_pass = True
    batch_size = 1

    # model = torch.compile(model)

    for num_frames in [8, 16]:
        # encoded = processor(video, return_tensors="pt")
        encoded = {
            "pixel_values_videos": torch.randn(
                (batch_size, num_frames, 3, 256, 256), device=device
            )
        }

        # test speed for forward pass
        if do_forward_pass:
            with torch.no_grad():
                for i in range(11):
                    t0 = time.time()
                    output = model(
                        pixel_values_videos=torch.randn_like(
                            encoded["pixel_values_videos"], device=device
                        ),
                        skip_predictor=True,
                    )
                    t1 = time.time()

                    if i == 0:
                        print(f"First forward pass in {t1 - t0:.3f}s")
                    else:
                        print(f"Forward pass in {t1 - t0:.3f}s")

        if do_backward_pass:
            # test speed for forward+backward pass
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            for i in range(11):
                t0 = time.time()
                output = model(
                    pixel_values_videos=torch.randn_like(
                        encoded["pixel_values_videos"], device=device
                    ),
                    skip_predictor=True,
                )
                labels = torch.randn(
                    (encoded["pixel_values_videos"].shape[0], 128 * num_frames, 1024),
                    device=device,
                )
                loss = F.mse_loss(output.last_hidden_state, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                t1 = time.time()

                if i == 0:
                    print(f"First forward+backward+optimizer step in {t1 - t0:.3f}s")
                else:
                    print(f"Forward forward+backward+optimizer step in {t1 - t0:.3f}s")


if __name__ == "__main__":
    main()
