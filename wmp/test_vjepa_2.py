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
        8 frames (1024 tokens): 1.2s
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
        8 frames (1024 tokens): 0.3s
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

When using a batch size of 1 and torch.compile, we get:

T4 GPU:
    forward pass:
        8 frames (1024 tokens):
    forward + backward + optimizer step:
        8 frames (1024 tokens):


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
    ckpt = "facebook/vjepa2-vitl-fpc64-256"
    model = VJEPA2Model.from_pretrained(ckpt)
    processor = cast(VJEPA2VideoProcessor, VJEPA2VideoProcessor.from_pretrained(ckpt))

    # load a video, 16x256x256x3 float32
    original_video = load_video("./data/pusht_noise/train/obses/episode_000.mp4")

    num_frames = [8, 16]
    do_forward_pass = False
    do_backward_pass = True

    for num_frames in [8, 16]:
        video = original_video[:num_frames]
        encoded = processor(video, return_tensors="pt")

        # skipping because of the following:
        # torch._inductor.exc.InductorError: ImportError: dlopen(/var/folders/pt/45xmzdh176jcsmxv9vb4by_c0000gn/T/torchinductor_michaelfatemi/kx/ckxhkaj6ldrff7qilnpkdfvtezlhokilim5t76qkkjafvjnhabbk.main.so, 0x0002): Library not loaded: @rpath/libc++.1.dylib
        # model = torch.compile(model)

        # test speed for forward pass
        if do_forward_pass:
            with torch.no_grad():
                # pass a dummy batch through it to load things up
                t0 = time.time()
                output = model(
                    pixel_values_videos=torch.randn_like(encoded["pixel_values_videos"])
                )
                t1 = time.time()

                print(f"First forward pass in {t1 - t0:.3f}s")

                # let's try encoding a video. use random batches every time to avoid caching issues.
                for _ in range(10):
                    t0 = time.time()
                    output = model(
                        pixel_values_videos=torch.randn_like(
                            encoded["pixel_values_videos"]
                        )
                    )
                    t1 = time.time()

                    print(f"Forward pass in {t1 - t0:.3f}s")

        if do_backward_pass:
            # test speed for forward+backward pass
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            # pass a dummy batch through it to load things up
            t0 = time.time()
            output = model(
                pixel_values_videos=torch.randn_like(encoded["pixel_values_videos"])
            )
            labels = torch.randn((1, 128 * len(video), 1024))
            loss = F.mse_loss(output.last_hidden_state, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t1 = time.time()

            print(f"First forward+backward+optimizer step in {t1 - t0:.3f}s")

            # let's try encoding a video. use random batches every time to avoid caching issues.
            for _ in range(10):
                t0 = time.time()
                output = model(
                    pixel_values_videos=torch.randn_like(encoded["pixel_values_videos"])
                )
                labels = torch.randn((1, 128 * len(video), 1024))
                loss = F.mse_loss(output.last_hidden_state, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                t1 = time.time()

                print(f"Forward forward+backward+optimizer step in {t1 - t0:.3f}s")


if __name__ == "__main__":
    main()
