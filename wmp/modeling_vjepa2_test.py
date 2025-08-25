from typing import cast

import torch
from transformers.models.vjepa2.configuration_vjepa2 import VJEPA2Config

from wmp.modeling_vjepa2 import (
    VJEPA2JointDiffuser,
    VJEPA2JointDiffuserLayer,
    VJEPA2JointDiffuserRopeAttention,
)

config = VJEPA2Config.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
config._attn_implementation = "eager"
diffuser = VJEPA2JointDiffuser(config, action_dim=2)
attn = VJEPA2JointDiffuserRopeAttention(
    config,
    hidden_size=1024,
    num_attention_heads=16,
)
# 8 frames
# 4 actions per frame, so (4 - 1) * 4 + 1 = 13 actions in total
# note, frames are encoded in pairs
attn_output = attn(
    video_hidden_states=torch.randn(1, 1024, 1024),
    action_hidden_states=torch.randn(1, 13, 1024),
)
layer = cast(VJEPA2JointDiffuserLayer, diffuser.layer[0])
layer_output = layer(
    video_hidden_states=torch.randn(1, 1024, 1024),
    action_hidden_states=torch.randn(1, 13, 1024),
    video_modulation=diffuser.video_modulation(torch.rand((1,))),
    action_modulation=diffuser.action_modulation(torch.rand((1,))),
)
diffuser_output = diffuser(
    pixel_values_videos=torch.randn(1, 8, 3, 256, 256),
    actions=torch.randn(1, 13, 2),
    video_t=torch.rand((1,)),
    actions_t=torch.rand((1,)),
)
