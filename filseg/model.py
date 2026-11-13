"""The segmentation network and its weights.

A U-Net (segmentation_models_pytorch) with a ConvNeXt-T encoder pretrained on ImageNet, one grey
input channel and two outputs: channel 0 is the filament probability, channel 1 an auxiliary
centre line that is only used in training, to keep thin filaments in one piece.
"""
from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


class FilamentNet(nn.Module):
    def __init__(self, encoder: str, pretrained: bool = True):
        super().__init__()
        self.net = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                            in_channels=1, classes=2)
        # 8-bit grey frames scaled to [0, 1], then a fixed mean and spread for all frames
        self.register_buffer("mean", torch.tensor(0.33).view(1, 1, 1, 1))
        self.register_buffer("std", torch.tensor(0.20).view(1, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)

    def param_groups(self, lr: float, encoder_mult: float, weight_decay: float) -> list[dict]:
        encoder = list(self.net.encoder.parameters())
        ids = {id(p) for p in encoder}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [{"params": encoder, "lr": lr * encoder_mult, "weight_decay": weight_decay},
                {"params": rest, "lr": lr, "weight_decay": weight_decay}]


def save_weights(net: nn.Module, path: str | Path) -> None:
    """Half precision, which is plenty for inference and halves the file."""
    state = {k: (v.half() if v.is_floating_point() else v).contiguous().cpu() for k, v in net.state_dict().items()}
    save_file(state, str(path))


def load_net(cfg: dict, path: str | Path, device: str = "cuda") -> FilamentNet:
    net = FilamentNet(cfg["model"]["encoder"], pretrained=False)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in load_file(str(path)).items()})
    return net.to(device).eval().to(memory_format=torch.channels_last)
