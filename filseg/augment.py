"""Training crops, augmented on the GPU.

All training frames sit on the GPU as uint8, so a batch never touches the disk. One crop is cut
out of a 2048 frame by a single `grid_sample` call that applies the zoom, the rotation and the
position at once. Then, per batch:

    flips and transposes (the Sun has no "up")
    copy-paste of real filaments from other training frames
    photometry: limb darkening, thin cloud, gamma, contrast, brightness, seeing blur, a lower
    resolution, and weak noise

Every one of these is something that really differs between GONG frames. Contrast is never
inverted, because a filament is dark by definition.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .rle import NATIVE


def dihedral(x: torch.Tensor, k: int) -> torch.Tensor:
    """One of the eight flips and rotations of the square (k = 0 is the identity)."""
    if k & 4:
        x = x.transpose(-1, -2)
    if k & 1:
        x = x.flip(-1)
    if k & 2:
        x = x.flip(-2)
    return x


def dihedral_inv(x: torch.Tensor, k: int) -> torch.Tensor:
    if k & 2:
        x = x.flip(-2)
    if k & 1:
        x = x.flip(-1)
    if k & 4:
        x = x.transpose(-1, -2)
    return x


def random_dihedral(x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    """Horizontal flip, vertical flip and transpose, each on or off per sample."""
    x = torch.where(flags[:, 0].view(-1, 1, 1, 1), x.flip(-1), x)
    x = torch.where(flags[:, 1].view(-1, 1, 1, 1), x.flip(-2), x)
    return torch.where(flags[:, 2].view(-1, 1, 1, 1), x.transpose(-1, -2), x)


class CopyPaste:
    """Pastes real filaments of the training frames onto training crops.

    A filament goes in as a darkening ratio, so it takes on the brightness of where it lands. It is
    only pasted on the disk and never over a filament that is already there, so the targets stay
    true.
    """

    def __init__(self, cfg: dict, bank: list[dict], train_frames: np.ndarray, device: str = "cuda",
                 max_side: int = 640, seed: int = 0):
        self.p = float(cfg["p"])
        self.n_lo, self.n_hi = cfg["n"]
        self.rotate = bool(cfg.get("rotate", True))
        allowed = set(int(i) for i in train_frames)            # never paste from a validation frame
        self.bank = []
        for e in bank:
            if e["frame"] in allowed and max(e["soft"].shape) <= max_side:
                ratio = torch.from_numpy(e["ratio"].astype(np.float32)).to(device)
                soft = torch.from_numpy(e["soft"]).to(device).float().div_(255)
                skel = torch.from_numpy(e["skel"]).to(device).float().div_(255)
                own = (soft > 0).float()[None, None]
                alpha = F.avg_pool2d(F.max_pool2d(own, 7, 1, 3), 5, 1, 2)[0, 0]      # widen, then soften the edge
                self.bank.append(torch.stack([ratio, soft, skel, alpha]))
        self.rng = np.random.default_rng(seed)

    def __call__(self, x, soft, skel, rad):
        if not self.bank:
            return x, soft, skel
        b, _, H, W = x.shape
        rng = self.rng
        x, soft, skel = x.clone(), soft.clone(), skel.clone()
        for i in range(b):
            if rng.random() >= self.p:
                continue
            for _ in range(int(rng.integers(self.n_lo, self.n_hi + 1))):
                patch = self.bank[int(rng.integers(len(self.bank)))]
                if self.rotate:
                    patch = torch.rot90(patch, int(rng.integers(4)), (1, 2))
                    if rng.random() < 0.5:
                        patch = patch.flip(2)
                ph, pw = patch.shape[1:]
                if ph >= H or pw >= W:
                    continue
                y0, x0 = int(rng.integers(H - ph)), int(rng.integers(W - pw))
                ratio, ps, pk, alpha = patch
                win = (slice(y0, y0 + ph), slice(x0, x0 + pw))
                blocked = (rad[i, 0][win] > 0.97) | (soft[i, 0][win] > 0.05)
                if bool((blocked & (alpha > 0)).any()):
                    continue
                x[i, 0][win] = x[i, 0][win] * (1 - alpha + alpha * ratio)
                soft[i, 0][win] = torch.maximum(soft[i, 0][win], ps)
                skel[i, 0][win] = torch.maximum(skel[i, 0][win], pk)
        return x, soft, skel


class CropSampler:
    def __init__(self, cache: dict, frames: np.ndarray, crop: int, aug: dict, copy_paste: CopyPaste | None = None,
                 device: str = "cuda", seed: int = 0):
        frames = np.asarray(frames)
        self.img = torch.from_numpy(np.ascontiguousarray(cache["img"][frames])).to(device)
        self.soft = torch.from_numpy(np.ascontiguousarray(cache["soft"][frames])).to(device)
        self.skel = torch.from_numpy(np.ascontiguousarray(cache["skel"][frames])).to(device)
        self.disk = torch.as_tensor(cache["disk"][frames], device=device, dtype=torch.float32)
        self.n = len(frames)
        self.crop, self.aug, self.copy_paste = crop, aug, copy_paste
        self.device = device
        self.g = torch.Generator(device=device).manual_seed(seed)

    def rand(self, *shape, lo=0.0, hi=1.0):
        return torch.rand(*shape, device=self.device, generator=self.g) * (hi - lo) + lo

    def _crops(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Crops of [image, soft target, centre line] in [0, 1], and each pixel's distance to disk centre / R."""
        b, out, a = len(ids), self.crop, self.aug
        lo, hi = a["scale"]
        half = (out / NATIVE) / torch.exp(self.rand(b, lo=math.log(lo), hi=math.log(hi)))
        ang = self.rand(b, lo=-math.pi, hi=math.pi) if a.get("rotate", False) else torch.zeros(b, device=self.device)

        # crop centre uniform over the solar disk, so a crop rarely shows only sky
        d = self.disk[ids]
        rr = d[:, 2] * torch.sqrt(self.rand(b)) * 0.95
        th = self.rand(b, lo=0, hi=2 * math.pi)
        cx = (d[:, 0] + rr * torch.cos(th)) / (NATIVE / 2) - 1
        cy = (d[:, 1] + rr * torch.sin(th)) / (NATIVE / 2) - 1

        theta = torch.zeros(b, 2, 3, device=self.device)
        theta[:, 0, 0], theta[:, 0, 1] = half * torch.cos(ang), -half * torch.sin(ang)
        theta[:, 1, 0], theta[:, 1, 1] = half * torch.sin(ang), half * torch.cos(ang)
        theta[:, 0, 2], theta[:, 1, 2] = cx, cy
        grid = F.affine_grid(theta, (b, 1, out, out), align_corners=False)
        stack = torch.stack([self.img[ids], self.soft[ids], self.skel[ids]], 1).float().div_(255)
        crops = F.grid_sample(stack, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        px = (grid[..., 0] + 1) * (NATIVE / 2) - 0.5
        py = (grid[..., 1] + 1) * (NATIVE / 2) - 0.5
        rad = torch.hypot(px - d[:, 0, None, None], py - d[:, 1, None, None]) / d[:, 2, None, None]
        return crops, rad[:, None]

    @torch.no_grad()
    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = torch.randint(0, self.n, (batch,), device=self.device, generator=self.g)
        crops, rad = self._crops(ids)
        if self.aug.get("dihedral", True):
            both = random_dihedral(torch.cat([crops, rad], 1), self.rand(batch, 3) < 0.5)
            crops, rad = both[:, :3], both[:, 3:]
        x, soft, skel = crops[:, :1], crops[:, 1:2], crops[:, 2:3]
        if self.copy_paste is not None:
            x, soft, skel = self.copy_paste(x, soft, skel, rad)
        x = self.photometric(x, rad)
        return x.contiguous(memory_format=torch.channels_last), soft, (skel > 0.5).float()

    def photometric(self, x: torch.Tensor, rad: torch.Tensor) -> torch.Tensor:
        b, a = x.shape[0], self.aug
        on_disk = (rad < 1.0).float()

        # limb darkening I/I0 = 1 - u (1 - mu): a different u per sample, as at another site
        mu = torch.sqrt((1 - rad.clamp(max=1) ** 2).clamp_min(0))
        du = self.rand(b, 1, 1, 1, lo=-a["limb_strength"], hi=a["limb_strength"]) * (self.rand(b, 1, 1, 1) < a["limb_p"]).float()
        x = x * (1 - du * (1 - mu) * on_disk)

        # thin cloud: a smooth multiplicative field
        field = F.interpolate(self.rand(b, 1, 6, 6), size=x.shape[-2:], mode="bicubic", align_corners=False)
        k = self.rand(b, 1, 1, 1, hi=a["haze_strength"]) * (self.rand(b, 1, 1, 1) < a["haze_p"]).float()
        x = x * (1 - k * field.clamp(0, 1))

        g0, g1 = a["gamma"]
        x = x.clamp(0, 1) ** torch.exp(self.rand(b, 1, 1, 1, lo=math.log(g0), hi=math.log(g1)))
        contrast = 1 + self.rand(b, 1, 1, 1, lo=-a["contrast"], hi=a["contrast"])
        bright = self.rand(b, 1, 1, 1, lo=-a["brightness"], hi=a["brightness"])
        mean = (x * on_disk).sum((2, 3), keepdim=True) / on_disk.sum((2, 3), keepdim=True).clamp_min(1)
        x = (x - mean) * contrast + mean + bright * on_disk

        x = self._seeing(x, a["blur_p"], a["blur_sigma"])
        x = self._low_res(x, a["lowres_p"], a["lowres_scale"])
        std = self.rand(b, 1, 1, 1, hi=a["noise_std"]) * (self.rand(b, 1, 1, 1) < a["noise_p"]).float()
        x = x + torch.randn(x.shape, device=x.device, generator=self.g) * std
        return x.clamp(0, 1)

    def _seeing(self, x: torch.Tensor, p: float, sigma: list[float]) -> torch.Tensor:
        """Gaussian blur with its own width for every sample (one separable grouped convolution)."""
        b = x.shape[0]
        s = self.rand(b, lo=sigma[0], hi=sigma[1])
        r = int(math.ceil(3 * sigma[1]))
        t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        k = torch.exp(-0.5 * (t[None] / s[:, None]) ** 2)
        k = k / k.sum(1, keepdim=True)
        xb = x.transpose(0, 1)
        xb = F.conv2d(F.pad(xb, (r, r, 0, 0), mode="reflect"), k[:, None, None, :], groups=b)
        xb = F.conv2d(F.pad(xb, (0, 0, r, r), mode="reflect"), k[:, None, :, None], groups=b)
        sel = (self.rand(b, 1, 1, 1) < p).float()
        return sel * xb.transpose(0, 1) + (1 - sel) * x

    def _low_res(self, x: torch.Tensor, p: float, scale: list[float]) -> torch.Tensor:
        """Down and up again by one random factor per batch, applied to some of its samples."""
        s = float(self.rand(1, lo=scale[0], hi=scale[1]))
        h, w = x.shape[-2:]
        small = F.interpolate(x.contiguous(), size=(max(8, int(h * s)), max(8, int(w * s))), mode="bilinear",
                              antialias=True, align_corners=False)
        back = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
        sel = (self.rand(x.shape[0], 1, 1, 1) < p).float()
        return sel * back + (1 - sel) * x
