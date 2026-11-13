"""Training the network on one fold (or on all frames) and writing its out-of-fold maps.

Each step draws twelve 1024 px crops at native scale, augmented on the GPU, and runs a bf16
forward and backward pass. An exponential moving average of the weights is what gets evaluated
and saved. Every `val_every` epochs it predicts the held-out frames of the fold exactly as the
test frames are predicted, and the checkpoint is chosen on Panoptic Quality, not on the loss.
"""
from __future__ import annotations

import json
import math
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .augment import CopyPaste, CropSampler
from .candidates import extract
from .config import seed_everything
from .data import load_records
from .folds import load_folds
from .inference import on_disk, probability_map, to_uint8
from .losses import FilamentLoss
from .metrics import GroundTruth, panoptic_quality
from .model import FilamentNet, load_net, save_weights

VAL_THRESHOLDS = (0.4, 0.5, 0.6)


class EMA:
    def __init__(self, net: torch.nn.Module, decay: float):
        self.decay = decay
        self.net = deepcopy(net).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net: torch.nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))          # follows the weights closely at the start
        for s, p in zip(self.net.state_dict().values(), net.state_dict().values()):
            if s.dtype.is_floating_point:
                s.lerp_(p, 1 - d)


def warmup_cosine(total: int, warmup: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t))
    return f


def split(cfg: dict, stems: list[str], fold: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the training and validation frames; `fold=None` trains on all frames."""
    if fold is None:
        return np.arange(len(stems)), np.zeros(0, int)
    of = load_folds(cfg)
    f = np.array([of[s] for s in stems])
    return np.flatnonzero(f != fold), np.flatnonzero(f == fold)


@torch.no_grad()
def validate(net, cache: dict, frames: np.ndarray, gt: GroundTruth) -> dict:
    """Validation BCE against the soft target and PQ of the plain components at a few thresholds."""
    maps, bce = {}, []
    for i in frames:
        img = np.asarray(cache["img"][i])
        p = probability_map(net, img, tta=1)
        soft = torch.as_tensor(np.asarray(cache["soft"][i]), device=p.device).float().div_(255)
        bce.append(float(F.binary_cross_entropy(p.clamp(1e-4, 1 - 1e-4), soft)))
        maps[cache["stems"][i]] = (on_disk(p, tuple(cache["disk"][i])), img, tuple(cache["disk"][i]))
    out = {"val_bce": round(float(np.mean(bce)), 5)}
    for thr in VAL_THRESHOLDS:
        pred = {s: [c.mask_rle for c in extract(p, img, disk, thr)] for s, (p, img, disk) in maps.items()}
        out[f"pq@{thr}"] = round(panoptic_quality(gt.subset(maps), pred).pq, 4)
    out["val_pq"] = max(out[f"pq@{t}"] for t in VAL_THRESHOLDS)
    return out


def train(cfg: dict, cache: dict, fold: int | None, epochs: int | None = None, device: str = "cuda") -> dict:
    t = cfg["train"]
    epochs = epochs or t["epochs"]
    k = -1 if fold is None else fold
    seed_everything(cfg["seed"] + 101 * k)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    tr, va = split(cfg, cache["stems"], fold)
    tag = "full" if fold is None else f"fold{fold}"
    runs = cfg["paths"]["runs"]
    runs.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"] + k
    sampler = CropSampler(cache, tr, t["crop"], cfg["augment"], CopyPaste(cfg["copy_paste"], cache["bank"], tr, device, seed=seed),
                          device, seed=seed)
    net = FilamentNet(cfg["model"]["encoder"]).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.param_groups(t["lr"], t["encoder_lr_mult"], t["weight_decay"]), lr=t["lr"], fused=True)
    total = epochs * t["steps_per_epoch"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total, int(total * t["warmup_frac"])))
    loss_fn = FilamentLoss(t["bce_weight"], t["dice_weight"], t["skel_weight"])
    ema = EMA(net, t["ema_decay"])
    gt = GroundTruth(load_records(cfg))

    best = {"val_pq": -1.0, "epoch": -1, "weights": str(runs / f"{tag}.safetensors")}
    history, step, start = [], 0, time.time()
    bar = tqdm(range(1, epochs + 1), desc=tag, unit="epoch", mininterval=30)
    for epoch in bar:
        net.train()
        running = 0.0
        for _ in range(t["steps_per_epoch"]):
            x, soft, skel = sampler.sample(t["batch_size"])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(x)
            loss = loss_fn(logits, soft, skel)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            ema.update(net, step)
            running += loss.item()
        rec = {"epoch": epoch, "loss": round(running / t["steps_per_epoch"], 4), "minutes": round((time.time() - start) / 60, 1)}
        if len(va) and (epoch % t["val_every"] == 0 or epoch == epochs):
            rec.update(validate(ema.net, cache, va, gt))
            if rec["val_pq"] > best["val_pq"]:
                best.update(val_pq=rec["val_pq"], epoch=epoch)
                save_weights(ema.net, best["weights"])
            print(f"[{tag}] {json.dumps(rec)}", flush=True)
        history.append(rec)
        bar.set_postfix(loss=rec["loss"], best_pq=best["val_pq"])
        (runs / f"{tag}_history.json").write_text(json.dumps({"best": best, "history": history}, indent=1))
    if not len(va):
        save_weights(ema.net, best["weights"])          # no validation frames: keep the last weights
    return best


@torch.no_grad()
def write_out_of_fold(cfg: dict, cache: dict, fold: int, device: str = "cuda") -> None:
    """Maps of the fold's frames from the network that never saw them, made as at test time."""
    _, va = split(cfg, cache["stems"], fold)
    runs = cfg["paths"]["runs"]
    net = load_net(cfg, runs / f"fold{fold}.safetensors", device)
    out = np.lib.format.open_memmap(runs / f"oof_fold{fold}.npy", mode="w+", dtype=np.uint8, shape=(len(va), 2048, 2048))
    for j, i in enumerate(va):
        out[j] = to_uint8(probability_map(net, np.asarray(cache["img"][i]), cfg["predict"]["tta"], device))
    out.flush()
    (runs / f"oof_fold{fold}_stems.json").write_text(json.dumps([cache["stems"][i] for i in va]))
