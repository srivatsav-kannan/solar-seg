"""Bounded scoring experiments; the released baseline implementation stays pinned."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

from research.models import DeepUNet
from solarseg.data import CompetitionData, sha256
from solarseg.engine import prepare_cache
from solarseg.model import device_auto, segmentation_loss
from solarseg.provenance import environment, source_hashes, utc_now


def train(
    data_root,
    manifest_path,
    out_dir,
    steps=1600,
    size=1024,
    crop=256,
    batch_size=6,
    width=16,
    seed=2026,
    device=None,
    depth=4,
    learning_rate=1e-3,
    initialize=None,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.pt").exists():
        raise FileExistsError("Use a new run directory; checkpoints are immutable")
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    data = CompetitionData(data_root)
    manifest = pd.read_csv(manifest_path)
    selected = manifest[manifest.role == "train"]
    if selected.empty or crop > size or crop % (2**depth):
        raise ValueError("Invalid training manifest or crop size")
    cache_dir = out.parent / f"cache-{size}"
    prepare_cache(data, selected, cache_dir, size)
    images, targets, positions = [], [], []
    for stem in selected.stem:
        with np.load(cache_dir / f"{stem}.npz") as z:
            images.append(z["image"])
            targets.append(z["target"])
            coords = np.argwhere(z["target"] >= 64)
            positions.append(coords[:: max(1, len(coords) // 5000)])
    device = device or device_auto()
    model = DeepUNet(width, depth).to(device)
    if initialize:
        initial = torch.load(initialize, map_location="cpu", weights_only=True)
        if initial["config"]["train_stems"] != selected.stem.tolist():
            raise ValueError("Warm start must have the same optimization observations")
        if initial["config"]["annotation_sha256"] != sha256(data.annotation_path):
            raise ValueError("Warm start annotation provenance differs")
        model.load_state_dict(initial["state_dict"])
    implementation = {str(p): sha256(p) for p in sorted(Path("research").glob("*.py"))}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=1e-5)
    cfg = {
        "created_at": utc_now(),
        "source_hashes": source_hashes(),
        "environment": environment(),
        "steps": steps,
        "size": size,
        "crop": crop,
        "batch_size": batch_size,
        "width": width,
        "seed": seed,
        "device": device,
        "learning_rate": learning_rate,
        "architecture": "DeepUNet",
        "depth": depth,
        "research_source_hashes": implementation,
        "initialize": str(initialize) if initialize else None,
        "initial_checkpoint_sha256": sha256(initialize) if initialize else None,
        "weight_decay": 1e-4,
        "train_stems": selected.stem.tolist(),
        "manifest_sha256": sha256(manifest_path),
        "annotation_sha256": sha256(data.annotation_path),
        "loss": "BCE + soft Dice",
        "target": "mean of per-annotator binary unions",
        "external_weights": False,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    history = []
    model.train()
    for step in range(1, steps + 1):
        xb, yb = [], []
        for _ in range(batch_size):
            idx = int(rng.integers(len(images)))
            if rng.random() < 0.5 and len(positions[idx]):
                cy, cx = positions[idx][rng.integers(len(positions[idx]))]
                y = int(np.clip(cy - rng.integers(crop), 0, size - crop))
                x = int(np.clip(cx - rng.integers(crop), 0, size - crop))
            else:
                y, x = rng.integers(0, size - crop + 1, 2)
            a = images[idx][y : y + crop, x : x + crop].astype(np.float32) / 255
            b = targets[idx][y : y + crop, x : x + crop].astype(np.float32) / 255
            k = int(rng.integers(4))
            a, b = np.rot90(a, k), np.rot90(b, k)
            if rng.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
            a = np.clip(a * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05), 0, 1)
            if rng.random() < 0.2:
                a = ndi.gaussian_filter(a, sigma=rng.uniform(0.1, 0.7))
            xb.append((a - 0.5) / 0.25)
            yb.append(b)
        xt = torch.from_numpy(np.stack(xb)[:, None]).to(device)
        yt = torch.from_numpy(np.stack(yb)[:, None]).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = segmentation_loss(model(xt), yt)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 50 == 0 or step == steps:
            record = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "seconds": time.monotonic() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=2))
    model.cpu()
    torch.save({"state_dict": model.state_dict(), "config": cfg}, out / "model.pt")
    cfg["training_seconds"] = time.monotonic() - started
    cfg["checkpoint_sha256"] = sha256(out / "model.pt")
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    return cfg


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data", default="data/raw")
    p.add_argument("--manifest", default="configs/manifests/train_manifest.csv")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--size", type=int, default=1536)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--initialize")
    a = p.parse_args()
    print(
        json.dumps(
            train(
                a.data,
                a.manifest,
                a.run,
                a.steps,
                a.size,
                a.crop,
                a.batch_size,
                a.width,
                a.seed,
                a.device,
                a.depth,
                a.learning_rate,
                a.initialize,
            ),
            indent=2,
        )
    )
