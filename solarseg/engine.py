"""Training, inference, calibration, and experiment records."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from tqdm.auto import tqdm

from solarseg.data import CompetitionData, read_image, sha256
from solarseg.metrics import aggregate, bootstrap_pq, evaluate_image
from solarseg.model import UNet, device_auto, segmentation_loss
from solarseg.postprocess import instances
from solarseg.provenance import environment, source_hashes, utc_now


def prepare_cache(data, manifest, cache_dir, size):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {"annotation_sha256": sha256(data.annotation_path), "size": size}
    meta = cache_dir / "metadata.json"
    if meta.exists() and json.loads(meta.read_text()) != fingerprint:
        raise ValueError("Cache fingerprint mismatch; use a new cache directory")
    meta.write_text(json.dumps(fingerprint, indent=2))
    for stem in tqdm(manifest.stem, desc="Preparing image/consensus cache"):
        target = cache_dir / f"{stem}.npz"
        if not target.exists():
            image = (read_image(data.image_path(stem), size) * 255).round().astype(np.uint8)
            mask = (data.soft_target(stem, size) * 255).round().astype(np.uint8)
            np.savez_compressed(target, image=image, target=mask)


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
    if selected.empty or crop > size or crop % 8:
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
    model = UNet(width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
        "learning_rate": 1e-3,
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


class Predictor:
    def __init__(self, checkpoint=None, size=1024, device=None, tta=False, tile=0):
        torch.set_num_threads(4)
        self.device = device or device_auto()
        self.tta = tta
        self.model = None
        self.checkpoint = checkpoint
        self.size = size
        self.tile = tile
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.size = state["config"]["size"]
            self.model = UNet(state["config"]["width"])
            self.model.load_state_dict(state["state_dict"])
            self.model.to(self.device).eval()
        if tile and (not checkpoint or tile % 8 or tile > self.size or tile < 32):
            raise ValueError("Tiling requires a checkpoint and a valid tile size divisible by 8")

    def _forward(self, tensor):
        probs = self.model(tensor).sigmoid()
        if self.tta:
            for dims in [[-1], [-2], [-1, -2]]:
                probs += self.model(torch.flip(tensor, dims)).sigmoid().flip(dims)
            probs /= 4
        return probs

    def _tiled(self, image):
        stride = self.tile * 3 // 4
        offsets = sorted(set(range(0, self.size - self.tile + 1, stride)) | {self.size - self.tile})
        coords = [(y, x) for y in offsets for x in offsets]
        # Positive Hann weights avoid zero coverage at the full-disk border.
        weight = np.maximum(np.outer(np.hanning(self.tile), np.hanning(self.tile)), 0.01)
        total = np.zeros_like(image)
        norm = np.zeros_like(image)
        for start in range(0, len(coords), 4):
            group = coords[start : start + 4]
            crops = np.stack([image[y : y + self.tile, x : x + self.tile] for y, x in group])
            tensor = torch.from_numpy((crops[:, None] - 0.5) / 0.25).to(self.device)
            probs = self._forward(tensor)[:, 0].cpu().numpy()
            for (y, x), prob in zip(group, probs, strict=True):
                total[y : y + self.tile, x : x + self.tile] += prob * weight
                norm[y : y + self.tile, x : x + self.tile] += weight
        return np.clip(total / norm, 0, 1)

    def __call__(self, path):
        image = read_image(path, self.size)
        if self.model is None:
            # Intensity deficit relative to a broad local background.
            background = ndi.gaussian_filter(image, 16)
            deficit = (background - image) / np.maximum(background, 0.1)
            prob = 1 / (1 + np.exp(-np.clip((deficit - 0.13) / 0.035, -20, 20)))
            prob[(image < 0.08) | (background < 0.2)] = 0
            return prob.astype(np.float32)
        tensor = torch.from_numpy((image[None, None] - 0.5) / 0.25).to(self.device)
        with torch.inference_mode():
            if self.tile:
                return self._tiled(image)
            probs = self._forward(tensor)
        return probs[0, 0].cpu().numpy()


def predict_cache(data, stems, predictor, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "schema": 2,
        "checkpoint": sha256(predictor.checkpoint) if predictor.checkpoint else "classical-v1",
        "size": predictor.size,
        "tta": predictor.tta,
        "tile": predictor.tile,
        "input": "grayscale JPEG",
        "input_sha256": {stem: sha256(data.image_path(stem)) for stem in stems},
        "implementation_sha256": hashlib.sha256(
            (
                inspect.getsource(Predictor)
                + inspect.getsource(read_image)
                + inspect.getsource(UNet)
            ).encode()
        ).hexdigest(),
    }
    meta = directory / "metadata.json"
    if meta.exists() and json.loads(meta.read_text()) != fingerprint:
        raise ValueError("Prediction cache fingerprint mismatch")
    meta.write_text(json.dumps(fingerprint, indent=2))
    for stem in tqdm(stems, desc="Inference"):
        path = directory / f"{stem}.npy"
        if not path.exists():
            np.save(path, predictor(data.image_path(stem)))


def evaluate_cached(data, stems, cache, params):
    records = []
    for stem in stems:
        prob = np.load(Path(cache) / f"{stem}.npy")
        records.extend(evaluate_image(data, stem, instances(prob, **params)))
    return records


def calibration_grid():
    """Initial 16 settings plus the nine settings motivated by v1 calibration."""
    initial = [
        {"threshold": t, "min_area": a, "closing": c}
        for t in [0.3, 0.45, 0.6, 0.75]
        for a in [80, 200]
        for c in [0, 3]
    ]
    return initial + [
        {"threshold": t, "min_area": a, "closing": 3}
        for t in [0.6, 0.8, 0.9]
        for a in [400, 800, 1600]
    ]


def calibrate(data, stems, cache, out_dir):
    """Recorded 25-point grid; no test or holdout labels are accessed here."""
    out_dir = Path(out_dir)
    results = []
    best = None
    for params in calibration_grid():
        rows = evaluate_cached(data, stems, cache, params)
        result = dict(params=params, **aggregate(rows))
        results.append(result)
        print(json.dumps(result), flush=True)
        if best is None or result["pq"] > best["pq"]:
            best = result
    (out_dir / "calibration.json").write_text(json.dumps(results, indent=2))
    (out_dir / "postprocess.json").write_text(json.dumps(best["params"], indent=2))
    return best


def evaluate_and_save(data, stems, cache, params, out_path):
    rows = evaluate_cached(data, stems, cache, params)
    result = dict(
        created_at=utc_now(),
        **aggregate(rows),
        bootstrap_95ci=bootstrap_pq(rows),
        images=len(stems),
        fragmented_gt=sum(r["fragmented_gt"] for r in rows),
        merged_pred=sum(r["merged_pred"] for r in rows),
        params=params,
        rows=rows,
    )
    Path(out_path).write_text(json.dumps(result, indent=2))
    return {k: v for k, v in result.items() if k != "rows"}
