"""Inference for subsequent experiments, with model and implementation fingerprints."""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from research.context import ContextUNet
from solarseg.data import sha256
from solarseg.engine import Predictor


class ContextPredictor(Predictor):
    def __init__(self, checkpoint, device=None, tta=False, tile=0, size=None):
        super().__init__(device=device, tta=tta)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = state["config"]
        self.checkpoint = str(checkpoint)
        self.size = size or cfg["size"]
        self.depth = cfg.get("depth", 3)
        self.tile = tile
        if self.size % 2**self.depth or (
            tile and (tile % 2**self.depth or tile > self.size or tile < 32)
        ):
            raise ValueError("Image and tile sizes must align with model stride")
        if cfg.get("architecture") != "ContextUNet":
            raise ValueError("Expected a context-model checkpoint")
        self.model = ContextUNet(cfg["width"], self.depth)
        self.model.load_state_dict(state["state_dict"], strict=True)
        self.model.to(self.device).eval()


def predict_cache(data, stems, predictor, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sources = [
        "research/models.py",
        "research/context.py",
        "research/context_predict.py",
        "research/predict.py",
        "solarseg/model.py",
        "solarseg/data.py",
        "solarseg/engine.py",
    ]
    fingerprint = {
        "schema": "context-v1",
        "checkpoint_sha256": sha256(predictor.checkpoint),
        "size": predictor.size,
        "depth": predictor.depth,
        "tile": predictor.tile,
        "tta": predictor.tta,
        "input_sha256": {s: sha256(data.image_path(s)) for s in stems},
        "inference_source_hashes": {p: sha256(p) for p in sources},
    }
    meta = directory / "metadata.json"
    if meta.exists() and json.loads(meta.read_text()) != fingerprint:
        raise ValueError("Research prediction cache fingerprint differs")
    meta.write_text(json.dumps(fingerprint, indent=2) + "\n")
    for stem in tqdm(stems, desc="Research inference"):
        path = directory / f"{stem}.npy"
        if not path.exists():
            prob = predictor(data.image_path(stem))
            if prob.shape != (predictor.size, predictor.size) or not np.isfinite(prob).all():
                raise ValueError(f"Invalid prediction for {stem}")
            np.save(path, prob)
