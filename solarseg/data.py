"""Competition-only inputs, physical-observation grouping, and COCO conversion."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as coco_mask
from sklearn.model_selection import GroupKFold


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_root(path):
    path = Path(path)
    if (path / "train").is_dir() and (path / "test").is_dir():
        return path
    candidates = list(path.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one competition dataset below {path}; got {len(candidates)}"
        )
    return candidates[0].parent.parent


def annotation_rle(annotation, height=2048, width=2048):
    seg = annotation["segmentation"]
    if isinstance(seg, list):
        return coco_mask.merge(coco_mask.frPyObjects(seg, height, width))
    if isinstance(seg["counts"], list):
        return coco_mask.frPyObjects(seg, height, width)
    return {
        "size": seg["size"],
        "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"],
    }


class CompetitionData:
    def __init__(self, root):
        self.root = find_root(root)
        self.annotation_path = self.root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        self.raw = json.loads(self.annotation_path.read_text())
        self.images = {str(v["id"]): v for v in self.raw["images"]}
        self.annotations = defaultdict(list)
        self.by_stem = defaultdict(list)
        self._ground_truth_cache = {}
        for image_id, im in self.images.items():
            self.by_stem[Path(im["file_name"]).stem].append(image_id)
        for annotation in self.raw["annotations"]:
            image_id = str(annotation["image_id"])
            if image_id not in self.images:
                raise ValueError(f"Orphan annotation {annotation['id']}")
            self.annotations[image_id].append(annotation)
        self.test_paths = {
            p.stem: p for p in sorted((self.root / "test/test_images").glob("*.jpeg"))
        }
        if set(self.by_stem) & set(self.test_paths):
            raise ValueError("Training and test image names overlap")

    def image_path(self, stem):
        if stem in self.test_paths:
            return self.test_paths[stem]
        return self.root / "train/train_images" / f"{stem}.jpeg"

    def ground_truth(self, stem):
        if stem not in self._ground_truth_cache:
            self._ground_truth_cache[stem] = {
                k: [annotation_rle(a) for a in self.annotations[k]] for k in self.by_stem[stem]
            }
        return self._ground_truth_cache[stem]

    def soft_target(self, stem, size):
        """Average independent annotator unions; never merge annotator instances for PQ."""
        result = np.zeros((size, size), np.float32)
        entries = self.ground_truth(stem)
        for rles in entries.values():
            union = (
                coco_mask.decode(coco_mask.merge(rles))
                if rles
                else np.zeros((2048, 2048), np.uint8)
            )
            result += np.asarray(
                Image.fromarray(union.astype(np.float32)).resize((size, size), Image.Resampling.BOX)
            )
        return result / len(entries)


def make_manifest(data, out_dir, seed=2026):
    """Five folds over ~solar-rotation time blocks, with exact-duplicate unioning."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, stems in [("train", sorted(data.by_stem)), ("test", sorted(data.test_paths))]:
        for stem in stems:
            path = data.image_path(stem)
            with Image.open(path) as im:
                if im.size != (2048, 2048):
                    raise ValueError(f"Unexpected shape: {path} {im.size}")
                im.load()
                pixel_hash = hashlib.sha256(np.asarray(im.convert("L")).tobytes()).hexdigest()
            date = pd.to_datetime(stem[:14], format="%Y%m%d%H%M%S")
            rows.append(
                {
                    "stem": stem,
                    "split": split,
                    "date": str(date),
                    "site": stem[14:],
                    "sha256": sha256(path),
                    "pixel_sha256": pixel_hash,
                    "block": int((date - pd.Timestamp("2010-01-01")).days // 27),
                    "annotators": len(data.by_stem.get(stem, [])),
                }
            )
    frame = pd.DataFrame(rows)
    train = frame[frame.split == "train"].copy()
    test = frame[frame.split == "test"]
    if set(train.pixel_sha256) & set(test.pixel_sha256):
        raise ValueError(
            "Exact image-pixel duplicate across train/test; quarantine before proceeding"
        )
    parent = {x: x for x in train.block.unique()}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for _, group in train.groupby("pixel_sha256"):
        blocks = group.block.tolist()
        for b in blocks[1:]:
            parent[find(b)] = find(blocks[0])
    train["group"] = train.block.map(find)
    train["fold"] = -1
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, idx) in enumerate(cv.split(train, groups=train.group)):
        train.iloc[idx, train.columns.get_loc("fold")] = fold
    # Folds 0 and 1 are locked holdout and calibration; remove training images
    # within three days of either, including adjacent blocks at fold boundaries.
    dates = pd.to_datetime(train.date).dt.as_unit("ns").astype("int64").to_numpy() / 86400e9
    held_dates = dates[train.fold.isin([0, 1]).to_numpy()]
    distance = np.abs(dates[:, None] - held_dates[None, :]).min(axis=1)
    train["role"] = np.select(
        [train.fold == 0, train.fold == 1, distance <= 3],
        ["holdout", "calibration", "embargo"],
        default="train",
    )
    if not (train.role == "train").any():
        raise ValueError("Split construction left no training observations")
    train.to_csv(out_dir / "train_manifest.csv", index=False)
    test.to_csv(out_dir / "test_manifest.csv", index=False)
    summary = {
        "physical_train": len(train),
        "test": len(test),
        "annotator_images": len(data.images),
        "annotations": len(data.raw["annotations"]),
        "roles": train.role.value_counts().to_dict(),
        "sites": train.site.value_counts().to_dict(),
        "train_annotation_sha256": sha256(data.annotation_path),
        "seed": seed,
        "split_manifest_sha256": sha256(out_dir / "train_manifest.csv"),
        "exact_duplicate_groups": int((train.pixel_sha256.value_counts() > 1).sum()),
    }
    (out_dir / "audit.json").write_text(json.dumps(summary, indent=2))
    return summary


def read_image(path, size=1024):
    with Image.open(path) as image:
        return (
            np.asarray(
                image.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32
            )
            / 255
        )
