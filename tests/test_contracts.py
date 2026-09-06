"""Analytical metric and serialization tests; no competition labels are committed."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from solarseg.data import annotation_rle
from solarseg.engine import Predictor
from solarseg.metrics import aggregate, counts, overlap
from solarseg.model import UNet, segmentation_loss
from solarseg.postprocess import decode, encode, encode_component, instances
from solarseg.submission import validate_submission


def rectangle(y0, y1, x0, x1, size=32):
    a = np.zeros((size, size), np.uint8)
    a[y0:y1, x0:x1] = 1
    return a


def test_rle_asymmetric_roundtrip_and_polygon():
    a = rectangle(2, 9, 13, 24)
    np.testing.assert_array_equal(decode(encode(a)), a)
    polygon = {"segmentation": [[13, 2, 24, 2, 24, 9, 13, 9]]}
    np.testing.assert_array_equal(decode(annotation_rle(polygon, 32, 32)), a)


def test_pq_perfect_missing_false_positive_and_strict_threshold():
    a, b = encode(rectangle(1, 4, 1, 5)), encode(rectangle(20, 24, 20, 24))
    assert aggregate([counts(overlap([a], [a]))])["pq"] == 1
    assert aggregate([counts(overlap([a], []))])["pq"] == 0
    assert aggregate([counts(overlap([], [a]))])["pq"] == 0
    assert aggregate([counts(overlap([a], [a, b]))])["pq"] == pytest.approx(2 / 3)
    assert counts(np.array([[0.5]])) == {"tp": 0, "fp": 1, "fn": 1, "iou_sum": 0}
    assert counts(np.array([[0.50001]]))["tp"] == 1


def test_fragmentation_and_merging_are_penalized():
    a = encode(rectangle(1, 11, 1, 11))
    b = encode(rectangle(1, 6, 1, 11))
    c = encode(rectangle(6, 11, 1, 11))
    assert aggregate([counts(overlap([a], [b, c]))])["pq"] == 0
    assert aggregate([counts(overlap([b, c], [a]))])["pq"] == 0


def test_pooled_score_is_not_macro_average():
    rows = [counts(np.array([[1.0]])), counts(np.zeros((3, 0)))]
    assert aggregate(rows)["pq"] == 0.4


def test_postprocessing_is_disjoint_at_native_resolution():
    prob = rectangle(1, 6, 1, 7).astype(np.float32)
    prob += rectangle(20, 27, 20, 29)
    masks = instances(prob, threshold=0.5, min_area=1, native_size=64)
    assert len(masks) == 2
    assert np.stack([decode(x) for x in masks]).sum(0).max() == 1


def test_invalid_probabilities_are_rejected():
    for value in [np.nan, np.inf, -0.1, 1.1]:
        with pytest.raises(ValueError, match="probability"):
            instances(np.full((8, 8), value))


def test_sparse_component_encoding_matches_dense_coco_at_image_edges():
    from scipy import ndimage as ndi

    rng = np.random.default_rng(2026)
    for mask in [np.ones((31, 29)), np.eye(31, 29), rng.random((31, 29)) > 0.5]:
        labels, n = ndi.label(mask, structure=np.ones((3, 3)))
        bounds = ndi.find_objects(labels)
        for label in range(1, n + 1):
            rle = encode_component(labels, label, bounds[label - 1])
            np.testing.assert_array_equal(decode(rle), labels == label)
            assert rle["counts"] == encode(labels == label)["counts"]


def test_submission_rejects_overlap_and_unknown_ids(tmp_path):
    a, b = rectangle(1, 6, 1, 6), rectangle(4, 9, 4, 9)
    path = tmp_path / "submission.csv"

    def write(stem="obs"):
        pd.DataFrame(
            {
                "filament_id": [stem + "_1", stem + "_2"],
                "segmentation_rle": [encode(x)["counts"].decode() for x in [a, b]],
            }
        ).to_csv(path, index=False)
        path.with_suffix(".images.json").write_text(json.dumps({"obs": 2}))

    write()
    with pytest.raises(ValueError, match="Overlapping"):
        validate_submission(path, ["obs"], (32, 32))
    write("alien")
    with pytest.raises(ValueError, match="Unknown"):
        validate_submission(path, ["obs"], (32, 32))


def test_submission_accepts_valid_empty_image_in_coverage_manifest(tmp_path):
    path = tmp_path / "submission.csv"
    rle = encode(rectangle(1, 6, 1, 6))["counts"].decode()
    pd.DataFrame({"filament_id": ["obs_1"], "segmentation_rle": [rle]}).to_csv(path, index=False)
    path.with_suffix(".images.json").write_text(json.dumps({"obs": 1, "empty_obs": 0}))
    result = validate_submission(path, ["obs", "empty_obs"], (32, 32))
    assert result["valid"] and result["zero_detection_images"] == 1


def test_model_forward_backward_is_finite():
    torch.set_num_threads(2)
    model = UNet(width=8)
    logits = model(torch.rand(2, 1, 40, 48))
    assert logits.shape == (2, 1, 40, 48)
    loss = segmentation_loss(logits, torch.zeros_like(logits))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_tiled_blending_covers_borders_and_preserves_pixel_alignment():
    predictor = Predictor(size=64, device="cpu")
    predictor.tile = 32
    predictor.model = torch.nn.Identity()
    image = np.random.default_rng(1).random((64, 64), dtype=np.float32)
    with torch.inference_mode():
        actual = predictor._tiled(image)
    expected = 1 / (1 + np.exp(-(image - 0.5) / 0.25))
    np.testing.assert_allclose(actual, expected, atol=2e-7)


@pytest.mark.skipif(
    not Path("references/official/self-evaluation-notebook.ipynb").exists(),
    reason="Download official evaluator to run parity check",
)
def test_parity_with_downloaded_organizer_evaluator():
    notebook = json.loads(Path("references/official/self-evaluation-notebook.ipynb").read_text())
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    parsed = ast.parse(source)
    names = {"get_pq_score", "fp_count_hit", "fn_count_hit", "get_overlap_matrices"}
    functions = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {"torch": torch, "pd": pd, "np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "official_evaluator", "exec"), env)  # noqa: S102 - reviewed organizer functions only
    matrices = [
        np.array([[1.0, 0.0], [0.0, 0.7]]),
        np.array([[0.5]]),
        np.zeros((2, 0)),
        np.zeros((0, 3)),
        np.array([[0.9], [0.8]]),
    ]
    frame = pd.DataFrame(
        [
            {"iou_matrix": torch.tensor(m), "n_gt": m.shape[0], "n_pred": m.shape[1]}
            for m in matrices
        ]
    )
    assert aggregate([counts(m) for m in matrices])["pq"] == pytest.approx(
        env["get_pq_score"](frame)
    )
    a, b = rectangle(1, 10, 1, 10), rectangle(3, 12, 4, 13)
    native, _ = env["get_overlap_matrices"](
        torch.tensor(a[None].astype(np.float32)), torch.tensor(b[None].astype(np.float32))
    )
    np.testing.assert_allclose(overlap([encode(a)], [encode(b)]), native.numpy(), rtol=1e-6)


@pytest.mark.skipif(
    not Path("artifacts/manifests/train_manifest.csv").exists(),
    reason="Local real-data split check",
)
def test_actual_manifest_group_integrity_and_time_embargo():
    m = pd.read_csv("artifacts/manifests/train_manifest.csv")
    assert m.stem.is_unique
    assert m.groupby("group").fold.nunique().max() == 1
    assert m.groupby("pixel_sha256").fold.nunique().max() == 1
    assert (m.role == "train").sum() > 100
    a = pd.to_datetime(m.loc[m.role == "train", "date"]).dt.as_unit("ns").astype("int64").to_numpy()
    b = (
        pd.to_datetime(m.loc[m.role.isin(["holdout", "calibration"]), "date"])
        .dt.as_unit("ns")
        .astype("int64")
        .to_numpy()
    )
    assert np.abs(a[:, None] - b[None, :]).min() > 3 * 86400 * 1e9
