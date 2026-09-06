"""Leakage contracts for the separate, not-yet-trained nested CV campaign."""

import numpy as np
import pandas as pd
import pytest

from scripts.make_nested_folds import build_folds


def observations():
    return pd.DataFrame([
        {"stem": f"image-{i}", "date": pd.Timestamp("2010-01-01") + pd.Timedelta(days=i),
         "group": i // 10, "fold": (i // 10) % 5, "pixel_sha256": f"pixels-{i}"}
        for i in range(300)
    ])


def test_nested_folds_keep_outer_groups_out_and_apply_real_three_day_embargo():
    original = observations()
    folds = build_folds(original)
    assert len(folds) == 20
    for name, frame in folds.items():
        outer = int(name.split("/")[0].split("-")[1])
        assert set(frame.loc[frame.role == "holdout", "stem"]) == set(
            original.loc[original.fold == outer, "stem"]
        )
        a = pd.to_datetime(frame.loc[frame.role == "train", "date"]).to_numpy(dtype="datetime64[ns]")
        b = pd.to_datetime(frame.loc[frame.role.isin(["holdout", "calibration"]), "date"]).to_numpy(dtype="datetime64[ns]")
        assert np.abs(a[:, None] - b[None, :]).min() > np.timedelta64(3, "D")
        train_groups = set(frame.loc[frame.role == "train", "group"])
        held_groups = set(frame.loc[frame.role.isin(["holdout", "calibration"]), "group"])
        assert not train_groups & held_groups
        assert (frame.role == "embargo").any()
        if (frame.role == "calibration").any():
            c = pd.to_datetime(frame.loc[frame.role == "calibration", "date"]).to_numpy(dtype="datetime64[ns]")
            d = pd.to_datetime(frame.loc[frame.role == "holdout", "date"]).to_numpy(dtype="datetime64[ns]")
            assert np.abs(c[:, None] - d[None, :]).min() > np.timedelta64(3, "D")
    for outer in range(5):
        calibration_stems = [
            stem for inner in range(3)
            for stem in folds[f"outer-{outer}/inner-{inner}.csv"].query("role == 'calibration'").stem
        ]
        assert len(calibration_stems) == len(set(calibration_stems))
        assert set(calibration_stems) == set(folds[f"outer-{outer}/refit.csv"].query("role == 'train'").stem)


def test_nested_folds_reject_cross_outer_duplicates():
    frame = observations()
    frame.loc[10, "pixel_sha256"] = frame.loc[0, "pixel_sha256"]
    with pytest.raises(ValueError, match="pixel_sha256"):
        build_folds(frame)
