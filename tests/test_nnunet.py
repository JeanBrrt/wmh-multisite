"""Tests of the nnU-Net export helpers (no nnU-Net install needed)."""

import numpy as np
import pandas as pd

from wmh_multisite import config
from wmh_multisite.seg.nnunet import LABELS, case_id, dataset_json, stratified_splits


def _cases(n_per_scanner=20, scanners=("a", "b", "c")):
    rng = np.random.default_rng(1)
    rows = [
        {"case_id": f"WMH_{s}{i:02d}", "scanner": s, "wmh_ml": float(rng.gamma(2, 8))}
        for s in scanners
        for i in range(n_per_scanner)
    ]
    return pd.DataFrame(rows)


def test_case_id_keeps_the_original_number():
    assert case_id("sub-000") == "WMH_000" and case_id("sub-169") == "WMH_169"


def test_splits_partition_the_cases():
    cases = _cases()
    splits = stratified_splits(cases, 5, seed=42)
    assert len(splits) == 5
    vals = [c for s in splits for c in s["val"]]
    assert sorted(vals) == sorted(cases.case_id)  # each case validated exactly once
    for s in splits:
        assert not set(s["train"]) & set(s["val"])
        assert len(s["train"]) + len(s["val"]) == len(cases)


def test_splits_are_stratified_by_scanner_and_load():
    cases = _cases().set_index("case_id")
    splits = stratified_splits(cases.reset_index(), 5, seed=42)
    overall = cases.wmh_ml.median()
    for s in splits:
        val = cases.loc[s["val"]]
        assert val.scanner.value_counts().tolist() == [4, 4, 4]  # 20 per scanner / 5 folds
        # each fold spans the whole load range of each scanner: one case per load quintile
        for scanner, group in cases.groupby("scanner"):
            ranks = group.wmh_ml.rank(method="first").astype(int)
            quintiles = sorted(((ranks.loc[val[val.scanner == scanner].index] - 1) // 5).tolist())
            assert quintiles == [0, 1, 2, 3]
        assert 0.5 * overall < val.wmh_ml.median() < 2 * overall


def test_splits_are_reproducible():
    cases = _cases()
    assert stratified_splits(cases, 5, seed=42) == stratified_splits(cases, 5, seed=42)
    assert stratified_splits(cases, 5, seed=42) != stratified_splits(cases, 5, seed=7)


def test_dataset_json_declares_the_ignore_label():
    cfg = config.load_config()
    d = dataset_json(cfg, 60)
    assert d["labels"] == {"background": 0, "WMH": 1, "ignore": 2} == LABELS
    assert max(d["labels"].values()) == d["labels"]["ignore"]  # nnU-Net: ignore must be the highest
    assert d["channel_names"] == {"0": "FLAIR", "1": "T1"}
    assert d["numTraining"] == 60 and d["file_ending"] == ".nii.gz"
