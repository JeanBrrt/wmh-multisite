"""Preprocessing experiment (inputs) and robustness study: case building, naming, subject choice."""

import pandas as pd

from wmh_multisite.qc import robustness
from wmh_multisite.seg import nnunet_infer


def test_challenge_inputs_get_their_own_prediction_name(tmp_path, monkeypatch):
    seen = {}

    def fake_predict(cfg, model, cases, staging_name):
        seen["cases"], seen["staging"] = cases, staging_name
        return pd.DataFrame()

    monkeypatch.setattr(nnunet_infer, "predict_cases", fake_predict)
    monkeypatch.setattr(nnunet_infer, "resolve_path", lambda cfg, key: tmp_path / key)
    nnunet_infer.run({}, ["sub-001"], "resencm", inputs="challenge")
    case = seen["cases"][0]
    assert case["out"].name == "sub-001_space-FLAIR_desc-resencmpre_dseg.nii.gz"
    assert case["flair"].name == "sub-001_desc-spm12_FLAIR.nii.gz"
    assert case["t1"].name == "sub-001_space-FLAIR_desc-spm12_T1w.nii.gz"
    nnunet_infer.run({}, ["sub-001"], "resencm", inputs="ours")
    assert seen["cases"][0]["out"].name == "sub-001_space-FLAIR_desc-resencm_dseg.nii.gz"


def test_robustness_uses_usable_test_subjects_per_scanner(tmp_path, monkeypatch):
    (tmp_path / "results" / "tables").mkdir(parents=True)
    (tmp_path / "bids").mkdir()
    pd.DataFrame(
        {
            "participant_id": ["a1", "a2", "a3", "b1", "b2", "t1"],
            "scanner": ["a", "a", "a", "b", "b", "a"],
            "qc_class": ["usable", "check", "usable", "usable", "usable", "usable"],
        }
    ).to_csv(tmp_path / "results" / "tables" / "qc_flags.csv", index=False)
    pd.DataFrame(
        {
            "participant_id": ["a1", "a2", "a3", "b1", "b2", "t1"],
            "split": ["test", "test", "test", "test", "test", "training"],
        }
    ).to_csv(tmp_path / "bids" / "participants.tsv", sep="\t", index=False)
    monkeypatch.setattr(robustness, "resolve_path", lambda cfg, key: tmp_path / key)
    chosen = robustness.choose_subjects({}, n_per_scanner=1)
    assert chosen == ["a1", "b1"]  # usable and test only (a2 flagged, t1 training)
