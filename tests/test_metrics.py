"""Official challenge metrics on tiny synthetic label maps (expected values computed by hand)."""

import numpy as np
import pytest

from wmh_multisite.evaluate.metrics import evaluate_case

SHAPE = (12, 12, 4)


def truth_and(prediction, nifti):
    truth = np.zeros(SHAPE, np.uint8)
    truth[2:5, 2:5, 1] = 1  # lesion A: 9 voxels
    truth[8:10, 8:10, 2] = 1  # lesion B: 4 voxels
    truth[0, 11, 3] = 2  # "other pathology": ignored
    return nifti(truth, "truth.nii.gz"), nifti(prediction.astype(np.uint8), "pred.nii.gz")


def test_perfect_prediction(nifti):
    pred = np.zeros(SHAPE)
    pred[2:5, 2:5, 1] = 1
    pred[8:10, 8:10, 2] = 1
    r = evaluate_case(*truth_and(pred, nifti))
    assert r["dice"] == pytest.approx(1) and r["avd_pct"] == 0 and r["hd95_mm"] == 0
    assert r["lesion_recall"] == 1 and r["lesion_f1"] == 1


def test_label_2_is_ignored(nifti):
    pred = np.zeros(SHAPE)
    pred[2:5, 2:5, 1] = 1
    pred[8:10, 8:10, 2] = 1
    pred[0, 11, 3] = 1  # predicted inside "other pathology": not a false positive
    r = evaluate_case(*truth_and(pred, nifti))
    assert r["dice"] == pytest.approx(1) and r["lesion_f1"] == 1


def test_one_lesion_missed_and_one_false_positive(nifti):
    pred = np.zeros(SHAPE)
    pred[2:5, 2:5, 1] = 1  # A found, B missed
    pred[10, 2, 3] = 1  # 1-voxel false positive
    r = evaluate_case(*truth_and(pred, nifti))
    assert r["dice"] == pytest.approx(2 * 9 / (13 + 10))
    assert r["avd_pct"] == pytest.approx(abs(13 - 10) / 13 * 100)
    assert r["lesion_recall"] == pytest.approx(0.5)  # 1 of 2 true lesions
    assert r["lesion_f1"] == pytest.approx(0.5)  # precision 1/2, recall 1/2
    assert r["true_ml"] == pytest.approx(13 * 3 / 1000)  # 1 x 1 x 3 mm voxels


def test_empty_prediction(nifti):
    r = evaluate_case(*truth_and(np.zeros(SHAPE), nifti))
    assert r["dice"] == 0 and np.isnan(r["hd95_mm"]) and r["lesion_recall"] == 0 and r["avd_pct"] == 100
