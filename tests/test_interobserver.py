"""Inter-observer agreement (6.3): ignore mask carried over, comparisons and summary on tiny volumes."""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from wmh_multisite.evaluate.interobserver import compare, summarize, with_ignore

SHAPE = (10, 10, 4)


def test_label_2_of_o1_is_ignored_between_two_other_observers(nifti, tmp_path):
    o1 = np.zeros(SHAPE, np.float32)
    o1[1:4, 1:4, 1] = 1
    o1[7:9, 7:9, 2] = 2  # other pathology, according to the reference
    o3 = np.zeros(SHAPE, np.float32)
    o3[1:4, 1:4, 1] = 1
    o4 = o3.copy()
    o4[7:9, 7:9, 2] = 1  # O4 marks the "other pathology" region as WMH: must not count as a disagreement
    p1, p3, p4 = nifti(o1, "o1.nii.gz"), nifti(o3, "o3.nii.gz"), nifti(o4, "o4.nii.gz")
    ref = with_ignore(p3, p1, tmp_path / "o3_ref.nii.gz")
    assert set(np.unique(np.asarray(nib.load(ref).dataobj))) == {0, 1, 2}
    with_mask = compare(("sub-x", "O4", "O3", str(p4), str(ref)))
    without_mask = compare(("sub-x", "O4", "O3", str(p4), str(p3)))
    assert with_mask["dice"] == pytest.approx(1) and with_mask["lesion_f1"] == 1
    assert without_mask["dice"] < 1  # the same pair, label 2 not carried over


def test_dice_is_symmetric_between_observers(nifti):
    a = np.zeros(SHAPE, np.float32)
    a[1:4, 1:4, 1] = 1
    b = np.zeros(SHAPE, np.float32)
    b[2:5, 1:4, 1] = 1
    pa, pb = nifti(a, "a.nii.gz"), nifti(b, "b.nii.gz")
    ab = compare(("s", "B", "A", str(pb), str(pa)))
    ba = compare(("s", "A", "B", str(pa), str(pb)))
    assert ab["dice"] == pytest.approx(ba["dice"]) == pytest.approx(2 * 6 / 18)
    assert ab["hd95_mm"] == pytest.approx(ba["hd95_mm"])


def test_summary_groups_pairs_scanners_and_shared_scans():
    rows = []
    for sub, scanner, shared in [("s1", "x", True), ("s2", "x", False), ("s3", "y", False)]:
        for rater, dice in [("O3", 0.7), ("O4", 0.8)]:
            rows.append(
                {
                    "participant_id": sub,
                    "scanner": scanner,
                    "rater": rater,
                    "reference": "O1",
                    "shared_with_m1": shared,
                    "dice": dice,
                    "hd95_mm": 1.0,
                    "avd_pct": 0.0,
                    "lesion_recall": 1.0,
                    "lesion_f1": 1.0,
                }
            )
    s = summarize(pd.DataFrame(rows))
    assert s.loc["O3 vs O1 | all", "n"] == 3 and s.loc["O4 vs O1 | all", "dice"] == pytest.approx(0.8)
    assert s.loc["O3 vs O1 | scanner=x", "n"] == 2
    assert s.loc["O3 vs O1 | 12 scans shared with M1", "n"] == 1
