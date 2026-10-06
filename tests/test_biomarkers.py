"""Biomarkers 7.1 (volumes) and 7.3 (lesions) on small masks whose answer is known in advance."""

import numpy as np
import pytest
import SimpleITK as sitk

from wmh_multisite.biomarkers.lesions import lesion_biomarkers, lesion_sizes_mm3
from wmh_multisite.biomarkers.table import subject_rows
from wmh_multisite.biomarkers.volumes import icv_ml, remove_other_pathology, volume_biomarkers

SHAPE = (12, 12, 6)


def masks():
    """A 'brain' of 400 voxels; WMH: a 12-voxel lesion, a 1-voxel speck, and 2 voxels touching by a corner."""
    brain = np.zeros(SHAPE, bool)
    brain[1:11, 1:11, 1:5] = True  # 10 x 10 x 4 = 400 voxels
    wmh = np.zeros(SHAPE, bool)
    wmh[2:5, 2:4, 2:4] = True  # 3 x 2 x 2 = 12 voxels
    wmh[8, 8, 2] = True  # 1 voxel
    wmh[6, 2, 1] = wmh[7, 3, 2] = True  # corner contact: one lesion under 26-connectivity
    return wmh, brain


def test_volumes_in_ml_and_fraction():
    wmh, brain = masks()
    v = volume_biomarkers(wmh, brain, voxel_mm3=3.0)
    assert v["wmh_ml"] == pytest.approx(15 * 3 / 1000)
    assert v["brain_ml"] == pytest.approx(400 * 3 / 1000)
    assert v["wmh_pct_brain"] == pytest.approx(100 * 15 / 400)


def test_label_2_of_the_reference_is_removed():
    wmh, _ = masks()
    ref = np.zeros(SHAPE, np.uint8)
    ref[8, 8, 2] = 2  # the speck lies in an "other pathology" region
    assert remove_other_pathology(wmh, ref).sum() == wmh.sum() - 1


def test_lesions_26_connectivity_sizes_and_threshold():
    wmh, _ = masks()
    sizes = sorted(lesion_sizes_mm3(wmh, voxel_mm3=3.0))
    assert sizes == [3.0, 6.0, 36.0]  # speck, corner pair, 12-voxel lesion
    b = lesion_biomarkers(wmh, voxel_mm3=3.0, min_lesion_mm3=10)
    assert b["n_lesions"] == 3 and b["n_lesions_ge_min"] == 1
    assert b["lesion_median_mm3"] == pytest.approx(6.0)
    assert b["largest_lesion_ml"] == pytest.approx(0.036)
    assert b["largest_share"] == pytest.approx(36 / 45)


def test_lesion_count_matches_the_official_connectivity():
    wmh, _ = masks()
    cc = sitk.ConnectedComponentImageFilter()
    cc.SetFullyConnected(True)  # as in wmh_challenge.getLesionDetection
    cc.Execute(sitk.GetImageFromArray(wmh.astype(np.uint8)))
    assert lesion_biomarkers(wmh, 1.0, 10)["n_lesions"] == cc.GetObjectCount()


def test_minimum_size_is_in_mm3_not_in_voxels():
    wmh, _ = masks()
    # 2-voxel corner pair: 3.5 mm3 on 1.75 mm3 voxels (not counted), 11 mm3 on 5.5 mm3 voxels (counted)
    fine = lesion_biomarkers(wmh, voxel_mm3=1.75, min_lesion_mm3=10)
    coarse = lesion_biomarkers(wmh, voxel_mm3=5.5, min_lesion_mm3=10)
    assert fine["n_lesions"] == coarse["n_lesions"] == 3
    assert fine["n_lesions_ge_min"] == 1 and coarse["n_lesions_ge_min"] == 2


def test_empty_mask():
    b = lesion_biomarkers(np.zeros(SHAPE, bool), 3.0, 10)
    assert b["n_lesions"] == 0 and b["largest_lesion_ml"] == 0 and np.isnan(b["largest_share"])


def test_subject_rows_reads_files(nifti):
    wmh, brain = masks()
    o1 = wmh.astype(np.float32)
    o1[8, 8, 2] = 2  # O1 calls the speck "other pathology"
    affine = np.diag([1.0, 1.0, 3.0, 1.0])
    paths = {
        "brain": nifti(brain.astype(np.uint8), "brain.nii.gz", affine),
        "O1": nifti(o1, "o1.nii.gz", affine),
        "M": nifti(wmh.astype(np.uint8), "m.nii.gz", affine),
    }
    vent = np.zeros(SHAPE, np.uint8)
    vent[2:5, 4:6, 2:4] = 1  # ventricles next to the 12-voxel lesion (1-2 mm), >= 3 mm from the corner pair
    paths["vent"] = nifti(vent, "vent.nii.gz", affine)
    task = {
        "sub": "sub-x",
        "brain": str(paths["brain"]),
        "sources": {"O1": str(paths["O1"]), "M": str(paths["M"])},
        "reference": str(paths["O1"]),
        "min_mm3": 10,
        "ventricles": str(paths["vent"]),
        "ventricle_source": "ventriclessynthseg",
        "pv_mm": 2,
    }
    rows = subject_rows(task)
    by = {r["source"]: r for r in rows}
    assert by["O1"]["wmh_ml"] == pytest.approx(14 * 3 / 1000)  # label-2 voxel is not WMH
    assert by["M"]["wmh_ml"] == pytest.approx(14 * 3 / 1000)  # the model's speck is removed like in 6.1
    assert by["M"]["n_lesions"] == 2
    assert by["M"]["ventricle_source"] == "ventriclessynthseg"
    # 12-voxel lesion: within 2 mm of the ventricles; corner pair: 3.2 and 4.1 mm away -> deep
    assert by["M"]["wmh_periventricular_ml"] == pytest.approx(12 * 3 / 1000)
    assert by["M"]["wmh_deep_ml"] == pytest.approx(2 * 3 / 1000)
    no_vent = subject_rows(task | {"ventricles": None, "ventricle_source": None})
    assert "wmh_deep_ml" not in no_vent[0]  # no ventricle mask: no split (NaN in the table)


def test_ventricle_source_priority(tmp_path, monkeypatch):
    from wmh_multisite.biomarkers import table

    monkeypatch.setattr(table, "resolve_path", lambda cfg, key: tmp_path)
    cfg = {"biomarkers": {"ventricles": ["ventriclessynthseg", "ventricles"]}}
    anat = tmp_path / table.PIPELINE / "sub-x" / "anat"
    anat.mkdir(parents=True)
    assert table.ventricles_for(cfg, "sub-x") == (None, None)
    (anat / "sub-x_space-FLAIR_desc-ventricles_mask.nii.gz").touch()
    assert table.ventricles_for(cfg, "sub-x")[0] == "ventricles"  # fallback: atlas C
    (anat / "sub-x_space-FLAIR_desc-ventriclessynthseg_mask.nii.gz").touch()
    assert table.ventricles_for(cfg, "sub-x")[0] == "ventriclessynthseg"  # preferred when present


def test_icv_counts_every_labelled_voxel():
    labels = np.zeros((10, 10, 10), np.uint8)
    labels[2:8, 2:8, 2:8] = 2  # brain
    labels[4:6, 4:6, 4:6] = 4  # ventricle
    labels[1, 1:9, 1:9] = 24  # extracerebral CSF: inside the skull, counted
    assert icv_ml(labels, voxel_mm3=1.0) == pytest.approx((216 + 64) / 1000)
