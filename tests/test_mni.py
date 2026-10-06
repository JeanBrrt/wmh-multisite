"""Lesion maps in MNI space (7.5): transport plumbing with identity transforms, map agreement."""

import numpy as np
import pandas as pd
import pytest

from wmh_multisite.biomarkers.mni import flair_to_mni, map_agreement


def test_identity_transforms_keep_the_lesion(nifti, tmp_path):
    import ants

    shape = (12, 12, 12)
    lesion = np.zeros(shape, np.float32)
    lesion[4:7, 4:7, 4:7] = 1
    flair = nifti(lesion, "flair_mask.nii.gz", np.eye(4))
    t1 = nifti(np.ones(shape, np.float32), "t1.nii.gz", np.eye(4))
    template = nifti(np.ones(shape, np.float32), "tpl.nii.gz", np.eye(4))
    identity = tmp_path / "identity.mat"
    ants.write_transform(
        ants.new_ants_transform(dimension=3, transform_type="AffineTransform"), str(identity)
    )
    zero_field = ants.from_numpy(np.zeros((*shape, 3), np.float32), has_components=True)
    warp = tmp_path / "warp.nii.gz"
    ants.image_write(zero_field, str(warp))
    reg = {"t1": t1, "rigid": identity, "affine": identity, "warp": warp}
    out = flair_to_mni(flair, lesion.astype(bool), reg, template)
    assert out.shape == shape
    assert out.sum() == pytest.approx(27, abs=1e-3)  # same lesion, same place
    assert out[5, 5, 5] == pytest.approx(1) and out[0, 0, 0] == 0


def test_map_agreement_correlates_each_method_with_o1():
    rng = np.random.default_rng(0)
    o1 = rng.random((8, 8, 8)).astype(np.float32)
    maps = {
        ("O1", "test", "all"): o1,
        ("M", "test", "all"): o1 * 2,  # same pattern, twice the frequency
        ("N", "test", "all"): rng.random((8, 8, 8)).astype(np.float32),
        ("O1", "all", "all"): o1,  # not a test map: ignored as a method
    }
    brain = np.ones((8, 8, 8), bool)
    a = map_agreement(maps, brain).set_index("source")
    assert a.loc["M", "pearson_r_vs_O1"] == pytest.approx(1.0)
    assert abs(a.loc["N", "pearson_r_vs_O1"]) < 0.3
    assert a.loc["M", "mean_freq"] == pytest.approx(2 * a.loc["M", "mean_freq_O1"])
    assert isinstance(a, pd.DataFrame) and "O1" not in a.index
