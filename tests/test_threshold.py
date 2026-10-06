"""FLAIR thresholding (M3) on small synthetic volumes whose answer is known in advance."""

import numpy as np
import pytest

from wmh_multisite.seg.threshold import depth_mm, remove_small, segment, zscore_in_brain

SHAPE = (20, 20, 10)
SPACING = (1.0, 1.0, 3.0)


def phantom():
    """A 'brain' box, flat tissue at 100 with noise, one deep bright lesion, one bright border rim."""
    rng = np.random.default_rng(0)
    brain = np.zeros(SHAPE, bool)
    brain[2:18, 2:18, 1:9] = True
    flair = np.where(brain, rng.normal(100, 5, SHAPE), 0).astype(np.float32)
    flair[9:12, 9:12, 4:6] = 160  # deep lesion: 18 voxels, 54 mm3
    flair[2, 2:18, 1:9] = 160  # bright rim on the brain border (cortex / meninges-like)
    return flair, brain


def test_zscore_uses_brain_only():
    flair, brain = phantom()
    z = zscore_in_brain(flair, brain)
    assert abs(np.median(z[brain])) < 0.05  # centred on the brain
    flair_out = flair.copy()
    flair_out[~brain] = 5000  # very bright outside the brain (e.g. face, fat): must not change z inside
    assert np.allclose(zscore_in_brain(flair_out, brain)[brain], z[brain])


def test_depth_is_distance_to_border_in_mm():
    _, brain = phantom()
    d = depth_mm(brain, SPACING)
    assert d[2, 10, 5] == pytest.approx(1.0)  # outermost x layer: 1 voxel of 1 mm from outside
    assert d[10, 10, 1] == pytest.approx(3.0)  # outermost z layer: 1 voxel of 3 mm
    assert d[0, 0, 0] == 0


def test_depth_removes_border_rim_and_keeps_deep_lesion():
    flair, brain = phantom()
    z, d = zscore_in_brain(flair, brain), depth_mm(brain, SPACING)
    shallow = segment(z, d, 3.0, threshold=4, min_depth_mm=0, min_size_mm3=0)
    deep = segment(z, d, 3.0, threshold=4, min_depth_mm=2, min_size_mm3=0)
    assert shallow[2].any() and not deep[2].any()  # rim gone
    assert deep[9:12, 9:12, 4:6].all() and deep.sum() == 18  # lesion only


def test_remove_small_counts_in_mm3():
    mask = np.zeros(SHAPE, bool)
    mask[1, 1, 1] = True  # 1 voxel = 3 mm3
    mask[5:7, 5:7, 5] = True  # 4 voxels = 12 mm3
    kept = remove_small(mask, min_size_mm3=10, voxel_mm3=3.0)
    assert not kept[1, 1, 1] and kept[5:7, 5:7, 5].all()
    diagonal = np.zeros(SHAPE, bool)
    diagonal[1, 1, 1] = diagonal[2, 2, 2] = True  # touching by a corner: one 26-connected component
    assert remove_small(diagonal, min_size_mm3=6, voxel_mm3=3.0).sum() == 2


def test_grid_counts_match_direct_segmentation(nifti, monkeypatch):
    """The fast grid search (one labelling per (t, e)) gives the same counts as segment()."""
    from wmh_multisite.seg import threshold as th

    flair, brain = phantom()
    truth = np.zeros(SHAPE, np.uint8)
    truth[9:12, 9:12, 4:6] = 1
    truth[10, 10, 6] = 1  # a lesion voxel the threshold misses
    truth[2, 5, 5] = 2  # "other pathology" on the bright rim: excluded from the counts
    affine = np.diag([*SPACING, 1.0])
    paths = {
        "flair_n4": nifti(flair, "flair.nii.gz", affine),
        "brain": nifti(brain.astype(np.uint8), "brain.nii.gz", affine),
        "truth": nifti(truth, "truth.nii.gz", affine),
    }
    monkeypatch.setattr(th, "load_config", lambda _=None: {})
    monkeypatch.setattr(th, "subject_paths", lambda cfg, sub: paths)
    grid = {"t": [2.0, 4.0], "e": [0, 2], "m": [0, 60]}
    counts = th.grid_counts("sub-x", grid)
    z, d = zscore_in_brain(flair, brain), depth_mm(brain, SPACING)
    for i, t in enumerate(grid["t"]):
        for j, e in enumerate(grid["e"]):
            for k, m in enumerate(grid["m"]):
                seg = segment(z, d, 3.0, t, e, m)
                expected = ((seg & (truth == 1)).sum(), (seg & (truth != 2)).sum(), (truth == 1).sum())
                assert tuple(counts[i, j, k]) == expected, (t, e, m)
    assert tuple(counts[1, 1, 0]) == (18, 18, 19)  # t=4, e=2: the deep lesion exactly, 1 voxel missed
