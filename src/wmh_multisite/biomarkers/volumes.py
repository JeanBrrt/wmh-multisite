"""WMH volume and its fraction of the brain and of the intracranial volume.

Volumes are computed with the voxel size of each image (from 1.75 to 4.72 mm3 depending on the scanner), so
voxel counts are never compared directly.

The brain fraction uses the HD-BET brain mask rather than a head mask, whose extent depends on the defacing
and the site. The brain shrinks with atrophy, which inflates this fraction, so we also give the fraction of
the intracranial volume (ICV), which does not shrink since CSF fills the space. The ICV is the set of labelled
voxels of the 1 mm WMH-SynthSeg segmentation.

As in the official evaluation, the voxels labelled 2 (other pathology) in O1 are removed from every
segmentation first.
"""

from __future__ import annotations

import numpy as np


def remove_other_pathology(wmh: np.ndarray, reference_labels: np.ndarray) -> np.ndarray:
    """WMH mask without the voxels that the reference labels as other pathology (label 2)."""
    return wmh.astype(bool) & (reference_labels != 2)


def volume_biomarkers(wmh: np.ndarray, brain: np.ndarray, voxel_mm3: float) -> dict[str, float]:
    """WMH volume (ml), brain volume (ml) and WMH fraction of the brain (%)."""
    wmh_ml = float(np.count_nonzero(wmh)) * voxel_mm3 / 1000
    brain_ml = float(np.count_nonzero(brain)) * voxel_mm3 / 1000
    return {
        "wmh_ml": wmh_ml,
        "brain_ml": brain_ml,
        "wmh_pct_brain": 100 * wmh_ml / brain_ml if brain_ml else float("nan"),
    }


def icv_ml(labels: np.ndarray, voxel_mm3: float) -> float:
    """Intracranial volume in ml: all non-zero voxels of a whole-head segmentation."""
    return float(np.count_nonzero(labels)) * voxel_mm3 / 1000
