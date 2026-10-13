"""WMH load: volume and fraction of the brain.

- WMH volume (ml) = WMH voxels x voxel volume. Voxels go from 1.75 mm3 (Amsterdam Philips) to 4.72 mm3
  (Amsterdam GE 1.5T), so voxel counts are never compared as such.
- Fraction of the brain (%) = WMH volume / HD-BET brain volume. Never the head mask, whose extent
  depends on defacing and on the site. The brain shrinks with atrophy, which inflates this fraction.
- Fraction of the intracranial volume (%), which does not shrink with atrophy (CSF fills the space).
  ICV = every labelled voxel of the 1 mm WMH-SynthSeg segmentation.

Label 2 of O1 ("other pathology") is removed from every segmentation first, as in the official
evaluation.
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
    """Intracranial volume (ml): every non-zero voxel of a whole-head anatomical segmentation."""
    return float(np.count_nonzero(labels)) * voxel_mm3 / 1000
