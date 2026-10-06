"""WMH load: volume and fraction of the brain (ROADMAP step 7.1, J-048).

- **WMH volume** (ml) = number of WMH voxels x voxel volume. The voxel volume differs between scanners
  (FLAIR grid: 1.75 mm3 at Amsterdam Philips to 4.72 mm3 at Amsterdam GE 1.5T), so voxels are never
  compared without this conversion.
- **Fraction of the brain** (%) = WMH volume / volume of the HD-BET brain mask (J-028, rule A: never the
  head mask, whose extent depends on defacing and on the site). Limit: the brain shrinks with atrophy,
  so at equal WMH volume the fraction rises.
- **Fraction of the intracranial volume** (%), the reference normalisation: the ICV does not shrink with
  atrophy (the space left by the brain fills with CSF). ICV = every labelled voxel of the 1 mm
  WMH-SynthSeg segmentation (brain, ventricles, extracerebral CSF; label 0 = outside the skull).

The "other pathology" regions (label 2 of the reference O1) are removed from every segmentation before
any count, as in the official evaluation.
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
