"""Lesion structure: number and size distribution of the WMH.

A lesion is a 26-connected component of the WMH mask, as in the official challenge script.

Two pitfalls shape the indicators:
- The smallest possible lesion is one voxel, and voxels range from 1.75 to 4.72 mm3 across scanners, so
  raw counts are not comparable between sites. ``n_lesions_ge_min`` only counts lesions of at least
  ``min_lesion_mm3`` (10 mm3: about 2 voxels on the coarsest grid, 6 on the finest).
- As the load grows, lesions merge into one periventricular plaque and the count can drop.
  ``largest_share`` (share of the volume in the largest lesion) tracks this.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

STRUCTURE = np.ones((3, 3, 3), bool)  # 26-connectivity, as the official script


def lesion_sizes_mm3(wmh: np.ndarray, voxel_mm3: float) -> np.ndarray:
    """Volume (mm3) of each connected component of the WMH mask."""
    lab, n = ndimage.label(wmh.astype(bool), structure=STRUCTURE)
    return np.bincount(lab.ravel(), minlength=n + 1)[1:] * voxel_mm3


def lesion_biomarkers(wmh: np.ndarray, voxel_mm3: float, min_lesion_mm3: float) -> dict[str, float]:
    sizes = lesion_sizes_mm3(wmh, voxel_mm3)
    total = sizes.sum()
    return {
        "n_lesions": int(len(sizes)),
        "n_lesions_ge_min": int((sizes >= min_lesion_mm3).sum()),
        "lesion_median_mm3": float(np.median(sizes)) if len(sizes) else float("nan"),
        "largest_lesion_ml": float(sizes.max()) / 1000 if len(sizes) else 0.0,
        "largest_share": float(sizes.max() / total) if total else float("nan"),
    }
