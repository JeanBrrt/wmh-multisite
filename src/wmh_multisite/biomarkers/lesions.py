"""Lesion count and size distribution.

A lesion is a 26-connected component of the WMH mask, as in the official challenge script.

Voxel sizes go from 1.75 to 4.72 mm3 depending on the scanner, so the smallest possible lesion is not the same
everywhere and raw counts are not comparable between sites. We therefore only count lesions of at least
min_lesion_mm3 (10 mm3, about 2 voxels on the coarsest grid).

As the lesion load grows, lesions merge into a single periventricular plaque and the count can go down.
largest_share, the share of the volume held by the largest lesion, keeps track of this.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

STRUCTURE = np.ones((3, 3, 3), bool)  # 26-connectivity, as the official script


def lesion_sizes_mm3(wmh: np.ndarray, voxel_mm3: float) -> np.ndarray:
    """Volume in mm3 of each connected component of the WMH mask."""
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
