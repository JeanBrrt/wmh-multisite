"""Lesion structure: number and size distribution of the WMH (ROADMAP step 7.3, J-048).

A lesion is a connected component of the WMH mask, with **26-connectivity** (voxels touching by a face,
an edge or a corner), exactly as the official challenge script counts lesions
(``ConnectedComponentImageFilter`` with ``SetFullyConnected(True)``).

Two pitfalls, hence the indicators:
- **The smallest lesion a mask can hold is one voxel, whose size depends on the scanner** (1.75 to
  4.72 mm3 on our FLAIR grids): a fine grid records tiny lesions that a coarse grid cannot represent.
  The raw count is therefore not comparable between sites; ``n_lesions_ge_min`` counts only lesions of
  at least ``min_lesion_mm3`` (10 mm3: about 2 voxels on the coarsest grid, 6 on the finest).
- **Confluence**: as the load grows, lesions merge into one large periventricular plaque, and the
  count can *decrease*. ``largest_share`` (share of the WMH volume in the largest lesion) measures it.
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
