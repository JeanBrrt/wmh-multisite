"""Periventricular and deep WMH.

Clinical scales such as Fazekas rate the two separately, because the same load can sit around the ventricles
or be scattered in the deep white matter. We use the most common rule in automated studies: a WMH voxel within
periventricular_distance_mm (10 mm) of the lateral ventricles is periventricular, otherwise it is deep. The
rule works voxel by voxel, so a confluent plaque that reaches far from the ventricles is split between the
two.

Distances are computed in millimetres with the real voxel size. On 3 mm slices, counting voxels would make 10
mm mean 10 mm in-plane but 30 mm across slices.

The ventricles come from WMH-SynthSeg (labels 4 and 43). All sources of a subject share the same ventricle
mask, so comparing sources compares their WMH and not their ventricles. WMH-SynthSeg gives the WMH their own
label, so its ventricles do not absorb the lesions.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def distance_to_ventricles_mm(ventricles: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    """Distance in mm from every voxel to the nearest ventricle voxel (0 inside the ventricles)."""
    if not ventricles.any():
        raise ValueError("empty ventricle mask")
    return ndimage.distance_transform_edt(~ventricles.astype(bool), sampling=spacing)


def location_biomarkers(
    wmh: np.ndarray, ventricles: np.ndarray, spacing: tuple[float, ...], threshold_mm: float
) -> dict[str, float]:
    """Periventricular and deep WMH volumes (ml) and the periventricular share of the volume."""
    voxel_ml = float(np.prod(spacing)) / 1000
    near = distance_to_ventricles_mm(ventricles, spacing) <= threshold_mm
    wmh = wmh.astype(bool)
    pv = float(np.count_nonzero(wmh & near)) * voxel_ml
    deep = float(np.count_nonzero(wmh & ~near)) * voxel_ml
    return {
        "wmh_periventricular_ml": pv,
        "wmh_deep_ml": deep,
        "periventricular_share": pv / (pv + deep) if pv + deep else float("nan"),
    }
