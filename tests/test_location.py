"""Periventricular / deep split (7.2) on small masks with known distances (1 x 1 x 3 mm voxels)."""

import numpy as np
import pytest

from wmh_multisite.biomarkers.location import distance_to_ventricles_mm, location_biomarkers

SHAPE = (40, 10, 10)
SPACING = (1.0, 1.0, 3.0)


def ventricles():
    v = np.zeros(SHAPE, bool)
    v[0:5, 4:6, 4:6] = True  # a small block; its edge is at x = 4, z = 4..5
    return v


def test_distance_uses_the_real_voxel_size():
    d = distance_to_ventricles_mm(ventricles(), SPACING)
    assert d[2, 5, 5] == 0  # inside
    assert d[14, 5, 5] == pytest.approx(10.0)  # 10 voxels of 1 mm along x
    assert d[2, 5, 8] == pytest.approx(9.0)  # 3 slices of 3 mm along z (counting voxels would say 3)


def test_ten_mm_rule_voxel_wise():
    wmh = np.zeros(SHAPE, bool)
    wmh[10, 5, 5] = True  # 6 mm in-plane: periventricular
    wmh[14, 5, 5] = True  # exactly 10 mm: periventricular (<=)
    wmh[20, 5, 5] = True  # 16 mm: deep
    wmh[2, 5, 9] = True  # 4 slices away = 12 mm across slices: deep, although only 4 voxels away
    b = location_biomarkers(wmh, ventricles(), SPACING, threshold_mm=10)
    assert b["wmh_periventricular_ml"] == pytest.approx(2 * 3 / 1000)
    assert b["wmh_deep_ml"] == pytest.approx(2 * 3 / 1000)
    assert b["periventricular_share"] == pytest.approx(0.5)


def test_a_confluent_plaque_is_split():
    wmh = np.zeros(SHAPE, bool)
    wmh[5:25, 5, 5] = True  # one lesion from the ventricle edge out to 20 mm
    b = location_biomarkers(wmh, ventricles(), SPACING, threshold_mm=10)
    assert b["wmh_periventricular_ml"] == pytest.approx(10 * 3 / 1000)  # x = 5..14 (1 to 10 mm)
    assert b["wmh_deep_ml"] == pytest.approx(10 * 3 / 1000)  # x = 15..24


def test_no_wmh_and_no_ventricles():
    b = location_biomarkers(np.zeros(SHAPE, bool), ventricles(), SPACING, 10)
    assert b["wmh_periventricular_ml"] == 0 and np.isnan(b["periventricular_share"])
    with pytest.raises(ValueError, match="empty ventricle mask"):
        distance_to_ventricles_mm(np.zeros(SHAPE, bool), SPACING)
