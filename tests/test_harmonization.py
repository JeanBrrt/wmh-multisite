"""Phase 8 (8.2, 8.3): NAWM features, site classifier and ComBat on simulated multi-site data."""

import numpy as np
import pandas as pd
import pytest

from wmh_multisite.harmonize.combat import eta_squared, site_classifier
from wmh_multisite.harmonize.features import tissue_features


def simulated(seed=0, n_per_site=20):
    """5 sites, 6 features: a strong additive and multiplicative site effect + a biological signal."""
    rng = np.random.default_rng(seed)
    rows, x = [], []
    for s in range(5):
        shift, scale = rng.normal(0, 2, 6), rng.uniform(0.7, 1.4, 6)
        for _ in range(n_per_site):
            load = rng.normal(0, 1)
            x.append((rng.normal(0, 1, 6) + 1.5 * load) * scale + shift)
            rows.append({"SITE": f"s{s}", "log_wmh": load, "icv": rng.normal(1450, 100)})
    return np.array(x), pd.DataFrame(rows)


def test_eta_squared():
    groups = np.array([0, 0, 1, 1])
    assert eta_squared(np.array([1.0, 1.0, 3.0, 3.0]), groups) == pytest.approx(1.0)  # all variance between
    assert eta_squared(np.array([1.0, 3.0, 1.0, 3.0]), groups) == pytest.approx(0.0)  # all within


def test_combat_removes_the_site_signature_learnt_on_training_folds():
    x, cov = simulated()
    site = cov.SITE.to_numpy()
    before = np.mean(site_classifier(x, site, None, seed=0))
    after = np.mean(site_classifier(x, site, cov, seed=0))
    assert before > 0.8  # the site is easy to guess
    assert after < 0.45  # close to chance (0.20) once harmonised


def test_combat_keeps_the_biological_signal():
    from neuroHarmonize import harmonizationLearn
    from scipy import stats

    x, cov = simulated(seed=1)
    _, xh = harmonizationLearn(x, cov)
    rho_before = np.mean([abs(stats.spearmanr(x[:, j], cov.log_wmh).statistic) for j in range(6)])
    rho_after = np.mean([abs(stats.spearmanr(xh[:, j], cov.log_wmh).statistic) for j in range(6)])
    assert rho_after > rho_before  # the site noise hid part of the biology


def test_tissue_features_on_known_values():
    image = np.zeros((10, 10, 10), np.float32)
    nawm = np.zeros(image.shape, bool)
    nawm[2:8, 2:8, 2:8] = True
    cortex = np.zeros(image.shape, bool)
    cortex[0:2] = True
    image[nawm] = np.linspace(0, 1, nawm.sum())
    image[cortex] = 2.0
    f = tissue_features(image, nawm, cortex)
    assert f["nawm_p50"] == pytest.approx(0.5, abs=0.01) and f["cortex_p50"] == 2.0
    assert f["gm_minus_wm"] == pytest.approx(1.5, abs=0.01) and f["nawm_iqr"] == pytest.approx(0.5, abs=0.01)
