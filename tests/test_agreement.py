"""Agreement statistics of 7.4 on small tables whose answer is known in advance."""

import numpy as np
import pandas as pd
import pytest

from wmh_multisite.biomarkers.agreement import (
    agreement,
    bland_altman,
    bland_altman_figure,
    by_scanner,
    icc_a1,
    pairs,
    stratified_indices,
)


def test_icc_a1_hand_computation():
    # constant offset of 1 between raters: MSR = 2, MSC = 1.5, MSE = 0 -> ICC(A,1) = 2 / (2 + 2/3 * 1.5) = 2/3
    x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    assert icc_a1(x) == pytest.approx(2 / 3)
    assert icc_a1(np.column_stack([x[:, 0], x[:, 0]])) == pytest.approx(1.0)  # identical raters


def test_bland_altman():
    ba = bland_altman(np.array([0.0, 0.2, 0.4]))
    assert ba["bias"] == pytest.approx(0.2) and ba["sd"] == pytest.approx(0.2)
    assert ba["loa_low"] == pytest.approx(0.2 - 1.96 * 0.2) and ba["loa_high"] == pytest.approx(
        0.2 + 1.96 * 0.2
    )


def test_stratified_indices_keep_strata_sizes():
    strata = np.array(["a", "b", "a", "b", "b"])
    idx = stratified_indices(strata, 50, np.random.default_rng(0))
    assert idx.shape == (50, 5)
    assert all((strata[row] == "a").sum() == 2 for row in idx)


def fake_pairs(seed=0):
    """Site s: rater = 1.0 x reference; site t: rater = 1.5 x reference (a site-dependent bias)."""
    rng = np.random.default_rng(seed)
    ref = rng.uniform(1, 50, 40)
    noise = np.exp(rng.normal(0, 0.05, 40))
    scanner = np.array(["s"] * 20 + ["t"] * 20)
    rater = ref * noise * np.where(scanner == "t", 1.5, 1.0)
    return pd.DataFrame(
        {"participant_id": [f"p{i}" for i in range(40)], "scanner": scanner, "rater": rater, "ref": ref}
    )


def test_site_dependent_bias_is_found():
    p = fake_pairs()
    sites = {r["scanner"]: r for r in by_scanner(p, "log", 500, 0.95, np.random.default_rng(1))}
    assert sites["s"]["bias"] == pytest.approx(1.0, abs=0.03)
    assert sites["t"]["bias"] == pytest.approx(1.5, abs=0.05)
    assert sites["t"]["ci_low"] > 1.4 and sites["t"]["p_wilcoxon"] < 1e-4
    a = agreement(p, "log", 300, 0.95, np.random.default_rng(1))
    assert a["p_kruskal_site"] < 1e-4  # the bias differs between sites
    assert a["bias"] == pytest.approx(np.sqrt(1.5), rel=0.05)  # geometric mean of 1.0 and 1.5
    assert a["icc_ci_low"] <= a["icc_a1"] <= a["icc_ci_high"]


def test_zero_volume_is_excluded_from_the_log_scale():
    p = fake_pairs()
    p.loc[0, "rater"] = 0.0
    a = agreement(p, "log", 100, 0.95, np.random.default_rng(1))
    assert a["n"] == 39 and a["n_excluded"] == 1
    assert agreement(p, "log1p", 100, 0.95, np.random.default_rng(1))["n_excluded"] == 0  # counts: ln(x + 1)


def test_pairs_join_each_rater_with_o1_by_split():
    rows = []
    for split, sources in [("test", ["O1", "M"]), ("training", ["O1", "O3"])]:
        for sub in ["a", "b"]:
            for src in sources:
                rows.append(
                    {
                        "participant_id": f"{split}{sub}",
                        "scanner": "s",
                        "split": split,
                        "source": src,
                        "wmh_ml": 2.0 if src == "O1" else 3.0,
                    }
                )
    out = pairs(pd.DataFrame(rows), "wmh_ml")
    assert set(out) == {("test", "M"), ("training", "O3")}
    assert (out[("test", "M")].rater == 3.0).all() and (out[("test", "M")].ref == 2.0).all()


def test_figure_is_written(tmp_path):
    path = tmp_path / "ba.png"
    bland_altman_figure(fake_pairs(), "test", path)
    assert path.is_file() and path.stat().st_size > 0
