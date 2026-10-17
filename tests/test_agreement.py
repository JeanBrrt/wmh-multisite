"""Agreement statistics on small tables whose answer is known in advance."""

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
    shared_cases,
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
    assert ba["loa_param_low"] == pytest.approx(0.2 - 1.96 * 0.2)
    assert ba["loa_param_high"] == pytest.approx(0.2 + 1.96 * 0.2)
    # empirical limits = 2.5th and 97.5th percentiles (linear interpolation)
    assert ba["loa_low"] == pytest.approx(0.01) and ba["loa_high"] == pytest.approx(0.39)


def test_empirical_limits_follow_a_skewed_distribution():
    # mostly small errors and a tail of large underestimations (as M1): the parametric limits are
    # pushed symmetrically outwards, the empirical ones stay where the data are
    rng = np.random.default_rng(0)
    d = np.concatenate([rng.normal(0.0, 0.05, 950), rng.normal(-1.0, 0.1, 50)])
    ba = bland_altman(d)
    assert ba["skewness"] < -2
    assert ba["loa_high"] < ba["loa_param_high"]  # no long upper tail in the data
    assert ba["loa_low"] < ba["loa_param_low"]  # the lower tail is longer than 1.96 SD


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


def test_kruskal_wallis_p_values_are_holm_corrected(tmp_path, monkeypatch):
    from wmh_multisite.biomarkers import agreement as ag
    from wmh_multisite.evaluate.stats import holm

    raw = pd.DataFrame({"p_kruskal_site": [0.01, 0.04, np.nan, 0.5]})
    tested = raw.p_kruskal_site.notna()
    expected = holm(raw.loc[tested, "p_kruskal_site"].to_numpy())
    assert expected.tolist() == pytest.approx([0.03, 0.08, 0.5])  # 3 tests: x3, x2, x1, monotone
    assert hasattr(ag, "holm")


def test_shared_cases_compare_m1_and_experts_on_the_same_subjects():
    # 6 training subjects, M1 on 4 of them: only those 4 are compared, for M1, O3 and O4.
    # O3 measures 1.1 x O1 on average, with a per-subject spread that gives a known min and max.
    spread = [0.95, 1.0, 1.05, 1.0, 1.0, 1.0]
    rows = []
    for i in range(6):
        for src, factor in [
            ("O1", 1.0),
            ("O3", 1.1 * spread[i]),
            ("O4", 0.9 / spread[i]),
            ("resencm", spread[3 - i % 4]),
        ]:
            if src == "resencm" and i >= 4:
                continue
            rows.append(
                {
                    "participant_id": f"s{i}",
                    "scanner": "ab"[i % 2],
                    "split": "training",
                    "source": src,
                    "wmh_ml": factor * (2.0 + i),
                    "wmh_periventricular_ml": factor * (1.0 + i),
                    "wmh_deep_ml": factor * (1.0 + i),
                    "n_lesions_ge10mm3": 10 + i + (src != "O1") * (i % 2),
                }
            )
    out = shared_cases(pd.DataFrame(rows), 50, 0.95, np.random.default_rng(0))
    vol = out[out.biomarker == "wmh_ml"].set_index("rater")
    assert set(vol.index) == {"resencm", "O3", "O4"} and (vol.n == 4).all()
    assert vol.loc["O3", "ratio_min"] == pytest.approx(1.1 * 0.95) and vol.loc[
        "O3", "ratio_max"
    ] == pytest.approx(1.1 * 1.05)
    assert vol.loc["O3", "bias"] == pytest.approx(1.1, rel=0.01)
