"""Test-set statistics on small tables whose answer is known in advance."""

import numpy as np
import pandas as pd
import pytest

from wmh_multisite.evaluate.metrics import METRICS
from wmh_multisite.evaluate.stats import (
    bootstrap_means,
    ci_table,
    gain_unseen_table,
    holm,
    paired_table,
    scanner_effect_table,
    seen_unseen_table,
)


def test_holm_matches_hand_computation():
    # sorted: 0.01 x 3 = 0.03 ; 0.03 x 2 = 0.06 ; 0.04 x 1 = 0.04 -> made monotone: 0.06
    assert holm(np.array([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])
    assert holm(np.array([0.5, 0.9])).max() <= 1


def test_bootstrap_is_stratified():
    # stratum "a": three 0, stratum "b": one 1. If the strata sizes are kept, every resample has
    # exactly three 0 and one 1, so every bootstrap mean is 0.25 (an unstratified bootstrap would vary).
    values, strata = np.array([0.0, 0.0, 0.0, 1.0]), np.array(["a", "a", "a", "b"])
    boot = bootstrap_means(values, strata, 500, np.random.default_rng(0))
    assert np.allclose(boot, 0.25)


def fake_table(n_per_scanner=10, seed=0):
    """Two models on the same subjects: B = A - 0.1 on the Dice; scanner u is 'unseen' and 0.2 lower."""
    rng = np.random.default_rng(seed)
    rows = []
    for scanner, seen, offset in [("s", True, 0.0), ("t", True, 0.0), ("u", False, -0.2)]:
        for i in range(n_per_scanner):
            base = {m: rng.uniform(0.5, 0.9) for m in METRICS}
            base["dice"] = 0.8 + offset + rng.normal(0, 0.02)
            for model, shift in [("A", 0.0), ("B", -0.1)]:
                row = {
                    "participant_id": f"{scanner}{i}",
                    "scanner": scanner,
                    "split": "test",
                    "seen_in_training": seen,
                    "model": model,
                    **base,
                }
                row["dice"] = base["dice"] + shift
                rows.append(row)
    return pd.DataFrame(rows)


def test_ci_means_equal_plain_means_and_contain_them():
    t = fake_table()
    ci = ci_table(t, 300, 0.95, seed=1)
    for (model, group), g in [
        (("A", "all"), t[t.model == "A"]),
        (("B", "scanner=u"), t[(t.model == "B") & (t.scanner == "u")]),
    ]:
        row = ci[(ci.model == model) & (ci.group == group) & (ci.metric == "dice")].iloc[0]
        assert row["mean"] == pytest.approx(g.dice.mean())
        assert row.ci_low <= row["mean"] <= row.ci_high


def test_ci_is_reproducible_with_the_seed():
    t = fake_table()
    assert ci_table(t, 200, 0.95, seed=3).equals(ci_table(t, 200, 0.95, seed=3))


def test_paired_difference_and_missing_hd95():
    t = fake_table()
    t.loc[(t.model == "B") & (t.participant_id == "s0"), "hd95_mm"] = np.nan  # B predicted nothing on s0
    p = paired_table(t, 300, 0.95, seed=1).set_index("metric")
    assert p.loc["dice", "mean_diff_a_minus_b"] == pytest.approx(0.1)
    assert p.loc["dice", "ci_low"] == pytest.approx(0.1) and p.loc["dice", "ci_high"] == pytest.approx(0.1)
    assert p.loc["dice", "p_wilcoxon"] < 1e-4
    assert p.loc["hd95_mm", "n"] == 29  # s0 left out for HD95 only
    assert p.loc["lesion_f1", "p_wilcoxon"] == 1.0  # identical values: no difference
    assert (p.p_holm >= p.p_wilcoxon).all()


def test_seen_unseen_difference():
    s = seen_unseen_table(fake_table(), 300, 0.95, seed=1).set_index(["model", "metric"])
    row = s.loc[("A", "dice")]
    assert row.n_seen == 20 and row.n_unseen == 10
    assert row.diff_unseen_minus_seen == pytest.approx(-0.2, abs=0.03)
    assert row.ci_low < -0.15 and row.ci_high < 0 and row.p_mannwhitney < 1e-3


def test_holm_is_applied_within_each_family():
    # C copies A: the pair (A, B) alone in family "f1", the pairs of A, B and C together in "f2"
    t = fake_table()
    t = pd.concat([t, t[t.model == "A"].assign(model="C")], ignore_index=True)
    p = paired_table(t, 100, 0.95, seed=1, families={"f1": ["A", "B"], "f2": ["A", "B", "C"]})
    assert set(p.family) == {"f1", "f2"}
    assert len(p[p.family == "f1"]) == len(METRICS) and len(p[p.family == "f2"]) == 3 * len(METRICS)
    f1, f2 = (
        p[(p.family == f) & (p.model_a == "A") & (p.model_b == "B")].set_index("metric") for f in ("f1", "f2")
    )
    assert f1.loc["dice", "p_wilcoxon"] == pytest.approx(f2.loc["dice", "p_wilcoxon"])
    assert f1.loc["dice", "p_holm"] < f2.loc["dice", "p_holm"]  # a smaller family, a milder correction
    s = seen_unseen_table(t, 100, 0.95, seed=1, families={"f1": ["A", "B"], "f2": ["C"]})
    assert s.groupby("family").size().to_dict() == {"f1": 2 * len(METRICS), "f2": len(METRICS)}


def test_gain_on_unseen_scanners():
    # C = A, except +0.1 Dice on the unseen scanner u only: the gain C - A is larger on unseen scanners
    t = fake_table()
    c = t[t.model == "A"].assign(model="C")
    c.loc[~c.seen_in_training, "dice"] += 0.1
    t = pd.concat([t, c], ignore_index=True)
    g = gain_unseen_table(t, [["C", "A"], ["C", "missing"]], 200, 0.95, seed=1).set_index("metric")
    assert len(g) == len(METRICS)  # the pair with a missing model is skipped
    row = g.loc["dice"]
    assert row.n_seen == 20 and row.n_unseen == 10
    assert row.gain_seen == pytest.approx(0) and row.gain_unseen == pytest.approx(0.1)
    assert row.ci_low == pytest.approx(0.1) and row.p_mannwhitney < 1e-3
    assert (g.p_holm >= g.p_mannwhitney).all()


def test_scanner_effect():
    # scanner u is 0.2 lower on the Dice: a strong scanner effect on the Dice of both models
    s = scanner_effect_table(fake_table()).set_index(["model", "metric"])
    assert s.loc[("A", "dice"), "n_scanners"] == 3
    assert s.loc[("A", "dice"), "p_kruskal"] < 1e-4 and s.loc[("B", "dice"), "p_kruskal"] < 1e-4


def test_scanner_effect_is_corrected_within_each_family():
    t = fake_table()
    s = scanner_effect_table(t, families={"f": ["A", "B"]})
    assert (s.family == "f").all() and (s.p_holm >= s.p_kruskal).all()
