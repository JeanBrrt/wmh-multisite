"""Uncertainty and comparisons on the test-set metrics.

With 10 to 30 test subjects per scanner, a gap between means may just reflect which subjects were
drawn. Every mean gets a 95 % confidence interval, and every comparison a test.

    results/tables/evaluation_<model>.csv  (one row per test subject and model)
        |
        |-- 1. mean + bootstrap CI     per model x group (all, each scanner, seen / unseen) x metric
        |-- 2. paired comparisons      per pair of models x metric: same subjects, per-subject differences
        |-- 3. seen vs unseen          per model x metric: different subjects
        v
    results/tables/stats_ci.csv, stats_paired.csv, stats_seen_unseen.csv

1. Percentile bootstrap: Dice is bounded and skewed, so no closed formula. Resampling is done within
   each scanner, so a resample cannot change the scanner mix and move the mean for that reason alone.
2. Paired comparisons: every model is scored on the same subjects, so per-subject differences cancel
   each subject's difficulty. Wilcoxon signed-rank test and bootstrap CI of the mean difference, Holm
   correction over all the paired tests.
3. Seen vs unseen scanners: different subjects (90 vs 20), so unpaired. Bootstrap CI of the difference
   and Mann-Whitney U. The 20 unseen subjects come from 2 scanners, and without demographics a gap may
   reflect the population as much as the scanner.

HD95 is undefined (NaN) when a method predicts nothing: such subjects are left out of that metric only
(as on the leaderboard) and counted in ``n_missing``.

Usage:
    uv run python -m wmh_multisite.evaluate.stats
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.evaluate.metrics import METRICS

log = logging.getLogger(__name__)


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment: k-th smallest p times (m - k + 1), made monotone, capped at 1."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adjusted_sorted
    return out


def bootstrap_means(
    values: np.ndarray, strata: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """``n_boot`` means of ``values`` resampled with replacement within each stratum (sizes preserved)."""
    total = np.zeros(n_boot)
    for s in np.unique(strata):
        v = values[strata == s]
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        total += v[idx].sum(axis=1)
    return total / len(values)


def mean_ci(
    values: np.ndarray, strata: np.ndarray, n_boot: int, level: float, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI (stratified)."""
    boot = bootstrap_means(values, strata, n_boot, rng)
    alpha = (1 - level) / 2
    return float(values.mean()), float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def groups(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"all": table}
    out |= {f"scanner={k}": g for k, g in table.groupby("scanner")}
    out |= {f"seen={k}": g for k, g in table.groupby("seen_in_training")}
    return out


def ci_table(table: pd.DataFrame, n_boot: int, level: float, seed: int) -> pd.DataFrame:
    """Block 1: mean and CI per model x group x metric."""
    rng = np.random.default_rng(seed)
    rows = []
    for model, tm in table.groupby("model"):
        for name, g in groups(tm).items():
            for metric in METRICS:
                ok = g[metric].notna()
                mean, lo, hi = mean_ci(
                    g.loc[ok, metric].to_numpy(), g.loc[ok, "scanner"].to_numpy(), n_boot, level, rng
                )
                rows.append(
                    {
                        "model": model,
                        "group": name,
                        "metric": metric,
                        "n": int(ok.sum()),
                        "n_missing": int((~ok).sum()),
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    return pd.DataFrame(rows)


def paired_table(table: pd.DataFrame, n_boot: int, level: float, seed: int) -> pd.DataFrame:
    """Block 2: per pair of models (a - b) and metric, on the subjects both models have."""
    rng = np.random.default_rng(seed)
    wide = table.pivot_table(index=["participant_id", "scanner"], columns="model", values=METRICS)
    rows = []
    for a, b in itertools.combinations(sorted(table.model.unique()), 2):
        for metric in METRICS:
            pair = wide[metric][[a, b]].dropna()
            diff = (pair[a] - pair[b]).to_numpy()
            strata = pair.index.get_level_values("scanner").to_numpy()
            mean, lo, hi = mean_ci(diff, strata, n_boot, level, rng)
            p = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(diff).pvalue)
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "n": len(diff),
                    "mean_a": float(pair[a].mean()),
                    "mean_b": float(pair[b].mean()),
                    "mean_diff_a_minus_b": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_wilcoxon": p,
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p_wilcoxon.to_numpy())
    return out


def seen_unseen_table(table: pd.DataFrame, n_boot: int, level: float, seed: int) -> pd.DataFrame:
    """Block 3: unseen - seen scanners, per model and metric (unpaired)."""
    rng = np.random.default_rng(seed)
    alpha = (1 - level) / 2
    rows = []
    for model, tm in table.groupby("model"):
        for metric in METRICS:
            ok = tm[tm[metric].notna()]
            seen, unseen = ok[ok.seen_in_training], ok[~ok.seen_in_training]
            if seen.empty or unseen.empty:
                continue
            bs = bootstrap_means(seen[metric].to_numpy(), seen.scanner.to_numpy(), n_boot, rng)
            bu = bootstrap_means(unseen[metric].to_numpy(), unseen.scanner.to_numpy(), n_boot, rng)
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_seen": len(seen),
                    "n_unseen": len(unseen),
                    "mean_seen": float(seen[metric].mean()),
                    "mean_unseen": float(unseen[metric].mean()),
                    "diff_unseen_minus_seen": float(unseen[metric].mean() - seen[metric].mean()),
                    "ci_low": float(np.quantile(bu - bs, alpha)),
                    "ci_high": float(np.quantile(bu - bs, 1 - alpha)),
                    "p_mannwhitney": float(stats.mannwhitneyu(unseen[metric], seen[metric]).pvalue),
                }
            )
    return pd.DataFrame(rows)


def load_evaluations(cfg: dict[str, Any]) -> pd.DataFrame:
    """All ``evaluation_<model>.csv`` tables (summaries excluded), test split only."""
    files = sorted(
        f
        for f in (resolve_path(cfg, "results") / "tables").glob("evaluation_*.csv")
        if not f.stem.endswith("_summary")
    )
    table = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return table[table.split == "test"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CIs and tests on the official metrics.")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    e = cfg["evaluation"]
    n_boot, level, seed = e["bootstrap_samples"], e["confidence_level"], e["seed"]
    table = load_evaluations(cfg)
    log.info("models: %s; %d subjects", sorted(table.model.unique()), table.participant_id.nunique())
    out = resolve_path(cfg, "results") / "tables"
    results = {
        "stats_ci": ci_table(table, n_boot, level, seed),
        "stats_paired": paired_table(table, n_boot, level, seed),
        "stats_seen_unseen": seen_unseen_table(table, n_boot, level, seed),
    }
    for name, df in results.items():
        df.round({c: 5 for c in df.columns if not c.startswith("p_")}).to_csv(  # p-values kept exact
            out / f"{name}.csv", index=False
        )
    pd.set_option("display.width", 200)
    ci = results["stats_ci"]
    log.info("Dice, mean [95 %% CI]:\n%s", ci[ci.metric == "dice"].round(3).to_string(index=False))
    log.info("paired:\n%s", results["stats_paired"].to_string(index=False, float_format="%.4g"))
    log.info("unseen - seen:\n%s", results["stats_seen_unseen"].to_string(index=False, float_format="%.4g"))


if __name__ == "__main__":
    main()
