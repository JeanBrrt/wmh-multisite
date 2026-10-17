"""Confidence intervals and statistical tests on the test-set metrics.

With 10 to 30 test subjects per scanner, a gap between two means can simply come from the subjects that
happened to be drawn. Every mean therefore gets a 95 % confidence interval and every comparison a test. Five
tables are produced from results/tables/evaluation_<model>.csv:

- stats_ci.csv: mean and bootstrap CI per model, group (all, each scanner, seen, unseen) and metric. The
  bootstrap resamples within each scanner, so that a resample cannot shift the mean just by changing the mix
  of scanners;
- stats_paired.csv: paired comparisons of two models on the same subjects, which removes each subject's own
  difficulty. Wilcoxon signed-rank test and bootstrap CI of the mean difference;
- stats_seen_unseen.csv: unseen minus seen scanners for each model. These are different subjects (20 against
  90), so the test is Mann-Whitney;
- stats_gain_unseen.csv: whether the gain of a variant over its base model is larger on the unseen scanners;
- stats_scanner_effect.csv: Kruskal-Wallis test of each metric across the five scanners.

Tests are grouped into families, one per question, as listed in evaluation.families (the main study M1/M2/M3,
then each side experiment). The Holm correction is applied within each family, so a side experiment does not
weigh on the correction of the main study and the other way round.

HD95 is undefined when a method predicts nothing. Those subjects are left out of that metric only, as on the
leaderboard, and counted in n_missing. The 20 unseen subjects come from 2 scanners, and without demographic
data a gap may reflect the population as much as the scanner.

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
    """Holm step-down correction: the k-th smallest p is multiplied by (m - k + 1), kept monotone and capped
    at 1.
    """
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
    """Bootstrap means of values, resampled with replacement within each stratum."""
    total = np.zeros(n_boot)
    for s in np.unique(strata):
        v = values[strata == s]
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        total += v[idx].sum(axis=1)
    return total / len(values)


def mean_ci(
    values: np.ndarray, strata: np.ndarray, n_boot: int, level: float, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Mean and stratified percentile bootstrap CI."""
    boot = bootstrap_means(values, strata, n_boot, rng)
    alpha = (1 - level) / 2
    return float(values.mean()), float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))


def groups(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"all": table}
    out |= {f"scanner={k}": g for k, g in table.groupby("scanner")}
    out |= {f"seen={k}": g for k, g in table.groupby("seen_in_training")}
    return out


def ci_table(table: pd.DataFrame, n_boot: int, level: float, seed: int) -> pd.DataFrame:
    """Mean and CI per model, group and metric."""
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


def model_families(models: list[str], families: dict[str, list[str]] | None) -> dict[str, list[str]]:
    """Families restricted to the evaluated models. Without families, all models form a single family."""
    if not families:
        return {"all": sorted(models)}
    return {name: [m for m in members if m in models] for name, members in families.items()}


def paired_table(
    table: pd.DataFrame, n_boot: int, level: float, seed: int, families: dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """Paired differences (a - b) for each pair of models in a family and each metric.

    Only the subjects that both models have are used. The Holm correction (p_holm) is applied within each
    family.
    """
    rng = np.random.default_rng(seed)
    wide = table.pivot_table(index=["participant_id", "scanner"], columns="model", values=METRICS)
    out = []
    for family, members in model_families(sorted(table.model.unique()), families).items():
        rows = []
        for a, b in itertools.combinations(sorted(members), 2):
            for metric in METRICS:
                pair = wide[metric][[a, b]].dropna()
                diff = (pair[a] - pair[b]).to_numpy()
                strata = pair.index.get_level_values("scanner").to_numpy()
                mean, lo, hi = mean_ci(diff, strata, n_boot, level, rng)
                p = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(diff).pvalue)
                rows.append(
                    {
                        "family": family,
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
        if rows:
            fam = pd.DataFrame(rows)
            fam["p_holm"] = holm(fam.p_wilcoxon.to_numpy())
            out.append(fam)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def first_family(models: list[str], families: dict[str, list[str]] | None) -> dict[str, str]:
    """First family listing each model, used for the per-model tests."""
    family_of: dict[str, str] = {}
    for family, members in model_families(models, families).items():
        for m in members:
            family_of.setdefault(m, family)
    return family_of


def seen_unseen_table(
    table: pd.DataFrame, n_boot: int, level: float, seed: int, families: dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """Unseen minus seen scanners, per model and metric (unpaired).

    The Holm correction is applied within the first family that lists the model.
    """
    rng = np.random.default_rng(seed)
    alpha = (1 - level) / 2
    family_of = first_family(sorted(table.model.unique()), families)
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
                    "family": family_of.get(model, "other"),
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
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = out.groupby("family").p_mannwhitney.transform(lambda p: holm(p.to_numpy()))
    return out


def scanner_effect_table(table: pd.DataFrame, families: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Kruskal-Wallis test of each metric across the scanners, per model.

    A low p means the metric depends on the scanner (for the threshold, an intensity scale that the
    normalisation did not align). As for the seen versus unseen tests, Holm is applied within the family of
    each model, which gives a third family of 15 tests for the main study.
    """
    family_of = first_family(sorted(table.model.unique()), families)
    rows = []
    for model, tm in table.groupby("model"):
        for metric in METRICS:
            groups = [g[metric].dropna().to_numpy() for _, g in tm.groupby("scanner")]
            groups = [g for g in groups if len(g)]
            rows.append(
                {
                    "family": family_of.get(model, "other"),
                    "model": model,
                    "metric": metric,
                    "n_scanners": len(groups),
                    "p_kruskal": float(stats.kruskal(*groups).pvalue) if len(groups) > 1 else float("nan"),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = out.groupby("family").p_kruskal.transform(lambda p: holm(p.to_numpy()))
    return out


def gain_unseen_table(
    table: pd.DataFrame, pairs: list[list[str]], n_boot: int, level: float, seed: int
) -> pd.DataFrame:
    """Is the per-subject gain A - B larger on the unseen scanners than on the seen ones?

    For a variant meant to generalise better, such as strong augmentation, a gain over all subjects is not the
    real question. The gain should be larger where the scanner was never seen. We compare the per-subject
    differences A - B of the two groups with a Mann-Whitney test and give a bootstrap CI of the difference of
    their means. Each pair is one question asked on the five metrics, so Holm is applied over its 5 tests.
    """
    rng = np.random.default_rng(seed)
    alpha = (1 - level) / 2
    rows = []
    for a, b in pairs:
        if not {a, b} <= set(table.model):
            continue
        for metric in METRICS:
            wide = table.pivot_table(
                index=["participant_id", "scanner", "seen_in_training"], columns="model", values=metric
            )
            pair = wide[[a, b]].dropna()
            diff = (pair[a] - pair[b]).reset_index()
            diff.columns = ["participant_id", "scanner", "seen_in_training", "gain"]
            seen, unseen = diff[diff.seen_in_training], diff[~diff.seen_in_training]
            bs = bootstrap_means(seen.gain.to_numpy(), seen.scanner.to_numpy(), n_boot, rng)
            bu = bootstrap_means(unseen.gain.to_numpy(), unseen.scanner.to_numpy(), n_boot, rng)
            rows.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "metric": metric,
                    "n_seen": len(seen),
                    "n_unseen": len(unseen),
                    "gain_seen": float(seen.gain.mean()),
                    "gain_unseen": float(unseen.gain.mean()),
                    "diff_unseen_minus_seen": float(unseen.gain.mean() - seen.gain.mean()),
                    "ci_low": float(np.quantile(bu - bs, alpha)),
                    "ci_high": float(np.quantile(bu - bs, 1 - alpha)),
                    "p_mannwhitney": float(stats.mannwhitneyu(unseen.gain, seen.gain).pvalue),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = out.groupby(["model_a", "model_b"]).p_mannwhitney.transform(
            lambda p: holm(p.to_numpy())
        )
    return out


def load_evaluations(cfg: dict[str, Any]) -> pd.DataFrame:
    """All evaluation_<model>.csv tables except the summaries, test split only."""
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
    families = e.get("families")
    results = {
        "stats_ci": ci_table(table, n_boot, level, seed),
        "stats_paired": paired_table(table, n_boot, level, seed, families),
        "stats_seen_unseen": seen_unseen_table(table, n_boot, level, seed, families),
        "stats_gain_unseen": gain_unseen_table(table, e.get("gain_unseen", []), n_boot, level, seed),
        "stats_scanner_effect": scanner_effect_table(table, families),
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
    log.info("gain on unseen:\n%s", results["stats_gain_unseen"].to_string(index=False, float_format="%.4g"))


if __name__ == "__main__":
    main()
