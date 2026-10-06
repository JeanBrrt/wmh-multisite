"""Agreement between predicted and manual biomarkers, overall and per site (ROADMAP step 7.4, J-049).

Why: a clinician or a multi-site study uses one number per patient, the WMH volume. The question is not
the voxel overlap (6.1) but "is the volume right?", and above all "is it equally right on every
scanner?": a site-dependent bias would create a site effect that does not come from the patients.

    biomarkers.csv (7.1, 7.3)
      test      : M1, M3 (then M2, DA5)  vs O1
      training  : O3, O4 (60) and M1 on the 12 fold-0 validation subjects  vs O1   (human reference)
        |   per subject: d = ln(rater / O1)        (volumes span 0.8 to 195 ml: errors grow with the
        |                                           volume, Spearman 0.71 in ml, -0.31 on the log scale)
        v
    1. Bland-Altman: bias = mean(d), limits of agreement = bias +/- 1.96 SD(d), back-transformed to ratios
    2. ICC(A,1), absolute agreement, on ln(value), with a bootstrap CI stratified by scanner
    3. bias per scanner (bootstrap CI, Wilcoxon against 0, Holm) + Kruskal-Wallis across scanners:
       does the bias depend on the site?
        v
    results/tables/stats_agreement.csv, stats_agreement_by_scanner.csv,
    results/figures/bland_altman_<split>_<rater>.png

ICC(A,1) (McGraw and Wong 1996, two-way model, absolute agreement, single measure), from the two-way
ANOVA of the n x k table (subjects x raters):

    ICC(A,1) = (MSR - MSE) / (MSR + (k - 1) MSE + k / n (MSC - MSE))

MSR, MSC, MSE: mean squares of rows (subjects), columns (raters) and residual. Unlike the "consistency"
ICC, a constant bias between raters lowers it.

Biomarkers: ``wmh_ml`` (primary), the periventricular and deep volumes of 7.2 (does a method measure
the scattered deep lesions as well as the periventricular plaques?), and the lesion count
``n_lesions_ge10mm3`` (secondary, comparable across sites; transformed with ln(x + 1) since a count can
be 0). A volume of 0 cannot be log-transformed:
such subjects are excluded and counted in ``n_excluded``. A site bias is reported as *associated* with
the site: without demographics, scanner and population cannot be separated (J-004).

Usage:
    uv run python -m wmh_multisite.biomarkers.agreement
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.evaluate.stats import holm

log = logging.getLogger(__name__)

MIN_SUBJECTS = 3  # below this, no bias, CI or test is reported (e.g. a scanner with 1 or 2 subjects)
BIOMARKERS = {
    "wmh_ml": "log",
    # 7.2: subjects with no deep (or no periventricular) WMH are excluded from these two (n_excluded)
    "wmh_periventricular_ml": "log",
    "wmh_deep_ml": "log",
    "n_lesions_ge10mm3": "log1p",
}


def icc_a1(x: np.ndarray) -> float:
    """ICC(A,1) of an n x k table (rows = subjects, columns = raters)."""
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((x - grand) ** 2).sum() - ss_rows - ss_cols
    msr, msc, mse = ss_rows / (n - 1), ss_cols / (k - 1), ss_err / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k / n * (msc - mse)))


def bland_altman(d: np.ndarray) -> dict[str, float]:
    """Bias and 95 % limits of agreement of the differences ``d``."""
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return {"bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}


def transform(values: pd.Series, kind: str) -> pd.Series:
    return np.log(values) if kind == "log" else np.log1p(values)


def stratified_indices(strata: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """``n_boot`` x n index arrays, resampled with replacement within each stratum."""
    out = np.empty((n_boot, len(strata)), int)
    col = 0
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        out[:, col : col + len(members)] = members[rng.integers(0, len(members), size=(n_boot, len(members)))]
        col += len(members)
    return out


def pairs(table: pd.DataFrame, biomarker: str) -> dict[tuple[str, str], pd.DataFrame]:
    """(split, rater) -> subjects with the rater's value and O1's value (inner join)."""
    out = {}
    for split, ts in table.groupby("split"):
        wide = ts.pivot_table(index=["participant_id", "scanner"], columns="source", values=biomarker)
        for rater in wide.columns.drop("O1", errors="ignore"):
            p = wide[[rater, "O1"]].dropna().rename(columns={rater: "rater", "O1": "ref"}).reset_index()
            if len(p):
                out[(split, rater)] = p
    return out


def agreement(
    p: pd.DataFrame, kind: str, n_boot: int, level: float, rng: np.random.Generator
) -> dict[str, Any]:
    """Blocks 1 and 2 for one rater vs O1, plus the Kruskal-Wallis test of block 3."""
    ok = (p.rater > 0) & (p.ref > 0) if kind == "log" else pd.Series(True, index=p.index)
    q = p[ok]
    if len(q) < MIN_SUBJECTS:
        return {"n": len(q), "n_excluded": int((~ok).sum())}
    tr, tref = transform(q.rater, kind).to_numpy(), transform(q.ref, kind).to_numpy()
    d = tr - tref
    ba = bland_altman(d)
    pair = np.column_stack([tr, tref])
    boot = [icc_a1(pair[idx]) for idx in stratified_indices(q.scanner.to_numpy(), n_boot, rng)]
    alpha = (1 - level) / 2
    by_site = [d[(q.scanner == s).to_numpy()] for s in sorted(q.scanner.unique())]
    back = np.exp if kind == "log" else (lambda v: v)  # log1p: differences stay on the log scale
    return {
        "n": len(q),
        "n_excluded": int((~ok).sum()),
        "scale": "ratio" if kind == "log" else "difference of ln(x + 1)",
        "bias": float(back(ba["bias"])),
        "loa_low": float(back(ba["loa_low"])),
        "loa_high": float(back(ba["loa_high"])),
        "mean_diff_raw": float((q.rater - q.ref).mean()),
        "icc_a1": icc_a1(pair),
        "icc_ci_low": float(np.quantile(boot, alpha)),
        "icc_ci_high": float(np.quantile(boot, 1 - alpha)),
        "p_kruskal_site": float(stats.kruskal(*by_site).pvalue) if len(by_site) > 1 else float("nan"),
    }


def by_scanner(
    p: pd.DataFrame, kind: str, n_boot: int, level: float, rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Block 3: bias per scanner (bootstrap CI, Wilcoxon test against no bias)."""
    ok = (p.rater > 0) & (p.ref > 0) if kind == "log" else pd.Series(True, index=p.index)
    q = p[ok]
    alpha = (1 - level) / 2
    back = np.exp if kind == "log" else (lambda v: v)
    rows = []
    for scanner, g in q.groupby("scanner"):
        if len(g) < MIN_SUBJECTS:
            continue
        d = (transform(g.rater, kind) - transform(g.ref, kind)).to_numpy()
        boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
        rows.append(
            {
                "scanner": scanner,
                "n": len(d),
                "bias": float(back(d.mean())),
                "ci_low": float(back(np.quantile(boot, alpha))),
                "ci_high": float(back(np.quantile(boot, 1 - alpha))),
                "mean_diff_raw": float((g.rater - g.ref).mean()),
                "p_wilcoxon": 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(d).pvalue),
            }
        )
    return rows


def bland_altman_figure(p: pd.DataFrame, title: str, path) -> None:
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = p[(p.rater > 0) & (p.ref > 0)]
    mean = np.sqrt(q.rater * q.ref)  # geometric mean of the two volumes
    ratio = q.rater / q.ref
    ba = bland_altman(np.log(ratio.to_numpy()))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for scanner, g in q.groupby("scanner"):
        ax.scatter(mean[g.index], ratio[g.index], s=18, label=scanner)
    for v, style in [(ba["bias"], "-"), (ba["loa_low"], "--"), (ba["loa_high"], "--")]:
        ax.axhline(np.exp(v), color="black", ls=style, lw=0.8)
        ax.text(mean.max(), np.exp(v), f" {np.exp(v):.2f}", va="center", fontsize=8)
    ax.axhline(1, color="grey", lw=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [
        t
        for t in (0.25, 0.33, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 3, 4, 6, 8)
        if ratio.min() / 1.1 <= t <= ratio.max() * 1.1
    ]
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.set_xlabel("WMH volume, geometric mean of the two (ml)")
    ax.set_ylabel("ratio to O1")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = cfg["evaluation"]
    n_boot, level = e["bootstrap_samples"], e["confidence_level"]
    rng = np.random.default_rng(e["seed"])
    table = pd.read_csv(resolve_path(cfg, "results") / "tables" / "biomarkers.csv")
    overall, per_site = [], []
    for biomarker, kind in BIOMARKERS.items():
        for (split, rater), p in pairs(table, biomarker).items():
            key = {"split": split, "rater": rater, "reference": "O1", "biomarker": biomarker}
            overall.append(key | agreement(p, kind, n_boot, level, rng))
            per_site += [key | r for r in by_scanner(p, kind, n_boot, level, rng)]
    per_site = pd.DataFrame(per_site)
    if len(per_site):
        per_site["p_holm"] = holm(per_site.p_wilcoxon.to_numpy())
    return pd.DataFrame(overall), per_site


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Agreement of predicted vs manual biomarkers (7.4).")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    overall, per_site = run(cfg)
    tables = resolve_path(cfg, "results") / "tables"
    overall.to_csv(tables / "stats_agreement.csv", index=False, float_format="%.5g")
    per_site.to_csv(tables / "stats_agreement_by_scanner.csv", index=False, float_format="%.5g")
    figures = resolve_path(cfg, "results") / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(tables / "biomarkers.csv")
    for (split, rater), p in pairs(table, "wmh_ml").items():
        bland_altman_figure(
            p,
            f"WMH volume: {rater} vs O1 ({split}, n={len(p)})",
            figures / f"bland_altman_{split}_{rater}.png",
        )
    pd.set_option("display.width", 220)
    log.info("overall:\n%s", overall.to_string(index=False, float_format="%.3g"))
    vol = per_site[per_site.biomarker == "wmh_ml"]
    log.info("WMH volume bias (ratio) per scanner:\n%s", vol.to_string(index=False, float_format="%.3g"))


if __name__ == "__main__":
    main()
