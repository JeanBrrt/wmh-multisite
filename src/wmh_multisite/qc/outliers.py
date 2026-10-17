"""Atypical images: robust z-scores per scanner and an Isolation Forest.

The IQMs depend strongly on the scanner (the 2D FLAIRs of Utrecht have almost no white/grey matter contrast),
so with all scanners pooled a whole site would be flagged. Each image is therefore judged against the images
of its own scanner.

For each scanner and modality, every IQM becomes a robust z-score, (x - median) / (1.4826 MAD), signed so that
a positive value always means a worse image. The median and MAD are used instead of the mean and SD because a
very bad image would pull the mean towards itself and hide.

An Isolation Forest is then fitted on these z-scores (FLAIR and T1 together). It isolates subjects with random
cuts, and an atypical subject is isolated in few cuts, which also catches unusual combinations that no single
IQM shows. The z-scores are clipped at 0 first so that, like the thresholds, the forest only looks at the
worse side. Without this it flagged images that were unusually good.

An image is classed as exclude if its largest z is at least z_exclude, as check if it is at least z_check or
if the forest flags it, and as usable otherwise. Scanners with fewer than min_per_scanner images are not
assessed. These classes are a prompt for a visual check, never an automatic exclusion. Finally we check
whether the flagged test subjects get a lower Dice with M1.

Output: results/tables/qc_flags.csv.

Usage:
    uv run python -m wmh_multisite.qc.outliers
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.qc.iqm import DIRECTION, JUDGED

log = logging.getLogger(__name__)


def robust_z(values: pd.Series, groups: pd.Series, min_group: int = 1) -> pd.Series:
    """Robust z-score (x - median) / (1.4826 MAD) within each group, 0 where the group has no spread.

    Groups smaller than min_group get NaN, because the median and MAD of a handful of images are too unstable
    to call an image atypical.
    """
    med = values.groupby(groups).transform("median")
    mad = (values - med).abs().groupby(groups).transform("median") * 1.4826
    z = ((values - med) / mad.replace(0, np.nan)).fillna(0.0)
    return z.where(groups.map(groups.value_counts()) >= min_group)


def badness_z(iqm: pd.DataFrame, min_group: int = 1) -> pd.DataFrame:
    """One column per modality and metric: z-score within the scanner, positive meaning worse."""
    cols = {}
    for modality, g in iqm.groupby("modality"):
        g = g.set_index("participant_id")
        for m in JUDGED[modality]:
            z = robust_z(g[m], g.scanner, min_group)
            cols[f"{modality}_{m}"] = -DIRECTION[m] * z if DIRECTION[m] else z.abs()
    return pd.DataFrame(cols)


def forest_features(z: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest input: z-scores clipped at 0 (only the worse side counts), missing values as 0."""
    return z.clip(lower=0).fillna(0.0)


def classify(z: pd.DataFrame, cfg_qc: dict[str, Any], seed: int) -> pd.DataFrame:
    forest = IsolationForest(
        n_estimators=500, contamination=cfg_qc["iforest_contamination"], random_state=seed
    )
    assessed = z.notna().any(axis=1)  # scanners with enough subjects
    out = pd.DataFrame(index=z.index)
    out["max_z"] = z.max(axis=1)
    out["worst_metric"] = z.loc[assessed].idxmax(axis=1).reindex(z.index)
    out["iforest_outlier"] = False
    out["iforest_score"] = np.nan
    if assessed.sum() >= 2:
        filled = forest_features(z.loc[assessed])
        out.loc[assessed, "iforest_outlier"] = forest.fit_predict(filled) == -1
        out.loc[assessed, "iforest_score"] = -forest.score_samples(filled)  # higher = more atypical
    out["qc_class"] = np.select(
        [~assessed, out.max_z >= cfg_qc["z_exclude"], (out.max_z >= cfg_qc["z_check"]) | out.iforest_outlier],
        ["not_assessed", "exclude", "check"],
        default="usable",
    )
    return out


def validate(flags: pd.DataFrame, evaluation: pd.DataFrame) -> dict[str, Any]:
    """Dice of M1 by QC class on the test subjects, and its rank correlation with the outlier score."""
    m = flags.join(evaluation.set_index("participant_id")[["dice"]], how="inner")
    usable, flagged = m[m.qc_class == "usable"].dice, m[m.qc_class.isin(["check", "exclude"])].dice
    out = {
        "n_test": len(m),
        "n_flagged": len(flagged),
        "dice_usable": float(usable.median()) if len(usable) else None,
        "dice_flagged": float(flagged.median()) if len(flagged) else None,
    }
    if len(usable) and len(flagged):
        out["p_mannwhitney"] = float(stats.mannwhitneyu(flagged, usable).pvalue)
    out["spearman_score_dice"] = float(stats.spearmanr(m.iforest_score, m.dice).statistic)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Atypical images per scanner.")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    tables = resolve_path(cfg, "results") / "tables"
    iqm = pd.read_csv(tables / "qc_iqm.csv")
    z = badness_z(iqm, cfg["qc"]["min_per_scanner"])
    flags = classify(z, cfg["qc"], cfg["evaluation"]["seed"])
    scanners = iqm.drop_duplicates("participant_id").set_index("participant_id").scanner
    flags.insert(0, "scanner", scanners.reindex(flags.index))
    flags = flags.join(z.add_prefix("z_"))
    flags.to_csv(tables / "qc_flags.csv", index_label="participant_id", float_format="%.4g")
    log.info("QC classes per scanner:\n%s", pd.crosstab(flags.scanner, flags.qc_class).to_string())
    log.info(
        "flagged:\n%s",
        flags[flags.qc_class.isin(["check", "exclude"])][["scanner", "qc_class", "max_z", "worst_metric"]]
        .round(2)
        .to_string(),
    )
    ev = tables / "evaluation_resencm.csv"
    if ev.is_file():
        log.info("validation against the Dice of M1 (test): %s", validate(flags, pd.read_csv(ev)))


if __name__ == "__main__":
    main()
