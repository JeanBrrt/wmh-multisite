"""Site classifier and ComBat harmonisation of the normal-appearing white matter features.

Can the scanner be guessed from features of the normal-appearing white matter, and can it still be guessed
after ComBat? And is the biological signal (the WMH load) kept?

The site classifier is a standardisation followed by a multinomial logistic regression, evaluated with 5-fold
cross-validation stratified by scanner. We report the balanced accuracy, for which chance is 1/5.

ComBat (neuroHarmonize) is learnt on the training folds and applied to the test fold of the same
cross-validation, so the classifier never sees data harmonised with its own test subjects. The protected
covariates are ln(1 + WMH volume of O1) and the ICV. The challenge's own training set could not be used for
this, because the 2 unseen scanners have no training subject and ComBat cannot harmonise a site it has never
seen.

To check that the biology is kept, we compare the Spearman correlation of each feature with ln(1 + WMH) before
and after ComBat (fitted here on all subjects), and the share of each feature's variance explained by the
scanner (eta squared).

Outputs: results/tables/harmonization_site_classifier.csv and harmonization_biology.csv.

Usage:
    uv run python -m wmh_multisite.harmonize.combat
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wmh_multisite.config import load_config, resolve_path

log = logging.getLogger(__name__)


def combat_learn_apply(
    train_x: np.ndarray, train_cov: pd.DataFrame, test_x: np.ndarray, test_cov: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Learn ComBat on the training set and apply it to both sets. The covariate tables hold SITE and the
    covariates.
    """
    from neuroHarmonize import harmonizationApply, harmonizationLearn

    model, train_h = harmonizationLearn(train_x, train_cov)
    test_h = harmonizationApply(test_x, test_cov, model)
    return train_h, test_h


def site_classifier(
    x: np.ndarray, site: np.ndarray, cov: pd.DataFrame | None, seed: int, n_splits: int = 5
) -> list[float]:
    """Balanced accuracy per fold. If cov is given, ComBat is learnt on each training fold first."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in folds.split(x, site):
        xtr, xte = x[tr], x[te]
        if cov is not None:
            xtr, xte = combat_learn_apply(
                xtr, cov.iloc[tr].reset_index(drop=True), xte, cov.iloc[te].reset_index(drop=True)
            )
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        clf.fit(xtr, site[tr])
        scores.append(balanced_accuracy_score(site[te], clf.predict(xte)))
    return scores


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Share of the variance explained by the groups (one-way ANOVA)."""
    grand = values.mean()
    between = sum(len(v) * (v.mean() - grand) ** 2 for v in (values[groups == g] for g in np.unique(groups)))
    total = ((values - grand) ** 2).sum()
    return float(between / total) if total else float("nan")


def covariates(table: pd.DataFrame, biomarkers: pd.DataFrame) -> pd.DataFrame:
    o1 = biomarkers[biomarkers.source == "O1"].set_index("participant_id")
    return pd.DataFrame(
        {
            "SITE": table.scanner.to_numpy(),
            "log_wmh": np.log1p(table.participant_id.map(o1.wmh_ml)).to_numpy(),
            "icv": table.participant_id.map(o1.icv_ml).to_numpy(),
        }
    )


def run(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = resolve_path(cfg, "results") / "tables"
    feats = pd.read_csv(tables / "nawm_features.csv")
    bio = pd.read_csv(tables / "biomarkers.csv")
    seed = cfg["evaluation"]["seed"]
    cols = [c for c in feats.columns if c.startswith(("FLAIR_", "T1w_"))]
    clf_rows, bio_rows = [], []
    for scale, t in feats.groupby("scale"):
        t = t.sort_values("participant_id").reset_index(drop=True)
        x, site, cov = t[cols].to_numpy(float), t.scanner.to_numpy(), covariates(t, bio)
        for harmonised in (False, True):
            s = site_classifier(x, site, cov if harmonised else None, seed)
            clf_rows.append(
                {
                    "scale": scale,
                    "combat": harmonised,
                    "balanced_accuracy": float(np.mean(s)),
                    "sd_over_folds": float(np.std(s)),
                    "chance": 1 / len(np.unique(site)),
                }
            )
        from neuroHarmonize import harmonizationLearn

        _, xh = harmonizationLearn(x, cov)
        for j, c in enumerate(cols):
            bio_rows.append(
                {
                    "scale": scale,
                    "feature": c,
                    "eta2_site_before": eta_squared(x[:, j], site),
                    "eta2_site_after": eta_squared(xh[:, j], site),
                    "rho_wmh_before": float(stats.spearmanr(x[:, j], cov.log_wmh).statistic),
                    "rho_wmh_after": float(stats.spearmanr(xh[:, j], cov.log_wmh).statistic),
                }
            )
    return pd.DataFrame(clf_rows), pd.DataFrame(bio_rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Site classifier and ComBat harmonisation (8.2, 8.3).")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    clf, bio = run(cfg)
    tables = resolve_path(cfg, "results") / "tables"
    clf.to_csv(tables / "harmonization_site_classifier.csv", index=False, float_format="%.4g")
    bio.to_csv(tables / "harmonization_biology.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    log.info("site classifier (balanced accuracy, 5-fold CV):\n%s", clf.round(3).to_string(index=False))
    summary = bio.groupby("scale")[["eta2_site_before", "eta2_site_after"]].mean()
    summary["mean_abs_rho_wmh_before"] = bio.assign(a=bio.rho_wmh_before.abs()).groupby("scale").a.mean()
    summary["mean_abs_rho_wmh_after"] = bio.assign(a=bio.rho_wmh_after.abs()).groupby("scale").a.mean()
    log.info(
        "site share of variance and association with WMH load (means over features):\n%s",
        summary.round(3).to_string(),
    )


if __name__ == "__main__":
    main()
