"""Normal-appearing white matter features, per subject (ROADMAP step 8.2, J-028, J-059).

Image-derived features carry the scanner: the same tissue reads differently on each site. These
features are what a multi-site study would compare between patients, and what 8.2 / 8.3 test for a
site signature before and after harmonisation.

    FLAIR N4 and T1 N4 (on the FLAIR grid), HD-BET brain mask (J-028, rule A)
    tissues of WMH-SynthSeg: white matter (2, 41; the WMH have their own label 77, so this is the
    normal-appearing white matter, NAWM), cortex (3, 42)
    intensity scale: ``none`` (I / median of the training brains, one constant per modality) or
    ``zscore`` (per-image robust z inside the brain, as M3), from ``harmonize.intensity``
      -> per modality: NAWM percentiles 5 / 25 / 50 / 75 / 95, IQR, skewness, kurtosis;
         cortex median and cortex - NAWM median (grey/white contrast)
      v
    results/tables/nawm_features.csv (one row per subject and scale)

No background statistic is used (J-028, rule B not involved).

Usage:
    uv run python -m wmh_multisite.harmonize.features
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.harmonize import intensity
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.utils import io

log = logging.getLogger(__name__)

SCALES = ("none", "zscore")
MODALITIES = {"FLAIR": "{sub}_desc-n4_FLAIR.nii.gz", "T1w": "{sub}_space-FLAIR_desc-n4_T1w.nii.gz"}


def tissue_features(image: np.ndarray, nawm: np.ndarray, cortex: np.ndarray) -> dict[str, float]:
    v = image[nawm].astype(np.float64)
    p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
    gm = float(np.median(image[cortex]))
    return {
        "nawm_p5": p5,
        "nawm_p25": p25,
        "nawm_p50": p50,
        "nawm_p75": p75,
        "nawm_p95": p95,
        "nawm_iqr": p75 - p25,
        "nawm_skew": float(stats.skew(v)),
        "nawm_kurtosis": float(stats.kurtosis(v)),
        "cortex_p50": gm,
        "gm_minus_wm": gm - p50,
    }


def subject_features(task: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = load_config(task["config"])
    sub = task["sub"]
    d = resolve_path(cfg, "derivatives") / PIPELINE / sub / "anat"
    brain = io.load_labels(d / f"{sub}_space-FLAIR_desc-brain_mask.nii.gz", allowed=(0, 1)).astype(bool)
    labels = io.load_labels(
        d / f"{sub}_space-FLAIR_desc-wmhsynthsegfull_dseg.nii.gz", allowed=tuple(range(256))
    )
    nawm = np.isin(labels, (2, 41)) & brain
    cortex = np.isin(labels, (3, 42)) & brain
    rows = []
    for scale in SCALES:
        row = {"participant_id": sub, "scale": scale}
        for modality, pattern in MODALITIES.items():
            image = io.load_data(d / pattern.format(sub=sub))
            norm = intensity.normalize(image, brain, scale, task["constants"][modality])
            row |= {f"{modality}_{k}": v for k, v in tissue_features(norm, nawm, cortex).items()}
        rows.append(row)
    return rows


def training_constants(cfg: dict[str, Any], train: list[str]) -> dict[str, dict[str, float]]:
    """``none`` scale: one constant per modality, the median of the training brains (J-013)."""
    d = resolve_path(cfg, "derivatives") / PIPELINE
    out = {}
    for modality, pattern in MODALITIES.items():
        pairs = []
        for sub in train:
            a = d / sub / "anat"
            brain = io.load_labels(a / f"{sub}_space-FLAIR_desc-brain_mask.nii.gz", allowed=(0, 1)).astype(
                bool
            )
            pairs.append((io.load_data(a / pattern.format(sub=sub)), brain))
        out[modality] = intensity.fit("none", pairs)
    return out


def build(cfg: dict[str, Any], jobs: int, config_path: str | None = None) -> pd.DataFrame:
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t").set_index("participant_id")
    constants = training_constants(cfg, part[part.split == "training"].index.tolist())
    tasks = [{"sub": s, "constants": constants, "config": config_path} for s in part.index]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = [r for rows in pool.map(subject_features, tasks) for r in rows]
    table = pd.DataFrame(rows)
    for i, col in enumerate(["scanner", "split", "seen_in_training"], start=1):
        table.insert(i, col, table.participant_id.map(part[col]))
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Normal-appearing white matter features (8.2).")
    parser.add_argument("--config")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    table = build(cfg, args.jobs, args.config)
    out = resolve_path(cfg, "results") / "tables" / "nawm_features.csv"
    table.to_csv(out, index=False, float_format="%.5g")
    log.info(
        "%s: %d rows, %d features per scale",
        out,
        len(table),
        len([c for c in table if c.startswith(("FLAIR_", "T1w_"))]),
    )


if __name__ == "__main__":
    main()
