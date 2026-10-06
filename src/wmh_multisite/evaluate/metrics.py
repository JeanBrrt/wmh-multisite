"""Evaluation of the segmentations with the official challenge metrics (ROADMAP step 6.1, J-012).

For each subject, the prediction (FLAIR grid, 0/1) is compared with the manual reference standard
(O1, FLAIR grid, 0 background / 1 WMH / 2 other pathology) by the official functions vendored in
``wmh_challenge.py``:

- **Dice** (DSC): voxel overlap, 2|A n B| / (|A| + |B|); 1 = perfect.
- **HD95** (mm): 95th percentile of the distances between the lesion *boundaries* of the two masks,
  taken in both directions (the larger of the two); 0 = perfect. Undefined (NaN) if nothing is
  predicted.
- **AVD** (%): absolute difference of total WMH volume, relative to the true volume; 0 = perfect.
- **Lesion recall**: share of the true lesions (3D connected components, 26-connectivity) touched
  by at least one predicted voxel; 1 = every lesion found.
- **Lesion F1**: harmonic mean of that recall and the lesion precision (share of predicted
  components touching a true lesion).

Voxels of label 2 are removed from the prediction before every metric (``getImages``), as in the
challenge. The leaderboard averages each metric over the test scans (HD95 NaN cases aside).

Output: ``results/tables/evaluation_<model>.csv`` (one row per subject, with scanner, split and
whether the scanner was seen in training).

Usage:
    uv run python -m wmh_multisite.evaluate.metrics --model resencm --split test
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.evaluate import wmh_challenge as official
from wmh_multisite.preproc import PIPELINE

log = logging.getLogger(__name__)

METRICS = ["dice", "hd95_mm", "avd_pct", "lesion_recall", "lesion_f1"]


def evaluate_case(truth: str | Path, prediction: str | Path) -> dict[str, float]:
    """The five official metrics for one subject, plus the true and predicted WMH volumes (ml).

    Pitfall of the official ``getImages``: it isolates WMH with ``BinaryThreshold(test, 0.5, 1.5)``.
    On an *integer* reference map, SimpleITK casts these thresholds to 0 and 1, so every voxel counts
    as WMH and the "other pathology" mask then erases the prediction: all metrics silently collapse.
    The challenge files are float32, so the leaderboard is not affected; to be safe whatever the
    storage type, the reference is always passed to the official code as a float32 copy.
    """
    with tempfile.TemporaryDirectory() as tmp:
        truth_f32 = Path(tmp) / "truth_float32.nii.gz"
        sitk.WriteImage(sitk.ReadImage(str(truth), sitk.sitkFloat32), str(truth_f32))
        test_image, result_image = official.getImages(str(truth_f32), str(prediction))
    recall, f1 = official.getLesionDetection(test_image, result_image)
    ml = float(np.prod(test_image.GetSpacing())) / 1000
    return {
        "dice": official.getDSC(test_image, result_image),
        "hd95_mm": official.getHausdorff(test_image, result_image),
        "avd_pct": official.getAVD(test_image, result_image),
        "lesion_recall": recall,
        "lesion_f1": f1,
        "true_ml": float(sitk.GetArrayFromImage(test_image).sum()) * ml,
        "pred_ml": float(sitk.GetArrayFromImage(result_image).sum()) * ml,
    }


def _paths(cfg: dict[str, Any], sub: str, model: str) -> tuple[Path, Path]:
    truth = (
        resolve_path(cfg, "bids")
        / "derivatives/manual"
        / sub
        / "anat"
        / f"{sub}_space-FLAIR_desc-O1_dseg.nii.gz"
    )
    pred = (
        resolve_path(cfg, "derivatives")
        / PIPELINE
        / sub
        / "anat"
        / f"{sub}_space-FLAIR_desc-{model}_dseg.nii.gz"
    )
    return truth, pred


def _one(args: tuple[str, str, str]) -> dict[str, Any]:
    sub, truth, pred = args
    return {"participant_id": sub, **evaluate_case(truth, pred)}


def evaluate(cfg: dict[str, Any], subjects: list[str], model: str, jobs: int = 4) -> pd.DataFrame:
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t").set_index("participant_id")
    todo, missing = [], []
    for sub in subjects:
        truth, pred = _paths(cfg, sub, model)
        (todo if pred.is_file() else missing).append((sub, str(truth), str(pred)))
    if missing:
        log.warning(
            "%d subjects without a %s prediction, skipped: %s",
            len(missing),
            model,
            [m[0] for m in missing][:10],
        )
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(_one, todo))
    table = pd.DataFrame(rows)
    table.insert(1, "scanner", table.participant_id.map(part.scanner))
    table.insert(2, "split", table.participant_id.map(part.split))
    table.insert(3, "seen_in_training", table.participant_id.map(part.seen_in_training))
    table.insert(4, "model", model)
    return table


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric overall, by scanner and by seen/unseen scanner (leaderboard convention)."""
    groups = {"all": table}
    groups |= {f"scanner={k}": g for k, g in table.groupby("scanner")}
    groups |= {f"seen={k}": g for k, g in table.groupby("seen_in_training")}
    return pd.DataFrame(
        {name: {"n": len(g), **{m: g[m].mean() for m in METRICS}} for name, g in groups.items()}
    ).T


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Official WMH challenge metrics for one model.")
    parser.add_argument("--config")
    parser.add_argument("--model", default="resencm")
    parser.add_argument("--split", choices=["training", "test", "all"], default="test")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t")
    splits = ("training", "test") if args.split == "all" else (args.split,)
    table = evaluate(cfg, part[part.split.isin(splits)].participant_id.tolist(), args.model, args.jobs)
    out = resolve_path(cfg, "results") / "tables" / f"evaluation_{args.model}.csv"
    table.round(4).to_csv(out, index=False)
    summary = summarize(table)
    summary.round(4).to_csv(out.with_name(f"evaluation_{args.model}_summary.csv"))
    log.info("%s (%d subjects)\n%s", out, len(table), summary.round(3).to_string())


if __name__ == "__main__":
    main()
