r"""FLAIR thresholding, method M3: the classical baseline.

WMH are bright on FLAIR by definition, so the simplest segmentation keeps the voxels above a
threshold. It shows what the learnt methods (M1 nnU-Net, M2 WMH-SynthSeg) add, and how much a fixed
intensity rule depends on the scanner.

    FLAIR N4 (FLAIR grid) + HD-BET brain mask (FLAIR grid)
       |
       |  1. z-score inside the brain (median, MAD)
       v
    z map ----- 2. z > t ----------------------------\
       |                                               \
       |  depth: distance (mm) to the brain border       >-- candidates
       \--------- 3. depth > e -------------------------/
                                                         |
                                 4. connected components smaller than m mm3 removed
                                                         v
                                        sub-XXX_space-FLAIR_desc-threshold_dseg.nii.gz

1. z = (FLAIR - median) / (1.4826 x MAD) over the brain voxels only. Unlike mean and SD, median and
   MAD barely move with the bright lesions, and the scale is the same on every scanner.
2. Threshold t on z.
3. Minimum depth e (mm to the brain border): the border is full of bright non-lesion voxels (cortex,
   leftover meninges or vessels, partial volume), while WMH sit deeper (5th percentile: 16 mm).
4. Minimum size m, in mm3 so that it means the same on 3 mm 2D and 3D FLAIRs (26-connectivity).

One global (t, e, m) is chosen by grid search on the 60 training subjects, maximising the mean Dice
(label 2 ignored, as in the official evaluation). The best threshold per scanner is reported to show
how scanner-dependent the rule is, but never used.

Outputs:
    results/tables/threshold_tuning.csv        mean Dice of every (t, e, m), overall and per scanner
    results/tables/threshold_params.json       chosen (t, e, m) and their training Dice
    data/derivatives/wmh-multisite/sub-XXX/anat/sub-XXX_space-FLAIR_desc-threshold_dseg.nii.gz

Usage:
    uv run python -m wmh_multisite.seg.threshold --tune                 # grid search on the training set
    uv run python -m wmh_multisite.seg.threshold --apply --split all    # writes the 170 segmentations
    uv run python -m wmh_multisite.evaluate.metrics --model threshold --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.harmonize import intensity
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.utils import io

log = logging.getLogger(__name__)

MODEL = "threshold"


def model_name(normalization: str) -> str:
    """BIDS desc of the predictions: ``threshold`` for the robust z, ``threshold<method>`` otherwise."""
    return MODEL if normalization == "zscore" else f"{MODEL}{normalization}"


def file_suffix(normalization: str) -> str:
    return "" if normalization == "zscore" else f"_{normalization}"


STRUCTURE = np.ones((3, 3, 3), bool)  # 26-connectivity


def zscore_in_brain(flair: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Robust z-score with statistics taken inside the brain only."""
    values = flair[brain]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826  # = standard deviation for a Gaussian
    if mad <= 0:
        raise ValueError("brain intensities have zero spread (MAD = 0)")
    return ((flair - median) / mad).astype(np.float32)


def depth_mm(brain: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    """Distance (mm) from each brain voxel to the nearest voxel outside the brain; 0 outside."""
    return ndimage.distance_transform_edt(brain, sampling=spacing).astype(np.float32)


def remove_small(mask: np.ndarray, min_size_mm3: float, voxel_mm3: float) -> np.ndarray:
    """Remove the connected components (26-connectivity) smaller than ``min_size_mm3``."""
    if min_size_mm3 <= 0 or not mask.any():
        return mask
    lab, _ = ndimage.label(mask, structure=STRUCTURE)
    sizes = np.bincount(lab.ravel()) * voxel_mm3
    keep = sizes >= min_size_mm3
    keep[0] = False
    return keep[lab]


def segment(
    z: np.ndarray,
    depth: np.ndarray,
    voxel_mm3: float,
    threshold: float,
    min_depth_mm: float,
    min_size_mm3: float,
) -> np.ndarray:
    """Steps 2 to 4: bright, deep enough, and part of a large enough component."""
    candidates = (z > threshold) & (depth > min_depth_mm)
    return remove_small(candidates, min_size_mm3, voxel_mm3)


def subject_paths(cfg: dict[str, Any], sub: str, normalization: str = "zscore") -> dict[str, Path]:
    d = resolve_path(cfg, "derivatives") / PIPELINE / sub / "anat"
    return {
        "labels": d / f"{sub}_space-FLAIR_desc-wmhsynthsegfull_dseg.nii.gz",
        "flair_n4": d / f"{sub}_desc-n4_FLAIR.nii.gz",
        "brain": d / f"{sub}_space-FLAIR_desc-brain_mask.nii.gz",
        "truth": resolve_path(cfg, "bids")
        / "derivatives/manual"
        / sub
        / "anat"
        / f"{sub}_space-FLAIR_desc-O1_dseg.nii.gz",
        "out": d / f"{sub}_space-FLAIR_desc-{model_name(normalization)}_dseg.nii.gz",
    }


def prepare(
    p: dict[str, Path], normalization: str = "zscore", norm_params: dict[str, Any] | None = None
) -> tuple[np.ndarray, np.ndarray, float, tuple[slice, ...]]:
    """Normalised FLAIR and depth map cropped to the brain bounding box (faster grid search), voxel volume.

    ``normalization``: ``zscore`` or another method of ``harmonize.intensity``.
    """
    brain = io.load_labels(p["brain"], allowed=(0, 1)).astype(bool)
    flair = io.load_data(p["flair_n4"])
    spacing = tuple(io.describe(p["flair_n4"])["voxel_size_mm"])
    box = ndimage.find_objects(brain.astype(np.uint8))[0]
    if normalization == "zscore":
        z = zscore_in_brain(flair, brain)[box]
    else:
        labels = None
        if normalization == "whitestripe":
            labels = io.load_labels(p["labels"], allowed=tuple(range(256)))
        z = intensity.normalize(flair, brain, normalization, norm_params or {}, labels)[box]
    depth = depth_mm(brain, spacing)[box]  # computed on the full grid: the box edge is not a border
    return z, depth, float(np.prod(spacing)), box


def grid_counts(
    sub: str,
    grid: dict[str, list[float]],
    config_path: str | None = None,
    normalization: str = "zscore",
    norm_params: dict[str, Any] | None = None,
) -> np.ndarray:
    """For one training subject: true positives, predicted and true volumes (voxels) for every
    (t, e, m) of the grid, label 2 excluded from the counts. Shape (n_t, n_e, n_m, 3).

    One labelling per (t, e): every minimum size m is then read from the component sizes.
    """
    cfg = load_config(config_path)
    p = subject_paths(cfg, sub)
    z, depth, voxel_mm3, box = prepare(p, normalization, norm_params)
    truth = io.load_labels(p["truth"])[box]
    lesion, scored = truth == 1, truth != 2
    n_true = int(lesion.sum())
    out = np.zeros((len(grid["t"]), len(grid["e"]), len(grid["m"]), 3), np.int64)
    for j, e in enumerate(grid["e"]):
        deep = depth > e
        for i, t in enumerate(grid["t"]):
            lab, n = ndimage.label((z > t) & deep, structure=STRUCTURE)
            size_mm3 = np.bincount(lab.ravel(), minlength=n + 1) * voxel_mm3
            tp = np.bincount(lab[lesion], minlength=n + 1)
            pred = np.bincount(lab[scored], minlength=n + 1)
            for k, m in enumerate(grid["m"]):
                keep = size_mm3 >= m
                keep[0] = False
                out[i, j, k] = (tp[keep].sum(), pred[keep].sum(), n_true)
    return out


def _grid(cfg: dict[str, Any], normalization: str = "zscore") -> dict[str, list[float]]:
    g = cfg["segmentation"]["threshold"]["grid"]
    start, stop, step = (
        (g["z_start"], g["z_stop"], g["z_step"])
        if normalization == "zscore"
        else cfg["segmentation"]["threshold"]["t_grid_by_normalization"][normalization]
    )
    t = np.round(np.arange(start, stop + 1e-9, step), 3).tolist()
    return {"t": t, "e": list(g["min_depth_mm"]), "m": list(g["min_size_mm3"])}


def fit_normalization(cfg: dict[str, Any], subjects: list[str], normalization: str) -> dict[str, Any]:
    """Normalisation parameters learnt inside the brain of the training subjects."""
    if normalization not in ("none", "histmatch"):
        return {}
    pairs = []
    for sub in subjects:
        p = subject_paths(cfg, sub)
        pairs.append((io.load_data(p["flair_n4"]), io.load_labels(p["brain"], allowed=(0, 1)).astype(bool)))
    return intensity.fit(normalization, pairs)


def tune(
    cfg: dict[str, Any], jobs: int, config_path: str | None = None, normalization: str = "zscore"
) -> dict[str, Any]:
    """Grid search on the training subjects; writes the tuning table and the chosen params."""
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t")
    train = part[part.split == "training"]
    grid = _grid(cfg, normalization)
    subjects = train.participant_id.tolist()
    norm_params = fit_normalization(cfg, subjects, normalization)
    log.info(
        "normalization %s; grid search on %d training subjects: %d thresholds x %d depths x %d sizes, %d jobs",  # noqa: E501
        normalization,
        len(subjects),
        len(grid["t"]),
        len(grid["e"]),
        len(grid["m"]),
        jobs,
    )
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        counts = np.stack(
            list(
                pool.map(
                    grid_counts,
                    subjects,
                    [grid] * len(subjects),
                    [config_path] * len(subjects),
                    [normalization] * len(subjects),
                    [norm_params] * len(subjects),
                )
            )
        )
    tp, pred, true = counts[..., 0], counts[..., 1], counts[..., 2]
    dice = 2 * tp / np.maximum(pred + true, 1)  # (subject, t, e, m)

    index = pd.MultiIndex.from_product(
        [grid["t"], grid["e"], grid["m"]], names=["z_threshold", "min_depth_mm", "min_size_mm3"]
    )
    table = pd.DataFrame({"dice_all": dice.mean(0).ravel()}, index=index)
    for scanner, rows in train.reset_index(drop=True).groupby("scanner").groups.items():
        table[f"dice_{scanner}"] = dice[list(rows)].mean(0).ravel()
    out_dir = resolve_path(cfg, "results") / "tables"
    table.round(4).to_csv(out_dir / f"threshold_tuning{file_suffix(normalization)}.csv")

    best = table.dice_all.idxmax()
    per_scanner = {
        c.removeprefix("dice_"): {
            **dict(zip(table.index.names, map(float, table[c].idxmax()), strict=True)),
            "dice": round(float(table[c].max()), 4),
        }
        for c in table.columns
        if c != "dice_all"
    }
    for name, value, axis in zip(
        ("z_threshold", "min_depth_mm", "min_size_mm3"), best, grid.values(), strict=True
    ):
        if value in (axis[0], axis[-1]) and len(axis) > 1:
            log.warning("best %s = %s is on the edge of the grid %s: widen it", name, value, axis)
    params = {
        "normalization": normalization,
        "norm_params": norm_params,
        "z_threshold": float(best[0]),
        "min_depth_mm": float(best[1]),
        "min_size_mm3": float(best[2]),
        "train_dice_mean": round(float(table.dice_all.max()), 4),
        "train_dice_by_scanner": {
            k.removeprefix("dice_"): round(float(v), 4) for k, v in table.loc[best].items() if k != "dice_all"
        },
        "best_per_scanner_for_information_only": per_scanner,
        "n_train_subjects": len(subjects),
    }
    (out_dir / f"threshold_params{file_suffix(normalization)}.json").write_text(json.dumps(params, indent=1))
    return params


def apply_subject(sub: str, params: dict[str, Any], config_path: str | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    normalization = params.get("normalization", "zscore")
    p = subject_paths(cfg, sub, normalization)
    brain = io.load_labels(p["brain"], allowed=(0, 1)).astype(bool)
    z, depth, voxel_mm3, box = prepare(p, normalization, params.get("norm_params"))
    seg_box = segment(
        z, depth, voxel_mm3, params["z_threshold"], params["min_depth_mm"], params["min_size_mm3"]
    )
    seg = np.zeros(brain.shape, np.uint8)
    seg[box] = seg_box
    io.save_like(seg, p["flair_n4"], p["out"], dtype=np.uint8)
    return {"participant_id": sub, "wmh_ml": round(float(seg.sum() * voxel_mm3 / 1000), 2)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FLAIR thresholding (M3): tuning on train, application.")
    parser.add_argument("--config")
    parser.add_argument("--tune", action="store_true", help="grid search on the training subjects")
    parser.add_argument("--apply", action="store_true", help="write the segmentations")
    parser.add_argument("--split", choices=["training", "test", "all"], default="all")
    parser.add_argument("--jobs", type=int)
    parser.add_argument(
        "--normalization", choices=intensity.METHODS, default="zscore", help="FLAIR normalisation"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    jobs = args.jobs or cfg["segmentation"]["threshold"]["jobs"]
    params_path = (
        resolve_path(cfg, "results") / "tables" / f"threshold_params{file_suffix(args.normalization)}.json"
    )
    if args.tune:
        params = tune(cfg, jobs, args.config, args.normalization)
        log.info("chosen on the training set:\n%s", json.dumps(params, indent=1))
    if args.apply:
        params = json.loads(params_path.read_text())
        part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t")
        splits = ("training", "test") if args.split == "all" else (args.split,)
        subjects = part[part.split.isin(splits)].participant_id.tolist()
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = list(
                pool.map(apply_subject, subjects, [params] * len(subjects), [args.config] * len(subjects))
            )
        volumes = pd.DataFrame(rows)
        log.info(
            "%d segmentations written (t=%s, e=%s mm, m=%s mm3); WMH volume median %.1f ml",
            len(rows),
            params["z_threshold"],
            params["min_depth_mm"],
            params["min_size_mm3"],
            volumes.wmh_ml.median(),
        )
    if not (args.tune or args.apply):
        parser.error("nothing to do: give --tune and/or --apply")


if __name__ == "__main__":
    main()
