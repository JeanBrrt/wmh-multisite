"""Biomarker table: volumes (7.1), location (7.2), lesion structure (7.3) per subject and source.

J-048, J-050.

    test (110)        : O1 (manual reference) + every model with predictions (M1 nnU-Net, M3 threshold,
                        then M2 WMH-SynthSeg and DA5)
    training (60)     : O1, O3, O4 (three experts: the human spread of the biomarkers, needed in 7.4)
                        + M1 on the 12 fold-0 validation subjects (held out from that model)
        |
        |  label 2 of O1 removed from every mask; HD-BET brain mask for the fraction (J-028, rule A);
        |  lateral ventricles of the subject (first available of ``biomarkers.ventricles``: WMH-SynthSeg,
        |  then the atlas refinement C) for the periventricular / deep split (``ventricle_source`` column;
        |  NaN if the subject has no ventricle mask yet)
        v
    results/tables/biomarkers.csv   one row per (subject, source)

Training-set predictions of M1 (other than the 12 held-out subjects) and of M3 (tuned on the training
set) are not used: they would be optimistic.

Check: on the test set, the WMH volumes must equal ``true_ml`` / ``pred_ml`` of the official evaluation
tables of 6.1 (same label-2 removal); the largest difference is logged.

Usage:
    uv run python -m wmh_multisite.biomarkers.table
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from wmh_multisite.biomarkers.lesions import lesion_biomarkers
from wmh_multisite.biomarkers.location import location_biomarkers
from wmh_multisite.biomarkers.volumes import icv_ml, remove_other_pathology, volume_biomarkers
from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.evaluate.interobserver import manual, validation_cases
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.utils import io

log = logging.getLogger(__name__)


def prediction(cfg: dict[str, Any], sub: str, model: str) -> Path:
    return (
        resolve_path(cfg, "derivatives")
        / PIPELINE
        / sub
        / "anat"
        / f"{sub}_space-FLAIR_desc-{model}_dseg.nii.gz"
    )


def brain_mask(cfg: dict[str, Any], sub: str) -> Path:
    return (
        resolve_path(cfg, "derivatives")
        / PIPELINE
        / sub
        / "anat"
        / f"{sub}_space-FLAIR_desc-brain_mask.nii.gz"
    )


def sources_for(cfg: dict[str, Any], sub: str, split: str, held_out: dict[str, Path]) -> dict[str, Path]:
    """Segmentations to measure for one subject (missing files are skipped by the caller)."""
    out = {"O1": manual(cfg, sub, "O1")}
    if split == "test":
        out |= {m: prediction(cfg, sub, m) for m in cfg["biomarkers"]["models"]}
    else:
        out |= {"O3": manual(cfg, sub, "O3"), "O4": manual(cfg, sub, "O4")}
        if sub in held_out:
            out["resencm"] = held_out[sub]
    return out


def ventricles_for(cfg: dict[str, Any], sub: str) -> tuple[str, Path] | tuple[None, None]:
    """First available ventricle mask of the subject, in the order of ``biomarkers.ventricles``."""
    d = resolve_path(cfg, "derivatives") / PIPELINE / sub / "anat"
    for desc in cfg["biomarkers"]["ventricles"]:
        path = d / f"{sub}_space-FLAIR_desc-{desc}_mask.nii.gz"
        if path.is_file():
            return desc, path
    return None, None


def subject_rows(task: dict[str, Any]) -> list[dict[str, Any]]:
    sub, brain_path, reference = task["sub"], task["brain"], task["reference"]
    brain = io.load_labels(brain_path, allowed=(0, 1)).astype(bool)
    voxel_mm3 = io.voxel_volume_mm3(brain_path)
    ref_labels = io.load_labels(reference)
    ventricles = None
    if task.get("ventricles"):
        if not io.same_grid(task["ventricles"], brain_path):
            raise ValueError(f"{sub}: ventricle mask is not on the FLAIR grid")
        ventricles = io.load_labels(task["ventricles"], allowed=(0, 1)).astype(bool)
        spacing = tuple(float(z) for z in io.load(brain_path).header.get_zooms()[:3])
    icv = float("nan")
    if task.get("icv_labels"):
        icv = icv_ml(
            io.load_labels(task["icv_labels"], allowed=tuple(range(256))),
            io.voxel_volume_mm3(task["icv_labels"]),
        )
    rows = []
    sources = task["sources"]
    for source, path in sources.items():
        if not io.same_grid(path, brain_path):
            raise ValueError(f"{sub} {source}: {Path(path).name} is not on the FLAIR grid")
        wmh = remove_other_pathology(io.load_labels(path) == 1, ref_labels)
        rows.append(
            {
                "participant_id": sub,
                "source": source,
                **volume_biomarkers(wmh, brain, voxel_mm3),
                "icv_ml": icv,
                **lesion_biomarkers(wmh, voxel_mm3, task["min_mm3"]),
                "ventricle_source": task.get("ventricle_source"),
                **(
                    location_biomarkers(wmh, ventricles, spacing, task["pv_mm"])
                    if ventricles is not None
                    else {}
                ),
            }
        )
    return rows


def build(cfg: dict[str, Any], jobs: int) -> pd.DataFrame:
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t").set_index("participant_id")
    held_out = validation_cases(cfg)
    min_mm3 = cfg["biomarkers"]["min_lesion_mm3"]
    tasks, missing, no_ventricles = [], [], []
    for sub, row in part.iterrows():
        sources = {}
        for source, path in sources_for(cfg, sub, row.split, held_out).items():
            if Path(path).is_file():
                sources[source] = str(path)
            else:
                missing.append(f"{sub}/{source}")
        vent_source, vent_path = ventricles_for(cfg, sub)
        icv_path = (
            resolve_path(cfg, "derivatives")
            / PIPELINE
            / sub
            / "anat"
            / f"{sub}_desc-wmhsynthseg1mm_dseg.nii.gz"
        )
        if vent_path is None:
            no_ventricles.append(sub)
        tasks.append(
            {
                "sub": sub,
                "brain": str(brain_mask(cfg, sub)),
                "sources": sources,
                "reference": sources["O1"],
                "min_mm3": min_mm3,
                "ventricles": str(vent_path) if vent_path else None,
                "ventricle_source": vent_source,
                "pv_mm": cfg["biomarkers"]["periventricular_distance_mm"],
                "icv_labels": str(icv_path) if icv_path.is_file() else None,
            }
        )
    if missing:
        log.warning("%d segmentations missing, skipped (e.g. %s)", len(missing), missing[:5])
    if no_ventricles:
        log.warning("%d subjects without ventricle mask: no periventricular / deep split", len(no_ventricles))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = [r for rows in pool.map(subject_rows, tasks) for r in rows]
    table = pd.DataFrame(rows).rename(columns={"n_lesions_ge_min": f"n_lesions_ge{min_mm3:g}mm3"})
    table.insert(table.columns.get_loc("icv_ml") + 1, "wmh_pct_icv", 100 * table.wmh_ml / table.icv_ml)
    for i, col in enumerate(["scanner", "split", "seen_in_training"], start=1):
        table.insert(i, col, table.participant_id.map(part[col]))
    return table


def check_against_evaluation(cfg: dict[str, Any], table: pd.DataFrame) -> dict[str, float]:
    """Largest |volume - official volume| (ml) per source on the test set (should be ~0)."""
    out = {}
    for f in sorted((resolve_path(cfg, "results") / "tables").glob("evaluation_*.csv")):
        if f.stem.endswith("_summary"):
            continue
        ev = pd.read_csv(f).set_index("participant_id")
        model = ev.model.iloc[0]
        test = table[table.split == "test"].set_index(["source", "participant_id"]).wmh_ml
        if model in test.index.get_level_values(0):
            out[model] = float((test[model] - ev.pred_ml).abs().max())
        out.setdefault("O1", 0.0)
        out["O1"] = max(out["O1"], float((test["O1"] - ev.true_ml).abs().max()))
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="WMH volumes (7.1) and lesion structure (7.3).")
    parser.add_argument("--config")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    table = build(cfg, args.jobs)
    out = resolve_path(cfg, "results") / "tables" / "biomarkers.csv"
    table.round(4).to_csv(out, index=False)
    log.info(
        "%s: %d rows; per split and source:\n%s",
        out,
        len(table),
        table.groupby(["split", "source"]).size().to_string(),
    )
    log.info(
        "largest |WMH volume - official volume| on the test set (ml): %s",
        check_against_evaluation(cfg, table),
    )
    pd.set_option("display.width", 200)
    log.info(
        "medians per split, source:\n%s",
        table.groupby(["split", "source"]).median(numeric_only=True).round(2).to_string(),
    )


if __name__ == "__main__":
    main()
