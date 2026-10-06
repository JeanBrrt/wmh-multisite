"""HD-BET brain masks, computed locally on the GPU (ROADMAP step 3.3).

HD-BET (Isensee et al., Hum Brain Mapp 2019) is a pretrained brain extraction network built on
nnU-Net: we only run inference. Its input is the raw 3D T1 at 1 mm (J-033); its output is a binary
mask on the grid of that T1.

How it runs (J-033):
- the project environment has a CPU-only PyTorch, so HD-BET runs in a separate, temporary uv
  environment (CUDA PyTorch + HD-BET), through ``hdbet_worker.py``;
- subjects are processed in batches; each batch is one worker process, run under the memory
  watchdog (RAM and GPU memory limits), with a hard GPU memory cap inside PyTorch;
- subjects that already have a mask are skipped, so an interrupted run can simply be restarted.

Every mask written by HD-BET is then checked (grid of its T1, values 0/1) before it enters the
derivatives, carried onto the FLAIR grid with the subject's rigid transform (step 3.2, ``genericLabel``
interpolation, never linear for a mask), and summarized in a QC table.

Outputs, in ``data/derivatives/wmh-multisite/sub-XXX/anat/``:
    sub-XXX_space-T1w_desc-brain_mask.nii.gz      HD-BET mask on the T1 grid
    sub-XXX_space-FLAIR_desc-brain_mask.nii.gz    the same mask on the FLAIR grid
and ``results/tables/brainmask_qc.csv``.

Usage:
    uv run python -m wmh_multisite.preproc.brainmask                    # all subjects
    uv run python -m wmh_multisite.preproc.brainmask --subjects sub-000 sub-050
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import ants
import numpy as np
import pandas as pd
from scipy import ndimage

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.utils import io
from wmh_multisite.utils.guard import run_guarded

log = logging.getLogger(__name__)

WORKER = Path(__file__).with_name("hdbet_worker.py")


def mask_paths(cfg: dict[str, Any], sub: str) -> dict[str, Path]:
    bids = resolve_path(cfg, "bids") / sub / "anat"
    out = resolve_path(cfg, "derivatives") / PIPELINE / sub / "anat"
    return {
        "t1w": bids / f"{sub}_T1w.nii.gz",
        "flair": bids / f"{sub}_FLAIR.nii.gz",
        "xfm": out / f"{sub}_from-T1w_to-FLAIR_mode-image_xfm.mat",
        "t1_head": out / f"{sub}_space-T1w_desc-head_mask.nii.gz",
        "brain_t1": out / f"{sub}_space-T1w_desc-brain_mask.nii.gz",
        "brain_flair": out / f"{sub}_space-FLAIR_desc-brain_mask.nii.gz",
    }


def worker_command(cfg: dict[str, Any], job_file: Path) -> list[str]:
    """``uv run`` command that executes the worker in the temporary CUDA environment."""
    rt = cfg["preproc"]["brainmask"]["runtime"]
    cmd = ["uv", "run", "--no-project", "--python", str(rt["python"])]
    for package in rt["packages"]:
        cmd += ["--with", package]
    cmd += [
        "--extra-index-url",
        rt["index_url"],
        "--index-strategy",
        "unsafe-best-match",
        "python",
        str(WORKER),
        str(job_file),
    ]
    return cmd


def accept_mask(cfg: dict[str, Any], sub: str, produced: Path) -> Path:
    """Check a mask written by HD-BET and store it in the derivatives (uint8, geometry of the T1)."""
    p = mask_paths(cfg, sub)
    if not io.same_grid(produced, p["t1w"]):
        raise ValueError(f"{sub}: the HD-BET mask is not on the grid of {p['t1w'].name}")
    mask = io.load_labels(produced, allowed=(0, 1))
    if not mask.any():
        raise ValueError(f"{sub}: empty HD-BET mask")
    io.save_like(mask, p["t1w"], p["brain_t1"], dtype=np.uint8)
    return p["brain_t1"]


def run_hdbet(cfg: dict[str, Any], subjects: list[str]) -> list[dict[str, Any]]:
    """Run HD-BET on the subjects without a mask, batch by batch, each batch under the watchdog."""
    bm = cfg["preproc"]["brainmask"]
    todo = [s for s in subjects if not mask_paths(cfg, s)["brain_t1"].is_file()]
    log.info("HD-BET: %d subjects to process (%d already done)", len(todo), len(subjects) - len(todo))
    staging = resolve_path(cfg, "derivatives") / PIPELINE / "_hdbet_staging"
    report = []
    for start in range(0, len(todo), bm["batch_size"]):
        batch = todo[start : start + bm["batch_size"]]
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        pairs = [[str(mask_paths(cfg, s)["t1w"]), str(staging / f"{s}_T1w_bet.nii.gz")] for s in batch]
        job = staging / "job.json"
        job.write_text(json.dumps({"pairs": pairs, "vram_cap_gb": bm["vram_cap_gb"], "tta": bm["tta"]}))
        t0 = time.time()
        result = run_guarded(
            worker_command(cfg, job),
            ram_limit_gb=bm["guard_ram_gb"],
            vram_limit_gb=bm["guard_vram_gb"],
            timeout_s=300 + 180 * len(batch),
            poll_s=0.5,
            log_path=staging / "worker.log",
        )
        accepted = []
        for s in batch:
            produced = staging / f"{s}_T1w_bet.nii.gz"
            if produced.is_file():
                accept_mask(cfg, s, produced)
                accepted.append(s)
        row = {
            "batch": start // bm["batch_size"],
            "subjects": len(batch),
            "accepted": len(accepted),
            "seconds": round(time.time() - t0, 1),
            "peak_ram_gb": result.peak_ram_gb,
            "peak_vram_gb": result.peak_vram_gb,
            "returncode": result.returncode,
            "killed": result.killed,
        }
        report.append(row)
        log.info("HD-BET batch %s", row)
        if not result.ok or len(accepted) < len(batch):
            tail = (staging / "worker.log").read_text(errors="replace")[-2000:]
            raise RuntimeError(
                f"HD-BET batch failed ({result.killed or result.returncode}); "
                f"{len(batch) - len(accepted)} mask(s) missing. Worker log tail:\n{tail}"
            )
    shutil.rmtree(staging, ignore_errors=True)
    return report


def to_flair_grid(cfg: dict[str, Any], sub: str, overwrite: bool = False) -> Path | None:
    """Carry the T1-grid brain mask onto the FLAIR grid with the rigid transform of step 3.2.

    Returns None if the subject has not been registered yet (its mask stays on the T1 grid).
    """
    p = mask_paths(cfg, sub)
    if not p["xfm"].is_file():
        return None
    if p["brain_flair"].is_file() and not overwrite:
        return p["brain_flair"]
    fixed = ants.image_read(str(p["flair"]))
    moving = ants.image_read(str(p["brain_t1"]))
    on_flair = ants.apply_transforms(fixed, moving, [str(p["xfm"])], interpolator="genericLabel")
    io.save_like(on_flair.numpy() > 0.5, p["flair"], p["brain_flair"])
    return p["brain_flair"]


def qc_row(cfg: dict[str, Any], sub: str) -> dict[str, Any]:
    p = mask_paths(cfg, sub)
    t1_mask = io.load_labels(p["brain_t1"], allowed=(0, 1)).astype(bool)
    ml_t1 = float(np.prod(io.describe(p["brain_t1"])["voxel_size_mm"])) / 1000
    labels, n = ndimage.label(t1_mask)
    sizes = np.bincount(labels.ravel())[1:] if n else np.array([0])
    row = {
        "participant_id": sub,
        "brain_ml": round(t1_mask.sum() * ml_t1, 1),
        "n_components": int(n),
        "largest_component_frac": round(float(sizes.max() / max(sizes.sum(), 1)), 4),
    }
    if p["t1_head"].is_file():  # the brain must lie inside the head (N4 mask, step 3.1)
        head = io.load_labels(p["t1_head"], allowed=(0, 1)).astype(bool)
        row["brain_inside_head_frac"] = round(float((t1_mask & head).sum() / max(t1_mask.sum(), 1)), 4)
    if p["brain_flair"].is_file():
        f_mask = io.load_labels(p["brain_flair"], allowed=(0, 1))
        ml_f = float(np.prod(io.describe(p["brain_flair"])["voxel_size_mm"])) / 1000
        row["brain_ml_flair_grid"] = round(float(f_mask.sum() * ml_f), 1)
        # share of the brain inside the FLAIR field of view (the FLAIR covers ~144 mm vertically)
        row["flair_coverage"] = round(row["brain_ml_flair_grid"] / max(row["brain_ml"], 1e-6), 3)
    return row


def run(cfg: dict[str, Any], subjects: list[str] | None = None) -> Path:
    participants = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t")
    participants = participants.set_index("participant_id")
    selected = subjects or participants.index.tolist()
    batches = run_hdbet(cfg, selected)
    done = [s for s in selected if mask_paths(cfg, s)["brain_t1"].is_file()]
    carried = sum(to_flair_grid(cfg, s) is not None for s in done)
    qc = pd.DataFrame([qc_row(cfg, s) for s in done])
    qc.insert(1, "scanner", qc.participant_id.map(participants.scanner))
    lo, hi = cfg["preproc"]["brainmask"]["qc_brain_ml"]
    qc["flagged"] = (qc.brain_ml < lo) | (qc.brain_ml > hi) | (qc.largest_component_frac < 0.99)
    out = resolve_path(cfg, "results") / "tables" / "brainmask_qc.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():  # keep the rows of subjects not processed this time
        old = pd.read_csv(out)
        qc = pd.concat([old[~old.participant_id.isin(qc.participant_id)], qc], ignore_index=True)
    qc = qc.sort_values("participant_id")
    qc.to_csv(out, index=False)
    if batches:
        pd.DataFrame(batches).to_csv(
            resolve_path(cfg, "derivatives") / PIPELINE / "hdbet_batches.csv", index=False
        )
    log.info(
        "%d masks (%d on the FLAIR grid) | brain volume median %.0f ml [%.0f-%.0f] | flagged: %s",
        len(qc),
        carried,
        qc.brain_ml.median(),
        qc.brain_ml.min(),
        qc.brain_ml.max(),
        qc[qc.flagged].participant_id.tolist() or "none",
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HD-BET brain masks on the local GPU, then FLAIR grid + QC.")
    parser.add_argument("--config", help="path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--subjects", nargs="*", help="only these participant ids (default: all)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(args.config), args.subjects)


if __name__ == "__main__":
    main()
