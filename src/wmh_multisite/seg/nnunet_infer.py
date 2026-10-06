"""nnU-Net inference on the local GPU (ROADMAP step 5.3).

The trained model (``data/nnunet/nnUNet_results/...``, downloaded from Kaggle) is applied to the
preprocessed images of each subject: FLAIR N4 (channel 0) and T1 N4 on the FLAIR grid (channel 1),
read directly from the derivatives. Predictions are written on the FLAIR grid, as BIDS derivatives:

    data/derivatives/wmh-multisite/sub-XXX/anat/sub-XXX_space-FLAIR_desc-<model>_dseg.nii.gz   (0 / 1)

Each batch runs ``nnunet_infer_worker.py`` in the temporary CUDA environment (the project's PyTorch is
CPU-only), under the memory watchdog, with a hard cap on GPU memory (as HD-BET, J-033). Subjects
already predicted are skipped; every prediction is checked (FLAIR grid, labels 0/1) before it is kept.

Usage:
    uv run python -m wmh_multisite.seg.nnunet_infer --split test
    uv run python -m wmh_multisite.seg.nnunet_infer --subjects sub-100 --model resencm
    uv run python -m wmh_multisite.seg.nnunet_infer --split test --inputs challenge   # desc-resencmpre
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.utils import io
from wmh_multisite.utils.guard import run_guarded

log = logging.getLogger(__name__)

WORKER = Path(__file__).with_name("nnunet_infer_worker.py")


def worker_command(cfg: dict[str, Any], job_file: Path) -> list[str]:
    rt = cfg["preproc"]["brainmask"]["runtime"]  # same temporary CUDA environment as HD-BET
    inf = cfg["segmentation"]["nnunet"]["inference"]
    cmd = ["uv", "run", "--no-project", "--python", rt["python"]]
    for pkg in inf["packages"]:
        cmd += ["--with", pkg]
    return cmd + [
        "--extra-index-url",
        rt["index_url"],
        "--index-strategy",
        "unsafe-best-match",
        "python",
        str(WORKER),
        str(job_file),
    ]


def model_dir(cfg: dict[str, Any], model: str) -> Path:
    nn = cfg["segmentation"]["nnunet"]
    spec = nn["models"][model]
    return (
        resolve_path(cfg, "nnunet")
        / "nnUNet_results"
        / nn["dataset_name"]
        / f"{spec['trainer']}__{nn['plans']}__{nn['configuration']}"
    )


def paths(cfg: dict[str, Any], sub: str, model: str) -> dict[str, Path]:
    d = resolve_path(cfg, "derivatives") / PIPELINE / sub / "anat"
    return {
        "flair": d / f"{sub}_desc-n4_FLAIR.nii.gz",
        "t1": d / f"{sub}_space-FLAIR_desc-n4_T1w.nii.gz",
        "pred": d / f"{sub}_space-FLAIR_desc-{model}_dseg.nii.gz",
    }


def check_prediction(pred: Path, flair: Path) -> None:
    if not io.same_grid(pred, flair):
        raise ValueError(f"{pred.name}: not on the FLAIR grid")
    io.load_labels(pred, allowed=(0, 1))


def predict_cases(
    cfg: dict[str, Any], model: str, cases: list[dict[str, Path]], staging_name: str
) -> pd.DataFrame:
    """Predict arbitrary cases: each ``{"id", "flair", "t1", "out"}`` (FLAIR grid). Used by ``run`` (our
    preprocessing), the preprocessing experiment (``--inputs challenge``) and the robustness study
    (``qc.robustness``). Cases whose output exists are skipped; each prediction is checked before it is kept.
    """
    inf = cfg["segmentation"]["nnunet"]["inference"]
    mdir = model_dir(cfg, model)
    if not (mdir / "plans.json").is_file():
        raise FileNotFoundError(f"model not found: {mdir}")
    staging = resolve_path(cfg, "derivatives") / PIPELINE / staging_name
    todo = [c for c in cases if not Path(c["out"]).is_file()]
    log.info("nnU-Net %s: %d cases to predict (%d already done)", model, len(todo), len(cases) - len(todo))
    batches = []
    for b, start in enumerate(range(0, len(todo), inf["batch_size"])):
        batch = todo[start : start + inf["batch_size"]]
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        job = {
            "model_dir": str(mdir),
            "fold": cfg["segmentation"]["nnunet"]["fold"],
            "checkpoint": inf["checkpoint"],
            "vram_cap_gb": inf["vram_cap_gb"],
            "use_mirroring": inf["use_mirroring"],
            "tile_step_size": inf["tile_step_size"],
            "cases": [[str(c["flair"]), str(c["t1"]), str(staging / c["id"])] for c in batch],
        }
        job_file = staging / "job.json"
        job_file.write_text(json.dumps(job, indent=1))
        timeout = inf["seconds_per_subject_max"] * len(batch) + 600
        r = run_guarded(
            worker_command(cfg, job_file),
            ram_limit_gb=inf["guard_ram_gb"],
            vram_limit_gb=inf["guard_vram_gb"],
            timeout_s=timeout,
            log_path=staging / "worker.log",
        )
        accepted = 0
        for c in batch:
            out = staging / f"{c['id']}.nii.gz"
            if out.is_file():
                check_prediction(out, Path(c["flair"]))
                Path(c["out"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(out), c["out"])
                accepted += 1
        row = {
            "batch": b,
            "cases": len(batch),
            "accepted": accepted,
            "seconds": r.seconds,
            "peak_ram_gb": r.peak_ram_gb,
            "peak_vram_gb": r.peak_vram_gb,
            "returncode": r.returncode,
            "killed": r.killed,
        }
        log.info("nnU-Net batch %s", row)
        batches.append(row)
        if accepted < len(batch):
            tail = (staging / "worker.log").read_text(errors="replace")[-2000:]
            raise RuntimeError(f"batch {b}: {accepted}/{len(batch)} predictions; worker log tail:\n{tail}")
    shutil.rmtree(staging, ignore_errors=True)
    return pd.DataFrame(batches)


def challenge_inputs(cfg: dict[str, Any], sub: str) -> tuple[Path, Path]:
    """The organizers' preprocessing (``pre/``: SPM12 bias correction, T1 registered with elastix)."""
    c = resolve_path(cfg, "bids") / "derivatives" / "challenge" / sub / "anat"
    return c / f"{sub}_desc-spm12_FLAIR.nii.gz", c / f"{sub}_space-FLAIR_desc-spm12_T1w.nii.gz"


def run(cfg: dict[str, Any], subjects: list[str], model: str, inputs: str = "ours") -> pd.DataFrame:
    """``inputs="ours"``: our preprocessing (5.3), predictions ``desc-<model>``; ``inputs="challenge"``:
    the organizers' ``pre/`` images (preprocessing experiment, J-010), predictions ``desc-<model>pre``."""
    cases = []
    for s in subjects:
        p = paths(cfg, s, model)
        if inputs == "ours":
            flair, t1, out = p["flair"], p["t1"], p["pred"]
        else:
            flair, t1 = challenge_inputs(cfg, s)
            out = p["pred"].with_name(p["pred"].name.replace(f"desc-{model}_", f"desc-{model}pre_"))
        cases.append({"id": s, "flair": flair, "t1": t1, "out": out})
    return predict_cases(cfg, model, cases, f"_nnunet_{model}_{inputs}_staging")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="nnU-Net inference on the local GPU.")
    parser.add_argument("--config")
    parser.add_argument("--model", default="resencm", help="key of segmentation.nnunet.models in the config")
    parser.add_argument("--split", choices=["training", "test", "all"], default="test")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument(
        "--inputs",
        choices=["ours", "challenge"],
        default="ours",
        help="our preprocessing, or the organizers' pre/ images (preprocessing experiment)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t")
    splits = ("training", "test") if args.split == "all" else (args.split,)
    subjects = part[part.split.isin(splits)].participant_id.tolist()
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]
    run(cfg, subjects, args.model, args.inputs)


if __name__ == "__main__":
    main()
