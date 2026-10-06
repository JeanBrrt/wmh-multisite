"""Robustness of nnU-Net (M1) to image artifacts (extension of ROADMAP step 4.3, J-057).

4.3 asked "does the QC detect a degraded image?"; the question that matters is "from which artifact
level does the segmentation itself degrade?". An artifact the QC misses but that does not harm the
segmentation is harmless; one that harms it before the QC sees it is a real risk.

    10 test subjects of class "usable" (``qc.robustness.n_per_scanner`` per scanner; test subjects so
    that the model has never seen them)
      x artifact (noise, motion, bias, ghosting) x level 1..4 (``qc.artifacts.levels``; level 0 = the
        existing prediction on the unmodified image)
      -> the **model input** (FLAIR N4) degraded with TorchIO (``qc.artifacts.degrade``, fixed seed); the
         T1 channel is left intact (only the FLAIR is degraded)
      -> nnU-Net (same model, same inference settings as 5.3)
      -> official metrics against O1 (6.1)
      v
    results/tables/robustness_nnunet.csv, results/figures/robustness_nnunet.png

Degraded images and predictions are written to ``data/derivatives/wmh-multisite/_robustness/`` (not
pipeline outputs). The bias artifact applied to the N4 image stands for a residual bias that N4 did not
remove.

Usage:
    uv run python -m wmh_multisite.qc.robustness
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.evaluate.interobserver import manual
from wmh_multisite.evaluate.metrics import evaluate_case
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.qc.artifacts import degrade
from wmh_multisite.seg.nnunet_infer import paths as nnunet_paths
from wmh_multisite.seg.nnunet_infer import predict_cases
from wmh_multisite.utils import io

log = logging.getLogger(__name__)
MODEL = "resencm"


def choose_subjects(cfg: dict[str, Any], n_per_scanner: int) -> list[str]:
    tables = resolve_path(cfg, "results") / "tables"
    flags = pd.read_csv(tables / "qc_flags.csv")
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t").set_index("participant_id")
    flags["split"] = flags.participant_id.map(part.split)
    usable = flags[(flags.qc_class == "usable") & (flags.split == "test")].sort_values("participant_id")
    return usable.groupby("scanner").head(n_per_scanner).participant_id.tolist()


def build_cases(cfg: dict[str, Any], subjects: list[str]) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Write the degraded FLAIRs; return the nnU-Net cases and the case table."""
    work = resolve_path(cfg, "derivatives") / PIPELINE / "_robustness"
    levels = cfg["qc"]["artifacts"]["levels"]
    seed = cfg["evaluation"]["seed"]
    cases, rows = [], []
    for sub in subjects:
        p = nnunet_paths(cfg, sub, MODEL)
        brain_path = (
            resolve_path(cfg, "derivatives")
            / PIPELINE
            / sub
            / "anat"
            / f"{sub}_space-FLAIR_desc-brain_mask.nii.gz"
        )
        flair = io.load_data(p["flair"])
        brain = io.load_labels(brain_path, allowed=(0, 1)).astype(bool)
        affine = io.load(p["flair"]).affine
        rows.append(
            {
                "participant_id": sub,
                "artifact": "none",
                "level_index": 0,
                "level": 0.0,
                "pred": str(p["pred"]),
            }
        )
        for artifact, values in levels.items():
            for k, level in enumerate(values):
                if k == 0:
                    continue  # level 0 = the existing prediction on the unmodified image
                case_id = f"{sub}_{artifact}{k}"
                d = work / sub
                d.mkdir(parents=True, exist_ok=True)
                image_path = d / f"{case_id}_FLAIR.nii.gz"
                if not image_path.is_file():
                    io.save_like(
                        degrade(flair, brain, artifact, level, seed + k, affine),
                        p["flair"],
                        image_path,
                        dtype=np.float32,
                    )
                out = d / f"{case_id}_pred.nii.gz"
                cases.append({"id": case_id, "flair": image_path, "t1": p["t1"], "out": out})
                rows.append(
                    {
                        "participant_id": sub,
                        "artifact": artifact,
                        "level_index": k,
                        "level": level,
                        "pred": str(out),
                    }
                )
    return cases, pd.DataFrame(rows)


def _score(args: tuple[str, str]) -> dict[str, float]:
    truth, pred = args
    return evaluate_case(truth, pred)


def figure(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = table[table.artifact == "none"].set_index("participant_id").dice
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for artifact, g in table[table.artifact != "none"].groupby("artifact"):
        drop = g.assign(drop=g.dice.to_numpy() - base.loc[g.participant_id].to_numpy())
        both = pd.concat([table[table.artifact == "none"].assign(artifact=artifact, drop=0.0), drop])
        m = both.groupby("level_index")[["dice", "drop"]].mean()
        axes[0].plot(m.index, m.dice, marker="o", label=artifact)
        axes[1].plot(m.index, m["drop"], marker="o", label=artifact)
    axes[0].set_ylabel("Dice vs O1 (mean)")
    axes[1].set_ylabel("Dice change vs unmodified image (mean)")
    for ax in axes:
        ax.set_xlabel("artifact level (0 = unmodified image)")
        ax.set_xticks(range(5))
        ax.legend(fontsize=8)
    fig.suptitle(
        f"nnU-Net (M1) on degraded FLAIRs ({table.participant_id.nunique()} test subjects)", fontsize=10
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Robustness of nnU-Net to simulated artifacts (4.3 extension)."
    )
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    subjects = choose_subjects(cfg, cfg["qc"]["robustness"]["n_per_scanner"])
    log.info("subjects: %s", subjects)
    cases, table = build_cases(cfg, subjects)
    predict_cases(cfg, MODEL, cases, "_nnunet_robustness_staging")
    tasks = [
        (str(manual(cfg, s, "O1")), pred) for s, pred in zip(table.participant_id, table.pred, strict=True)
    ]
    with ProcessPoolExecutor(max_workers=4) as pool:
        scores = pd.DataFrame(list(pool.map(_score, tasks)))
    table = pd.concat([table.drop(columns="pred").reset_index(drop=True), scores], axis=1)
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t").set_index("participant_id")
    table.insert(1, "scanner", table.participant_id.map(part.scanner))
    results = resolve_path(cfg, "results")
    table.to_csv(results / "tables" / "robustness_nnunet.csv", index=False, float_format="%.4g")
    figure(table, results / "figures" / "robustness_nnunet.png")
    pd.set_option("display.width", 200)
    summary = table.groupby(["artifact", "level_index"])[["dice", "avd_pct", "lesion_recall"]].mean().round(3)
    log.info("mean metrics by artifact and level:\n%s", summary.to_string())


if __name__ == "__main__":
    main()
