"""Inter-observer agreement on the training set: the human ceiling (ROADMAP step 6.3, J-046).

Why: the reference standard (O1) is itself one expert's delineation. Two experts never draw exactly
the same lesions, so no method can be expected to reach Dice = 1 against O1. Scoring two other experts
(O3, O4, who delineated the 60 training scans independently) against O1 **with the same official
metrics** gives the level a human reaches: the ceiling against which a method is read.

    60 training scans                                       (official metrics, label 2 of O1 ignored)
      O1 (reference) <---- O3                               "how far is one human from the reference?"
      O1 (reference) <---- O4
      O3 <---- O4  (O1's label 2 also ignored)              "how far are two humans from each other?"
    12 of them never seen by nnU-Net (fold-0 validation cases)
      O1 (reference) <---- M1 (nnU-Net)                     paired with O3 and O4 on the same 12 scans

Reading the metrics in this setting:
- Dice, HD95 and lesion F1 are **symmetric** (swapping the two masks gives the same value), so the
  O3-O4 comparison is well defined for them.
- AVD (normalised by the reference volume) and lesion recall (share of the reference lesions found)
  are **directional**; for O3-O4 they depend on which observer is taken as reference (here O3), and are
  reported for completeness only.

The M1 rows use the fold-0 validation predictions written by nnU-Net at the end of training
(``fold_0/validation``): the 12 subjects were held out from that model, which is the model used on the
test set (final checkpoint, not selected on them). They are the only scans where M1 and the humans can
be compared on the same images; the test-set scores of 6.1 are on other subjects.

Outputs:
    results/tables/interobserver.csv           one row per (subject, rater, reference)
    results/tables/interobserver_summary.csv   means overall, by scanner, and on the 12 shared scans

Usage:
    uv run python -m wmh_multisite.evaluate.interobserver
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

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.evaluate.metrics import METRICS, evaluate_case
from wmh_multisite.seg.nnunet_infer import model_dir
from wmh_multisite.utils import io

log = logging.getLogger(__name__)

SYMMETRIC = ["dice", "hd95_mm", "lesion_f1"]


def manual(cfg: dict[str, Any], sub: str, observer: str) -> Path:
    return (
        resolve_path(cfg, "bids")
        / "derivatives/manual"
        / sub
        / "anat"
        / f"{sub}_space-FLAIR_desc-{observer}_dseg.nii.gz"
    )


def with_ignore(observer: Path, reference: Path, out: Path) -> Path:
    """Copy of ``observer`` (0/1) where the "other pathology" voxels of ``reference`` (label 2) are set
    to 2, so that they are ignored when ``observer`` serves as the reference."""
    obs = io.load_labels(observer, allowed=(0, 1))
    ref = io.load_labels(reference)
    return io.save_like(np.where(ref == 2, 2, obs).astype(np.float32), observer, out)


def compare(args: tuple[str, str, str, str, str]) -> dict[str, Any]:
    """One comparison: ``rater`` scored against ``reference`` (official metrics)."""
    sub, rater, reference, rater_path, reference_path = args
    return {
        "participant_id": sub,
        "rater": rater,
        "reference": reference,
        **evaluate_case(reference_path, rater_path),
    }


def validation_cases(cfg: dict[str, Any], model: str = "resencm") -> dict[str, Path]:
    """participant_id -> fold-0 validation prediction of nnU-Net (subjects held out from that model)."""
    nn = cfg["segmentation"]["nnunet"]
    folder = model_dir(cfg, model) / f"fold_{nn['fold']}" / "validation"
    mapping = pd.read_csv(
        resolve_path(cfg, "nnunet") / "nnUNet_raw" / nn["dataset_name"] / "case_mapping.csv"
    ).set_index("case_id")
    out = {}
    for f in sorted(folder.glob("*.nii*")):
        case = f.name.split(".")[0]
        out[mapping.loc[case, "participant_id"]] = f
    return out


def run(cfg: dict[str, Any], jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t").set_index("participant_id")
    train = part[part.split == "training"].index.tolist()
    m1 = validation_cases(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        tasks = []
        for sub in train:
            o1, o3, o4 = (manual(cfg, sub, o) for o in ("O1", "O3", "O4"))
            tasks += [(sub, "O3", "O1", str(o3), str(o1)), (sub, "O4", "O1", str(o4), str(o1))]
            o3_ref = with_ignore(o3, o1, Path(tmp) / f"{sub}_O3_ref.nii.gz")
            tasks.append((sub, "O4", "O3", str(o4), str(o3_ref)))
            if sub in m1:
                tasks.append((sub, "M1", "O1", str(m1[sub]), str(o1)))
        log.info(
            "%d comparisons (%d training subjects, %d with an M1 validation prediction)",
            len(tasks),
            len(train),
            len(m1),
        )
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(compare, tasks))
    table = pd.DataFrame(rows)
    table.insert(1, "scanner", table.participant_id.map(part.scanner))
    table["shared_with_m1"] = table.participant_id.isin(m1)
    return table, summarize(table)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Mean of each metric per (rater vs reference), overall, by scanner, and on the scans shared with M1."""
    pairs = table.rater + " vs " + table.reference
    groups: dict[str, pd.DataFrame] = {}
    for pair, g in table.groupby(pairs):
        groups[f"{pair} | all"] = g
        for scanner, gs in g.groupby("scanner"):
            groups[f"{pair} | scanner={scanner}"] = gs
        shared = g[g.shared_with_m1]
        if len(shared):
            groups[f"{pair} | 12 scans shared with M1"] = shared
    return pd.DataFrame(
        {name: {"n": len(g), **{m: g[m].mean() for m in METRICS}} for name, g in groups.items()}
    ).T


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Inter-observer agreement (human ceiling) on the training set."
    )
    parser.add_argument("--config")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    table, summary = run(cfg, args.jobs)
    out = resolve_path(cfg, "results") / "tables" / "interobserver.csv"
    table.round(4).to_csv(out, index=False)
    summary.round(4).to_csv(out.with_name("interobserver_summary.csv"))
    log.info("%s\n%s", out, summary.round(3).to_string())


if __name__ == "__main__":
    main()
