"""Agreement between experts on the training set, used as the human ceiling.

O1 is itself one expert's delineation and two experts never draw exactly the same lesions, so no method can
reach a Dice of 1 against it. We score the two other experts, O3 and O4, who delineated the 60 training scans
independently, against O1 with the official metrics, and also against each other.

Dice, HD95 and lesion F1 are symmetric, so the O3 versus O4 comparison makes sense for them. AVD and lesion
recall depend on which observer is taken as the reference and are only given for completeness.

M1 is added on its 12 fold-0 validation subjects, using nnU-Net's own validation predictions. These subjects
were held out from the model used on the test set, so they are the only scans where M1 and the experts can be
compared on the same images. On them, M1 is tested against each expert with a paired Wilcoxon test on the
symmetric metrics and a Holm correction over the 6 tests.

Outputs: results/tables/interobserver.csv (one row per subject, rater and reference),
interobserver_summary.csv and interobserver_paired.csv.

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
from scipy import stats

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.evaluate.metrics import METRICS, evaluate_case
from wmh_multisite.evaluate.stats import holm, mean_ci
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
    """Copy of observer in which the other-pathology voxels of the reference are set to 2.

    This way those voxels are ignored when the observer itself is used as the reference.
    """
    obs = io.load_labels(observer, allowed=(0, 1))
    ref = io.load_labels(reference)
    return io.save_like(np.where(ref == 2, 2, obs).astype(np.float32), observer, out)


def compare(args: tuple[str, str, str, str, str]) -> dict[str, Any]:
    """Official metrics of one rater against one reference."""
    sub, rater, reference, rater_path, reference_path = args
    return {
        "participant_id": sub,
        "rater": rater,
        "reference": reference,
        **evaluate_case(reference_path, rater_path),
    }


def validation_cases(cfg: dict[str, Any], model: str = "resencm") -> dict[str, Path]:
    """Fold-0 validation predictions of nnU-Net, by participant id."""
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
    """Mean of each metric per comparison, overall, by scanner and on the scans shared with M1."""
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


def paired_with_m1(
    table: pd.DataFrame, n_boot: int = 2000, level: float = 0.95, seed: int = 42
) -> pd.DataFrame:
    """M1 against each expert on the scans they share, with O1 as the reference.

    For each symmetric metric we take the per-subject difference M1 - expert, its mean with a bootstrap CI
    (resampled within each scanner) and a Wilcoxon signed-rank test, then apply Holm over the 6 tests.
    n_m1_better counts the subjects where M1 is closer to O1.
    """
    rng = np.random.default_rng(seed)
    shared = table[table.shared_with_m1 & (table.reference == "O1")]
    wide = shared.pivot_table(index=["participant_id", "scanner"], columns="rater", values=SYMMETRIC)
    rows = []
    for rater in ("O3", "O4"):
        for metric in SYMMETRIC:
            pair = wide[metric][["M1", rater]].dropna()
            diff = pair["M1"] - pair[rater]
            better = (diff < 0) if metric == "hd95_mm" else (diff > 0)
            strata = pair.index.get_level_values("scanner").to_numpy()
            mean, lo, hi = mean_ci(diff.to_numpy(), strata, n_boot, level, rng)
            rows.append(
                {
                    "rater": rater,
                    "metric": metric,
                    "n": len(diff),
                    "mean_m1": float(pair["M1"].mean()),
                    "mean_rater": float(pair[rater].mean()),
                    "mean_diff_m1_minus_rater": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_m1_better": int(better.sum()),
                    "p_wilcoxon": float(stats.wilcoxon(diff).pvalue),
                }
            )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p_wilcoxon.to_numpy())
    return out


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
    e = cfg["evaluation"]
    paired = paired_with_m1(table, e["bootstrap_samples"], e["confidence_level"], e["seed"])
    paired.to_csv(out.with_name("interobserver_paired.csv"), index=False, float_format="%.4g")
    log.info("%s\n%s", out, summary.round(3).to_string())
    log.info("M1 vs experts on the shared scans:\n%s", paired.to_string(index=False, float_format="%.3g"))


if __name__ == "__main__":
    main()
