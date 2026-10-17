"""Does the QC catch degraded images? A test with simulated artifacts.

The challenge images were selected by the organizers, so few of them are really bad. To test the detector of
qc.outliers we create bad images ourselves: clean raw images receive TorchIO artifacts (noise, motion, bias
field, ghosting) at four increasing levels, then their IQMs are recomputed and judged by the detector fitted
on the real images. The detection rate is the share of degraded images that are not classed as usable, and
level 0 gives the false-alarm rate.

The images are those of n_per_scanner usable test subjects per scanner, the same as in qc.robustness, so that
detection and segmentation can be read on the same images. Each degraded image goes through N4 with the
pipeline settings, like a real one would, and its IQMs follow the MRIQC protocol. Its z-scores are computed
against the clean images of the same scanner.

A simulated bias shows in inu_range, which N4 estimates, and not in the tissue metrics, which are measured
after N4 as in MRIQC. The other modality keeps its real IQMs and the regions (brain, tissues, transform) are
those of the clean image, so this tests the metrics and not HD-BET, the registration or the segmentation. Only
one modality is degraded at a time: the FLAIR, or the T1 where the motion-sensitive CJV is used.

Outputs: results/tables/qc_artifacts.csv (qc_artifacts_T1w.csv for the T1) and
results/figures/qc_artifact_detection.png.

Usage:
    uv run python -m wmh_multisite.qc.artifacts                    # FLAIR
    uv run python -m wmh_multisite.qc.artifacts --modality T1w     # T1
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.qc.iqm import DIRECTION, JUDGED, labels_to_t1, measure, subject_paths
from wmh_multisite.qc.outliers import badness_z, forest_features
from wmh_multisite.utils import io

log = logging.getLogger(__name__)


def choose_subjects(cfg: dict[str, Any], n_per_scanner: int) -> list[str]:
    """The first n_per_scanner usable test subjects of each scanner (also used by qc.robustness)."""
    tables = resolve_path(cfg, "results") / "tables"
    flags = pd.read_csv(tables / "qc_flags.csv")
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t").set_index("participant_id")
    flags["split"] = flags.participant_id.map(part.split)
    usable = flags[(flags.qc_class == "usable") & (flags.split == "test")].sort_values("participant_id")
    return usable.groupby("scanner").head(n_per_scanner).participant_id.tolist()


def degrade(
    image: np.ndarray,
    brain: np.ndarray,
    artifact: str,
    level: float,
    seed: int,
    affine: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one TorchIO artifact at the given level (level 0 returns the image unchanged)."""
    if level == 0:
        return image.copy()
    import torch
    import torchio as tio

    scale = float(np.median(image[brain]))
    transform = {
        "noise": lambda: tio.RandomNoise(std=(level, level)),
        "motion": lambda: tio.RandomMotion(degrees=level, translation=level, num_transforms=2),
        "bias": lambda: tio.RandomBiasField(coefficients=(level, level)),
        "ghosting": lambda: tio.RandomGhosting(num_ghosts=(4, 4), axes=(0, 1), intensity=(level, level)),
    }[artifact]()
    torch.manual_seed(seed)
    tensor = torch.from_numpy((image / scale).astype(np.float32))[None]
    out = (
        transform(tio.ScalarImage(tensor=tensor, affine=np.eye(4) if affine is None else affine))
        .data[0]
        .numpy()
    )
    return np.clip(out, 0, None) * scale  # magnitude images are non-negative


def n4_field(image: np.ndarray, brain: np.ndarray, reference: Path) -> np.ndarray:
    """N4 field of an in-memory image, with the same code and settings as preproc.bias."""
    from wmh_multisite.preproc.bias import n4_correct

    with tempfile.TemporaryDirectory() as tmp:
        path = io.save_like(image.astype(np.float32), reference, Path(tmp) / "degraded.nii.gz")
        return n4_correct(path, brain)[1]


def badness(rows: pd.DataFrame, clean: pd.DataFrame, modality: str) -> pd.DataFrame:
    """z-scores of the degraded IQMs against the clean images of the same scanner and modality (positive is
    worse).
    """
    ref = clean[clean.modality == modality]
    out = pd.DataFrame(index=rows.index)
    for m in JUDGED[modality]:
        med = rows.scanner.map(ref.groupby("scanner")[m].median())
        mad = rows.scanner.map(
            ref.groupby("scanner")[m].apply(lambda v: (v - v.median()).abs().median() * 1.4826)
        )
        z = ((rows[m] - med) / mad.replace(0, np.nan)).fillna(0.0)
        out[f"{modality}_{m}"] = -DIRECTION[m] * z if DIRECTION[m] else z.abs()
    return out


def images(cfg: dict[str, Any], sub: str, modality: str) -> dict[str, Any]:
    """Raw image, brain mask, tissue labels and voxel size of one modality, on its own grid."""
    p = subject_paths(cfg, sub)
    nii = io.load(p[modality])
    if modality == "FLAIR":
        labels = io.load_labels(p["labels_FLAIR"], allowed=tuple(range(256)))
    else:
        labels = labels_to_t1(p["T1w"], p["labels_1mm"], p["xfm"])
    return {
        "path": p[modality],
        "image": io.load_data(p[modality]),
        "affine": nii.affine,  # real voxel size: motion in mm
        "spacing": nii.header.get_zooms()[:3],
        "brain": io.load_labels(p[f"brain_{modality}"], allowed=(0, 1)).astype(bool),
        "labels": labels,
    }


def run(cfg: dict[str, Any], modality: str = "FLAIR") -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest

    tables = resolve_path(cfg, "results") / "tables"
    clean = pd.read_csv(tables / "qc_iqm.csv")
    flags = pd.read_csv(tables / "qc_flags.csv", index_col="participant_id")
    q, seed = cfg["qc"], cfg["evaluation"]["seed"]
    a = q["artifacts"]
    z_clean = badness_z(clean, q["min_per_scanner"]).dropna(how="all")
    if z_clean.empty:
        log.warning(
            "no scanner with at least %d assessed images yet: nothing to degrade", q["min_per_scanner"]
        )
        return pd.DataFrame()
    forest = IsolationForest(n_estimators=500, contamination=q["iforest_contamination"], random_state=seed)
    forest.fit(forest_features(z_clean))
    chosen = choose_subjects(cfg, a["n_per_scanner"])
    log.info("degrading the %s of %d usable subjects: %s", modality, len(chosen), chosen)
    if not chosen:
        log.warning("no usable subject on an assessed scanner yet: nothing to degrade")
        return pd.DataFrame()
    rows = []
    for sub in chosen:
        im = images(cfg, sub, modality)
        brain = im["brain"]
        for artifact, levels in a["levels"].items():
            for k, level in enumerate(levels):
                degraded = degrade(im["image"], brain, artifact, level, seed + k, im["affine"])
                field = n4_field(degraded, brain, im["path"])
                iqm = measure(degraded, im["labels"], brain, field, im["spacing"], q["wmh_margin_mm"])
                log.info("%s %s %s level %d done", sub, modality, artifact, k)
                rows.append(
                    {
                        "participant_id": sub,
                        "scanner": flags.loc[sub, "scanner"],
                        "modality": modality,
                        "artifact": artifact,
                        "level_index": k,
                        "level": level,
                        **iqm,
                    }
                )
    table = pd.DataFrame(rows)
    z = badness(table, clean, modality)
    other = z_clean.loc[  # the other modality keeps the z of its real image
        table.participant_id, [c for c in z_clean.columns if not c.startswith(f"{modality}_")]
    ].reset_index(drop=True)
    features = forest_features(pd.concat([z.reset_index(drop=True), other], axis=1)[z_clean.columns])
    table["max_z"] = z.max(axis=1).to_numpy()
    table["worst_metric"] = z.idxmax(axis=1).to_numpy()
    table["iforest_outlier"] = forest.predict(features) == -1
    table["qc_class"] = np.select(
        [table.max_z >= q["z_exclude"], (table.max_z >= q["z_check"]) | table.iforest_outlier],
        ["exclude", "check"],
        default="usable",
    )
    table["detected"] = table.qc_class.isin(["check", "exclude"])
    return table


def detection_figure(table: pd.DataFrame, path, modality: str = "FLAIR") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rate = table.groupby(["artifact", "level_index"]).detected.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    for artifact in rate.columns:
        ax.plot(rate.index, rate[artifact], marker="o", label=artifact)
    ax.set_xticks(rate.index)
    ax.set_xlabel("artifact level (0 = unmodified image)")
    ax.set_ylabel("share of images flagged")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(
        f"QC detection of simulated artifacts ({modality}, n = {table.participant_id.nunique()} subjects)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detection of simulated artifacts by the QC.")
    parser.add_argument("--config")
    parser.add_argument("--modality", choices=["FLAIR", "T1w"], default="FLAIR")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    table = run(cfg, args.modality)
    if table.empty:
        return
    results = resolve_path(cfg, "results")
    suffix = "" if args.modality == "FLAIR" else f"_{args.modality}"  # FLAIR keeps the original file names
    table.to_csv(results / "tables" / f"qc_artifacts{suffix}.csv", index=False, float_format="%.5g")
    (results / "figures").mkdir(exist_ok=True)
    detection_figure(table, results / "figures" / f"qc_artifact_detection{suffix}.png", args.modality)
    log.info(
        "detection rate (artifact x level):\n%s",
        table.groupby(["artifact", "level_index"]).detected.mean().unstack(1).round(2).to_string(),
    )


if __name__ == "__main__":
    main()
