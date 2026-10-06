"""Export of the preprocessed images to the nnU-Net v2 dataset format (ROADMAP step 5.1).

nnU-Net v2 expects (documentation: "dataset_format.md"):

    nnUNet_raw/Dataset001_WMH/
        dataset.json
        imagesTr/WMH_000_0000.nii.gz    channel 0: FLAIR, N4-corrected (FLAIR grid)
        imagesTr/WMH_000_0001.nii.gz    channel 1: T1w, N4-corrected, registered onto the FLAIR grid
        labelsTr/WMH_000.nii.gz         0 background, 1 WMH, 2 ignore (uint8, FLAIR grid)

- The case identifier keeps the original challenge id (``WMH_000`` <-> ``sub-000``), and a table
  ``case_mapping.csv`` gives the correspondence with BIDS, scanner and split.
- Label 2 ("other pathology" in the challenge) is declared as nnU-Net's *ignore* label: those voxels
  contribute neither to the loss nor to the validation metric, as in the official evaluation. nnU-Net
  requires the ignore label to be the highest one, which is the case.
- Normalization is left to nnU-Net (z-score per image for MRI channels).
- ``splits_final.json``: 5 cross-validation folds stratified by scanner, so that every validation fold
  has the same number of subjects from each site (J-032). nnU-Net uses this file if it finds it in
  ``nnUNet_preprocessed/Dataset001_WMH/``; the Kaggle notebook copies it there.

Images are hard links to the derivatives (no extra disk space); labels are rewritten as uint8.

Usage:
    uv run python -m wmh_multisite.seg.nnunet            # build + check + zip
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.data.bids import link_or_copy
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.utils import io

log = logging.getLogger(__name__)

LABELS = {"background": 0, "WMH": 1, "ignore": 2}


def case_id(participant_id: str) -> str:
    """``sub-000`` -> ``WMH_000``."""
    return "WMH_" + participant_id.split("-", 1)[1]


def stratified_splits(cases: pd.DataFrame, n_folds: int, seed: int) -> list[dict[str, list[str]]]:
    """K-fold splits stratified by scanner and lesion load, in the format of ``splits_final.json``.

    Inside each scanner, cases are sorted by WMH volume and cut into consecutive blocks of
    ``n_folds`` cases of similar load; each block is dealt to the folds in a random order (fixed
    seed). Every validation fold thus receives ``n_per_scanner / n_folds`` subjects of each scanner
    (4 here) spanning the whole range of lesion loads.
    """
    rng = np.random.default_rng(seed)
    fold_of: dict[str, int] = {}
    for _, group in cases.groupby("scanner", sort=True):
        ordered = group.sort_values(["wmh_ml", "case_id"]).case_id.tolist()
        for start in range(0, len(ordered), n_folds):
            block = ordered[start : start + n_folds]
            for cid, fold in zip(block, rng.permutation(n_folds)[: len(block)], strict=True):
                fold_of[str(cid)] = int(fold)
    all_ids = sorted(fold_of)
    return [
        {"train": [c for c in all_ids if fold_of[c] != k], "val": [c for c in all_ids if fold_of[c] == k]}
        for k in range(n_folds)
    ]


def dataset_json(cfg: dict[str, Any], n_training: int) -> dict[str, Any]:
    nn = cfg["segmentation"]["nnunet"]
    return {
        "name": nn["dataset_name"],
        "description": "MICCAI WMH 2017, FLAIR + T1 (N4-corrected, T1 rigidly registered to the FLAIR grid) "
        "by wmh-multisite; label 2 (other pathology) ignored as in the official evaluation",
        "reference": "Kuijf et al., IEEE TMI 2019; data DOI " + cfg["dataset"]["doi"],
        "licence": cfg["dataset"]["license"],
        "channel_names": {k: v for k, v in nn["channels"].items()},
        "labels": LABELS,
        "numTraining": n_training,
        "file_ending": ".nii.gz",
    }


def check_case(flair: Path, t1: Path, label: Path) -> dict[str, Any]:
    """Same grid for the three files, labels in {0, 1, 2}. Raises if not."""
    for other in (t1, label):
        if not io.same_grid(flair, other):
            raise ValueError(f"{other.name} is not on the grid of {flair.name}")
    lab = io.load_labels(label, allowed=tuple(LABELS.values()))
    ml = float(np.prod(io.describe(flair)["voxel_size_mm"])) / 1000
    return {
        "wmh_ml": round(float((lab == 1).sum() * ml), 2),
        "ignore_ml": round(float((lab == 2).sum() * ml), 2),
    }


def build(cfg: dict[str, Any], split: str = "training") -> Path:
    """Write ``nnUNet_raw/<dataset>/`` for the subjects of ``split`` (only 'training' has labels)."""
    nn = cfg["segmentation"]["nnunet"]
    bids = resolve_path(cfg, "bids")
    deriv = resolve_path(cfg, "derivatives") / PIPELINE
    root = resolve_path(cfg, "nnunet") / "nnUNet_raw" / nn["dataset_name"]
    images, labels = root / "imagesTr", root / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    participants = pd.read_csv(bids / "participants.tsv", sep="\t")
    selected = participants[participants.split == split]
    rows = []
    for _, p in selected.iterrows():
        sub, cid = p.participant_id, case_id(p.participant_id)
        src = {
            "flair": deriv / sub / "anat" / f"{sub}_desc-n4_FLAIR.nii.gz",
            "t1": deriv / sub / "anat" / f"{sub}_space-FLAIR_desc-n4_T1w.nii.gz",
            "label": bids / "derivatives/manual" / sub / "anat" / f"{sub}_space-FLAIR_desc-O1_dseg.nii.gz",
        }
        missing = [k for k, v in src.items() if not v.is_file()]
        if missing:
            raise FileNotFoundError(f"{sub}: missing {missing} (run N4 and registration first)")
        dst = {
            "flair": images / f"{cid}_0000.nii.gz",
            "t1": images / f"{cid}_0001.nii.gz",
            "label": labels / f"{cid}.nii.gz",
        }
        link_or_copy(src["flair"], dst["flair"])
        link_or_copy(src["t1"], dst["t1"])
        if not dst["label"].is_file():  # the challenge stores labels as float32: rewrite as uint8
            io.save_like(io.load_labels(src["label"]), src["flair"], dst["label"], dtype=np.uint8)
        rows.append(
            {
                "case_id": cid,
                "participant_id": sub,
                "scanner": p.scanner,
                "split": p.split,
                **check_case(dst["flair"], dst["t1"], dst["label"]),
            }
        )

    cases = pd.DataFrame(rows)
    cases.to_csv(root / "case_mapping.csv", index=False)
    (root / "dataset.json").write_text(json.dumps(dataset_json(cfg, len(cases)), indent=2) + "\n")
    splits = stratified_splits(cases, nn["n_folds"], cfg["seed"])
    (root / "splits_final.json").write_text(json.dumps(splits, indent=1) + "\n")
    log.info(
        "%s: %d cases | WMH median %.1f ml | fold sizes (val) %s",
        root,
        len(cases),
        cases.wmh_ml.median(),
        [len(s["val"]) for s in splits],
    )
    return root


def zip_dataset(root: Path) -> Path:
    """Archive for the Kaggle upload (images are already gzip-compressed: store, do not recompress)."""
    archive = root.parent.parent / f"{root.name}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                z.write(path, Path(root.name) / path.relative_to(root))
    log.info("Archive %s (%.0f MB)", archive, archive.stat().st_size / 1e6)
    return archive


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export the training cases to the nnU-Net v2 format.")
    parser.add_argument("--config", help="path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--rebuild", action="store_true", help="delete the dataset folder first")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    root = resolve_path(cfg, "nnunet") / "nnUNet_raw" / cfg["segmentation"]["nnunet"]["dataset_name"]
    if args.rebuild and root.exists():
        shutil.rmtree(root)  # only links and rewritten labels: the derivatives are untouched
    root = build(cfg)
    if not args.no_zip:
        zip_dataset(root)


if __name__ == "__main__":
    main()
