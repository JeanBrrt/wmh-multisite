"""N4 bias field correction with the HD-BET brain mask.

This is the last preprocessing step, once the brain masks exist on both grids. The raw FLAIR and the raw 1 mm
T1 are corrected with N4 (Tustison et al., 2010), which fits a smooth multiplicative field on the voxels of a
mask. We use the HD-BET brain mask because it means the same thing on every site, whereas an Otsu head mask
changed with the scanner contrast (preproc.n4.mask: otsu still gives that earlier variant). The corrected T1
is then resampled onto the FLAIR grid to serve as the second nnU-Net channel.

Outputs, in data/derivatives/wmh-multisite/sub-XXX/anat/:
    sub-XXX_desc-n4_FLAIR.nii.gz             corrected FLAIR (float32)
    sub-XXX_desc-biasfield_FLAIR.nii.gz      estimated field (corrected = raw / field)
    sub-XXX_desc-n4_T1w.nii.gz               corrected T1, on its own grid
    sub-XXX_desc-biasfield_T1w.nii.gz        T1 field
    sub-XXX_space-FLAIR_desc-n4_T1w.nii.gz   corrected T1 resampled onto the FLAIR grid

Usage:
    uv run python -m wmh_multisite.preproc.bias --split training --jobs 2
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import ants
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.filters import threshold_otsu

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.utils import io
from wmh_multisite.utils.guard import call_guarded

log = logging.getLogger(__name__)

MODALITIES = {"FLAIR": "FLAIR", "T1w": "T1w"}  # BIDS suffix -> space label of its mask


def head_mask(volume: np.ndarray) -> np.ndarray:
    """Rough head mask: Otsu threshold, largest connected component, holes filled.

    Otsu separates the air from the much brighter head, and keeping the largest component removes isolated
    bright voxels in the air. Holes are filled in 3D and then slice by slice along each axis, because dark
    structures inside the head (ventricles on the FLAIR, CSF, bone) often reach the edge of the field of view,
    where a 3D fill alone would miss them.
    """
    above = volume > threshold_otsu(volume)
    labels, n = ndimage.label(above)
    if n == 0:
        raise ValueError("Empty foreground: the image has no voxel above the Otsu threshold")
    sizes = np.bincount(labels.ravel())[1:]
    mask = labels == (np.argmax(sizes) + 1)
    mask = ndimage.binary_fill_holes(mask)
    for axis in range(3):
        mask = np.stack([ndimage.binary_fill_holes(s) for s in np.moveaxis(mask, axis, 0)], axis=0)
        mask = np.moveaxis(mask, 0, axis)
    return mask


def n4_correct(
    image_path: Path, mask: np.ndarray | None = None, shrink_factor: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the corrected image, the field and the mask, as arrays in the voxel order of the file.

    mask sets where N4 estimates the field (None means the Otsu head mask). The correction is raw / field, so
    the saved field and the saved image always agree.
    """
    image = ants.image_read(str(image_path), pixeltype="float")
    mask = head_mask(image.numpy()) if mask is None else mask.astype(bool)
    field = ants.n4_bias_field_correction(
        image,
        mask=image.new_image_like(mask.astype("float32")),
        shrink_factor=shrink_factor,
        return_bias_field=True,
    ).numpy()
    raw = image.numpy()
    corrected = np.divide(raw, field, out=np.zeros_like(raw), where=field > 0)
    return corrected.astype(np.float32), field.astype(np.float32), mask


def _anat_dir(cfg: dict[str, Any], sub: str) -> Path:
    return resolve_path(cfg, "derivatives") / PIPELINE / sub / "anat"


def correct_subject(sub: str, config_path: str | None = None, shrink_factor: int = 4) -> list[dict[str, Any]]:
    """N4 on the FLAIR and the T1 of one subject, then the T1 resampled onto the FLAIR grid.

    Images that are already corrected are skipped. The function lives at module level and takes plain
    arguments so that it can run in a guarded child process. Returns the QC rows.
    """
    from wmh_multisite.preproc.register import resample_onto_flair  # avoids a circular import

    cfg = load_config(config_path)
    method = cfg["preproc"]["n4"]["mask"]
    bids_anat = resolve_path(cfg, "bids") / sub / "anat"
    out = _anat_dir(cfg, sub)
    out.mkdir(parents=True, exist_ok=True)
    xfm = out / f"{sub}_from-T1w_to-FLAIR_mode-image_xfm.mat"
    rows = []
    for suffix, space in MODALITIES.items():
        raw_path = bids_anat / f"{sub}_{suffix}.nii.gz"
        dst = {
            "n4": out / f"{sub}_desc-n4_{suffix}.nii.gz",
            "field": out / f"{sub}_desc-biasfield_{suffix}.nii.gz",
        }
        mask_path = out / f"{sub}_space-{space}_desc-{'brain' if method == 'hdbet' else 'head'}_mask.nii.gz"
        if all(p.is_file() for p in dst.values()):
            field = io.load_data(dst["field"])
            mask = io.load_labels(mask_path, allowed=(0, 1)).astype(bool)
            rows.append({"participant_id": sub, "image": suffix, "status": "present", **_qc(field, mask)})
            continue
        if method == "hdbet":
            if not mask_path.is_file():
                raise FileNotFoundError(f"{sub}: missing {mask_path.name} (run brainmask after register)")
            mask = io.load_labels(mask_path, allowed=(0, 1))
        else:
            mask = None
        t0 = time.time()
        corrected, field, mask = n4_correct(raw_path, mask, shrink_factor)
        io.save_like(corrected, raw_path, dst["n4"], dtype=np.float32)
        io.save_like(field, raw_path, dst["field"], dtype=np.float32)
        if method == "otsu":
            io.save_like(mask, raw_path, mask_path)
        rows.append(
            {
                "participant_id": sub,
                "image": suffix,
                "status": "corrected",
                "seconds": round(time.time() - t0, 1),
                **_qc(field, mask),
            }
        )
    # Second nnU-Net channel: the corrected T1 on the FLAIR grid.
    t1_on_flair = out / f"{sub}_space-FLAIR_desc-n4_T1w.nii.gz"
    if not t1_on_flair.is_file():
        if not xfm.is_file():
            raise FileNotFoundError(f"{sub}: missing {xfm.name} (run register first)")
        resample_onto_flair(
            bids_anat / f"{sub}_FLAIR.nii.gz", out / f"{sub}_desc-n4_T1w.nii.gz", xfm, t1_on_flair
        )
    return rows


def _qc(field: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Mask coverage and spread of the field inside the mask (5th and 95th percentiles relative to the
    median).
    """
    inside = field[mask]
    return {
        "mask_fraction": round(float(mask.mean()), 3),
        "field_p5": round(float(np.percentile(inside, 5) / np.median(inside)), 3),
        "field_p95": round(float(np.percentile(inside, 95) / np.median(inside)), 3),
    }


def _write_dataset_description(cfg: dict[str, Any]) -> None:
    path = resolve_path(cfg, "derivatives") / PIPELINE / "dataset_description.json"
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "Name": "wmh-multisite derivatives",
                "BIDSVersion": cfg["bids"]["version"],
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {"Name": "wmh-multisite", "CodeURL": "https://github.com/JeanBrrt/wmh-multisite"}
                ],
                "SourceDatasets": [{"URL": "../../bids"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run(
    cfg: dict[str, Any],
    splits: tuple[str, ...],
    jobs: int,
    subjects: list[str] | None = None,
    config_path: str | None = None,
) -> Path:
    """Correct every selected subject, at most jobs at a time, each in a guarded process."""
    participants = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t")
    selected = participants[participants.split.isin(splits)].participant_id.tolist()
    if subjects:
        selected = [s for s in selected if s in subjects]
    _write_dataset_description(cfg)
    limit = cfg["guard"]["ram_limit_gb"]
    log.info("N4 on %d subjects (%d at a time, RAM limit %s GB each)", len(selected), jobs, limit)

    def one(sub: str) -> list[dict[str, Any]]:
        result = call_guarded(correct_subject, sub, config_path, ram_limit_gb=limit)
        if not result.ok:
            log.error("%s failed: %s", sub, result.killed or result.error)
            return [{"participant_id": sub, "image": "*", "status": f"failed: {result.killed or 'error'}"}]
        for row in result.value:
            log.info(
                "%s %-5s %s %s",
                sub,
                row["image"],
                row["status"],
                {k: v for k, v in row.items() if k not in ("participant_id", "image", "status")},
            )
        return result.value

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = [row for rows in pool.map(one, selected) for row in rows]

    report = resolve_path(cfg, "derivatives") / PIPELINE / "n4_report.csv"
    new = pd.DataFrame(rows)
    if report.is_file():  # keep previous rows of subjects not processed this time
        old = pd.read_csv(report)
        new = pd.concat([old[~old.participant_id.isin(new.participant_id)], new], ignore_index=True)
    new.sort_values(["participant_id", "image"]).to_csv(report, index=False)
    failed = new.status.str.startswith("failed").sum()
    log.info("Report %s: %d rows, %d failed", report, len(new), failed)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="N4 bias field correction (HD-BET brain mask) of the raw FLAIR and T1w."
    )
    parser.add_argument("--config", help="path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--split", choices=["training", "test", "all"], default="all")
    parser.add_argument("--subjects", nargs="*", help="only these participant ids (e.g. sub-000 sub-050)")
    parser.add_argument(
        "--jobs", type=int, default=None, help="parallel subjects (default: guard.max_parallel_jobs)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    splits = ("training", "test") if args.split == "all" else (args.split,)
    run(cfg, splits, args.jobs or cfg["guard"]["max_parallel_jobs"], args.subjects, args.config)


if __name__ == "__main__":
    main()
