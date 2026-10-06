"""Rigid registration of the raw 3D T1 to the raw FLAIR (ROADMAP step 3.2).

Pipeline order (J-036): HD-BET brain mask on the raw T1 (3.3) -> **this step** -> brain mask carried
onto the FLAIR grid (3.3) -> N4 with the brain masks, then N4 T1 resampled onto the FLAIR grid (3.1).
Registering the *raw* images makes the transform independent of the bias correction, so that N4 can
use a brain mask defined on the FLAIR grid. Mutual information is robust to the smooth bias field
(raw images registered at 0.98-0.99 against elastix in the feasibility test, J-016).

The 3D T1 (moving, 1 mm) is registered to the FLAIR (fixed) with a rigid transform (3 rotations +
3 translations: same subject, same session, the head only moved) and Mattes mutual information (the
two contrasts differ). The FLAIR is never resampled (it is the evaluation grid, J-010).

Quality control against the organizers' elastix registration (J-016, J-031):
- the transform is applied to the raw T1 and correlated with ``orig/T1`` (raw T1 aligned by elastix);
- the displacement between our transform and elastix's is measured over the head (mean and max, mm),
  together with the angle of the relative rotation.

Documented exception: subjects listed in ``preproc.rigid.use_elastix`` (sub-146, J-039) use the
organizers' elastix transform; the ANTs transform is kept as ``..._desc-ants_xfm.mat``.

Output, in ``data/derivatives/wmh-multisite/sub-XXX/anat/``:
    sub-XXX_from-T1w_to-FLAIR_mode-image_xfm.mat    rigid transform (ITK format)

Usage:
    uv run python -m wmh_multisite.preproc.register --split training --jobs 4
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import ants
import numpy as np
import pandas as pd
import SimpleITK as sitk

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.preproc.bias import head_mask
from wmh_multisite.utils import io
from wmh_multisite.utils.guard import call_guarded

log = logging.getLogger(__name__)


def remove_registration_files(reg: dict[str, Any]) -> int:
    """Delete the transform files ``ants.registration`` left in the temporary folder (after copying them).

    ANTsPy writes every transform to the system temporary folder and never removes it: 807 files (5.4 GB)
    had accumulated after the registrations of 3.2, 3.6 and 7.5 and filled the disk (2026-10-06). Only files
    located in the temporary folder are removed.
    """
    import tempfile

    tmp = Path(tempfile.gettempdir()).resolve()
    removed = 0
    for f in {*reg.get("fwdtransforms", []), *reg.get("invtransforms", [])}:
        f = Path(f)
        if f.is_file() and f.resolve().parent == tmp:
            f.unlink()
            removed += 1
    return removed


def subject_paths(cfg: dict[str, Any], sub: str) -> dict[str, Path]:
    bids = resolve_path(cfg, "bids")
    out = resolve_path(cfg, "derivatives") / PIPELINE / sub / "anat"
    challenge = bids / "derivatives" / "challenge" / sub / "anat"
    return {
        "flair_raw": bids / sub / "anat" / f"{sub}_FLAIR.nii.gz",
        "t1_raw": bids / sub / "anat" / f"{sub}_T1w.nii.gz",
        "t1_elastix": challenge / f"{sub}_space-FLAIR_T1w.nii.gz",
        "elastix_params": challenge / f"{sub}_from-T1w_to-FLAIR_mode-image_xfm.txt",
        "xfm": out / f"{sub}_from-T1w_to-FLAIR_mode-image_xfm.mat",  # output
    }


def resample_onto_flair(flair: Path, moving: Path, xfm: Path, out: Path, labels: bool = False) -> Path:
    """Resample a T1-grid image onto the FLAIR grid with the rigid transform.

    Images: linear interpolation (as the organizers, J-031). Masks: ``genericLabel`` (never linear for
    labels, notebook 01). ANTs and nibabel arrays share the voxel order of the file, so the result is
    saved with the FLAIR geometry by ``io.save_like``.
    """
    fixed = ants.image_read(str(flair))
    result = ants.apply_transforms(
        fixed,
        ants.image_read(str(moving), pixeltype="float"),
        [str(xfm)],
        interpolator="genericLabel" if labels else "linear",
    ).numpy()
    if labels:
        io.save_like(result > 0.5, flair, out)
    else:
        io.save_like(result, flair, out, dtype=np.float32)
    return out


def elastix_euler(param_file: Path) -> sitk.Euler3DTransform:
    """Rebuild the organizers' rigid transform from its elastix parameter file (notebook 02, section 4).

    Like an ANTs forward transform, it maps a point of the FLAIR (fixed) space to the T1 (moving)
    space, in LPS millimeters.
    """
    text = param_file.read_text()

    def values(name: str) -> list[float]:
        match = re.search(rf"\({name} ([^)]*)\)", text)
        if match is None:
            raise ValueError(f"{name} not found in {param_file}")
        return [float(v) for v in match.group(1).split()]

    params = values("TransformParameters")
    euler = sitk.Euler3DTransform()
    euler.SetCenter(values("CenterOfRotationPoint"))
    euler.SetRotation(*params[:3])
    euler.SetTranslation(params[3:])
    return euler


def compare_transforms(
    ours: sitk.Transform, reference: sitk.Transform, points: np.ndarray
) -> dict[str, float]:
    """Displacement between two transforms over physical points (LPS mm), and their relative rotation.

    Comparing parameters directly would be misleading: a rotation and a translation only mean
    something with respect to a center, and the two tools do not use the same center. Mapping the
    same points with both transforms gives a center-independent answer, in millimeters.
    """
    diff = np.array(
        [np.subtract(ours.TransformPoint(tuple(p)), reference.TransformPoint(tuple(p))) for p in points]
    )
    dist = np.linalg.norm(diff, axis=1)
    rel = np.reshape(ours.GetMatrix(), (3, 3)) @ np.reshape(reference.GetMatrix(), (3, 3)).T
    angle = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    return {
        "disp_mean_mm": round(float(dist.mean()), 3),
        "disp_max_mm": round(float(dist.max()), 3),
        "rot_diff_deg": round(float(angle), 3),
    }


def head_points(image: ants.ANTsImage, mask: np.ndarray, n: int = 5000, seed: int = 0) -> np.ndarray:
    """Physical (LPS) coordinates of ``n`` random voxels of ``mask`` in ``image`` (ANTs voxel order)."""
    idx = np.argwhere(mask)
    idx = idx[np.random.default_rng(seed).choice(len(idx), size=min(n, len(idx)), replace=False)]
    direction = np.asarray(image.direction)
    return np.asarray(image.origin) + (idx * np.asarray(image.spacing)) @ direction.T


def register_subject(sub: str, config_path: str | None = None) -> dict[str, Any]:
    """Register one subject (raw T1 -> raw FLAIR); skip it if the transform exists. Returns one QC row.

    Kept at module level with plain arguments so that it can run in a guarded child process.
    """
    cfg = load_config(config_path)
    rigid = cfg["preproc"]["rigid"]
    p = subject_paths(cfg, sub)
    p["xfm"].parent.mkdir(parents=True, exist_ok=True)
    fixed = ants.image_read(str(p["flair_raw"]), pixeltype="float")
    moving = ants.image_read(str(p["t1_raw"]), pixeltype="float")
    t0 = time.time()
    if sub in rigid.get("use_elastix", []):
        # Documented exception (J-039): ANTs is unreliable on this subject's atypical FLAIR; use the
        # organizers' elastix transform, converted to an ITK file that ANTs applies like its own.
        if not p["xfm"].is_file() or sitk.ReadTransform(str(p["xfm"])).GetName() != "Euler3DTransform":
            if p["xfm"].is_file():
                ants_copy = p["xfm"].with_name(p["xfm"].name.replace("_xfm.mat", "_desc-ants_xfm.mat"))
                shutil.copyfile(p["xfm"], ants_copy)
            sitk.WriteTransform(elastix_euler(p["elastix_params"]), str(p["xfm"]))
            status = "elastix"
        else:
            status = "present"
    elif p["xfm"].is_file():
        status = "present"
    else:
        # The metric samples 20 % of the voxels at random: fix ANTs' seed for reproducibility.
        ants.config._random_seed = rigid["seed"]
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=rigid["type_of_transform"])
        shutil.copyfile(reg["fwdtransforms"][0], p["xfm"])  # ANTs writes it in a temporary folder
        remove_registration_files(reg)
        status = "registered"
    seconds = round(time.time() - t0, 1)
    transforms = [str(p["xfm"])]

    # QC 1: our transform applied to the raw T1, compared with the raw T1 aligned by elastix.
    raw_on_flair = ants.apply_transforms(fixed, moving, transforms, interpolator="linear").numpy()
    reference = io.load_data(p["t1_elastix"])
    inside = (reference > np.percentile(reference, 50)) & (raw_on_flair > 0)
    corr = float(np.corrcoef(raw_on_flair[inside], reference[inside])[0, 1])

    # QC 2: geometric difference with elastix over the head (rough Otsu mask of the FLAIR).
    ours = sitk.ReadTransform(str(p["xfm"])).Downcast()
    geometry = compare_transforms(
        ours, elastix_euler(p["elastix_params"]), head_points(fixed, head_mask(fixed.numpy()))
    )

    flagged = (
        corr < rigid["qc_min_corr_vs_elastix"]
        or geometry["rot_diff_deg"] > rigid["qc_max_rotation_deg"]
        or geometry["disp_max_mm"] > rigid["qc_max_translation_mm"]
    )
    if status == "elastix" or sub in rigid.get("use_elastix", []):
        flagged = False  # the transform is elastix itself: the comparison is trivially perfect
    return {
        "participant_id": sub,
        "status": status,
        "seconds": seconds if status == "registered" else None,
        "corr_vs_elastix": round(corr, 4),
        **geometry,
        "flagged": bool(flagged),
    }


def run(
    cfg: dict[str, Any],
    splits: tuple[str, ...],
    jobs: int,
    subjects: list[str] | None = None,
    config_path: str | None = None,
) -> Path:
    """Register every selected subject, at most ``jobs`` at a time, each in a guarded process."""
    participants = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t")
    selected = participants[participants.split.isin(splits)].participant_id.tolist()
    if subjects:
        selected = [s for s in selected if s in subjects]
    limit = cfg["guard"]["ram_limit_gb"]
    # ITK sums the metric over threads in a varying order: results differ slightly from run to run
    # unless a single thread is used. Set before the child processes start; they inherit it.
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(cfg["preproc"]["rigid"]["threads"])
    log.info(
        "Rigid T1 -> FLAIR on %d subjects (%d at a time, RAM limit %s GB each)", len(selected), jobs, limit
    )

    def one(sub: str) -> dict[str, Any]:
        result = call_guarded(register_subject, sub, config_path, ram_limit_gb=limit)
        if not result.ok:
            log.error("%s failed: %s", sub, result.killed or result.error)
            return {"participant_id": sub, "status": f"failed: {result.killed or 'error'}"}
        row = result.value
        log.info(
            "%s %s %s",
            sub,
            "FLAGGED" if row["flagged"] else "ok",
            {k: v for k, v in row.items() if k not in ("participant_id", "flagged")},
        )
        return row

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(one, selected))

    report = resolve_path(cfg, "derivatives") / PIPELINE / "registration_report.csv"
    new = pd.DataFrame(rows)
    if report.is_file():  # keep previous rows of subjects not processed this time
        old = pd.read_csv(report)
        new = pd.concat([old[~old.participant_id.isin(new.participant_id)], new], ignore_index=True)
    new = new.sort_values("participant_id")
    new.to_csv(report, index=False)
    failed = int(new.status.str.startswith("failed").sum())
    flagged = new.loc[
        new.get("flagged", pd.Series(False, index=new.index)).fillna(False).astype(bool), "participant_id"
    ].tolist()
    log.info("Report %s: %d subjects, %d failed, flagged: %s", report, len(new), failed, flagged or "none")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rigid registration of the raw T1w to the raw FLAIR.")
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
