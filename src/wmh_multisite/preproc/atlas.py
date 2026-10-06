"""Registration to the MNI atlas and lateral ventricles of each subject (ROADMAP step 3.6, J-041).

Why: step 7.2 splits the WMH into *periventricular* and *deep* lesions by their distance to the
lateral ventricles, and step 7.5 maps all lesions into a common space. Both need each brain registered
to an atlas.

    TemplateFlow MNI152NLin2009cAsym (1 mm):  T1 template + brain mask + Harvard-Oxford labels
              ^
              |  A. SyN (rigid + affine + non-linear), brain to brain: subject T1 N4 -> template T1
              |
    subject T1 N4 (1 mm) ----- rigid (3.2) -----> subject FLAIR grid

    B. atlas ventricles -> subject T1 grid (inverse of A) -> FLAIR grid (rigid of 3.2), genericLabel
    C. refinement on the FLAIR: dark voxels (subject-specific threshold) connected to the atlas
       ventricles, within a band around them (the atlas under-estimates enlarged, atrophic ventricles)

A. **Brain to brain.** The subject T1 (N4, HD-BET mask) is registered to the template T1 restricted to
   the template brain mask: skull, scalp and neck vary between people and are not in the atlas.
   ANTs ``SyN`` chains rigid, affine and a smooth invertible deformation (Mattes mutual information,
   update field smoothed by a Gaussian of 3 voxels). One thread and a fixed seed (J-031).
B. **Atlas labels to the subject**, in two resamplings that use ANTs' documented conventions instead
   of a hand-made composition: ``invtransforms`` (template -> subject T1 grid), then the rigid
   transform (T1 grid -> FLAIR grid, ``register.resample_onto_flair``). Labels are always resampled
   with ``genericLabel`` (never linear).
C. **Refinement.** Threshold specific to each subject, learnt on its own FLAIR, inside the brain
   (J-028, rule A): halfway between the median of the atlas-ventricle *core* (eroded by ``core_mm``:
   certainly CSF) and the median of the brain tissue far from the ventricles (``tissue_min_mm``; the
   median is robust to the few bright lesions, and no manual label is used, J-013). Dark voxels within
   ``zone_mm`` of the atlas ventricles that are connected to them form the refined ventricles; the band
   prevents a leak into the sulcal CSF.

Outputs, in ``data/derivatives/wmh-multisite/sub-XXX/anat/``:
    sub-XXX_from-T1w_to-MNI152NLin2009cAsym_mode-image_desc-affine_xfm.mat
    sub-XXX_from-T1w_to-MNI152NLin2009cAsym_mode-image_desc-warp_xfm.nii.gz      (and desc-invwarp)
    sub-XXX_space-T1w_desc-ventriclesatlas_mask.nii.gz       atlas ventricles, T1 grid
    sub-XXX_space-FLAIR_desc-ventriclesatlas_mask.nii.gz     atlas ventricles, FLAIR grid
    sub-XXX_space-FLAIR_desc-ventricles_mask.nii.gz          refined ventricles, FLAIR grid (used in 7.2)
and ``results/tables/atlas_qc.csv``.

Usage:
    uv run python -m wmh_multisite.preproc.atlas --subjects sub-000
    uv run python -m wmh_multisite.preproc.atlas --split all --jobs 4
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import ants
import numpy as np
import pandas as pd
from scipy import ndimage

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.preproc.register import remove_registration_files, resample_onto_flair
from wmh_multisite.utils import io
from wmh_multisite.utils.guard import call_guarded, wait_for_free_ram

log = logging.getLogger(__name__)


def template_files(cfg: dict[str, Any]) -> dict[str, Path]:
    """Template T1, brain mask and ventricle atlas from TemplateFlow (downloaded once, then cached)."""
    os.environ["TEMPLATEFLOW_HOME"] = str(resolve_path(cfg, "templateflow"))
    import templateflow.api as tf  # imported after TEMPLATEFLOW_HOME is set: it reads it at import

    a = cfg["preproc"]["atlas"]
    res = a["resolution"]
    return {
        "t1": Path(tf.get(a["template"], resolution=res, desc=None, suffix="T1w", extension=".nii.gz")),
        "brain": Path(
            tf.get(a["template"], resolution=res, desc="brain", suffix="mask", extension=".nii.gz")
        ),
        "labels": Path(
            tf.get(
                a["template"],
                resolution=res,
                atlas=a["ventricle_atlas"]["atlas"],
                desc=a["ventricle_atlas"]["desc"],
                suffix="dseg",
                extension=".nii.gz",
            )
        ),
    }


def subject_paths(cfg: dict[str, Any], sub: str) -> dict[str, Path]:
    d = resolve_path(cfg, "derivatives") / PIPELINE / sub / "anat"
    tpl = cfg["preproc"]["atlas"]["template"]
    stem = f"{sub}_from-T1w_to-{tpl}_mode-image"
    return {
        "t1_n4": d / f"{sub}_desc-n4_T1w.nii.gz",
        "t1_brain": d / f"{sub}_space-T1w_desc-brain_mask.nii.gz",
        "flair": resolve_path(cfg, "bids") / sub / "anat" / f"{sub}_FLAIR.nii.gz",
        "flair_n4": d / f"{sub}_desc-n4_FLAIR.nii.gz",
        "flair_brain": d / f"{sub}_space-FLAIR_desc-brain_mask.nii.gz",
        "rigid": d / f"{sub}_from-T1w_to-FLAIR_mode-image_xfm.mat",
        "affine": d / f"{stem}_desc-affine_xfm.mat",
        "warp": d / f"{stem}_desc-warp_xfm.nii.gz",
        "invwarp": d / f"{stem}_desc-invwarp_xfm.nii.gz",
        "vent_t1": d / f"{sub}_space-T1w_desc-ventriclesatlas_mask.nii.gz",
        "vent_atlas": d / f"{sub}_space-FLAIR_desc-ventriclesatlas_mask.nii.gz",
        "vent": d / f"{sub}_space-FLAIR_desc-ventricles_mask.nii.gz",
    }


def masked(image: ants.ANTsImage, mask: ants.ANTsImage) -> ants.ANTsImage:
    return image * (mask > 0.5)


def register_to_template(cfg: dict[str, Any], p: dict[str, Path], tpl: dict[str, Path]) -> float:
    """Step A. Writes the affine and the two warp fields; returns the brain correlation with the template."""
    a = cfg["preproc"]["atlas"]
    fixed = masked(ants.image_read(str(tpl["t1"]), pixeltype="float"), ants.image_read(str(tpl["brain"])))
    moving = masked(ants.image_read(str(p["t1_n4"]), pixeltype="float"), ants.image_read(str(p["t1_brain"])))
    ants.config._random_seed = a["seed"]
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform=a["type_of_transform"])
    # fwdtransforms = [warp, affine]; invtransforms = [affine, inverse warp] (ANTsPy conventions)
    shutil.copyfile(reg["fwdtransforms"][1], p["affine"])
    shutil.copyfile(reg["fwdtransforms"][0], p["warp"])
    shutil.copyfile(reg["invtransforms"][1], p["invwarp"])
    remove_registration_files(reg)
    warped = reg["warpedmovout"].numpy()
    brain = ants.image_read(str(tpl["brain"])).numpy() > 0.5
    return float(np.corrcoef(warped[brain], fixed.numpy()[brain])[0, 1])


def ventricles_to_flair(cfg: dict[str, Any], p: dict[str, Path], tpl: dict[str, Path]) -> None:
    """Step B. Atlas ventricles -> subject T1 grid (inverse of A) -> FLAIR grid (rigid of 3.2)."""
    labels = ants.image_read(str(tpl["labels"]))
    vent = labels.new_image_like(
        np.isin(labels.numpy(), cfg["preproc"]["atlas"]["ventricle_labels"]).astype("float32")
    )
    t1 = ants.image_read(str(p["t1_n4"]))
    on_t1 = ants.apply_transforms(
        fixed=t1,
        moving=vent,
        transformlist=[str(p["affine"]), str(p["invwarp"])],
        whichtoinvert=[True, False],
        interpolator="genericLabel",
    )
    io.save_like(on_t1.numpy() > 0.5, p["t1_n4"], p["vent_t1"])
    resample_onto_flair(p["flair"], p["vent_t1"], p["rigid"], p["vent_atlas"], labels=True)


def refine_ventricles(cfg: dict[str, Any], p: dict[str, Path]) -> dict[str, float]:
    """Step C. Subject-specific CSF threshold, dark voxels connected to the atlas ventricles."""
    a = cfg["preproc"]["atlas"]
    flair = io.load_data(p["flair_n4"])
    brain = io.load_labels(p["flair_brain"], allowed=(0, 1)).astype(bool)
    atlas = io.load_labels(p["vent_atlas"], allowed=(0, 1)).astype(bool) & brain
    spacing = io.describe(p["flair_n4"])["voxel_size_mm"]
    inside = ndimage.distance_transform_edt(atlas, sampling=spacing)  # mm to the atlas border, inside
    outside = ndimage.distance_transform_edt(~atlas, sampling=spacing)  # mm to the atlas ventricles
    core = atlas & (inside > a["core_mm"])
    if core.sum() < 50:  # very thin atlas ventricles: fall back to the whole atlas mask
        core = atlas
    tissue = brain & (outside > a["tissue_min_mm"])
    csf, tis = float(np.median(flair[core])), float(np.median(flair[tissue]))
    threshold = (csf + tis) / 2
    zone = brain & (outside <= a["zone_mm"])
    dark = zone & (flair < threshold)
    lab, _ = ndimage.label(dark)
    keep = np.unique(lab[atlas & dark])
    vent = np.isin(lab, keep[keep > 0])
    vent = ndimage.binary_fill_holes(vent) & brain  # choroid plexus: brighter spots inside the ventricles
    io.save_like(vent, p["flair_n4"], p["vent"])
    ml = float(np.prod(spacing)) / 1000
    return {
        "csf_median": round(csf, 1),
        "tissue_median": round(tis, 1),
        "threshold": round(threshold, 1),
        "contrast_csf_tissue": round(csf / tis, 3) if tis else None,
        "vent_atlas_ml": round(float(atlas.sum() * ml), 1),
        "vent_ml": round(float(vent.sum() * ml), 1),
        "dice_atlas_refined": round(float(2 * (atlas & vent).sum() / max(atlas.sum() + vent.sum(), 1)), 3),
        "vent_flair_median_rel": round(float(np.median(flair[vent]) / tis), 3)
        if vent.any() and tis
        else None,
    }


def process_subject(sub: str, config_path: str | None = None) -> dict[str, Any]:
    """A + B + C for one subject; skips the parts already done. Runs in a guarded child process."""
    cfg = load_config(config_path)
    p, tpl = subject_paths(cfg, sub), template_files(cfg)
    missing = [k for k in ("t1_n4", "t1_brain", "flair_n4", "flair_brain", "rigid") if not p[k].is_file()]
    if missing:
        raise FileNotFoundError(f"{sub}: missing {missing} (run register, brainmask and bias first)")
    t0, corr = time.time(), None
    if not all(p[k].is_file() for k in ("affine", "warp", "invwarp")):
        corr = register_to_template(cfg, p, tpl)
    seconds = round(time.time() - t0, 1)
    if not p["vent_atlas"].is_file():
        ventricles_to_flair(cfg, p, tpl)
    stats = refine_ventricles(cfg, p)
    return {
        "participant_id": sub,
        "syn_seconds": seconds if corr is not None else None,
        "syn_corr_brain": round(corr, 4) if corr is not None else None,
        **stats,
    }


def run(cfg: dict[str, Any], subjects: list[str], jobs: int, config_path: str | None = None) -> Path:
    template_files(cfg)  # download once, before the parallel workers
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(cfg["preproc"]["atlas"]["threads"])
    a = cfg["preproc"]["atlas"]
    limit, floor = a["ram_limit_gb"], cfg["guard"]["min_free_ram_gb"]
    log.info(
        "Atlas registration on %d subjects (%d at a time, RAM limit %s GB each, start only if >= %s GB free)",
        len(subjects),
        jobs,
        limit,
        floor,
    )

    def one(sub: str) -> dict[str, Any]:
        wait_for_free_ram(floor, log=lambda m: log.info("%s %s", sub, m))
        r = call_guarded(process_subject, sub, config_path, ram_limit_gb=limit)
        if not r.ok:
            log.error("%s failed: %s", sub, r.killed or r.error)
            return {"participant_id": sub, "error": str(r.killed or r.error)[-300:]}
        log.info("%s %s", sub, {k: v for k, v in r.value.items() if k != "participant_id"})
        return r.value | {"peak_ram_gb": r.peak_ram_gb}

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(one, subjects))
    out = resolve_path(cfg, "results") / "tables" / "atlas_qc.csv"
    new = pd.DataFrame(rows)
    if out.is_file():  # keep the SyN figures of a previous run when only step C was redone
        old = pd.read_csv(out).set_index("participant_id")
        new = new.set_index("participant_id")
        for col in ("syn_seconds", "syn_corr_brain"):
            if col in old and col in new:
                new[col] = new[col].fillna(old[col].reindex(new.index))
        new = pd.concat([old[~old.index.isin(new.index)], new]).reset_index()
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t").set_index("participant_id")
    new["scanner"] = new.participant_id.map(part.scanner)
    new.sort_values("participant_id").to_csv(out, index=False)
    failed = int(new["error"].notna().sum()) if "error" in new else 0
    log.info("QC %s: %d subjects, %d failed", out, len(new), failed)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SyN registration to the MNI atlas and lateral ventricles.")
    parser.add_argument("--config")
    parser.add_argument("--split", choices=["training", "test", "all"], default="all")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t")
    splits = ("training", "test") if args.split == "all" else (args.split,)
    subjects = part[part.split.isin(splits)].participant_id.tolist()
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]
    run(cfg, subjects, args.jobs or cfg["preproc"]["atlas"]["jobs"], args.config)


if __name__ == "__main__":
    main()
