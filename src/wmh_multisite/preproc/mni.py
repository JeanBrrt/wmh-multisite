"""Registration of each subject to the 2 mm MNI template, used for the lesion frequency maps.

The N4-corrected T1, restricted to the brain, is registered with SyN to the brain of the MNI152NLin2009cAsym
template at 2 mm. Working brain to brain avoids the skull, scalp and neck, which vary between people and are
not in the template.

We use cross-correlation rather than the default Mattes mutual information because it follows the enlarged
ventricles of atrophic subjects, which the default setting left inside the template contour. It is slower, but
the 2 mm template (8 times fewer voxels) brings it back to about 6 minutes per subject for a nearly identical
result. The comparison of settings on a few subjects is kept in results/tables/mni_syn_settings.csv.

With several ITK threads per subject the result is not reproducible to the last digit of the deformation. The
maps are averaged over tens of subjects, so this does not show in the results.

Outputs, in data/derivatives/wmh-multisite/sub-XXX/anat/:
    sub-XXX_from-T1w_to-MNI152NLin2009cAsym_res-2_mode-image_desc-affine_xfm.mat
    sub-XXX_from-T1w_to-MNI152NLin2009cAsym_res-2_mode-image_desc-warp_xfm.nii.gz
and results/tables/mni_registration_qc.csv (time, memory and brain correlation with the template). Subjects
that already have both transforms are skipped.

Usage:
    uv run python -m wmh_multisite.preproc.mni --subjects sub-000
    uv run python -m wmh_multisite.preproc.mni
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.utils.guard import call_guarded, wait_for_free_ram

log = logging.getLogger(__name__)


def template_files(cfg: dict[str, Any]) -> dict[str, Path]:
    """Template T1 and brain mask from TemplateFlow, downloaded once and then cached."""
    os.environ["TEMPLATEFLOW_HOME"] = str(resolve_path(cfg, "templateflow"))
    import templateflow.api as tf  # imported after TEMPLATEFLOW_HOME is set: it reads it at import

    m = cfg["preproc"]["mni"]
    get = {"resolution": m["resolution"], "extension": ".nii.gz"}
    return {
        "t1": Path(tf.get(m["template"], desc=None, suffix="T1w", **get)),
        "brain": Path(tf.get(m["template"], desc="brain", suffix="mask", **get)),
    }


def subject_t1(cfg: dict[str, Any], sub: str) -> dict[str, Path]:
    """Paths of the N4-corrected T1 and of its brain mask."""
    d = resolve_path(cfg, "derivatives") / PIPELINE / sub / "anat"
    return {
        "t1_n4": d / f"{sub}_desc-n4_T1w.nii.gz",
        "t1_brain": d / f"{sub}_space-T1w_desc-brain_mask.nii.gz",
    }


def registration_paths(cfg: dict[str, Any], sub: str) -> dict[str, Path]:
    m = cfg["preproc"]["mni"]
    d = resolve_path(cfg, "derivatives") / PIPELINE / sub / "anat"
    stem = f"{sub}_from-T1w_to-{m['template']}_res-{m['resolution']}_mode-image"
    return {"affine": d / f"{stem}_desc-affine_xfm.mat", "warp": d / f"{stem}_desc-warp_xfm.nii.gz"}


def done(cfg: dict[str, Any], sub: str) -> bool:
    return all(p.is_file() for p in registration_paths(cfg, sub).values())


def register_subject(sub: str, config_path: str | None = None) -> dict[str, Any]:
    """Register one subject. Runs in a guarded child process."""
    import ants
    import numpy as np

    from wmh_multisite.preproc.register import remove_registration_files

    cfg = load_config(config_path)
    m = cfg["preproc"]["mni"]
    tpl = template_files(cfg)
    p = subject_t1(cfg, sub)
    out = registration_paths(cfg, sub)
    missing = [k for k in ("t1_n4", "t1_brain") if not p[k].is_file()]
    if missing:
        raise FileNotFoundError(f"{sub}: missing {missing} (run brainmask and bias first)")
    fixed = ants.image_read(str(tpl["t1"]), pixeltype="float") * (ants.image_read(str(tpl["brain"])) > 0.5)
    moving = ants.image_read(str(p["t1_n4"]), pixeltype="float") * (ants.image_read(str(p["t1_brain"])) > 0.5)
    ants.config._random_seed = m["seed"]
    t0 = time.time()
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN",
        reg_iterations=tuple(m["reg_iterations"]),
        syn_metric=m["syn_metric"],
        syn_sampling=m["syn_sampling"],
    )
    seconds = time.time() - t0
    out["affine"].parent.mkdir(parents=True, exist_ok=True)
    # fwdtransforms = [warp, affine] (ANTsPy conventions)
    shutil.copyfile(reg["fwdtransforms"][1], out["affine"])
    shutil.copyfile(reg["fwdtransforms"][0], out["warp"])
    remove_registration_files(reg)
    brain = ants.image_read(str(tpl["brain"])).numpy() > 0.5
    warped = reg["warpedmovout"].numpy()
    corr = float(np.corrcoef(warped[brain], fixed.numpy()[brain])[0, 1])
    return {"participant_id": sub, "syn_seconds": round(seconds, 1), "syn_corr_brain": round(corr, 4)}


def interleaved(part: pd.DataFrame) -> list[str]:
    """Subjects ordered with the scanners alternated, so that a run stopped early still covers every site."""
    by_scanner = [g.participant_id.tolist() for _, g in part.groupby("scanner")]
    return [s for group in itertools.zip_longest(*by_scanner) for s in group if s]


def run(cfg: dict[str, Any], subjects: list[str], config_path: str | None = None) -> Path:
    m = cfg["preproc"]["mni"]
    template_files(cfg)  # download once, before the parallel workers
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(m["threads"])
    limit, floor = cfg["guard"]["ram_limit_gb"], cfg["guard"]["min_free_ram_gb"]
    todo = [s for s in subjects if not done(cfg, s)]
    log.info(
        "MNI registration: %d subjects to do (%d already done), %d at a time, %d ITK threads each",
        len(todo),
        len(subjects) - len(todo),
        m["jobs"],
        m["threads"],
    )

    def one(sub: str) -> dict[str, Any]:
        wait_for_free_ram(floor, log=lambda msg: log.info("%s %s", sub, msg))
        r = call_guarded(register_subject, sub, config_path, ram_limit_gb=limit)
        if not r.ok:
            log.error("%s failed: %s", sub, r.killed or r.error)
            return {"participant_id": sub, "error": str(r.killed or r.error)[-300:]}
        log.info("%s %s", sub, {k: v for k, v in r.value.items() if k != "participant_id"})
        return r.value | {"peak_ram_gb": r.peak_ram_gb}

    with ThreadPoolExecutor(max_workers=m["jobs"]) as pool:
        rows = list(pool.map(one, todo))
    out = resolve_path(cfg, "results") / "tables" / "mni_registration_qc.csv"
    new = pd.DataFrame(
        rows, columns=["participant_id", "syn_seconds", "peak_ram_gb", "syn_corr_brain", "error"]
    )
    if out.is_file():  # keep the rows of the subjects registered in a previous run
        old = pd.read_csv(out)
        new = pd.concat([old[~old.participant_id.isin(new.participant_id)], new])
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t").set_index("participant_id")
    new["scanner"] = new.participant_id.map(part.scanner)
    new.sort_values("participant_id").to_csv(out, index=False)
    failed = int(new["error"].notna().sum())
    log.info("QC %s: %d subjects, %d failed", out, len(new), failed)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="SyN registration to the 2 mm MNI template.")
    parser.add_argument("--config")
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t")
    subjects = interleaved(part)
    if args.subjects:
        subjects = [s for s in subjects if s in args.subjects]
    run(cfg, subjects, args.config)


if __name__ == "__main__":
    main()
