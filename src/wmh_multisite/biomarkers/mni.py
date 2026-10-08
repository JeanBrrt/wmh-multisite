"""Lesion frequency maps in MNI space, per site and per source (ROADMAP step 7.5, J-054, J-056).

Where do the WMH lie, in a common space, for each site and each segmentation method?

    lesion mask of a source (FLAIR grid; label 2 of O1 removed, as everywhere)
      1. -> raw T1 grid: inverse of the rigid transform of 3.2
      2. -> MNI152NLin2009cAsym 2 mm: affine + SyN CC deformation of the night run (J-054)
         ``linear`` interpolation both times: a 2 mm voxel half covered by a lesion reads 0.5 (the
         covered fraction is kept instead of being rounded to 0 or 1)
      -> sub-XXX_space-MNI152NLin2009cAsym_res-2_label-WMH_desc-<source>_probseg.nii.gz
    frequency map = mean of these fractions over the subjects of a group (source x site, source x all)
    agreement of a method with O1 = voxel-wise Pearson correlation of the two maps inside the brain,
    on the same (test) subjects; difference map (method - O1) shows where a method over-/under-segments

Two resamplings in a row instead of one composed transform: each step reuses a convention already
checked in the project (``qc.iqm.labels_to_t1`` for the inverse rigid, notebook 06 for [warp, affine]),
at the cost of a little extra smoothing, harmless for a frequency map.

Sources: O1 for the 170 subjects (training and test); the models on the 110 test subjects; the experts
O3 and O4 on the 60 training subjects (human reference for the map agreement, compared with the O1 map
of the same 60 subjects).

Usage:
    uv run python -m wmh_multisite.biomarkers.mni
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wmh_multisite.biomarkers.table import prediction
from wmh_multisite.biomarkers.volumes import remove_other_pathology
from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.evaluate.interobserver import manual
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.utils import io

log = logging.getLogger(__name__)
SPACE = "MNI152NLin2009cAsym"


def registration_paths(cfg: dict[str, Any], sub: str) -> dict[str, Path]:
    m = cfg["biomarkers"]["mni"]
    d = resolve_path(cfg, "derivatives") / PIPELINE
    reg = d / m["registration_dir"] / f"{sub}__{m['variant']}"
    stem = f"{sub}_from-T1w_to-{SPACE}_mode-image"
    return {
        "t1": resolve_path(cfg, "bids") / sub / "anat" / f"{sub}_T1w.nii.gz",
        "rigid": d / sub / "anat" / f"{sub}_from-T1w_to-FLAIR_mode-image_xfm.mat",
        "affine": reg / f"{stem}_desc-affine_xfm.mat",
        "warp": reg / f"{stem}_desc-warp_xfm.nii.gz",
    }


def output_path(cfg: dict[str, Any], sub: str, source: str) -> Path:
    res = cfg["biomarkers"]["mni"]["resolution"]
    return (
        resolve_path(cfg, "derivatives")
        / PIPELINE
        / sub
        / "anat"
        / f"{sub}_space-{SPACE}_res-{res}_label-WMH_desc-{source}_probseg.nii.gz"
    )


def flair_to_mni(mask_flair: Path, wmh: np.ndarray, reg: dict[str, Path], template: Path) -> np.ndarray:
    """Fraction of each template voxel covered by the WMH (two linear resamplings)."""
    import ants

    moving = ants.image_read(str(mask_flair), pixeltype="float")
    moving = moving.new_image_like(wmh.astype(np.float32))
    on_t1 = ants.apply_transforms(
        ants.image_read(str(reg["t1"])),
        moving,
        [str(reg["rigid"])],
        whichtoinvert=[True],
        interpolator="linear",
    )
    on_mni = ants.apply_transforms(
        ants.image_read(str(template)), on_t1, [str(reg["warp"]), str(reg["affine"])], interpolator="linear"
    )
    return np.clip(on_mni.numpy(), 0, 1).astype(np.float32)


def subject_maps(task: dict[str, Any]) -> list[dict[str, Any]]:
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")
    import nibabel as nib

    cfg = load_config(task["config"])
    sub = task["sub"]
    reg = registration_paths(cfg, sub)
    ref = io.load_labels(task["sources"]["O1"])
    template_img = nib.load(task["template"])
    voxel_ml = io.voxel_volume_mm3(task["sources"]["O1"]) / 1000
    rows = []
    for source, path in task["sources"].items():
        wmh = remove_other_pathology(io.load_labels(path) == 1, ref)
        frac = flair_to_mni(Path(path), wmh, reg, Path(task["template"]))
        out = output_path(cfg, sub, source)
        nib.save(nib.Nifti1Image(frac, template_img.affine), out)
        rows.append(
            {
                "participant_id": sub,
                "source": source,
                "native_ml": float(wmh.sum()) * voxel_ml,
                "mni_ml": float(frac.sum()) * float(np.prod(template_img.header.get_zooms()[:3])) / 1000,
            }
        )
    return rows


def frequency_maps(cfg: dict[str, Any], table: pd.DataFrame) -> dict[tuple[str, str, str], np.ndarray]:
    """(source, split group, site) -> mean fraction map. Split group: 'training', 'test', or 'all' (O1)."""
    import nibabel as nib

    maps = {}
    for source, g in table.groupby("source"):
        groups = {split: g[g.split == split] for split in sorted(g.split.unique())}
        if source == "O1":
            groups["all"] = g
        for split_group, gs in groups.items():
            for site, gg in [("all", gs), *gs.groupby("scanner")]:
                stack = [
                    nib.load(output_path(cfg, s, source)).get_fdata(dtype=np.float32)
                    for s in gg.participant_id
                ]
                maps[(source, split_group, site)] = np.mean(stack, axis=0)
    return maps


def map_agreement(maps: dict, brain: np.ndarray) -> pd.DataFrame:
    """Voxel-wise Pearson correlation of each source's map with the O1 map of the same subjects, per site."""
    rows = []
    for (source, split_group, site), m in maps.items():
        if source == "O1" or split_group == "all":
            continue
        ref = maps[("O1", split_group, site)]
        rows.append(
            {
                "source": source,
                "split": split_group,
                "site": site,
                "pearson_r_vs_O1": float(np.corrcoef(m[brain], ref[brain])[0, 1]),
                "mean_freq": float(m[brain].mean()),
                "mean_freq_O1": float(ref[brain].mean()),
            }
        )
    return pd.DataFrame(rows)


def figure(
    arrays: list[np.ndarray], titles: list[str], template: np.ndarray, path: Path, diff: bool = False
) -> None:
    """Axial slices (through the lateral ventricles) of maps over the template."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zs = [int(template.shape[2] * f) for f in (0.5, 0.58, 0.66)]
    fig, axes = plt.subplots(len(arrays), len(zs), figsize=(3.2 * len(zs), 3.0 * len(arrays)), squeeze=False)
    vmax = 0.2 if diff else 0.4
    for row, (data, title) in enumerate(zip(arrays, titles, strict=True)):
        for col, z in enumerate(zs):
            ax = axes[row, col]
            ax.imshow(template[:, :, z].T, cmap="gray", origin="lower")
            if diff:
                im = ax.imshow(
                    data[:, :, z].T, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax, alpha=0.85
                )
            else:
                shown = np.ma.masked_where(data[:, :, z].T < 0.01, data[:, :, z].T)
                im = ax.imshow(shown, cmap="hot", origin="lower", vmin=0, vmax=vmax)
            ax.set_title(f"{title} (z={z})", fontsize=8)
            ax.axis("off")
    label = "difference of lesion frequency" if diff else "lesion frequency (share of subjects)"
    fig.colorbar(im, ax=axes, shrink=0.5, label=label)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lesion frequency maps in MNI space (7.5).")
    parser.add_argument("--config")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    from wmh_multisite.preproc import atlas

    m = cfg["biomarkers"]["mni"]
    cfg_tpl = {
        **cfg,
        "preproc": {**cfg["preproc"], "atlas": {**cfg["preproc"]["atlas"], "resolution": m["resolution"]}},
    }
    tpl = atlas.template_files(cfg_tpl)
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t").set_index("participant_id")
    tasks, skipped = [], []
    for sub, row in part.iterrows():
        reg = registration_paths(cfg, sub)
        if not (reg["warp"].is_file() and reg["affine"].is_file()):
            skipped.append(sub)
            continue
        sources = {"O1": manual(cfg, sub, "O1")}
        if row.split == "test":
            sources |= {s: prediction(cfg, sub, s) for s in m["models"] if prediction(cfg, sub, s).is_file()}
        else:  # experts of the training set (human reference)
            sources |= {o: manual(cfg, sub, o) for o in ("O3", "O4") if manual(cfg, sub, o).is_file()}
        tasks.append(
            {
                "sub": sub,
                "sources": {k: str(v) for k, v in sources.items()},
                "template": str(tpl["t1"]),
                "config": args.config,
            }
        )
    if skipped:
        log.warning("%d subjects without MNI registration: skipped (%s...)", len(skipped), skipped[:5])
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = [r for rows in pool.map(subject_maps, tasks) for r in rows]
    table = pd.DataFrame(rows)
    table.insert(1, "scanner", table.participant_id.map(part.scanner))
    table.insert(2, "split", table.participant_id.map(part.split))
    out_tables = resolve_path(cfg, "results") / "tables"
    table.to_csv(out_tables / "mni_lesion_volumes.csv", index=False, float_format="%.4g")
    log.info(
        "MNI / native volume ratio per source (median): %s",
        (table.mni_ml / table.native_ml).groupby(table.source).median().round(3).to_dict(),
    )

    import nibabel as nib

    template = nib.load(tpl["t1"]).get_fdata()
    brain = nib.load(tpl["brain"]).get_fdata() > 0.5
    maps = frequency_maps(cfg, table)
    agreement = map_agreement(maps, brain)
    agreement.to_csv(out_tables / "mni_map_agreement.csv", index=False, float_format="%.4g")
    log.info("map agreement with O1 (test):\n%s", agreement.round(3).to_string(index=False))
    maps_dir = resolve_path(cfg, "results") / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    for (source, split_group, site), arr in maps.items():
        nib.save(
            nib.Nifti1Image(arr, nib.load(tpl["t1"]).affine),
            maps_dir / f"freq_{source}_{split_group}_{site}.nii.gz",
        )
    figures = resolve_path(cfg, "results") / "figures"
    sites = sorted(part.scanner.unique())
    n_o1 = int((table.source == "O1").sum())
    figure(
        [maps[("O1", "all", "all")]] + [maps[("O1", "all", s)] for s in sites],
        [f"O1, all sites (n={n_o1})"] + [f"O1, {s}" for s in sites],
        template,
        figures / "mni_frequency_O1_by_site.png",
    )
    methods = [s for s in m["models"] if (s, "test", "all") in maps]
    figure(
        [maps[("O1", "test", "all")]] + [maps[(s, "test", "all")] for s in methods],
        ["O1 (test)"] + [f"{s} (test)" for s in methods],
        template,
        figures / "mni_frequency_methods_test.png",
    )
    figure(
        [maps[(s, "test", "all")] - maps[("O1", "test", "all")] for s in methods],
        [f"{s} - O1 (test)" for s in methods],
        template,
        figures / "mni_frequency_methods_minus_O1.png",
        diff=True,
    )
    log.info("maps in %s, figures in %s", maps_dir, figures)


if __name__ == "__main__":
    main()
