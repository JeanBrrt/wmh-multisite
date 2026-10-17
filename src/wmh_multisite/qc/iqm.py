"""Image quality metrics (IQMs) of every FLAIR and T1, following MRIQC.

A degraded image (motion, noise, strong bias, ghosting) can silently give a wrong biomarker, so every image is
measured before the analysis. The formulas are those of MRIQC (Esteban et al., 2017) for its main metrics. As
in MRIQC, the bias is measured on the N4 field only and the noise and contrast on the image after N4, so the
bias is not counted twice. The corrected image is scaled so that the white matter median is 1000 and rounded
to integers before the tissue statistics, as MRIQC does.

The metrics, with med, sd and mad the median, standard deviation and MAD of a region:

    snr_wm     med_WM / (sd_WM * sqrt(n_WM / (n_WM - 1)))                   higher is better
    snrd_wm    0.6551364 * med_WM / mad_air (Dietrich et al., 2007)          higher is better
    cnr        |med_WM - med_GM| / sqrt(sd_air^2 + sd_WM^2 + sd_GM^2)        higher is better
    cjv        (mad_WM + mad_GM) / |med_WM - med_GM|                         lower is better (T1 only)
    efc        entropy of the voxel intensities, normalised                  lower is better
    fber       median(x^2 in the head) / median(x^2 in the air)              higher is better
    inu_range  |p95 - p5| of the N4 field in the brain                       lower is better
    fwhm       in-plane smoothness of the raw image, as AFNI 3dFWHMx         lower is better

Our choices differ from MRIQC in a few places. The tissues come from WMH-SynthSeg instead of Atropos, and the
WMH plus a margin of qc.wmh_margin_mm are removed from the white matter, otherwise the SNR followed the lesion
load. The grey matter is the cortex only and is not eroded, since it is only 2 to 3 mm thick. Tissue
statistics are taken inside the HD-BET brain. The air is everything outside the dilated head except exact
zeros (defacing), because our FLAIRs are only about 144 mm high and do not contain MRIQC's region above the
head. The MRIQC classifier is not used, as it was trained on T1 images only.

The T1 stays on its own grid, since resampling would smooth its noise and inflate its SNR. The WMH-SynthSeg
labels are carried to it with the inverse rigid transform.

Output: results/tables/qc_iqm.csv, one row per subject and modality. Subjects without a WMH-SynthSeg
segmentation are skipped and logged.

Usage:
    uv run python -m wmh_multisite.qc.iqm
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
from scipy import ndimage

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.preproc import PIPELINE
from wmh_multisite.preproc.bias import head_mask
from wmh_multisite.utils import io

log = logging.getLogger(__name__)

WM_LABELS, GM_LABELS, WMH_LABEL = (2, 41), (3, 42), 77
DIETRICH_FACTOR = 0.6551364  # MRIQC: 1 / sqrt(2 / (4 - pi)); std of a Rayleigh variable / its sigma
METRICS = ["snr_wm", "snrd_wm", "cnr", "cjv", "efc", "fber", "inu_range", "fwhm"]
# Metrics judged per modality. The FLAIR has almost no WM/GM contrast (median ratio 0.87 to 0.99), so
# CJV, which divides by |med_WM - med_GM|, blows up for reasons unrelated to quality (z up to 209). It
# is measured for both but judged for the T1 only.
JUDGED = {"FLAIR": ["snr_wm", "snrd_wm", "cnr", "efc", "fber", "inu_range", "fwhm"], "T1w": METRICS}
# direction of "better": +1 higher is better, -1 lower is better, 0 no direction
DIRECTION = {
    "snr_wm": 1,
    "snrd_wm": 1,
    "cnr": 1,
    "cjv": -1,
    "efc": -1,
    "fber": 1,
    "inu_range": -1,
    "fwhm": -1,
}
# MRIQC Harmonize erodes the white matter with this structure (scipy: 3D, connectivity 2)
ERODE_STRUCTURE = ndimage.generate_binary_structure(3, 2)


def background_mask(image: np.ndarray, head: np.ndarray, margin_vox: int = 3) -> np.ndarray:
    """Air: everything outside the head dilated by margin_vox voxels, except exact zeros (defacing)."""
    near_head = ndimage.binary_dilation(head, iterations=margin_vox)
    return ~near_head & (image != 0)


def harmonize(corrected: np.ndarray, wm: np.ndarray) -> np.ndarray:
    """Scale the image so that the white matter median is 1000, as MRIQC does.

    The ratios do not depend on the scale, but the integer rounding of the statistics and the mad_air > 1
    switch of Dietrich's SNR do, so we scale first to match MRIQC.
    """
    return corrected.astype(np.float64) * (1000.0 / np.median(corrected[wm]))


def region_stats(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """MRIQC summary statistics of a region.

    Median and standard deviation of the values rounded to integers, MAD of the values around that median
    (normalised by 0.6745, as in statsmodels) and number of voxels.
    """
    from statsmodels.robust.scale import mad
    from statsmodels.stats.weightstats import DescrStatsW

    values = image[mask]
    if values.size < 2:
        return {"median": float("nan"), "stdv": float("nan"), "mad": float("nan"), "n": float(values.size)}
    stats = DescrStatsW(np.round(values, 0))
    median = float(stats.quantile(np.array([0.5]), return_pandas=False)[0])
    return {
        "median": median,
        "stdv": float(stats.std),
        "mad": float(mad(values, center=median)),
        "n": float(values.size),
    }


def snr(median: float, stdv: float, n: float) -> float:
    return median / (stdv * np.sqrt(n / (n - 1)))


def snr_dietrich(median: float, mad_air: float, sd_air: float) -> float:
    if mad_air > 1.0:
        return DIETRICH_FACTOR * median / mad_air
    return DIETRICH_FACTOR * median / sd_air if sd_air > 1e-3 else float("nan")  # MRIQC returns -1


def efc(image: np.ndarray) -> float:
    """Entropy focus criterion of MRIQC, over the non-zero voxels (which also leaves out the defacing)."""
    x = image[image != 0].astype(np.float64)
    n = x.size
    if n < 2:
        return float("nan")
    b = np.sqrt((x**2).sum())
    efc_max = n * (1 / np.sqrt(n)) * np.log(1 / np.sqrt(n))
    return float((x / b * np.log((x + 1e-16) / b)).sum() / efc_max)


def fber(image: np.ndarray, head: np.ndarray, air: np.ndarray) -> float:
    fg = float(np.median(np.abs(image[head]) ** 2))
    bg = float(np.median(np.abs(image[air]) ** 2)) if air.any() else 0.0
    return fg / bg if bg >= 1e-3 else float("nan")  # MRIQC returns -1


def tissue_iqms(
    corrected: np.ndarray, wm: np.ndarray, gm: np.ndarray, air: np.ndarray, head: np.ndarray
) -> dict[str, float]:
    """All IQMs except the bias one, from the corrected image and its masks."""
    image = harmonize(corrected, wm)
    s_wm, s_gm, s_air = region_stats(image, wm), region_stats(image, gm), region_stats(image, air)
    contrast = abs(s_wm["median"] - s_gm["median"])
    return {
        "snr_wm": snr(s_wm["median"], s_wm["stdv"], s_wm["n"]),
        "snrd_wm": snr_dietrich(s_wm["median"], s_air["mad"], s_air["stdv"]),
        "cnr": contrast / np.sqrt(s_air["stdv"] ** 2 + s_wm["stdv"] ** 2 + s_gm["stdv"] ** 2),
        "cjv": (s_wm["mad"] + s_gm["mad"]) / contrast if contrast else float("nan"),
        "efc": efc(image),
        "fber": fber(image, head, air),
        "n_wm_voxels": int(s_wm["n"]),
        "n_air_voxels": int(s_air["n"]),
    }


def fwhm(image: np.ndarray, mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    """In-plane smoothness of image inside mask (mm), with the classic AFNI 3dFWHMx estimate.

    For a smoothed noise field, the variance of the differences between neighbours shrinks as the smoothing
    grows, which gives FWHM = sqrt(-2 ln 2 / ln(1 - var(d) / (2 var))) times the voxel size. The result is
    averaged over the in-plane axes when one axis is much coarser (3 mm slices), otherwise over all three.
    """
    x = image.astype(np.float64)
    var = x[mask].var()
    if var <= 0:
        return float("nan")
    axes = [0, 1, 2]
    if max(spacing) > 1.5 * min(spacing):
        axes.remove(int(np.argmax(spacing)))
    values = []
    for k in axes:
        lead = [slice(None)] * 3
        lead[k] = slice(1, None)
        trail = [slice(None)] * 3
        trail[k] = slice(None, -1)
        both = mask[tuple(lead)] & mask[tuple(trail)]
        ratio = np.diff(x, axis=k)[both].var() / (2 * var)
        if 0 < ratio < 1:
            values.append(np.sqrt(-2 * np.log(2) / np.log(1 - ratio)) * spacing[k])
    return float(np.mean(values)) if values else float("nan")


def inu_range(field: np.ndarray, brain: np.ndarray) -> float:
    lo, hi = np.percentile(field[brain], [5, 95])
    return float(abs(hi - lo))


def tissues(
    labels: np.ndarray,
    brain: np.ndarray,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    wmh_margin_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """White matter and cortex masks for the IQMs.

    The white matter is eroded and loses the WMH and a margin of wmh_margin_mm around them, because the white
    matter next to a lesion is often already abnormal and would lower the SNR of subjects with many lesions.
    Distances use the real voxel size.
    """
    wm = ndimage.binary_erosion(np.isin(labels, WM_LABELS), structure=ERODE_STRUCTURE) & brain
    wmh = labels == WMH_LABEL
    if wmh_margin_mm > 0 and wmh.any():
        wm &= ndimage.distance_transform_edt(~wmh, sampling=spacing) > wmh_margin_mm
    gm = np.isin(labels, GM_LABELS) & brain
    return wm, gm


def correct(raw: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Bias-corrected image, raw / N4 field, as in bias.n4_correct."""
    raw = raw.astype(np.float64)
    return np.divide(raw, field, out=np.zeros_like(raw), where=field > 0)


def labels_to_t1(t1_raw: Path, labels_1mm: Path, xfm: Path) -> np.ndarray:
    """WMH-SynthSeg labels carried to the raw T1 grid with the inverse rigid transform."""
    import ants

    out = ants.apply_transforms(
        ants.image_read(str(t1_raw)),
        ants.image_read(str(labels_1mm), pixeltype="float"),
        [str(xfm)],
        whichtoinvert=[True],
        interpolator="genericLabel",
    )
    return np.rint(out.numpy()).astype(np.int16)


def subject_paths(cfg: dict[str, Any], sub: str) -> dict[str, Path]:
    d = resolve_path(cfg, "derivatives") / PIPELINE / sub / "anat"
    b = resolve_path(cfg, "bids") / sub / "anat"
    return {
        "FLAIR": b / f"{sub}_FLAIR.nii.gz",
        "T1w": b / f"{sub}_T1w.nii.gz",
        "brain_FLAIR": d / f"{sub}_space-FLAIR_desc-brain_mask.nii.gz",
        "brain_T1w": d / f"{sub}_space-T1w_desc-brain_mask.nii.gz",
        "bias_FLAIR": d / f"{sub}_desc-biasfield_FLAIR.nii.gz",
        "bias_T1w": d / f"{sub}_desc-biasfield_T1w.nii.gz",
        "labels_FLAIR": d / f"{sub}_space-FLAIR_desc-wmhsynthsegfull_dseg.nii.gz",
        "labels_1mm": d / f"{sub}_desc-wmhsynthseg1mm_dseg.nii.gz",
        "xfm": d / f"{sub}_from-T1w_to-FLAIR_mode-image_xfm.mat",
    }


def measure(
    raw: np.ndarray,
    labels: np.ndarray,
    brain: np.ndarray,
    field: np.ndarray,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    wmh_margin_mm: float = 0.0,
) -> dict[str, float]:
    """IQMs of one image from its raw version and its N4 field. Also used on degraded copies."""
    head = head_mask(raw)
    wm, gm = tissues(labels, brain, spacing, wmh_margin_mm)
    out = tissue_iqms(correct(raw, field), wm, gm, background_mask(raw, head), head)
    out["inu_range"] = inu_range(field, brain)
    out["fwhm"] = fwhm(raw, brain, spacing)  # on the raw image, as MRIQC
    return out


def subject_iqms(task: tuple[str, str | None]) -> list[dict[str, Any]]:
    sub, config_path = task
    import os

    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    cfg = load_config(config_path)
    p = subject_paths(cfg, sub)
    rows = []
    flair_labels = io.load_labels(p["labels_FLAIR"], allowed=tuple(range(256)))
    t1_labels = labels_to_t1(p["T1w"], p["labels_1mm"], p["xfm"])
    margin = cfg["qc"]["wmh_margin_mm"]
    for modality, labels in [("FLAIR", flair_labels), ("T1w", t1_labels)]:
        image = io.load_data(p[modality])
        spacing = io.load(p[modality]).header.get_zooms()[:3]
        brain = io.load_labels(p[f"brain_{modality}"], allowed=(0, 1)).astype(bool)
        field = io.load_data(p[f"bias_{modality}"])
        iqm = measure(image, labels, brain, field, spacing, margin)
        rows.append({"participant_id": sub, "modality": modality, **iqm})
    return rows


def run(cfg: dict[str, Any], jobs: int, config_path: str | None = None) -> pd.DataFrame:
    part = pd.read_csv(resolve_path(cfg, "bids") / "participants.tsv", sep="\t").set_index("participant_id")
    ready = [s for s in part.index if subject_paths(cfg, s)["labels_1mm"].is_file()]
    if len(ready) < len(part):
        log.warning(
            "%d of %d subjects without the WMH-SynthSeg segmentation yet: skipped",
            len(part) - len(ready),
            len(part),
        )
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = [r for rows in pool.map(subject_iqms, [(s, config_path) for s in ready]) for r in rows]
    table = pd.DataFrame(rows)
    table.insert(1, "scanner", table.participant_id.map(part.scanner))
    table.insert(2, "split", table.participant_id.map(part.split))
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Image quality metrics of the FLAIR and T1, MRIQC protocol.")
    parser.add_argument("--config")
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    table = run(cfg, args.jobs, args.config)
    out = resolve_path(cfg, "results") / "tables" / "qc_iqm.csv"
    table.to_csv(out, index=False, float_format="%.5g")
    pd.set_option("display.width", 200)
    log.info(
        "%s: %d rows\n%s",
        out,
        len(table),
        table.groupby(["modality", "scanner"])[METRICS].median().round(3).to_string(),
    )


if __name__ == "__main__":
    main()
