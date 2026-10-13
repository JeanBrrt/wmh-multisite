"""Image quality metrics (IQMs) of every FLAIR and T1, following MRIQC.

A degraded image (motion, noise, strong bias, ghosts) silently gives a wrong biomarker; IQMs let every
image be checked before analysis.

The formulas are MRIQC's (Esteban et al. 2017; nipreps/mriqc ``qc/anatomical.py`` and
``interfaces/anatomical.py``, StructuralQC and Harmonize), for its main IQMs. The bias is measured on
the N4 field only, noise and contrast on the image after N4, so the bias is not counted twice.

    raw image / N4 field                               = INU-corrected image
      x 1000 / median(corrected in eroded WM)          = "harmonized" image (MRIQC Harmonize)
      rounded to integers for the tissue statistics    (MRIQC summary_stats, rprec_data=0)

    per region R: med_R (median), sd_R (standard deviation, ddof 0), mad_R (statsmodels ``mad``, i.e.
    median |x - med_R| / 0.6745, centred on med_R), n_R (number of voxels)

    snr_wm   med_WM / (sd_WM * sqrt(n_WM / (n_WM - 1)))                    higher is better
    snrd_wm  0.6551364 * med_WM / mad_air   (sd_air if mad_air <= 1)      higher (Dietrich 2007:
             0.6551364 = 1 / sqrt(2 / (4 - pi)), air noise is Rayleigh)
    cnr      |med_WM - med_GM| / sqrt(sd_air^2 + sd_WM^2 + sd_GM^2)        higher
    cjv      (mad_WM + mad_GM) / |med_WM - med_GM|                         lower (T1 only, see JUDGED)
    efc      sum(x / b log(x / b)) / (N / sqrt(N) log(1 / sqrt(N))), b = sqrt(sum x^2)   lower
    fber     median(x^2 in the head) / median(x^2 in the air)              higher
    inu_range |p95 - p5| of the N4 field in the brain                     lower

Deliberate differences from MRIQC:
- Tissues come from WMH-SynthSeg, not from Atropos. White matter = labels 2/41 eroded like MRIQC
  Harmonize's WM mask, minus the WMH (label 77) and a ring of ``qc.wmh_margin_mm`` around them
  (otherwise the SNR tracked the lesion load). Grey matter = cortex only (3/42), not eroded: it is
  2-3 mm thick and one erosion would leave 14-23 % of it on the FLAIR grid.
- Tissue statistics inside the HD-BET brain.
- Air = outside the dilated head, exact zeros excluded (only some FLAIRs are defaced), instead of
  MRIQC's "hat" above the nasion-inion plane: our FLAIRs only cover ~144 mm in height.
- No MRIQC classifier: it was trained on T1 only.

The T1 stays on its own grid, since resampling would smooth its noise and inflate its SNR: the
WMH-SynthSeg labels are carried to it by the inverse rigid transform (``genericLabel``).

Output: ``results/tables/qc_iqm.csv`` (one row per subject and modality). Subjects without the
WMH-SynthSeg segmentation yet are skipped (logged).

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
METRICS = ["snr_wm", "snrd_wm", "cnr", "cjv", "efc", "fber", "inu_range"]
# Metrics judged per modality. The FLAIR has almost no WM/GM contrast (median ratio 0.87 to 0.99), so
# CJV, which divides by |med_WM - med_GM|, blows up for reasons unrelated to quality (z up to 209). It
# is measured for both but judged for the T1 only.
JUDGED = {"FLAIR": ["snr_wm", "snrd_wm", "cnr", "efc", "fber", "inu_range"], "T1w": METRICS}
# direction of "better": +1 higher is better, -1 lower is better, 0 no direction
DIRECTION = {"snr_wm": 1, "snrd_wm": 1, "cnr": 1, "cjv": -1, "efc": -1, "fber": 1, "inu_range": -1}
# MRIQC Harmonize erodes the white matter with this structure (scipy: 3D, connectivity 2)
ERODE_STRUCTURE = ndimage.generate_binary_structure(3, 2)


def background_mask(image: np.ndarray, head: np.ndarray, margin_vox: int = 3) -> np.ndarray:
    """Air: outside the head dilated by ``margin_vox`` voxels, exact zeros (defacing) excluded."""
    near_head = ndimage.binary_dilation(head, iterations=margin_vox)
    return ~near_head & (image != 0)


def harmonize(corrected: np.ndarray, wm: np.ndarray) -> np.ndarray:
    """MRIQC Harmonize: scale so that the median of the white matter is 1000.

    The ratios below do not depend on the scale, but the integer rounding of the statistics and the
    ``mad_air > 1`` switch of Dietrich's SNR do: harmonizing first keeps them as in MRIQC.
    """
    return corrected.astype(np.float64) * (1000.0 / np.median(corrected[wm]))


def region_stats(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """MRIQC ``summary_stats`` for a binary region: median and std of the integer-rounded values,
    MAD (statsmodels, normalized by 0.6745) of the values around that median, number of voxels."""
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
    """Entropy focus criterion (MRIQC), over the non-zero voxels, which also drops the defacing zeros."""
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
    """All IQMs but the bias one, from the INU-corrected image and its masks (boolean, same grid)."""
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


def inu_range(field: np.ndarray, brain: np.ndarray) -> float:
    lo, hi = np.percentile(field[brain], [5, 95])
    return float(abs(hi - lo))


def tissues(
    labels: np.ndarray,
    brain: np.ndarray,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
    wmh_margin_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """White matter (eroded, minus the WMH and a ring of ``wmh_margin_mm`` around them) and cortex.

    White matter next to a lesion is often already abnormal; left in, it lowers the SNR of subjects with
    many lesions. Distances in mm, with the real voxel size (3 mm slices on the FLAIR grid).
    """
    wm = ndimage.binary_erosion(np.isin(labels, WM_LABELS), structure=ERODE_STRUCTURE) & brain
    wmh = labels == WMH_LABEL
    if wmh_margin_mm > 0 and wmh.any():
        wm &= ndimage.distance_transform_edt(~wmh, sampling=spacing) > wmh_margin_mm
    gm = np.isin(labels, GM_LABELS) & brain
    return wm, gm


def correct(raw: np.ndarray, field: np.ndarray) -> np.ndarray:
    """INU-corrected image: raw / N4 field, as in ``bias.n4_correct``."""
    raw = raw.astype(np.float64)
    return np.divide(raw, field, out=np.zeros_like(raw), where=field > 0)


def labels_to_t1(t1_raw: Path, labels_1mm: Path, xfm: Path) -> np.ndarray:
    """WMH-SynthSeg labels (1 mm grid, FLAIR space) carried to the raw T1 grid (inverse rigid transform)."""
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
    """IQMs of one image from its raw version and its N4 field (also used on degraded copies)."""
    head = head_mask(raw)
    wm, gm = tissues(labels, brain, spacing, wmh_margin_mm)
    out = tissue_iqms(correct(raw, field), wm, gm, background_mask(raw, head), head)
    out["inu_range"] = inu_range(field, brain)
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
