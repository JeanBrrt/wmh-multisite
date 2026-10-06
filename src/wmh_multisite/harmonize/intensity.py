"""Intensity normalisation of the FLAIR (ROADMAP step 8.1, J-028, J-055).

MRI intensities have no unit: the same tissue reads 400 on one scanner and 900 on another. A rule
applied to intensities (the threshold M3, the features of 8.2) needs them on a common scale. Four ways,
all computed **inside the HD-BET brain mask** (J-028, rule A: the head mask, with or without the face,
would differ between sites):

    none        I / C, C = median of the training brains (one constant for everybody: no per-image
                correction; the baseline that shows what normalisation brings)
    zscore      (I - median) / (1.4826 MAD), over the brain of the image (the M3 of 5.5)
    whitestripe (I - mean) / SD over the normal-appearing white matter of the image (WMH-SynthSeg white
                matter, labels 2 and 41; the WMH have their own label 77). A variant of WhiteStripe
                (Shinohara et al. 2014), which finds the white-matter peak in the histogram: here the
                segmentation gives the white matter directly
    histmatch   piecewise-linear mapping of the brain percentiles (1, 10, 20, ..., 90, 99) of the image
                onto a standard scale learnt on the training images (Nyul and Udupa 2000): each training
                image's [p1, p99] is mapped linearly onto [0, 100], and the standard landmarks are the
                means of the mapped percentiles

Parameters learnt on the training set only (J-013): the constant of ``none`` and the standard
landmarks of ``histmatch``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

METHODS = ("none", "zscore", "whitestripe", "histmatch")
PERCENTILES = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99)
WM_LABELS = (2, 41)


def zscore(image: np.ndarray, brain: np.ndarray) -> np.ndarray:
    values = image[brain]
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) * 1.4826
    if mad <= 0:
        raise ValueError("brain intensities have zero spread (MAD = 0)")
    return ((image - med) / mad).astype(np.float32)


def whitestripe(image: np.ndarray, brain: np.ndarray, labels: np.ndarray) -> np.ndarray:
    wm = np.isin(labels, WM_LABELS) & brain
    if wm.sum() < 100:
        raise ValueError("too few white-matter voxels for WhiteStripe")
    mu, sd = float(image[wm].mean()), float(image[wm].std())
    return ((image - mu) / sd).astype(np.float32)


def landmarks(image: np.ndarray, brain: np.ndarray) -> np.ndarray:
    return np.percentile(image[brain], PERCENTILES)


def fit_histogram_standard(landmark_sets: list[np.ndarray]) -> np.ndarray:
    """Standard landmarks: every image's [p1, p99] mapped onto [0, 100], landmarks averaged."""
    mapped = [100 * (lm - lm[0]) / (lm[-1] - lm[0]) for lm in landmark_sets]
    return np.mean(mapped, axis=0)


def histmatch(image: np.ndarray, brain: np.ndarray, standard: np.ndarray) -> np.ndarray:
    """Piecewise-linear map of the image landmarks onto the standard ones; linear extrapolation outside."""
    lm = landmarks(image, brain)
    out = np.interp(image, lm, standard)
    lo_slope = (standard[1] - standard[0]) / (lm[1] - lm[0])
    hi_slope = (standard[-1] - standard[-2]) / (lm[-1] - lm[-2])
    out = np.where(image < lm[0], standard[0] + (image - lm[0]) * lo_slope, out)
    out = np.where(image > lm[-1], standard[-1] + (image - lm[-1]) * hi_slope, out)
    return out.astype(np.float32)


def normalize(
    image: np.ndarray,
    brain: np.ndarray,
    method: str,
    params: dict[str, Any],
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """``image`` on the common scale of ``method`` (``params``: learnt on the training set)."""
    if method == "none":
        return (image / params["constant"]).astype(np.float32)
    if method == "zscore":
        return zscore(image, brain)
    if method == "whitestripe":
        if labels is None:
            raise ValueError("whitestripe needs the tissue labels")
        return whitestripe(image, brain, labels)
    if method == "histmatch":
        return histmatch(image, brain, np.asarray(params["standard"]))
    raise ValueError(f"unknown normalisation {method!r} (expected one of {METHODS})")


def fit(method: str, images: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """Parameters of ``method`` learnt on (image, brain) pairs of the training set."""
    if method == "none":
        return {"constant": float(np.median([np.median(im[b]) for im, b in images]))}
    if method == "histmatch":
        return {"standard": fit_histogram_standard([landmarks(im, b) for im, b in images]).tolist()}
    return {}
