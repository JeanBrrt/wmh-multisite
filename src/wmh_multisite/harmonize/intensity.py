"""Intensity normalisation of the FLAIR.

MRI intensities have no unit: the same tissue reads 400 on one scanner and 900 on another, so the M3
threshold and the NAWM features need a common scale. All four methods work inside the HD-BET brain
mask, since a head mask (with or without the face) would differ between sites:

    none        I / C, C = median of the training brains (one constant for everybody: no per-image
                correction; the baseline that shows what normalisation brings)
    zscore      (I - median) / (1.4826 MAD), over the brain of the image (as in M3)
    whitestripe WhiteStripe (Shinohara et al. 2014): the main mode of the smoothed brain histogram (the
                white matter), then the "stripe" of brain voxels whose quantile lies within +/- tau = 5 %
                of the mode's quantile; (I - mean) / SD of that stripe. A first version used the mean and
                SD of the whole WMH-SynthSeg white matter; the narrow stripe leaves out partial-volume and
                atypical voxels and gave a better threshold (Dice 0.611 vs 0.564)
    histmatch   piecewise-linear mapping of the brain percentiles (1, 10, 20, ..., 90, 99) of the image
                onto a standard scale learnt on the training images (Nyul and Udupa 2000): each training
                image's [p1, p99] is mapped linearly onto [0, 100], and the standard landmarks are the
                means of the mapped percentiles

The constant of ``none`` and the landmarks of ``histmatch`` are learnt on the training set only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

METHODS = ("none", "zscore", "whitestripe", "histmatch")
PERCENTILES = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99)
WM_LABELS = (2, 41)
WHITESTRIPE_TAU = 0.05  # half-width of the stripe, in quantiles (WhiteStripe default)


def zscore(image: np.ndarray, brain: np.ndarray) -> np.ndarray:
    values = image[brain]
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) * 1.4826
    if mad <= 0:
        raise ValueError("brain intensities have zero spread (MAD = 0)")
    return ((image - med) / mad).astype(np.float32)


def whitestripe(
    image: np.ndarray, brain: np.ndarray, labels: np.ndarray | None = None, tau: float = WHITESTRIPE_TAU
) -> np.ndarray:
    """WhiteStripe: mean and SD of the voxels around the main mode of the brain histogram.

    The histogram (1000 bins between the 0.5th and 99.5th brain percentiles) is smoothed, its main mode
    taken as the white-matter intensity, and the stripe is the brain voxels whose quantile lies within
    ``tau`` of that mode's quantile. ``labels`` is unused (kept for a uniform call).
    """
    from scipy import ndimage

    v = image[brain].astype(np.float64)
    lo, hi = np.percentile(v, [0.5, 99.5])
    hist, edges = np.histogram(v, bins=1000, range=(lo, hi))
    smooth = ndimage.gaussian_filter1d(hist.astype(np.float64), 8)
    mode = 0.5 * (edges[:-1] + edges[1:])[int(np.argmax(smooth))]
    q = float((v < mode).mean())
    a, b = np.quantile(v, [max(q - tau, 0.0), min(q + tau, 1.0)])
    stripe = v[(v >= a) & (v <= b)]
    if stripe.size < 100 or stripe.std() <= 0:
        raise ValueError("WhiteStripe: stripe too small or flat")
    return ((image - stripe.mean()) / stripe.std()).astype(np.float32)


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
        return whitestripe(image, brain)
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
