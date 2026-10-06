"""Intensity normalisations of 8.1 on synthetic images whose answer is known in advance."""

import numpy as np
import pytest

from wmh_multisite.harmonize import intensity as hi


def brain_image(scale=1.0, offset=0.0, seed=0):
    """Brain: white matter (100 +/- 5) and grey matter (130 +/- 5); bright face outside the brain."""
    rng = np.random.default_rng(seed)
    image = np.zeros((20, 20, 10), np.float32)
    brain = np.zeros(image.shape, bool)
    brain[2:18, 2:18, 2:8] = True
    labels = np.zeros(image.shape, np.int16)
    labels[brain] = 3
    labels[5:15, 5:15, 3:7] = 2
    image[labels == 3] = 130 + rng.normal(0, 5, (labels == 3).sum())
    image[labels == 2] = 100 + rng.normal(0, 5, (labels == 2).sum())
    image[0, :, :] = 2000  # face / fat outside the brain
    return image * scale + offset, brain, labels


def test_zscore_and_whitestripe_cancel_a_scanner_gain_and_offset():
    a, brain, labels = brain_image()
    b, *_ = brain_image(scale=3.0, offset=50.0)  # same brain, other scanner scale
    for method in ("zscore", "whitestripe"):
        za = hi.normalize(a, brain, method, {}, labels)
        zb = hi.normalize(b, brain, method, {}, labels)
        assert np.allclose(za[brain], zb[brain], atol=1e-3)


def test_whitestripe_puts_white_matter_at_zero_mean_unit_sd():
    image, brain, labels = brain_image()
    z = hi.whitestripe(image, brain, labels)
    wm = np.isin(labels, (2, 41)) & brain
    assert z[wm].mean() == pytest.approx(0, abs=1e-5) and z[wm].std() == pytest.approx(1, abs=1e-4)


def test_rule_a_outside_the_brain_does_not_change_the_parameters():
    image, brain, labels = brain_image()
    brighter_face = image.copy()
    brighter_face[0] = 9000
    for method in ("zscore", "whitestripe"):
        assert np.allclose(
            hi.normalize(image, brain, method, {}, labels)[brain],
            hi.normalize(brighter_face, brain, method, {}, labels)[brain],
        )


def test_histogram_matching_maps_landmarks_onto_the_standard():
    images = [brain_image(scale=s, seed=i)[:2] for i, s in enumerate((1.0, 2.0, 0.5))]
    params = hi.fit("histmatch", images)
    standard = np.asarray(params["standard"])
    assert standard[0] == pytest.approx(0) and standard[-1] == pytest.approx(100)
    image, brain = brain_image(scale=4.0, seed=7)[:2]
    out = hi.histmatch(image, brain, standard)
    assert np.allclose(np.percentile(out[brain], hi.PERCENTILES), standard, atol=1.0)


def test_none_is_one_constant_learnt_on_train():
    images = [brain_image(scale=s)[:2] for s in (1.0, 2.0, 3.0)]
    params = hi.fit("none", images)
    image, brain, _ = brain_image(scale=2.0)
    assert params["constant"] == pytest.approx(
        np.median(image[brain]), rel=0.01
    )  # median of the three medians
    assert np.allclose(hi.normalize(image, brain, "none", params), image / params["constant"])


def test_unknown_method_and_missing_labels():
    image, brain, _ = brain_image()
    with pytest.raises(ValueError, match="unknown normalisation"):
        hi.normalize(image, brain, "foo", {})
    with pytest.raises(ValueError, match="tissue labels"):
        hi.normalize(image, brain, "whitestripe", {})
