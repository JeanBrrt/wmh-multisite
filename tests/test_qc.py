"""Phase 4 (image quality control) on synthetic volumes whose answer is known in advance."""

import numpy as np
import pandas as pd
import pytest

from wmh_multisite.qc.artifacts import degrade
from wmh_multisite.qc.iqm import (
    DIETRICH_FACTOR,
    background_mask,
    efc,
    harmonize,
    inu_range,
    labels_to_t1,
    measure,
    tissue_iqms,
    tissues,
)
from wmh_multisite.qc.outliers import badness_z, classify, robust_z, validate
from wmh_multisite.viz.report import build_html, thumbnail

SHAPE = (40, 40, 20)


def phantom(seed=0):
    """Air with Rayleigh noise, a head, a brain with white matter (100 +/- 5) and grey matter (120 +/- 5)."""
    rng = np.random.default_rng(seed)
    image = rng.rayleigh(4.0, SHAPE)  # air: Rayleigh with sigma 4
    head = np.zeros(SHAPE, bool)
    head[8:32, 8:32, 4:16] = True
    brain = np.zeros(SHAPE, bool)
    brain[11:29, 11:29, 6:14] = True
    labels = np.zeros(SHAPE, np.int16)
    labels[brain] = 3  # cortex
    labels[14:26, 14:26, 7:13] = 2  # white matter inside
    image[head] = 60 + rng.normal(0, 5, head.sum())  # scalp / CSF
    image[labels == 3] = 120 + rng.normal(0, 5, (labels == 3).sum())
    image[labels == 2] = 100 + rng.normal(0, 5, (labels == 2).sum())
    return image, labels, brain, head


def iqms(image, labels, brain, head):
    wm, gm = tissues(labels, brain)
    return tissue_iqms(image, wm, gm, background_mask(image, head), head)


def test_tissue_metrics_match_their_definitions():
    image, labels, brain, head = phantom()
    q = iqms(image, labels, brain, head)
    assert q["snr_wm"] == pytest.approx(100 / 5, rel=0.1)
    assert q["cjv"] == pytest.approx((5 + 5) / 20, rel=0.1)  # MAD of a normal = its sigma
    # air: Rayleigh(sigma = 4), normalized MAD ~0.665 sigma, so snrd_wm ~ 0.655 * 100 / (0.665 * 4)
    assert q["snrd_wm"] == pytest.approx(100 / 4, rel=0.1)


def test_metrics_follow_the_mriqc_formulas():
    """Same numbers as MRIQC's summary_stats / snr / cnr / cjv, recomputed by hand."""
    from statsmodels.robust.scale import mad

    image, labels, brain, head = phantom()
    wm, gm = tissues(labels, brain)
    air = background_mask(image, head)
    h = harmonize(image, wm)
    assert np.median(h[wm]) == pytest.approx(1000.0)  # MRIQC Harmonize
    med = {k: np.median(np.round(h[m])) for k, m in {"wm": wm, "gm": gm}.items()}
    sd = {k: np.round(h[m]).std() for k, m in {"wm": wm, "gm": gm, "air": air}.items()}
    n = wm.sum()
    q = tissue_iqms(image, wm, gm, air, head)
    assert q["snr_wm"] == pytest.approx(med["wm"] / (sd["wm"] * np.sqrt(n / (n - 1))), rel=1e-3)
    assert q["cnr"] == pytest.approx(
        abs(med["wm"] - med["gm"]) / np.sqrt(sd["air"] ** 2 + sd["wm"] ** 2 + sd["gm"] ** 2), rel=1e-3
    )
    mad_air = mad(h[air], center=np.median(np.round(h[air])))
    assert q["snrd_wm"] == pytest.approx(DIETRICH_FACTOR * med["wm"] / mad_air, rel=1e-3)
    cjv = (mad(h[wm], center=med["wm"]) + mad(h[gm], center=med["gm"])) / abs(med["wm"] - med["gm"])
    assert q["cjv"] == pytest.approx(cjv, rel=1e-3)


def test_metrics_do_not_depend_on_the_intensity_scale():
    image, labels, brain, head = phantom()
    before, after = iqms(image, labels, brain, head), iqms(image * 7.3, labels, brain, head)
    for m in ("snr_wm", "snrd_wm", "cnr", "cjv", "efc", "fber"):
        assert after[m] == pytest.approx(before[m], rel=1e-2)


def test_white_matter_ring_around_wmh_is_left_out():
    image, labels, brain, head = phantom()
    labels[19:21, 19:21, 9:11] = 77  # a small lesion inside the white matter
    spacing = (1.0, 1.0, 3.0)  # FLAIR-like: 3 mm slices
    wm0, _ = tissues(labels, brain, spacing, 0.0)
    wm2, _ = tissues(labels, brain, spacing, 2.0)
    assert not wm0[labels == 77].any() and wm2.sum() < wm0.sum()
    removed = wm0 & ~wm2
    assert removed.any() and removed[:, :, 9:11].sum() == removed.sum()  # ring in the lesion slices only
    assert not removed[:, :, :8].any() and not removed[:, :, 12:].any()  # 3 mm slices: no ring above/below


def test_bias_is_seen_by_inu_range_only():
    """MRIQC protocol: tissue metrics on the N4-corrected image, the bias on the field alone."""
    image, labels, brain, head = phantom()
    flat = np.ones(SHAPE)
    ramp = np.broadcast_to(np.linspace(0.7, 1.3, SHAPE[0])[:, None, None], SHAPE).copy()
    clean = measure(image, labels, brain, flat)
    biased = measure(image * ramp, labels, brain, ramp)  # field perfectly estimated by N4
    for m in ("snr_wm", "snrd_wm", "cnr", "cjv"):
        assert biased[m] == pytest.approx(clean[m], rel=0.02)
    assert clean["inu_range"] == 0 and biased["inu_range"] > 0.2  # ramp 0.82-1.18 across the brain


def test_rule_b_defacing_zeros_do_not_change_the_background_metrics():
    image, labels, brain, head = phantom()
    before = iqms(image, labels, brain, head)
    defaced = image.copy()
    defaced[:6, :, :] = 0  # a "defaced" slab of air set to exactly zero
    after = iqms(defaced, labels, brain, head)
    assert after["n_air_voxels"] < before["n_air_voxels"]
    for m in ("snrd_wm", "fber"):
        assert after[m] == pytest.approx(before[m], rel=0.03)
    assert efc(np.concatenate([image.ravel(), np.zeros(5000)])) == pytest.approx(efc(image))  # zeros left out


def test_rule_a_outside_the_brain_does_not_change_the_tissue_metrics():
    image, labels, brain, head = phantom()
    before = iqms(image, labels, brain, head)
    face = image.copy()
    face[8:11, 8:32, 4:16] = 900  # very bright tissue outside the brain (e.g. face, fat)
    after = iqms(face, labels, brain, head)
    for m in ("snr_wm", "cjv"):
        assert after[m] == pytest.approx(before[m])


def test_efc_bounds_and_inu_range():
    assert efc(np.ones(1000)) == pytest.approx(1.0)  # energy spread evenly: maximal entropy
    focused = np.full(1000, 1e-3)
    focused[0] = 1000.0
    assert efc(focused) < 0.01  # almost all the energy in one voxel
    assert np.isnan(efc(np.array([0.0, 0.0, 5.0])))  # one non-zero voxel: undefined (zeros left out)
    field = np.tile(np.linspace(0.8, 1.2, 101), (2, 1))
    assert inu_range(field, np.ones_like(field, bool)) == pytest.approx(
        0.36, abs=0.01
    )  # 95th - 5th percentile


def test_labels_carried_to_t1_with_an_identity_transform(nifti, tmp_path):
    import ants

    labels = np.zeros((10, 10, 10), np.float32)
    labels[3:6, 3:6, 3:6] = 41
    lab_path = nifti(labels, "lab.nii.gz", np.eye(4))
    t1_path = nifti(np.ones((10, 10, 10), np.float32), "t1.nii.gz", np.eye(4))
    xfm = tmp_path / "identity.mat"
    ants.write_transform(ants.new_ants_transform(dimension=3, transform_type="AffineTransform"), str(xfm))
    out = labels_to_t1(t1_path, lab_path, xfm)
    assert set(np.unique(out)) == {0, 41} and (out == 41).sum() == 27


def iqm_table(n_per_scanner=12, seed=0):
    """Two scanners with different typical SNR; one very bad image in scanner a."""
    rng = np.random.default_rng(seed)
    rows = []
    for scanner, snr in [("a", 10.0), ("b", 30.0)]:
        for i in range(n_per_scanner):
            for modality in ("FLAIR", "T1w"):
                row = {"participant_id": f"{scanner}{i}", "scanner": scanner, "modality": modality}
                row |= {
                    m: rng.normal(1.0, 0.05) for m in ["snrd_wm", "cnr", "cjv", "efc", "fber", "inu_range"]
                }
                row["snr_wm"] = snr + rng.normal(0, 1)
                rows.append(row)
    table = pd.DataFrame(rows)
    noisy = (table.participant_id == "a0") & (table.modality == "FLAIR")
    table.loc[noisy, "snr_wm"] = 2.0  # very noisy image
    return table


def test_robust_z_is_computed_within_each_scanner():
    t = iqm_table()
    flair = t[t.modality == "FLAIR"].set_index("participant_id")
    z = robust_z(flair.snr_wm, flair.scanner)
    assert abs(z.drop("a0")).max() < 4  # scanner b's higher SNR is not "atypical"
    assert z["a0"] < -5


def test_bad_image_is_classed_exclude_and_sign_means_worse():
    t = iqm_table()
    z = badness_z(t)
    assert z.loc["a0", "FLAIR_snr_wm"] > 5  # low SNR -> positive (worse)
    flags = classify(z, {"z_check": 3, "z_exclude": 5, "iforest_contamination": 0.05}, seed=0)
    assert flags.loc["a0", "qc_class"] == "exclude" and flags.loc["a0", "worst_metric"] == "FLAIR_snr_wm"
    assert (flags.qc_class == "usable").sum() >= 18


def test_validation_compares_dice_by_class():
    flags = pd.DataFrame(
        {"qc_class": ["usable"] * 4 + ["check"] * 2, "iforest_score": [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]},
        index=[f"s{i}" for i in range(6)],
    )
    ev = pd.DataFrame(
        {"participant_id": [f"s{i}" for i in range(6)], "dice": [0.8, 0.82, 0.79, 0.81, 0.5, 0.55]}
    )
    v = validate(flags, ev)
    assert v["n_flagged"] == 2 and v["dice_flagged"] < v["dice_usable"] and v["spearman_score_dice"] < 0


def test_artifacts_degrade_and_level_zero_is_identity():
    image, labels, brain, head = phantom()
    assert np.array_equal(degrade(image, brain, "noise", 0, seed=1), image)
    wm, _ = tissues(labels, brain)
    snrs = [image[wm].mean() / degrade(image, brain, "noise", s, seed=1)[wm].std() for s in (0.03, 0.1)]
    assert snrs[0] > snrs[1]  # stronger noise, lower SNR
    for artifact, level in [("motion", 4), ("bias", 0.3), ("ghosting", 0.5)]:
        out = degrade(image, brain, artifact, level, seed=1)
        assert out.shape == image.shape and (out >= 0).all() and not np.allclose(out, image)


def test_report_html_lists_flagged_subjects(nifti):
    flags = pd.DataFrame(
        {
            "scanner": ["a", "a"],
            "qc_class": ["usable", "check"],
            "max_z": [1.0, 3.5],
            "worst_metric": ["FLAIR_snr", "FLAIR_cjv"],
            "iforest_score": [0.4, 0.7],
            "z_FLAIR_snr": [1.0, 0.2],
            "z_FLAIR_cjv": [0.1, 3.5],
        },
        index=["sub-1", "sub-<2>"],
    )
    image, *_ = phantom()
    thumb = thumbnail(nifti(image.astype(np.float32), "img.nii.gz"))
    page = build_html(flags, {"sub-<2>": {"FLAIR": thumb}})
    assert "sub-&lt;2&gt;" in page and "FLAIR_cjv = 3.5" in page and "data:image/png;base64," in page
    assert "<h3 class='check'>sub-1" not in page  # usable subjects get no thumbnail section


def test_small_scanners_are_not_assessed():
    t = iqm_table(n_per_scanner=12)
    small = t[~t.participant_id.isin([f"b{i}" for i in range(4, 12)])]  # scanner b keeps 4 subjects
    z = badness_z(small, min_group=8)
    assert z.loc[[f"b{i}" for i in range(4)]].isna().all().all()
    flags = classify(z, {"z_check": 3, "z_exclude": 5, "iforest_contamination": 0.05}, seed=0)
    assert (flags.loc[[f"b{i}" for i in range(4)], "qc_class"] == "not_assessed").all()
    assert flags.loc["a0", "qc_class"] == "exclude"  # scanner a (12 subjects) is still judged


def test_flair_is_not_judged_on_its_white_grey_contrast():
    z = badness_z(iqm_table())
    assert "FLAIR_cjv" not in z.columns
    assert "T1w_cjv" in z.columns and "FLAIR_snr_wm" in z.columns


def test_artifact_badness_uses_the_metrics_judged_for_each_modality():
    from wmh_multisite.qc.artifacts import badness

    t = iqm_table()
    rows = t[(t.modality == "T1w") & (t.participant_id == "a1")].copy()
    rows["cjv"] = 5.0  # much worse than the clean T1s of scanner a
    z = badness(rows, t, "T1w")
    assert "T1w_cjv" in z.columns and z["T1w_cjv"].iloc[0] > 5
    assert "FLAIR_cjv" not in badness(t[t.modality == "FLAIR"], t, "FLAIR").columns
