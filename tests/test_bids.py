"""data/bids.py: conversion of a fake challenge tree (3 subjects, 2 sites) into BIDS."""

import csv
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from conftest import FLAIR_AFFINE
from wmh_multisite.config import resolve_path
from wmh_multisite.data import bids


@pytest.fixture
def converted(fake_project):
    cfg, raw = fake_project
    inventory = bids.convert(cfg)
    return cfg, raw, resolve_path(cfg, "bids"), pd.read_csv(inventory)


def test_bids_tree_and_names(converted):
    _, _, b, _ = converted
    for path in [
        "dataset_description.json",
        "README",
        "participants.tsv",
        "participants.json",
        "sub-000/anat/sub-000_FLAIR.nii.gz",
        "sub-000/anat/sub-000_FLAIR.json",
        "sub-000/anat/sub-000_T1w.nii.gz",
        "sub-000/anat/sub-000_T1w.json",
        "derivatives/manual/dataset_description.json",
        "derivatives/manual/dseg.tsv",
        "derivatives/manual/sub-000/anat/sub-000_space-FLAIR_desc-O1_dseg.nii.gz",
        "derivatives/manual/sub-000/anat/sub-000_space-FLAIR_desc-O3_dseg.nii.gz",
        "derivatives/challenge/sub-000/anat/sub-000_space-FLAIR_T1w.nii.gz",
        "derivatives/challenge/sub-000/anat/sub-000_from-T1w_to-FLAIR_mode-image_xfm.txt",
        "derivatives/challenge/sub-000/anat/sub-000_space-T1w_desc-deface_mask.nii.gz",
        "derivatives/challenge/sub-000/anat/sub-000_desc-spm12_FLAIR.nii.gz",
        "derivatives/challenge/sub-000/anat/sub-000_space-FLAIR_desc-spm12_T1w.nii.gz",
        "derivatives/challenge/sub-100/anat/sub-100_space-FLAIR_desc-deface_mask.nii.gz",
        "sub-160/anat/sub-160_FLAIR.nii.gz",  # published folder with a space and a trailing dot
    ]:
        assert (b / path).is_file(), path
    # FLAIR defacing masks only where they were published (Amsterdam)
    assert not (b / "derivatives/challenge/sub-000/anat/sub-000_space-FLAIR_desc-deface_mask.nii.gz").exists()


def test_files_are_hard_links_to_raw(converted):
    _, raw, b, _ = converted
    src = raw / "training/Utrecht/0/orig/FLAIR.nii.gz"
    dst = b / "sub-000/anat/sub-000_FLAIR.nii.gz"
    assert dst.read_bytes() == src.read_bytes()
    assert os.stat(dst).st_nlink >= 2  # same file on disk, no copy


def test_sidecar_units_and_metadata(converted):
    cfg, _, b, _ = converted
    meta = json.loads((b / "sub-100/anat/sub-100_FLAIR.json").read_text(encoding="utf-8"))
    proto = cfg["scanners"]["amsterdam_ge_3t"]["protocol"]["flair"]
    assert meta["RepetitionTime"] == pytest.approx(proto["tr_ms"] / 1000)  # BIDS: seconds
    assert meta["InversionTime"] == pytest.approx(proto["ti_ms"] / 1000)
    assert meta["MagneticFieldStrength"] == 3.0 and meta["Manufacturer"] == "GE"
    assert "resampled" in meta["ProcessingNote"]  # 3D FLAIR resampled by the organizers
    t1 = json.loads((b / "sub-000/anat/sub-000_T1w.json").read_text(encoding="utf-8"))
    assert "Face removed" in t1["ProcessingNote"]


def test_participants(converted):
    _, _, b, _ = converted
    p = pd.read_csv(b / "participants.tsv", sep="\t")
    assert list(p.participant_id) == ["sub-000", "sub-100", "sub-160"]
    assert list(p.scanner) == ["utrecht_philips_3t", "amsterdam_ge_3t", "amsterdam_philips_3t"]
    assert list(p.seen_in_training) == [True, True, False]
    described = json.loads((b / "participants.json").read_text(encoding="utf-8"))
    assert set(described) == set(p.columns)


def test_dseg_lookup_table(converted):
    _, _, b, _ = converted
    with open(b / "derivatives/manual/dseg.tsv", encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    assert rows == [["index", "name"], ["0", "background"], ["1", "WMH"], ["2", "other_pathology"]]


def test_inventory_checks(converted):
    _, _, _, inv = converted
    inv = inv.set_index("participant_id")
    assert inv.qform_sform_agree.all() and inv.O1_valid.all() and inv.O1_same_grid.all()
    # 8 voxels of 1 x 1 x 3 mm = 0.024 ml; the inventory rounds volumes to 0.01 ml
    assert inv.loc["sub-000", "O1_wmh_ml"] == pytest.approx(8 * 3 / 1000, abs=0.005)
    assert inv.loc["sub-000", "O3_valid"] and inv.loc["sub-000", "O3_same_grid"]
    assert not inv.loc["sub-000", "flair_defaced"]
    amsterdam = inv.loc[["sub-100", "sub-160"]]
    assert amsterdam.flair_defaced.all() and amsterdam.flair_mask_same_grid.all()
    assert amsterdam.flair_zero_where_defaced.all()
    # one row of 6 removed, one interpolated border row (fractions rounded in the inventory)
    assert amsterdam.flair_defaced_fraction.iloc[0] == pytest.approx(1 / 6, abs=1e-3)
    assert amsterdam.flair_mask_partial_fraction.iloc[0] == pytest.approx(1 / 6, abs=1e-3)


def test_inventory_flags_a_label_map_off_grid(fake_project):
    cfg, raw = fake_project
    path = raw / "training/Utrecht/0/wmh.nii.gz"
    data = np.asanyarray(nib.load(path).dataobj)
    shifted = FLAIR_AFFINE.copy()
    shifted[0, 3] += 1.0
    nib.save(nib.Nifti1Image(data, shifted), path)  # ground truth no longer on the FLAIR grid
    inv = pd.read_csv(bids.convert(cfg)).set_index("participant_id")
    assert not inv.loc["sub-000", "O1_same_grid"]


def test_inventory_flags_an_invalid_label_map(fake_project):
    cfg, raw = fake_project
    path = raw / "training/Utrecht/0/wmh.nii.gz"
    img = nib.load(path)
    nib.save(nib.Nifti1Image(np.asanyarray(img.dataobj) * 0.5, img.affine), path)  # "interpolated"
    inv = pd.read_csv(bids.convert(cfg)).set_index("participant_id")
    assert not inv.loc["sub-000", "O1_valid"]


def test_conversion_is_idempotent_and_prunes_stale_files(converted):
    cfg, _, b, _ = converted
    stale = b / "derivatives/challenge/sub-000/anat/sub-000_desc-deface_mask.nii.gz"  # old naming
    stale.write_bytes(b"old")
    bids.convert(cfg)
    assert not stale.exists()
    assert (b / "sub-000/anat/sub-000_FLAIR.nii.gz").is_file()


def test_incomplete_subject_is_skipped(fake_project, caplog):
    cfg, raw = fake_project
    manifest = raw / "manifest.csv"
    rows = [
        r
        for r in csv.DictReader(open(manifest, encoding="utf-8"))
        if r["remote_path"] != "training/Amsterdam/GE3T/100/wmh.nii.gz"
    ]
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    inv = pd.read_csv(bids.convert(cfg))
    assert list(inv.participant_id) == ["sub-000", "sub-160"]
    assert "Subject 100 skipped" in caplog.text


def test_unknown_site_is_an_error(fake_project):
    cfg, raw = fake_project
    rows = list(csv.DictReader(open(raw / "manifest.csv", encoding="utf-8")))
    rows[0]["remote_path"] = rows[0]["remote_path"].replace("Utrecht", "Paris")
    with pytest.raises(ValueError, match="Unknown site"):
        bids.collect_subjects(rows, raw, cfg["scanners"])


def test_failed_downloads_are_ignored(fake_project):
    cfg, raw = fake_project
    rows = list(csv.DictReader(open(raw / "manifest.csv", encoding="utf-8")))
    for r in rows:
        if r["remote_path"].startswith("test/"):
            r["status"] = "failed"
    subjects = bids.collect_subjects(rows, raw, cfg["scanners"])
    assert [s.original_id for s in subjects] == [0, 100]


def test_missing_manifest(tmp_path, fake_project):
    cfg, raw = fake_project
    (raw / "manifest.csv").unlink()
    with pytest.raises(FileNotFoundError, match="manifest"):
        bids.convert(cfg)
