"""Shared fixtures: tiny synthetic NIfTI images and a fake copy of the challenge tree.

No test needs the real dataset or the network: every image is a few voxels built with numpy,
with a geometry chosen so that the expected answer is known in advance.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from wmh_multisite.config import load_config
from wmh_multisite.data.download import MANIFEST_FIELDS, RemoteFile

# FLAIR-like grid: 1 x 1 x 3 mm voxels, small oblique-free affine
FLAIR_AFFINE = np.diag([1.0, 1.0, 3.0, 1.0])
FLAIR_SHAPE = (6, 6, 4)
# T1-like grid: 1 mm isotropic, shifted origin
T1_AFFINE = np.array([[1.0, 0, 0, -1], [0, 1.0, 0, -1], [0, 0, 1.0, -2], [0, 0, 0, 1]])
T1_SHAPE = (8, 8, 8)


def write_nifti(path: Path, data: np.ndarray, affine: np.ndarray = FLAIR_AFFINE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), affine), path)
    return path


@pytest.fixture
def nifti(tmp_path):
    """Factory: nifti(data, name="img.nii.gz", affine=FLAIR_AFFINE) -> path of a written NIfTI file."""

    def _make(data, name="img.nii.gz", affine=FLAIR_AFFINE):
        return write_nifti(tmp_path / name, data, affine)

    return _make


@pytest.fixture
def labels():
    """A label map on the FLAIR grid: a 2x2x2 'lesion' (label 1) and a 'other pathology' block (label 2)."""
    lab = np.zeros(FLAIR_SHAPE, np.float32)  # float32 on purpose: like the challenge files
    lab[1:3, 1:3, 1:3] = 1
    lab[4:6, 4:6, 0:2] = 2
    return lab


# ---------------------------------------------------------------------------------------------
# Fake challenge tree for the BIDS conversion
# ---------------------------------------------------------------------------------------------

FAKE_SUBJECTS = [
    # (split, published site folder, subject id, has FLAIR_mask, has observers)
    ("training", "Utrecht", 0, False, True),
    ("training", "Amsterdam/GE3T", 100, True, False),
    ("test", "Amsterdam/Philips_VU .PETMR_01.", 160, True, False),  # name that Windows cannot store as is
]


def _fake_subject_files(rng: np.random.Generator, has_flair_mask: bool) -> dict[str, object]:
    flair = rng.integers(50, 1000, FLAIR_SHAPE).astype(np.uint16)
    files: dict[str, object] = {
        "orig/FLAIR.nii.gz": (flair, FLAIR_AFFINE),
        "orig/3DT1.nii.gz": (rng.integers(0, 1500, T1_SHAPE).astype(np.uint16), T1_AFFINE),
        "orig/3DT1_mask.nii.gz": (np.ones(T1_SHAPE, np.float32), T1_AFFINE),
        "orig/T1.nii.gz": (rng.integers(0, 1500, FLAIR_SHAPE).astype(np.uint16), FLAIR_AFFINE),
        "orig/reg_3DT1_to_FLAIR.txt": '(Transform "EulerTransform")\n',
        "pre/FLAIR.nii.gz": ((flair * 0.9).astype(np.float32), FLAIR_AFFINE),
        "pre/T1.nii.gz": (rng.integers(0, 1500, FLAIR_SHAPE).astype(np.uint16), FLAIR_AFFINE),
    }
    lab = np.zeros(FLAIR_SHAPE, np.float32)
    lab[1:3, 1:3, 1:3] = 1
    files["wmh.nii.gz"] = (lab, FLAIR_AFFINE)
    if has_flair_mask:
        # Like the published Amsterdam masks: 1 kept, 0 defaced, interpolated values on the border,
        # and the FLAIR exactly 0 where the mask is 0.
        mask = np.ones(FLAIR_SHAPE, np.float32)
        mask[:, 0, :] = 0
        mask[:, 1, :] = 0.6
        flair[mask == 0] = 0
        files["orig/FLAIR.nii.gz"] = (flair, FLAIR_AFFINE)
        files["pre/FLAIR.nii.gz"] = ((flair * 0.9).astype(np.float32), FLAIR_AFFINE)
        files["orig/FLAIR_mask.nii.gz"] = (mask, FLAIR_AFFINE)
    return files


def _write_any(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        path.write_text(content)
    else:
        data, affine = content
        nib.save(nib.Nifti1Image(data, affine), path)


@pytest.fixture
def fake_project(tmp_path):
    """A project root with data/raw (challenge layout + manifest.csv) and a config pointing at it.

    Returns (cfg, raw_dir). Three subjects: Utrecht 0 (with observer O3), Amsterdam GE3T 100 and
    Amsterdam Philips 160 (defaced FLAIR with a non-binary mask).
    """
    cfg = load_config()
    cfg["root"] = tmp_path
    raw = tmp_path / cfg["paths"]["raw"]
    rng = np.random.default_rng(0)
    rows = []

    def add(remote_path: str, content: object) -> None:
        rf = RemoteFile(len(rows) + 1, remote_path, 0, "")
        local = raw / rf.local_path
        _write_any(local, content)
        rows.append(
            {
                "remote_path": remote_path,
                "local_path": rf.local_path.as_posix(),
                "file_id": rf.file_id,
                "size": local.stat().st_size,
                "sha1": hashlib.sha1(local.read_bytes()).hexdigest(),
                "status": "downloaded",
                "dataset_version": "1.0",
                "checked_at": "2026-10-03T00:00:00+00:00",
            }
        )

    for split, site, sid, has_mask, has_obs in FAKE_SUBJECTS:
        for rel, content in _fake_subject_files(rng, has_mask).items():
            add(f"{split}/{site}/{sid}/{rel}", content)
        if has_obs:
            obs = np.zeros(FLAIR_SHAPE, np.float32)
            obs[1:3, 1:3, 1:2] = 1
            add(f"additional_annotations/observer_o3/{split}/{site}/{sid}/result.nii.gz", (obs, FLAIR_AFFINE))

    with open(raw / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return cfg, raw
