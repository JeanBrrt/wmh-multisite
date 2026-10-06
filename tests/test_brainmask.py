"""Tests of the HD-BET step without a GPU: the worker run is replaced by a fake that writes masks.

What is tested: mask checks before they enter the derivatives, batching and resumption, failure
handling, the worker command, the carry onto the FLAIR grid and the QC table.
"""

import json
from pathlib import Path

import ants
import numpy as np
import pytest

from conftest import FLAIR_AFFINE, FLAIR_SHAPE, T1_AFFINE, T1_SHAPE, write_nifti
from wmh_multisite.data import bids
from wmh_multisite.preproc import brainmask
from wmh_multisite.utils import io
from wmh_multisite.utils.guard import GuardResult


@pytest.fixture
def cfg(fake_project):
    cfg, _ = fake_project
    bids.convert(cfg)
    return cfg


def _brain(shape=T1_SHAPE):
    m = np.zeros(shape, np.uint8)
    m[2:6, 2:6, 2:6] = 1  # 64 voxels of 1 mm3
    return m


def _fake_worker(calls, mask=None, affine=T1_AFFINE, fail=False):
    """Replacement for run_guarded: reads job.json and writes one mask per pair."""

    def run(cmd, **kwargs):
        job = json.loads(Path(cmd[-1]).read_text())
        calls.append([Path(src).name for src, _ in job["pairs"]])
        if not fail:
            for _, dst in job["pairs"]:
                write_nifti(Path(dst), _brain() if mask is None else mask, affine)
        Path(kwargs["log_path"]).write_text("fake worker log")
        return GuardResult(1 if fail else 0, 0.1, 1.0, 1.0, None)

    return run


def test_worker_command_uses_the_temporary_cuda_environment(cfg, tmp_path):
    cmd = brainmask.worker_command(cfg, tmp_path / "job.json")
    assert cmd[:3] == ["uv", "run", "--no-project"]
    assert "torch==2.14.1+cu130" in cmd and "hd-bet==2.0.1" in cmd
    assert cmd[-2].endswith("hdbet_worker.py") and cmd[-1].endswith("job.json")


def test_run_writes_checked_masks_in_batches_and_resumes(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(brainmask, "run_guarded", _fake_worker(calls))
    cfg["preproc"]["brainmask"]["batch_size"] = 2
    subjects = ["sub-000", "sub-100", "sub-160"]
    brainmask.run_hdbet(cfg, subjects)
    assert [len(c) for c in calls] == [2, 1]  # 3 subjects, batches of 2
    for s in subjects:
        p = brainmask.mask_paths(cfg, s)
        assert io.same_grid(p["brain_t1"], p["t1w"])
        assert io.describe(p["brain_t1"])["dtype_on_disk"] == "uint8"
    brainmask.run_hdbet(cfg, subjects)  # second run: nothing left to do
    assert len(calls) == 2


def test_a_mask_on_another_grid_is_refused(cfg, monkeypatch):
    monkeypatch.setattr(brainmask, "run_guarded", _fake_worker([], _brain(FLAIR_SHAPE), FLAIR_AFFINE))
    with pytest.raises(ValueError, match="grid"):
        brainmask.run_hdbet(cfg, ["sub-000"])
    assert not brainmask.mask_paths(cfg, "sub-000")["brain_t1"].exists()


def test_a_non_binary_or_empty_mask_is_refused(cfg, monkeypatch):
    monkeypatch.setattr(brainmask, "run_guarded", _fake_worker([], _brain().astype(np.float32) * 0.6))
    with pytest.raises(ValueError):
        brainmask.run_hdbet(cfg, ["sub-000"])
    monkeypatch.setattr(brainmask, "run_guarded", _fake_worker([], np.zeros(T1_SHAPE, np.uint8)))
    with pytest.raises(ValueError, match="empty"):
        brainmask.run_hdbet(cfg, ["sub-000"])


def test_a_failed_batch_stops_with_the_worker_log(cfg, monkeypatch):
    monkeypatch.setattr(brainmask, "run_guarded", _fake_worker([], fail=True))
    with pytest.raises(RuntimeError, match="fake worker log"):
        brainmask.run_hdbet(cfg, ["sub-000"])


def test_carry_onto_the_flair_grid_and_qc(cfg, monkeypatch):
    monkeypatch.setattr(brainmask, "run_guarded", _fake_worker([]))
    brainmask.run_hdbet(cfg, ["sub-000"])
    p = brainmask.mask_paths(cfg, "sub-000")
    assert brainmask.to_flair_grid(cfg, "sub-000") is None  # not registered yet
    p["xfm"].parent.mkdir(parents=True, exist_ok=True)
    ants.write_transform(ants.new_ants_transform(dimension=3), str(p["xfm"]))  # identity
    out = brainmask.to_flair_grid(cfg, "sub-000")
    assert io.same_grid(out, p["flair"])
    assert set(np.unique(io.load_labels(out, allowed=(0, 1)))) <= {0, 1}
    row = brainmask.qc_row(cfg, "sub-000")
    assert row["brain_ml"] == pytest.approx(0.064, abs=0.05)  # volumes are rounded to 0.1 ml
    assert row["n_components"] == 1
    assert row["largest_component_frac"] == 1.0
    # identity transform, 1 mm -> 1 x 1 x 3 mm grid: volume preserved up to the coarser sampling
    assert 0.5 < row["flair_coverage"] < 1.5
