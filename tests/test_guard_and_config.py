"""utils/guard.py (memory watchdog) and config.py (configuration loading)."""

import sys
from pathlib import Path

import pytest

from wmh_multisite import config
from wmh_multisite.utils.guard import call_guarded, run_guarded

# --- guard: the watched jobs allocate a few hundred MB at most ---------------------------------


def test_call_guarded_returns_the_value():
    r = call_guarded(sum, [1, 2, 3], ram_limit_gb=1.0, poll_s=0.1)
    assert r.ok and r.value == 6 and r.killed is None


def test_call_guarded_reports_the_exception():
    r = call_guarded(int, "abc", ram_limit_gb=1.0, poll_s=0.1)
    assert not r.ok and "ValueError" in r.error


def test_call_guarded_kills_above_the_ram_limit():
    r = call_guarded(bytearray, 600_000_000, ram_limit_gb=0.2, poll_s=0.05)
    assert r.killed is not None and r.killed.startswith("RAM")
    assert not r.ok


def test_run_guarded_command():
    assert run_guarded([sys.executable, "-c", "print(1)"], ram_limit_gb=1.0, poll_s=0.1).ok
    big = run_guarded(
        [sys.executable, "-c", "x = bytearray(600_000_000); import time; time.sleep(5)"],
        ram_limit_gb=0.2,
        poll_s=0.1,
    )
    assert big.killed is not None and not big.ok


def test_run_guarded_timeout():
    r = run_guarded([sys.executable, "-c", "import time; time.sleep(10)"], timeout_s=0.5, poll_s=0.1)
    assert r.killed and "timeout" in r.killed


# --- config ----------------------------------------------------------------------------------


def test_load_config_from_repository():
    cfg = config.load_config()
    assert (cfg["root"] / "pyproject.toml").is_file()
    assert sum(s["n_train"] for s in cfg["scanners"].values()) == 60
    assert sum(s["n_test"] for s in cfg["scanners"].values()) == 110
    assert sum(not s["seen_in_training"] for s in cfg["scanners"].values()) == 2


def test_resolve_path_is_relative_to_root(tmp_path):
    cfg = {"root": tmp_path, "paths": {"raw": "data/raw"}}
    assert config.resolve_path(cfg, "raw") == tmp_path / "data" / "raw"


def test_project_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("WMH_ROOT", str(tmp_path))
    assert config.project_root() == tmp_path.resolve()


def test_every_scanner_has_a_protocol():
    for key, scanner in config.load_config()["scanners"].items():
        for modality in ("t1w", "flair"):
            proto = scanner["protocol"][modality]
            assert {"acquisition", "voxel_mm", "tr_ms", "te_ms"} <= proto.keys(), (key, modality)


@pytest.mark.parametrize("key", ["raw", "bids", "derivatives", "results"])
def test_paths_are_relative(key):
    assert not config.load_config()["paths"][key].startswith(("/", "C:"))


def test_rss_of_a_finished_process_is_zero_not_an_error():
    """Race fixed on 2026-10-05: a fast child can exit between two psutil calls."""
    import subprocess

    import psutil

    from wmh_multisite.utils.guard import _tree_rss_gb

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    proc = psutil.Process(child.pid)
    child.wait()
    assert _tree_rss_gb(proc) == 0.0


def test_registration_temp_files_are_removed_only_in_the_temp_folder(tmp_path):
    import os
    import tempfile

    from wmh_multisite.preproc.register import remove_registration_files

    fd, name = tempfile.mkstemp(suffix="Warp.nii.gz")
    os.close(fd)  # Windows cannot delete an open file
    in_tmp = Path(name)
    kept = tmp_path / "kept_GenericAffine.mat"
    kept.write_text("x")
    n = remove_registration_files({"fwdtransforms": [str(in_tmp), str(kept)], "invtransforms": [str(in_tmp)]})
    assert n == 1 and not in_tmp.exists() and kept.exists()
