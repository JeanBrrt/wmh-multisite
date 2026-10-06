"""Download the MICCAI WMH 2017 dataset from DataverseNL (ROADMAP step 2.1).

The dataset is published file by file (doi:10.34894/AECRSD). This module:
1. lists every file through the Dataverse API (path, size, SHA-1 checksum);
2. keeps only the files listed in config.yaml (``dataset.keep``, observers, extra files);
3. downloads them in parallel into ``data/raw/``, mirroring the published tree;
4. verifies each file against its published SHA-1 (a truncated or corrupted download is retried);
5. writes ``data/raw/manifest.csv`` (one row per file: paths, size, checksum, status).

Re-running is cheap: files already present with the right checksum are skipped.

Usage:
    uv run python -m wmh_multisite.data.download --split training   # 60 training subjects first
    uv run python -m wmh_multisite.data.download                     # everything (~5 GB)
    uv run python -m wmh_multisite.data.download --dry-run           # list what would be downloaded
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
import re
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from wmh_multisite.config import load_config, resolve_path

log = logging.getLogger(__name__)

CHUNK_BYTES = 1 << 20  # 1 MiB
SPLITS = ("training", "test")
MANIFEST_FIELDS = [
    "remote_path",
    "local_path",
    "file_id",
    "size",
    "sha1",
    "status",
    "dataset_version",
    "checked_at",
]

# training/<site>[/<scanner>]/<subject id>/<file inside the subject folder>
SUBJECT_FILE = re.compile(r"^(?P<split>training|test)/(?P<site>.+)/(?P<subject>\d+)/(?P<rel>.+)$")
# additional_annotations/<observer>/<split>/<site>[/<scanner>]/<subject id>/result.nii.gz
OBSERVER_FILE = re.compile(
    r"^additional_annotations/(?P<observer>observer_o\d)/(?P<split>training|test)/.+/\d+/result\.nii\.gz$"
)


def sanitize(component: str) -> str:
    """Make one published folder/file name safe on every operating system.

    Windows silently drops trailing dots and spaces from names, so the published folder
    "Philips_VU .PETMR_01." would be written as "Philips_VU .PETMR_01" and the mirror would no
    longer match the remote tree. Spaces become "_", and a trailing dot becomes "_":
    "Philips_VU .PETMR_01." -> "Philips_VU_.PETMR_01_". Other names are unchanged.
    """
    clean = component.replace(" ", "_")
    if clean.endswith("."):
        clean = clean[:-1] + "_"
    return clean


@dataclass(frozen=True)
class RemoteFile:
    file_id: int
    remote_path: str  # path inside the published dataset, e.g. "training/Utrecht/0/orig/FLAIR.nii.gz"
    size: int
    sha1: str

    @property
    def local_path(self) -> Path:
        """Path relative to data/raw/, with every component sanitized."""
        return Path(*(sanitize(part) for part in self.remote_path.split("/")))


def list_remote_files(api_url: str, session: requests.Session) -> tuple[str, list[RemoteFile]]:
    """Return (dataset version, every published file) from the Dataverse API."""
    response = session.get(api_url, timeout=60)
    response.raise_for_status()
    latest = response.json()["data"]["latestVersion"]
    version = f"{latest['versionNumber']}.{latest['versionMinorNumber']}"
    files = []
    for entry in latest["files"]:
        data_file = entry["dataFile"]
        checksum = data_file.get("checksum") or {}
        if checksum.get("type") != "SHA-1":
            raise ValueError(f"Unexpected checksum {checksum} for {data_file['filename']}")
        remote_path = "/".join(part for part in (entry.get("directoryLabel"), data_file["filename"]) if part)
        files.append(
            RemoteFile(int(data_file["id"]), remote_path, int(data_file["filesize"]), checksum["value"])
        )
    return version, files


def select_files(
    files: Iterable[RemoteFile],
    keep: Iterable[str],
    observers: Iterable[str] = (),
    extra_files: Iterable[str] = (),
    splits: Iterable[str] = SPLITS,
) -> list[RemoteFile]:
    """Keep the files the pipeline needs.

    - subject files whose path inside the subject folder is in ``keep`` (e.g. "orig/FLAIR.nii.gz");
    - annotations of the listed ``observers``;
    - top-level ``extra_files`` (e.g. "readme.pdf").
    Only subjects of the requested ``splits`` are kept.
    """
    keep, observers, extra_files, splits = set(keep), set(observers), set(extra_files), set(splits)
    selected = []
    for f in files:
        subject = SUBJECT_FILE.match(f.remote_path)
        observer = OBSERVER_FILE.match(f.remote_path)
        if (
            f.remote_path in extra_files
            or (subject and subject["split"] in splits and subject["rel"] in keep)
            or (observer and observer["observer"] in observers and observer["split"] in splits)
        ):
            selected.append(f)
    return sorted(selected, key=lambda f: f.remote_path)


def sha1_of(path: Path) -> str:
    """SHA-1 of a file, read by chunks (constant memory)."""
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


_thread_local = threading.local()


def _session() -> requests.Session:
    """One HTTP session per worker thread (requests.Session is not guaranteed thread-safe)."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def download_file(
    rf: RemoteFile, raw_dir: Path, file_url: str, retries: int = 3, verify_existing: bool = True
) -> str:
    """Download one file into ``raw_dir`` and check its SHA-1. Returns "present" or "downloaded".

    The file is streamed to "<name>.part", hashed on the fly, and renamed only if the checksum
    matches: an interrupted run never leaves a truncated file under the final name.
    """
    dest = raw_dir / rf.local_path
    complete = dest.is_file() and dest.stat().st_size == rf.size
    if complete and (not verify_existing or sha1_of(dest) == rf.sha1):
        return "present"

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            digest = hashlib.sha1()
            with _session().get(file_url.format(file_id=rf.file_id), stream=True, timeout=(15, 120)) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(CHUNK_BYTES):
                        fh.write(chunk)
                        digest.update(chunk)
            if digest.hexdigest() != rf.sha1:
                raise OSError(f"SHA-1 mismatch (got {digest.hexdigest()}, expected {rf.sha1})")
            os.replace(tmp, dest)  # atomic rename
            return "downloaded"
        except (requests.RequestException, OSError) as err:
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"{rf.remote_path}: failed after {retries} attempts ({err})") from err
            log.warning("%s: attempt %d/%d failed (%s), retrying", rf.remote_path, attempt, retries, err)
            time.sleep(2**attempt)  # 2 s, 4 s, ... between attempts
    raise AssertionError("unreachable")


def _read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["remote_path"]: row for row in csv.DictReader(fh)}


def _write_manifest(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    os.replace(tmp, path)


def download_dataset(
    cfg: dict[str, Any],
    splits: Iterable[str] = SPLITS,
    workers: int | None = None,
    verify_existing: bool = True,
    dry_run: bool = False,
) -> Path | None:
    """Download the selected files; return the manifest path (None for a dry run).

    Failures do not stop the other downloads: they are recorded in the manifest and reported at
    the end with a RuntimeError, so a re-run only fetches what is missing.
    """
    ds = cfg["dataset"]
    raw_dir = resolve_path(cfg, "raw")
    workers = workers or ds["download"]["workers"]

    with requests.Session() as session:
        version, remote = list_remote_files(ds["api_url"], session)
    selected = select_files(remote, ds["keep"], ds["additional_observers"], ds["extra_files"], splits)
    total = sum(f.size for f in selected)
    log.info(
        "Dataset version %s: %d published files, %d selected (%.2f GB) for splits %s",
        version,
        len(remote),
        len(selected),
        total / 1e9,
        sorted(set(splits)),
    )
    if dry_run:
        return None

    manifest_path = raw_dir / "manifest.csv"
    rows = _read_manifest(manifest_path)
    failures = []
    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        tqdm(total=total, unit="B", unit_scale=True, desc="WMH dataset") as bar,
    ):
        futures = {
            pool.submit(
                download_file, f, raw_dir, ds["file_url"], ds["download"]["retries"], verify_existing
            ): f
            for f in selected
        }
        for future in as_completed(futures):
            rf = futures[future]
            try:
                status = future.result()
            except RuntimeError as err:
                status = "failed"
                failures.append(str(err))
            rows[rf.remote_path] = {
                "remote_path": rf.remote_path,
                "local_path": rf.local_path.as_posix(),
                "file_id": rf.file_id,
                "size": rf.size,
                "sha1": rf.sha1,
                "status": status,
                "dataset_version": version,
                "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            bar.update(rf.size)
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path, rows)

    counts = {s: sum(r["status"] == s for r in rows.values()) for s in ("present", "downloaded", "failed")}
    log.info("Manifest %s: %s", manifest_path, counts)
    if failures:
        raise RuntimeError(f"{len(failures)} file(s) failed, re-run to retry:\n" + "\n".join(failures))
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the MICCAI WMH 2017 dataset (DataverseNL).")
    parser.add_argument("--config", help="path to config.yaml (default: config/config.yaml)")
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--workers", type=int, help="parallel downloads (default: from config)")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="trust existing files of the right size instead of re-hashing them",
    )
    parser.add_argument("--dry-run", action="store_true", help="only list what would be downloaded")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    splits = SPLITS if args.split == "all" else (args.split,)
    download_dataset(load_config(args.config), splits, args.workers, not args.no_verify, args.dry_run)


if __name__ == "__main__":
    main()
