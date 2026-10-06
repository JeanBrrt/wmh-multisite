"""data/download.py: selection, checksum and download logic, with a fake HTTP server (no network)."""

import csv
import hashlib
from pathlib import Path

import pytest
import requests

from wmh_multisite.data import download as dl
from wmh_multisite.data.download import OBSERVER_FILE, SUBJECT_FILE, RemoteFile


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


# --- pure helpers ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Philips_VU .PETMR_01.", "Philips_VU_.PETMR_01_"),  # space and trailing dot (dropped by Windows)
        ("GE3T", "GE3T"),
        ("FLAIR.nii.gz", "FLAIR.nii.gz"),  # dots inside a name are kept
    ],
)
def test_sanitize(name, expected):
    assert dl.sanitize(name) == expected


def test_local_path_is_sanitized():
    rf = RemoteFile(1, "test/Amsterdam/Philips_VU .PETMR_01./160/orig/FLAIR.nii.gz", 0, "")
    assert rf.local_path.as_posix() == "test/Amsterdam/Philips_VU_.PETMR_01_/160/orig/FLAIR.nii.gz"


@pytest.mark.parametrize(
    "path, site, subject, rel",
    [
        ("training/Utrecht/0/wmh.nii.gz", "Utrecht", "0", "wmh.nii.gz"),
        ("training/Amsterdam/GE3T/100/orig/FLAIR.nii.gz", "Amsterdam/GE3T", "100", "orig/FLAIR.nii.gz"),
        (
            "test/Amsterdam/Philips_VU .PETMR_01./160/pre/T1.nii.gz",
            "Amsterdam/Philips_VU .PETMR_01.",
            "160",
            "pre/T1.nii.gz",
        ),
    ],
)
def test_subject_file_regex(path, site, subject, rel):
    m = SUBJECT_FILE.match(path)
    assert (m["site"], m["subject"], m["rel"]) == (site, subject, rel)
    assert OBSERVER_FILE.match(path) is None


def test_observer_file_regex():
    m = OBSERVER_FILE.match("additional_annotations/observer_o4/training/Amsterdam/GE3T/101/result.nii.gz")
    assert (m["observer"], m["split"]) == ("observer_o4", "training")
    assert SUBJECT_FILE.match("additional_annotations/observer_o4/training/Utrecht/0/result.nii.gz") is None


REMOTE = [
    RemoteFile(i, p, 10, "x")
    for i, p in enumerate(
        [
            "readme.pdf",
            "training/Utrecht/0/orig/FLAIR.nii.gz",
            "training/Utrecht/0/orig/T1.nii.gz",
            "training/Utrecht/0/pre/3DT1.nii.gz",  # not in keep (J-015)
            "training/Utrecht/0/wmh.nii.gz",
            "test/Amsterdam/GE1T5/150/orig/FLAIR.nii.gz",
            "test/Amsterdam/GE1T5/150/orig/FLAIR_mask.nii.gz",
            "additional_annotations/observer_o3/training/Utrecht/0/result.nii.gz",
            "additional_annotations/observer_o9/training/Utrecht/0/result.nii.gz",  # observer not requested
        ]
    )
]
KEEP = ["orig/FLAIR.nii.gz", "orig/T1.nii.gz", "orig/FLAIR_mask.nii.gz", "wmh.nii.gz"]


def test_select_files_all_splits():
    paths = [f.remote_path for f in dl.select_files(REMOTE, KEEP, ["observer_o3"], ["readme.pdf"])]
    assert "training/Utrecht/0/pre/3DT1.nii.gz" not in paths
    assert "additional_annotations/observer_o9/training/Utrecht/0/result.nii.gz" not in paths
    assert "training/Utrecht/0/orig/T1.nii.gz" in paths  # orig/T1 is not confused with orig/3DT1
    assert len(paths) == 7
    assert paths == sorted(paths)  # deterministic order


def test_select_files_one_split():
    paths = [f.remote_path for f in dl.select_files(REMOTE, KEEP, ["observer_o3"], [], splits=["test"])]
    assert paths == [
        "test/Amsterdam/GE1T5/150/orig/FLAIR.nii.gz",
        "test/Amsterdam/GE1T5/150/orig/FLAIR_mask.nii.gz",
    ]


def test_sha1_of(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc" * 1_000_000)
    assert dl.sha1_of(path) == sha1(b"abc" * 1_000_000)


# --- fake HTTP --------------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, payload=b"", json_data=None, status=200):
        self.payload, self.json_data, self.status = payload, json_data, status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self.json_data

    def iter_content(self, chunk_size):
        for i in range(0, len(self.payload), 4):  # tiny chunks: hashing must work across chunks
            yield self.payload[i : i + 4]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Serves the responses given in order; records the calls."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    def get(self, url, **kwargs):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)


def use_session(monkeypatch, session):
    monkeypatch.setattr(dl, "_session", lambda: session)
    return session


def test_list_remote_files():
    api = {
        "data": {
            "latestVersion": {
                "versionNumber": 1,
                "versionMinorNumber": 0,
                "files": [
                    {
                        "directoryLabel": "training/Utrecht/0/orig",
                        "dataFile": {
                            "id": 7,
                            "filename": "FLAIR.nii.gz",
                            "filesize": 42,
                            "checksum": {"type": "SHA-1", "value": "ab"},
                        },
                    },
                    {
                        "dataFile": {
                            "id": 8,
                            "filename": "readme.pdf",
                            "filesize": 1,
                            "checksum": {"type": "SHA-1", "value": "cd"},
                        }
                    },
                ],
            }
        }
    }
    version, files = dl.list_remote_files("url", FakeSession(FakeResponse(json_data=api)))
    assert version == "1.0"
    assert files == [
        RemoteFile(7, "training/Utrecht/0/orig/FLAIR.nii.gz", 42, "ab"),
        RemoteFile(8, "readme.pdf", 1, "cd"),
    ]


def test_list_remote_files_rejects_unknown_checksum():
    api = {
        "data": {
            "latestVersion": {
                "versionNumber": 1,
                "versionMinorNumber": 0,
                "files": [
                    {
                        "dataFile": {
                            "id": 1,
                            "filename": "x",
                            "filesize": 1,
                            "checksum": {"type": "MD5", "value": "ab"},
                        }
                    }
                ],
            }
        }
    }
    with pytest.raises(ValueError, match="Unexpected checksum"):
        dl.list_remote_files("url", FakeSession(FakeResponse(json_data=api)))


def test_download_file_success(tmp_path, monkeypatch):
    data = b"some nifti bytes"
    rf = RemoteFile(1, "training/Utrecht/0/wmh.nii.gz", len(data), sha1(data))
    session = use_session(monkeypatch, FakeSession(FakeResponse(data)))
    assert dl.download_file(rf, tmp_path, "{file_id}") == "downloaded"
    assert (tmp_path / rf.local_path).read_bytes() == data
    assert session.calls == 1
    assert not list(tmp_path.rglob("*.part"))


def test_download_file_present_is_skipped(tmp_path, monkeypatch):
    data = b"already here"
    rf = RemoteFile(1, "readme.pdf", len(data), sha1(data))
    (tmp_path / "readme.pdf").write_bytes(data)
    session = use_session(monkeypatch, FakeSession())
    assert dl.download_file(rf, tmp_path, "{file_id}") == "present"
    assert session.calls == 0


def test_download_file_present_but_corrupted_is_downloaded_again(tmp_path, monkeypatch):
    data = b"good content"
    rf = RemoteFile(1, "readme.pdf", len(data), sha1(data))
    (tmp_path / "readme.pdf").write_bytes(b"bad content!")  # same size, wrong checksum
    use_session(monkeypatch, FakeSession(FakeResponse(data)))
    assert dl.download_file(rf, tmp_path, "{file_id}") == "downloaded"
    assert (tmp_path / "readme.pdf").read_bytes() == data


def test_download_file_retries_then_succeeds(tmp_path, monkeypatch, no_sleep):
    data = b"payload"
    rf = RemoteFile(1, "readme.pdf", len(data), sha1(data))
    session = use_session(monkeypatch, FakeSession(requests.ConnectionError("reset"), FakeResponse(data)))
    assert dl.download_file(rf, tmp_path, "{file_id}", retries=3) == "downloaded"
    assert session.calls == 2


def test_download_file_checksum_mismatch_never_leaves_a_file(tmp_path, monkeypatch, no_sleep):
    rf = RemoteFile(1, "training/Utrecht/0/wmh.nii.gz", 9, sha1(b"expected!"))
    session = use_session(monkeypatch, FakeSession(*[FakeResponse(b"truncated") for _ in range(3)]))
    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        dl.download_file(rf, tmp_path, "{file_id}", retries=3)
    assert session.calls == 3
    assert not (tmp_path / rf.local_path).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_download_dataset_dry_run_and_manifest(tmp_path, monkeypatch):
    payloads = {1: b"flair", 2: b"label", 3: b"pdf"}
    remote = [
        RemoteFile(1, "training/Utrecht/0/orig/FLAIR.nii.gz", 5, sha1(payloads[1])),
        RemoteFile(2, "training/Utrecht/0/wmh.nii.gz", 5, sha1(payloads[2])),
        RemoteFile(3, "readme.pdf", 3, sha1(payloads[3])),
    ]
    monkeypatch.setattr(dl, "list_remote_files", lambda url, session: ("1.0", remote))

    class ById:  # serves the bytes of the requested file id, whatever the thread
        def get(self, url, **kw):
            return FakeResponse(payloads[int(url)])

    monkeypatch.setattr(dl, "_session", lambda: ById())

    cfg = {
        "root": tmp_path,
        "paths": {"raw": "raw"},
        "dataset": {
            "api_url": "api",
            "file_url": "{file_id}",
            "keep": ["orig/FLAIR.nii.gz", "wmh.nii.gz"],
            "additional_observers": [],
            "extra_files": ["readme.pdf"],
            "download": {"workers": 2, "retries": 1},
        },
    }

    assert dl.download_dataset(cfg, dry_run=True) is None
    assert not (tmp_path / "raw").exists()  # dry run writes nothing

    manifest = dl.download_dataset(cfg)
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    assert [r["status"] for r in rows] == ["downloaded"] * 3
    assert all((tmp_path / "raw" / Path(r["local_path"])).is_file() for r in rows)

    rows = list(csv.DictReader(open(dl.download_dataset(cfg), encoding="utf-8")))  # second run
    assert [r["status"] for r in rows] == ["present"] * 3
