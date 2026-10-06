"""Convert the downloaded WMH dataset into a BIDS dataset (ROADMAP step 2.2).

BIDS (Brain Imaging Data Structure, https://bids-specification.readthedocs.io) is the community
standard for organizing neuroimaging data. Layout produced in ``data/bids/``:

    dataset_description.json, README, participants.tsv, participants.json
    sub-000/anat/sub-000_FLAIR.nii.gz (+ .json)       raw FLAIR (orig/FLAIR)
    sub-000/anat/sub-000_T1w.nii.gz   (+ .json)       raw 3D T1, defaced (orig/3DT1)
    derivatives/manual/                               manual annotations (reference O1, observers O3/O4)
        sub-000/anat/sub-000_space-FLAIR_desc-O1_dseg.nii.gz
    derivatives/challenge/                            processing done by the challenge organizers
        sub-000/anat/sub-000_space-FLAIR_T1w.nii.gz                (3D T1 aligned with elastix)
        sub-000/anat/sub-000_from-T1w_to-FLAIR_mode-image_xfm.txt  (elastix parameters)
        sub-000/anat/sub-000_space-T1w_desc-deface_mask.nii.gz     (defacing mask of the 3D T1)
        sub-000/anat/sub-000_space-FLAIR_desc-deface_mask.nii.gz   (defacing mask of the FLAIR, Amsterdam)
        sub-000/anat/sub-000_desc-spm12_FLAIR.nii.gz               (bias-corrected FLAIR, pre/)
        sub-000/anat/sub-000_space-FLAIR_desc-spm12_T1w.nii.gz     (bias-corrected aligned T1, pre/)

Subjects keep their original challenge id (unique across sites): ``sub-{id:03d}``.
Files are hard links to ``data/raw/`` (same bytes, no extra disk space), or copies if the file
system refuses links. The conversion also writes a QC inventory (``results/tables/``): geometry
of every image, consistency of the label maps and their grids, WMH volumes.

Usage:
    uv run python -m wmh_multisite.data.bids
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wmh_multisite.config import load_config, resolve_path
from wmh_multisite.data.download import OBSERVER_FILE, SUBJECT_FILE
from wmh_multisite.utils import io

log = logging.getLogger(__name__)

# file inside the published subject folder -> (BIDS dataset relative to data/bids, file name template)
TARGETS: dict[str, tuple[str, str]] = {
    "orig/FLAIR.nii.gz": ("", "{sub}_FLAIR.nii.gz"),
    "orig/3DT1.nii.gz": ("", "{sub}_T1w.nii.gz"),
    "wmh.nii.gz": ("derivatives/manual", "{sub}_space-FLAIR_desc-O1_dseg.nii.gz"),
    "orig/T1.nii.gz": ("derivatives/challenge", "{sub}_space-FLAIR_T1w.nii.gz"),
    "orig/reg_3DT1_to_FLAIR.txt": ("derivatives/challenge", "{sub}_from-T1w_to-FLAIR_mode-image_xfm.txt"),
    "orig/3DT1_mask.nii.gz": ("derivatives/challenge", "{sub}_space-T1w_desc-deface_mask.nii.gz"),
    "orig/FLAIR_mask.nii.gz": ("derivatives/challenge", "{sub}_space-FLAIR_desc-deface_mask.nii.gz"),
    "pre/FLAIR.nii.gz": ("derivatives/challenge", "{sub}_desc-spm12_FLAIR.nii.gz"),
    "pre/T1.nii.gz": ("derivatives/challenge", "{sub}_space-FLAIR_desc-spm12_T1w.nii.gz"),
}
OBSERVER_TARGET = ("derivatives/manual", "{sub}_space-FLAIR_desc-{obs}_dseg.nii.gz")
REQUIRED = ("orig/FLAIR.nii.gz", "orig/3DT1.nii.gz", "wmh.nii.gz")
LABELS = {0: "background", 1: "WMH", 2: "other_pathology"}


@dataclass
class Subject:
    original_id: int
    split: str
    scanner: str  # key of config["scanners"]
    files: dict[str, Path] = field(default_factory=dict)  # "orig/FLAIR.nii.gz" -> local raw path
    observers: dict[str, Path] = field(default_factory=dict)  # "O3" -> local raw path

    def participant_id(self, naming: str) -> str:
        return naming.format(id=self.original_id)


def _scanner_by_site(scanners: dict[str, Any]) -> dict[str, str]:
    """Published site folder ("Amsterdam/GE3T") -> scanner key ("amsterdam_ge_3t")."""
    return {cfg["folder"]: key for key, cfg in scanners.items()}


def collect_subjects(
    manifest_rows: list[dict[str, str]], raw_dir: Path, scanners: dict[str, Any]
) -> list[Subject]:
    """Group the files of the download manifest by subject (only files downloaded successfully)."""
    by_site = _scanner_by_site(scanners)
    subjects: dict[int, Subject] = {}
    observer_rows = []
    for row in manifest_rows:
        if row["status"] == "failed":
            continue
        match = SUBJECT_FILE.match(row["remote_path"])
        if match is None:
            if OBSERVER_FILE.match(row["remote_path"]):
                observer_rows.append(row)
            continue
        if match["site"] not in by_site:
            raise ValueError(f"Unknown site folder {match['site']!r} (not in config scanners)")
        sid = int(match["subject"])
        subject = subjects.setdefault(sid, Subject(sid, match["split"], by_site[match["site"]]))
        subject.files[match["rel"]] = raw_dir / row["local_path"]

    for row in observer_rows:
        observer = OBSERVER_FILE.match(row["remote_path"])["observer"]  # "observer_o3"
        sid = int(Path(row["remote_path"]).parent.name)
        if sid in subjects:
            subjects[sid].observers[observer.removeprefix("observer_").upper()] = raw_dir / row["local_path"]

    complete = []
    for sid in sorted(subjects):
        missing = [rel for rel in REQUIRED if rel not in subjects[sid].files]
        if missing:
            log.warning("Subject %d skipped, missing %s", sid, missing)
        else:
            complete.append(subjects[sid])
    return complete


def link_or_copy(src: Path, dst: Path) -> str:
    """Hard link ``src`` to ``dst`` (no extra disk space); copy if links are not supported."""
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "present"
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sidecar(scanner: dict[str, Any], modality: str) -> dict[str, Any]:
    """BIDS JSON sidecar of a raw image, from the acquisition protocol in the config (times in s)."""
    proto = scanner["protocol"][modality]
    meta: dict[str, Any] = {
        "Manufacturer": scanner["vendor"],
        "ManufacturersModelName": scanner["model"],
        "InstitutionName": scanner["institute"],
        "MagneticFieldStrength": scanner["field_strength_t"],
        "MRAcquisitionType": proto["acquisition"],
        "RepetitionTime": proto["tr_ms"] / 1000,
        "EchoTime": proto["te_ms"] / 1000,
    }
    if "ti_ms" in proto:
        meta["InversionTime"] = proto["ti_ms"] / 1000
    meta["AcquisitionVoxelSize"] = proto["voxel_mm"]  # not a BIDS key: protocol value, not the file's
    if modality == "flair" and proto["acquisition"] == "3D":
        meta["ProcessingNote"] = (
            "Acquired 3D sagittal; reoriented to transversal and resampled to 3 mm "
            "slices by the challenge organizers."
        )
    if modality == "t1w":
        meta["ProcessingNote"] = "Face removed by the challenge organizers (mask in derivatives/challenge)."
    return meta


def write_dataset_files(bids_dir: Path, cfg: dict[str, Any]) -> None:
    """Top-level BIDS files and the description of both derivative datasets."""
    ds, version = cfg["dataset"], cfg["bids"]["version"]
    reference = (
        'Kuijf, H. J., et al. "Standardized Assessment of Automatic Segmentation of White Matter '
        'Hyperintensities; Results of the WMH Segmentation Challenge." IEEE Transactions on '
        "Medical Imaging (2019)."
    )
    _write_json(
        bids_dir / "dataset_description.json",
        {
            "Name": ds["name"],
            "BIDSVersion": version,
            "DatasetType": "raw",
            "License": "CC-BY-NC-4.0",
            "DatasetDOI": f"doi:{ds['doi']}",
            "HowToAcknowledge": reference,
            "ReferencesAndLinks": [reference],
            "GeneratedBy": [
                {"Name": "wmh-multisite", "Description": "BIDS conversion of the published files"}
            ],
        },
    )
    (bids_dir / "README").write_text(
        f"{ds['name']} (doi:{ds['doi']}, {ds['license']}), reorganized in BIDS by wmh-multisite.\n\n"
        "Raw images are the 'orig' files published by the organizers: FLAIR as published (3D FLAIRs of\n"
        "Amsterdam were reoriented and resampled to 3 mm slices by the organizers) and defaced 3D T1.\n"
        "derivatives/manual: reference WMH annotations (desc-O1, labels 0 background, 1 WMH,\n"
        "2 other pathology) and two additional observers on the training set (desc-O3, desc-O4).\n"
        "derivatives/challenge: preprocessing distributed by the organizers (elastix alignment of\n"
        "the T1 to the FLAIR, SPM12 bias correction, defacing masks: space-T1w for every subject,\n"
        "space-FLAIR for the Amsterdam subjects, whose 3D FLAIRs were defaced).\n",
        encoding="utf-8",
    )
    for name, description in [
        ("manual", "Manual WMH annotations (STRIVE criteria)"),
        ("challenge", "Preprocessing distributed by the challenge organizers"),
    ]:
        _write_json(
            bids_dir / "derivatives" / name / "dataset_description.json",
            {
                "Name": f"{ds['name']} - {description}",
                "BIDSVersion": version,
                "DatasetType": "derivative",
                "GeneratedBy": [{"Name": "WMH Segmentation Challenge organizers"}],
            },
        )
    with open(bids_dir / "derivatives" / "manual" / "dseg.tsv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["index", "name"])
        writer.writerows(LABELS.items())


PARTICIPANT_COLUMNS = {
    "participant_id": "BIDS subject label (original challenge id, zero-padded)",
    "original_id": "Subject id in the published challenge data",
    "split": "Challenge split: training or test",
    "scanner": "Scanner key in config/config.yaml",
    "institute": "Acquisition site",
    "manufacturer": "Scanner manufacturer",
    "model": "Scanner model",
    "field_strength_t": "Main magnetic field strength (tesla)",
    "seen_in_training": "Whether this scanner appears in the training split",
}


def write_participants(bids_dir: Path, subjects: list[Subject], cfg: dict[str, Any]) -> None:
    naming = cfg["bids"]["subject_naming"]
    with open(bids_dir / "participants.tsv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(PARTICIPANT_COLUMNS)
        for s in subjects:
            sc = cfg["scanners"][s.scanner]
            writer.writerow(
                [
                    s.participant_id(naming),
                    s.original_id,
                    s.split,
                    s.scanner,
                    sc["institute"],
                    sc["vendor"],
                    sc["model"],
                    sc["field_strength_t"],
                    str(sc["seen_in_training"]).lower(),
                ]
            )
    _write_json(
        bids_dir / "participants.json", {k: {"Description": v} for k, v in PARTICIPANT_COLUMNS.items()}
    )


def convert_subject(subject: Subject, bids_dir: Path, cfg: dict[str, Any]) -> dict[str, Path]:
    """Link every file of one subject into the BIDS tree; return {key: BIDS path}."""
    sub = subject.participant_id(cfg["bids"]["subject_naming"])
    scanner = cfg["scanners"][subject.scanner]
    out: dict[str, Path] = {}
    for rel, src in subject.files.items():
        if rel not in TARGETS:
            continue
        dataset, template = TARGETS[rel]
        dst = bids_dir / dataset / sub / "anat" / template.format(sub=sub)
        link_or_copy(src, dst)
        out[rel] = dst
    for obs, src in subject.observers.items():
        dataset, template = OBSERVER_TARGET
        dst = bids_dir / dataset / sub / "anat" / template.format(sub=sub, obs=obs)
        link_or_copy(src, dst)
        out[f"observer_{obs}"] = dst
    _write_json(out["orig/FLAIR.nii.gz"].with_name(f"{sub}_FLAIR.json"), sidecar(scanner, "flair"))
    _write_json(out["orig/3DT1.nii.gz"].with_name(f"{sub}_T1w.json"), sidecar(scanner, "t1w"))
    return out


def inventory_row(subject: Subject, paths: dict[str, Path], naming: str) -> dict[str, Any]:
    """QC of one subject: geometry of the raw images, validity and grid of the label maps, volumes."""
    flair, t1 = io.describe(paths["orig/FLAIR.nii.gz"]), io.describe(paths["orig/3DT1.nii.gz"])
    row: dict[str, Any] = {
        "participant_id": subject.participant_id(naming),
        "split": subject.split,
        "scanner": subject.scanner,
        "flair_shape": "x".join(map(str, flair["shape"])),
        "flair_voxel_mm": "x".join(f"{z:.2f}" for z in flair["voxel_size_mm"]),
        "flair_axcodes": flair["axcodes"],
        "flair_dtype": flair["dtype_on_disk"],
        "t1_shape": "x".join(map(str, t1["shape"])),
        "t1_voxel_mm": "x".join(f"{z:.2f}" for z in t1["voxel_size_mm"]),
        "t1_axcodes": t1["axcodes"],
        "qform_sform_agree": all(d["qform_sform_agree"] is not False for d in (flair, t1)),
    }
    voxel_ml = float(np.prod(flair["voxel_size_mm"])) / 1000
    for key, prefix in [("wmh.nii.gz", "O1"), ("observer_O3", "O3"), ("observer_O4", "O4")]:
        if key not in paths:
            continue
        same = io.same_grid(paths[key], paths["orig/FLAIR.nii.gz"])
        try:
            labels = io.load_labels(paths[key])
            row[f"{prefix}_valid"] = True
            row[f"{prefix}_wmh_ml"] = round(float((labels == 1).sum()) * voxel_ml, 2)
            if prefix == "O1":
                row["O1_other_ml"] = round(float((labels == 2).sum()) * voxel_ml, 2)
        except ValueError as err:
            row[f"{prefix}_valid"] = False
            log.warning("%s %s: %s", row["participant_id"], prefix, err)
        row[f"{prefix}_same_grid"] = same
    if "orig/T1.nii.gz" in paths:
        row["challenge_t1_same_grid"] = io.same_grid(paths["orig/T1.nii.gz"], paths["orig/FLAIR.nii.gz"])
    row["flair_defaced"] = "orig/FLAIR_mask.nii.gz" in paths
    if row["flair_defaced"]:
        mask_path = paths["orig/FLAIR_mask.nii.gz"]
        row["flair_mask_same_grid"] = io.same_grid(mask_path, paths["orig/FLAIR.nii.gz"])
        # The published FLAIR masks are not binary: ~1% of voxels on their border hold interpolated
        # values in (0, 1). The FLAIR was zeroed exactly where the mask is 0 (verified on all 70), so
        # the defaced region is defined as mask == 0 (J-028).
        mask = io.load_data(mask_path)
        removed = mask == 0
        row["flair_mask_partial_fraction"] = round(float(((mask > 0) & (mask < 1)).mean()), 4)
        row["flair_defaced_fraction"] = round(float(removed.mean()), 3)
        row["flair_zero_where_defaced"] = bool((io.load_data(paths["orig/FLAIR.nii.gz"])[removed] == 0).all())
    return row


def prune_stale(bids_dir: Path, produced: set[Path]) -> None:
    """Remove subject files left by a previous run under names that are no longer produced
    (e.g. after a renaming). Only links/copies inside data/bids are removed, never data/raw."""
    stale = [
        p
        for p in bids_dir.rglob("sub-*/anat/*")
        if p.suffix in (".gz", ".txt") and p.resolve() not in produced
    ]
    for path in stale:
        path.unlink()
    if stale:
        log.info("Removed %d stale file(s) from a previous conversion, e.g. %s", len(stale), stale[0].name)


def convert(cfg: dict[str, Any]) -> Path:
    """Build data/bids from data/raw (manifest of step 2.1); return the inventory path."""
    raw_dir, bids_dir = resolve_path(cfg, "raw"), resolve_path(cfg, "bids")
    manifest = raw_dir / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} not found: run `python -m wmh_multisite.data.download` first")
    with open(manifest, newline="", encoding="utf-8") as fh:
        subjects = collect_subjects(list(csv.DictReader(fh)), raw_dir, cfg["scanners"])
    log.info("%d complete subjects in %s", len(subjects), manifest)

    write_dataset_files(bids_dir, cfg)
    write_participants(bids_dir, subjects, cfg)
    naming = cfg["bids"]["subject_naming"]
    rows, produced = [], set()
    for subject in subjects:
        paths = convert_subject(subject, bids_dir, cfg)
        produced.update(p.resolve() for p in paths.values())
        rows.append(inventory_row(subject, paths, naming))
    prune_stale(bids_dir, produced)

    inventory = resolve_path(cfg, "results") / "tables" / "bids_inventory.csv"
    inventory.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(inventory, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="n/a")
        writer.writeheader()
        writer.writerows(rows)
    _log_summary(rows)
    return inventory


def _log_summary(rows: list[dict[str, Any]]) -> None:
    by_scanner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scanner[row["scanner"]].append(row)
    for scanner, group in sorted(by_scanner.items()):
        volumes = [r["O1_wmh_ml"] for r in group if "O1_wmh_ml" in r]
        problems = [
            r["participant_id"]
            for r in group
            if not (r["qform_sform_agree"] and r.get("O1_valid") and r.get("O1_same_grid"))
        ]
        # a group whose label maps are all invalid has no volume: report it instead of crashing
        wmh = (
            f"{np.median(volumes):.1f} ml [{min(volumes):.1f}-{max(volumes):.1f}]"
            if volumes
            else "n/a (no valid O1)"
        )
        log.info(
            "%-22s n=%3d | FLAIR voxels %s | axcodes %s | WMH O1 median %s | problems: %s",
            scanner,
            len(group),
            sorted({r["flair_voxel_mm"] for r in group}),
            sorted({r["flair_axcodes"] for r in group}),
            wmh,
            problems or "none",
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert the downloaded WMH dataset to BIDS.")
    parser.add_argument("--config", help="path to config.yaml (default: config/config.yaml)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    convert(load_config(args.config))


if __name__ == "__main__":
    main()
