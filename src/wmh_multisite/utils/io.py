"""NIfTI input/output: the only way the pipeline reads and writes images.

Why this module exists (see notebooks/01_explore_nifti.ipynb):
- an array without its affine is meaningless: saving with the wrong affine moves the image silently;
- two arrays with the same shape are not necessarily on the same grid;
- ``get_fdata()`` returns float64 by default (4x the disk size of uint16 data);
- label maps of the challenge are stored as float32 and must become integers;
- qform and sform can disagree, and different libraries do not pick the same one.

Every function accepts either a path or an already loaded ``nib.Nifti1Image`` (type ``ImageLike``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

ImageLike = str | os.PathLike | nib.Nifti1Image

NIFTI_SUFFIXES = (".nii", ".nii.gz")


def _as_image(img: ImageLike) -> nib.Nifti1Image:
    """Return ``img`` unchanged if it is already an image, otherwise ``load(img)``."""
    if not isinstance(img, nib.Nifti1Image):
        return load(img)
    return img


def load(path: str | os.PathLike) -> nib.Nifti1Image:
    """Open a NIfTI file lazily (header only; voxels are read on first access).

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the file name does not end with .nii or .nii.gz.
        nibabel.filebasedimages.ImageFileError: the file is not a readable NIfTI (raised by nibabel,
            whose message already names the file).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    if not path.name.endswith(NIFTI_SUFFIXES):
        raise ValueError(f"Not a NIfTI file (expected {' or '.join(NIFTI_SUFFIXES)}): {path}")
    return nib.load(path)


def load_data(img: ImageLike, dtype: np.dtype | type = np.float32) -> np.ndarray:
    """Return the voxel array as ``dtype`` (float32 by default), scaling applied.

    Never returns float64 unless explicitly asked: a 1 mm T1 weighs 100 MB in float64.
    """
    return _as_image(img).get_fdata(dtype=dtype, caching="unchanged")


def load_labels(img: ImageLike, allowed: tuple[int, ...] = (0, 1, 2), atol: float = 1e-3) -> np.ndarray:
    """Return a label map as uint8.

    Values must be (almost) integers and belong to ``allowed``; anything else means the file
    is not a label map, or was interpolated linearly somewhere upstream.

    Raises:
        ValueError: a value is not within ``atol`` of an integer, or is not in ``allowed``.
    """
    img = _as_image(img)
    name = img.get_filename() or "in-memory image"
    values = load_data(img, dtype=np.float32)
    rounded = np.rint(values)

    # NaN/inf would slip through the tolerance test (nan > atol is False): flag them explicitly.
    not_integer = ~np.isfinite(values) | (np.abs(values - rounded) > atol)
    if not_integer.any():
        examples = np.unique(values[not_integer])[:5]
        raise ValueError(
            f"{int(not_integer.sum())} voxels are not integer labels (e.g. {examples}) "
            f"in {name}; was this map interpolated linearly?"
        )

    present = np.unique(rounded).astype(int)
    unexpected = present[~np.isin(present, allowed)]
    if unexpected.size:
        shown = unexpected[:10].tolist()
        more = f" and {unexpected.size - 10} more" if unexpected.size > 10 else ""
        raise ValueError(f"Unexpected label values {shown}{more} in {name} (allowed: {list(allowed)})")

    return rounded.astype(np.uint8)


def save_like(
    data: np.ndarray, reference: ImageLike, path: str | os.PathLike, dtype: np.dtype | type | None = None
) -> Path:
    """Save ``data`` with the geometry (affine and header) of ``reference``.

    ``data`` must have the spatial shape of ``reference``. The on-disk type is ``dtype`` if
    given, else ``data.dtype``. Parent folders are created. Returns the written path.

    Raises:
        ValueError: shape mismatch between ``data`` and ``reference``, or ``path`` is not .nii/.nii.gz.
    """
    reference = _as_image(reference)
    data = np.asarray(data)
    if data.shape != reference.shape[:3]:
        raise ValueError(
            f"Shape mismatch: data {data.shape} vs reference {reference.shape[:3]} "
            f"({reference.get_filename() or 'in-memory image'})"
        )
    path = Path(path)
    if not path.name.endswith(NIFTI_SUFFIXES):
        raise ValueError(f"Output must end with {' or '.join(NIFTI_SUFFIXES)}: {path}")

    # NIfTI has no boolean type: masks are written as uint8.
    out_dtype = np.dtype(np.uint8 if data.dtype == bool and dtype is None else (dtype or data.dtype))

    header = reference.header.copy()
    header.set_data_dtype(out_dtype)
    img = nib.Nifti1Image(data.astype(out_dtype, copy=False), reference.affine, header=header)
    # Write the same affine in qform and sform, keeping the reference's codes (1 = scanner if unset).
    _, qcode = reference.header.get_qform(coded=True)
    _, scode = reference.header.get_sform(coded=True)
    img.set_qform(reference.affine, code=int(qcode) or 1)
    img.set_sform(reference.affine, code=int(scode) or 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, path)
    return path


def same_grid(a: ImageLike, b: ImageLike, atol: float = 1e-3) -> bool:
    """True if ``a`` and ``b`` share spatial shape and affine (within ``atol`` mm).

    Call it before any voxel-wise comparison (Dice, difference, overlay).
    """
    a, b = _as_image(a), _as_image(b)
    return bool(a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=atol))


def voxel_volume_mm3(img: ImageLike) -> float:
    """Exact voxel volume (mm3) from the header; ``describe`` rounds voxel sizes for display only."""
    return float(np.prod(np.asarray(_as_image(img).header.get_zooms()[:3], dtype=np.float64)))


def describe(img: ImageLike, with_stats: bool = False) -> dict[str, Any]:
    """Summarize an image for QC tables (used by the BIDS conversion, step 2.2).

    Keys: shape, voxel_size_mm, axcodes, dtype_on_disk, qform_code, sform_code, qform_sform_agree
    (None when fewer than two of the forms are valid). With ``with_stats``: min, p1, p50, p99,
    max of the finite voxel values (reads the data). Values are plain Python types (JSON-ready).
    """
    img = _as_image(img)
    header = img.header
    qform, qcode = header.get_qform(coded=True)
    sform, scode = header.get_sform(coded=True)
    agree = bool(np.allclose(qform, sform, atol=1e-3)) if qcode > 0 and scode > 0 else None

    info: dict[str, Any] = {
        "shape": tuple(int(n) for n in img.shape),
        "voxel_size_mm": tuple(round(float(z), 4) for z in header.get_zooms()[:3]),
        "axcodes": "".join(nib.aff2axcodes(img.affine)),
        "dtype_on_disk": str(img.get_data_dtype()),
        "qform_code": int(qcode),
        "sform_code": int(scode),
        "qform_sform_agree": agree,
    }
    if with_stats:
        values = load_data(img)
        values = values[np.isfinite(values)]
        p1, p50, p99 = np.percentile(values, [1, 50, 99])
        info.update(
            min=float(values.min()), p1=float(p1), p50=float(p50), p99=float(p99), max=float(values.max())
        )
    return info
