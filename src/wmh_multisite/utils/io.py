"""Reading and writing NIfTI images. The whole pipeline goes through these functions.

They exist to avoid a few classic mistakes: saving an array with the wrong affine silently moves the image,
two arrays of the same shape are not necessarily on the same grid, get_fdata() returns float64 by default, the
challenge label maps are stored as float32 and must become integers, and the qform and sform can disagree.

Every function accepts either a path or an image already loaded with nibabel.
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
    """Return img as is if it is already an image, otherwise load it."""
    if not isinstance(img, nib.Nifti1Image):
        return load(img)
    return img


def load(path: str | os.PathLike) -> nib.Nifti1Image:
    """Open a NIfTI file lazily: only the header is read until the voxels are needed.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the file name does not end with .nii or .nii.gz.
        nibabel.filebasedimages.ImageFileError: the file is not a readable NIfTI.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    if not path.name.endswith(NIFTI_SUFFIXES):
        raise ValueError(f"Not a NIfTI file (expected {' or '.join(NIFTI_SUFFIXES)}): {path}")
    return nib.load(path)


def load_data(img: ImageLike, dtype: np.dtype | type = np.float32) -> np.ndarray:
    """Voxel array as dtype (float32 by default), with the scaling applied.

    float64 is only returned when explicitly asked for, since a 1 mm T1 takes 100 MB in float64.
    """
    return _as_image(img).get_fdata(dtype=dtype, caching="unchanged")


def load_labels(img: ImageLike, allowed: tuple[int, ...] = (0, 1, 2), atol: float = 1e-3) -> np.ndarray:
    """Label map as uint8.

    The values must be integers (up to atol) and belong to allowed. Anything else means the file is not a
    label map, or was interpolated linearly at some point.

    Raises:
        ValueError: a value is not close to an integer, or is not in allowed.
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
    """Save data with the affine and header of reference, and return the written path.

    data must have the spatial shape of reference. It is saved as dtype if given, otherwise as data.dtype, and
    missing parent folders are created.

    Raises:
        ValueError: the shapes differ, or path does not end with .nii or .nii.gz.
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
    """True if a and b have the same spatial shape and the same affine (up to atol mm).

    Call it before any voxel-wise comparison (Dice, difference, overlay).
    """
    a, b = _as_image(a), _as_image(b)
    return bool(a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=atol))


def voxel_volume_mm3(img: ImageLike) -> float:
    """Exact voxel volume in mm3, from the header (describe rounds the voxel sizes for display)."""
    return float(np.prod(np.asarray(_as_image(img).header.get_zooms()[:3], dtype=np.float64)))


def describe(img: ImageLike, with_stats: bool = False) -> dict[str, Any]:
    """Summary of an image for the QC tables, used by the BIDS conversion.

    Gives the shape, voxel size, orientation, on-disk type, qform and sform codes and whether the two forms
    agree (None when fewer than two are valid). With with_stats it also reads the data and adds the min, 1st
    percentile, median, 99th percentile and max of the finite values. All values are plain Python types.
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
