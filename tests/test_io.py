"""utils/io.py: reading, writing and checking NIfTI images."""

import json

import nibabel as nib
import numpy as np
import pytest
from nibabel.filebasedimages import ImageFileError

from conftest import FLAIR_AFFINE, FLAIR_SHAPE, T1_AFFINE
from wmh_multisite.utils import io

# --- load -------------------------------------------------------------------------------------


def test_load_returns_lazy_image(nifti, labels):
    img = io.load(nifti(labels))
    assert isinstance(img, nib.Nifti1Image)
    assert img.shape == FLAIR_SHAPE


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        io.load(tmp_path / "missing.nii.gz")


def test_load_directory_is_not_a_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.load(tmp_path)


def test_load_wrong_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Not a NIfTI"):
        io.load(path)


def test_load_corrupted_nifti(tmp_path):
    path = tmp_path / "broken.nii.gz"
    path.write_bytes(b"not a nifti")
    with pytest.raises(ImageFileError):
        io.load(path)


# --- load_data --------------------------------------------------------------------------------


def test_load_data_is_float32_by_default(nifti):
    path = nifti(np.arange(np.prod(FLAIR_SHAPE), dtype=np.uint16).reshape(FLAIR_SHAPE))
    data = io.load_data(path)
    assert data.dtype == np.float32
    assert data[5, 5, 3] == np.prod(FLAIR_SHAPE) - 1


def test_load_data_accepts_path_and_image(nifti, labels):
    path = nifti(labels)
    np.testing.assert_array_equal(io.load_data(path), io.load_data(nib.load(path)))


def test_load_data_does_not_fill_the_cache(nifti, labels):
    img = nib.load(nifti(labels))
    io.load_data(img)
    assert not img.in_memory  # caching="unchanged": the image object stays light


# --- load_labels ------------------------------------------------------------------------------


def test_load_labels_accepts_float_stored_labels(nifti, labels):
    lab = io.load_labels(nifti(labels))
    assert lab.dtype == np.uint8
    assert set(np.unique(lab)) == {0, 1, 2}
    assert (lab == 1).sum() == 8


@pytest.mark.parametrize(
    "bad, message",
    [
        (0.62, "not integer labels"),  # interpolated border, like the Amsterdam FLAIR masks
        (np.nan, "not integer labels"),  # NaN would slip through a tolerance test
        (3.0, "Unexpected label values"),  # integer but not allowed
    ],
)
def test_load_labels_rejects_invalid_values(nifti, labels, bad, message):
    labels[0, 0, 0] = bad
    with pytest.raises(ValueError, match=message):
        io.load_labels(nifti(labels))


def test_load_labels_custom_allowed_set(nifti, labels):
    with pytest.raises(ValueError, match=r"\[2\]"):
        io.load_labels(nifti(labels), allowed=(0, 1))


def test_load_labels_rejects_an_intensity_image(nifti):
    image = np.random.default_rng(0).integers(0, 2000, FLAIR_SHAPE).astype(np.uint16)
    with pytest.raises(ValueError, match="Unexpected label values"):
        io.load_labels(nifti(image))


# --- save_like --------------------------------------------------------------------------------


def test_save_like_roundtrip_labels_with_float_reference(tmp_path, nifti, labels):
    reference = nifti(labels.astype(np.float32), "ref.nii.gz")
    lab = io.load_labels(reference)
    out = io.save_like(lab, reference, tmp_path / "sub" / "dir" / "out.nii.gz")  # parent folders created
    assert out.is_file()
    assert io.describe(out)["dtype_on_disk"] == "uint8"
    assert io.same_grid(out, reference)
    np.testing.assert_array_equal(io.load_labels(out), lab)


def test_save_like_keeps_float_values(tmp_path, nifti):
    reference = nifti(np.ones(FLAIR_SHAPE, np.uint16), "ref.nii.gz")
    values = np.random.default_rng(1).random(FLAIR_SHAPE).astype(np.float32) * 2.5
    out = io.save_like(values, reference, tmp_path / "f.nii.gz")
    assert io.describe(out)["dtype_on_disk"] == "float32"
    np.testing.assert_array_equal(io.load_data(out), values)


def test_save_like_bool_becomes_uint8(tmp_path, nifti, labels):
    out = io.save_like(labels == 1, nifti(labels), tmp_path / "mask.nii.gz")
    assert io.describe(out)["dtype_on_disk"] == "uint8"
    assert set(np.unique(io.load_labels(out, allowed=(0, 1)))) == {0, 1}


def test_save_like_ignores_scaling_of_the_reference(tmp_path, labels):
    ref = nib.Nifti1Image(np.ones(FLAIR_SHAPE, np.int16), FLAIR_AFFINE)
    ref.header.set_slope_inter(2.0, 10.0)
    out = io.save_like(io.load_labels(nib.Nifti1Image(labels, FLAIR_AFFINE)), ref, tmp_path / "x.nii.gz")
    np.testing.assert_array_equal(np.unique(io.load_data(out)), [0, 1, 2])  # not 10, 12, 14


def test_save_like_writes_consistent_qform_and_sform(tmp_path, nifti, labels):
    out = io.save_like(labels, nifti(labels), tmp_path / "x.nii.gz")
    info = io.describe(out)
    assert info["qform_code"] > 0 and info["sform_code"] > 0 and info["qform_sform_agree"] is True


def test_save_like_shape_mismatch(tmp_path, nifti, labels):
    with pytest.raises(ValueError, match="Shape mismatch"):
        io.save_like(labels[:, :, :2], nifti(labels), tmp_path / "x.nii.gz")
    assert not (tmp_path / "x.nii.gz").exists()


def test_save_like_wrong_extension(tmp_path, nifti, labels):
    with pytest.raises(ValueError, match="Output must end"):
        io.save_like(labels, nifti(labels), tmp_path / "x.png")


# --- same_grid --------------------------------------------------------------------------------


def test_same_grid(nifti, labels):
    a = nifti(labels, "a.nii.gz")
    assert io.same_grid(a, nifti(labels * 0, "b.nii.gz"))
    shifted = FLAIR_AFFINE.copy()
    shifted[0, 3] += 0.5  # same shape, origin moved by half a voxel
    assert not io.same_grid(a, nifti(labels, "c.nii.gz", shifted))
    assert not io.same_grid(a, nifti(labels[:, :, :2], "d.nii.gz"))
    tiny = FLAIR_AFFINE.copy()
    tiny[0, 3] += 1e-5  # storage rounding is tolerated
    assert io.same_grid(a, nifti(labels, "e.nii.gz", tiny))


# --- describe ---------------------------------------------------------------------------------


def test_describe_geometry_and_json(nifti, labels):
    info = io.describe(nifti(labels, affine=T1_AFFINE), with_stats=True)
    assert info["shape"] == FLAIR_SHAPE
    assert info["voxel_size_mm"] == (1.0, 1.0, 1.0)
    assert info["axcodes"] == "RAS"
    assert info["dtype_on_disk"] == "float32"
    assert info["min"] == 0.0 and info["max"] == 2.0
    json.dumps(info)  # plain Python types only


def test_describe_orientation_codes(nifti, labels):
    lps = np.diag([-1.0, -1.0, 3.0, 1.0])  # i -> left, j -> posterior, k -> superior
    assert io.describe(nifti(labels, affine=lps))["axcodes"] == "LPS"


def test_describe_qform_sform_disagreement(tmp_path, labels):
    img = nib.Nifti1Image(labels, FLAIR_AFFINE)
    other = FLAIR_AFFINE.copy()
    other[0, 3] = 5.0
    img.set_qform(other, code=1)
    img.set_sform(FLAIR_AFFINE, code=1)
    nib.save(img, tmp_path / "x.nii.gz")
    assert io.describe(tmp_path / "x.nii.gz")["qform_sform_agree"] is False


def test_voxel_volume_is_exact_not_rounded(nifti):
    path = nifti(np.zeros((2, 2, 2), np.uint8), "v.nii.gz", np.diag([0.9583333, 0.9583333, 3.0, 1.0]))
    assert io.voxel_volume_mm3(path) == pytest.approx(0.9583333**2 * 3.0, rel=1e-6)
    assert io.describe(path)["voxel_size_mm"][0] == 0.9583  # display value is rounded, on purpose
