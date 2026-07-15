"""Tests for the shared lev1 image-dtype helper."""

import nibabel as nib
import numpy as np

from network_glm.lev1.processing.imaging import cast_nifti_to_float32


def test_volumetric_recast_to_float32():
    # int16-stored NIfTI (the case that triggers nibabel's lossy auto-scaling)
    img = nib.Nifti1Image(np.arange(8, dtype=np.int16).reshape(2, 2, 2), np.eye(4))
    out = cast_nifti_to_float32(img, is_surface=False)
    assert out.get_data_dtype() == np.float32
    assert np.allclose(out.get_fdata(), np.arange(8).reshape(2, 2, 2))


def test_surface_passthrough_is_unchanged_object():
    img = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), np.eye(4))
    out = cast_nifti_to_float32(img, is_surface=True)
    assert out is img  # GIFTI/surface left exactly as-is
