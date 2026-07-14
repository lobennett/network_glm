"""Shared image-dtype helpers for lev1 statistical-map saving."""

from __future__ import annotations


def cast_nifti_to_float32(img, *, is_surface: bool):
    """Return ``img`` recast to float32 for volumetric NIfTIs; surface GIFTIs as-is.

    ``nibabel``'s ``to_filename()`` auto-scales output to the BOLD header's
    integer storage type (~256 quantization levels across cal_min..cal_max),
    which destroys variance maps and degrades z/effect-size precision. Recasting
    the data + header to float32 avoids this. Surface (GIFTI) outputs are
    unaffected, so they pass through unchanged.
    """
    if is_surface:
        return img
    new_img = img.__class__(
        img.get_fdata().astype("float32"),
        img.affine,
        header=img.header,
    )
    new_img.set_data_dtype("float32")
    return new_img
