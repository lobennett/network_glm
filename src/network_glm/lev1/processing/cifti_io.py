"""CIFTI dense-timeseries (.dtseries.nii) IO for fsLR den-91k residuals.

The task GLM runs on the (T, 91282) data matrix and the residual is the same
shape, so the residual dtseries reuses the input image's CIFTI + NIfTI headers
verbatim — the robust way to preserve the BrainModelAxis/SeriesAxis.
"""
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def load_dtseries(path) -> Tuple[np.ndarray, "nib.cifti2.Cifti2Image"]:
    """Load a .dtseries.nii into (data[T, n_grayordinates] float32, template image)."""
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    return data, img


def write_dtseries(data: np.ndarray, template: "nib.cifti2.Cifti2Image", out_path) -> Path:
    """Write a same-shape residual matrix back to a valid .dtseries.nii, reusing
    the template's CIFTI + NIfTI headers (SeriesAxis + BrainModelAxis)."""
    out_path = Path(out_path)
    if tuple(data.shape) != tuple(template.shape):
        raise ValueError(
            f"residual shape {tuple(data.shape)} != template shape {tuple(template.shape)}; "
            "header reuse requires identical (timepoints, grayordinates)."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.cifti2.Cifti2Image(
        np.asarray(data, dtype=np.float32),
        header=template.header,
        nifti_header=template.nifti_header,
    ).to_filename(str(out_path))
    return out_path
