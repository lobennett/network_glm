"""Public synthetic NIfTI fixtures for group-analysis tests."""

import gzip

import nibabel as nib
import numpy as np

CONTRAST = "task-rest_contrast-task"
CORRP = "onesample_2sided_tfce_corrp_fstat1.nii.gz"


def save(path, data=None, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(
        np.ones((3, 3, 3), np.float32) if data is None else np.asarray(data, np.float32),
        np.eye(4) if affine is None else affine,
    ), path)
    return str(path)


def effect(root, subject, value=1, arm="RTDur", contrast=CONTRAST):
    return save(root / f"sub-{subject}/task-rest/fixed_effects/"
                f"sub-{subject}_{contrast}_rtmodel-{arm}_stat-fixed-effects.nii.gz",
                np.full((3, 3, 3), value))


def corrupt_gzip(path):
    """Keep a readable NIfTI header, then fail decompression of the voxel data."""
    image = nib.Nifti1Image(np.ones((3, 3, 3), np.float32), np.eye(4))
    # The second gzip member has a valid header but a reserved DEFLATE block type.
    path.write_bytes(gzip.compress(image.to_bytes()[:352])
                     + bytes.fromhex("1f8b08000000000002ff07"))
    return path
