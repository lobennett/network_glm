"""Pytest configuration and fixtures for parcelextract tests."""

import pytest

try:
    import nibabel  # noqa: F401 -- availability guard, skips module if missing
except ImportError:
    pytest.skip(
        "neuroimaging dependencies not installed (install with: uv pip install -e '.[lev1]')",
        allow_module_level=True,
    )

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_events_data():
    """Create sample events data for testing RT calculations."""
    import pandas as pd

    # Create realistic task-switching events
    events_data = {
        "onset": [2.0, 4.5, 7.2, 9.8, 12.1, 15.0, 17.3, 20.1],
        "duration": [1.5] * 8,
        "trial_id": ["test_trial"] * 6 + ["practice_trial"] * 2,
        "trial_type": [
            "switch",
            "repeat",
            "switch",
            "repeat",
            "switch",
            "repeat",
            "switch",
            "repeat",
        ],
        "key_press": [1, 2, 1, 2, -1, 2, 1, 2],  # -1 = no response (omission)
        "correct_response": [1, 2, 1, 2, 1, 2, 1, 2],
        "response_time": [
            0.65,
            0.52,
            0.78,
            0.61,
            -1,
            0.49,
            0.71,
            0.58,
        ],  # -1 = no response
        "junk": [0, 0, 0, 0, 0, 0, 0, 0],
    }

    return pd.DataFrame(events_data)


@pytest.fixture
def sample_events_file(sample_events_data, temp_dir):
    """Create a sample events TSV file."""
    filepath = temp_dir / "events.tsv"
    sample_events_data.to_csv(filepath, sep="\t", index=False)
    return filepath


@pytest.fixture
def multiple_brain_masks(temp_dir):
    """Create multiple brain mask files with different overlaps."""
    affine = np.eye(4)
    affine[0, 0] = 2.0
    affine[1, 1] = 2.0
    affine[2, 2] = 2.0

    mask_files = []

    # Mask 1 - larger region
    mask1 = np.zeros((10, 10, 10), dtype=np.int32)
    mask1[2:8, 2:8, 2:8] = 1
    img1 = nib.Nifti1Image(mask1, affine)
    filepath1 = temp_dir / "mask1.nii.gz"
    nib.save(img1, filepath1)
    mask_files.append(filepath1)

    # Mask 2 - overlapping region
    mask2 = np.zeros((10, 10, 10), dtype=np.int32)
    mask2[3:7, 3:7, 3:7] = 1
    img2 = nib.Nifti1Image(mask2, affine)
    filepath2 = temp_dir / "mask2.nii.gz"
    nib.save(img2, filepath2)
    mask_files.append(filepath2)

    # Mask 3 - smaller overlap
    mask3 = np.zeros((10, 10, 10), dtype=np.int32)
    mask3[4:6, 4:6, 4:6] = 1
    img3 = nib.Nifti1Image(mask3, affine)
    filepath3 = temp_dir / "mask3.nii.gz"
    nib.save(img3, filepath3)
    mask_files.append(filepath3)

    return mask_files


@pytest.fixture
def files_dict_with_masks(multiple_brain_masks, sample_events_file):
    """Create a files dictionary structure with mask files for testing."""
    return {
        "ses-01": {
            "run-01": {
                "mni_brain_mask": multiple_brain_masks[0],
                "t1w_brain_mask": multiple_brain_masks[0],
                "events": sample_events_file,
            },
            "run-02": {
                "mni_brain_mask": multiple_brain_masks[1],
                "t1w_brain_mask": multiple_brain_masks[1],
                "events": sample_events_file,
            },
        },
        "ses-02": {
            "run-01": {
                "mni_brain_mask": multiple_brain_masks[2],
                "t1w_brain_mask": multiple_brain_masks[2],
                "events": sample_events_file,
            }
        },
    }
