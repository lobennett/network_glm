"""Slice-timing reference resolution for the lev1 frame_times offset.

fMRIPrep slice-time corrects to a reference time it records as ``StartTime``
in the derivative sidecar. Our regressors must be sampled at that same
reference, otherwise the model is misaligned with the data.

The pipeline previously hardcoded ``+TR/2``, which is only equal to the
reference when the slices span the whole TR. Our acquisition leaves ~88 ms
of dead time at the end of each TR, so ``TR/2`` (0.745 s) sits 44 ms past
the reference fMRIPrep actually used (0.701 s).
"""

import json

import numpy as np
import pandas as pd
import pytest

from network_glm.acquisition import resolve_slice_time_ref
from network_glm.lev1.processing.design import create_regressor


def _bold_with_sidecar(tmp_path, bold_name: str, sidecar: dict | None):
    """Create an (empty) BOLD file and its JSON sidecar; return the BOLD path."""
    bold = tmp_path / bold_name
    bold.write_bytes(b"")
    if sidecar is not None:
        stem = bold_name.split(".")[0]
        (tmp_path / f"{stem}.json").write_text(json.dumps(sidecar))
    return bold


# --------------------------------------------------------------------------
# resolve_slice_time_ref
# --------------------------------------------------------------------------


def test_returns_start_time_when_slice_timing_corrected(tmp_path):
    """STC was applied -> use the reference fMRIPrep corrected to."""
    bold = _bold_with_sidecar(
        tmp_path,
        "sub-s03_task-flanker_space-T1w_desc-preproc_bold.nii.gz",
        {"RepetitionTime": 1.49, "StartTime": 0.701, "SliceTimingCorrected": True},
    )
    assert resolve_slice_time_ref(bold) == pytest.approx(0.701)


def test_returns_zero_when_slice_timing_not_corrected(tmp_path):
    """No STC -> data are at volume onset, so the model needs no offset."""
    bold = _bold_with_sidecar(
        tmp_path,
        "sub-s03_task-flanker_space-T1w_desc-preproc_bold.nii.gz",
        {"RepetitionTime": 1.49, "SliceTimingCorrected": False},
    )
    assert resolve_slice_time_ref(bold) == 0.0


def test_returns_zero_when_slice_timing_field_absent(tmp_path):
    """Sidecar that never mentions STC is treated as un-corrected."""
    bold = _bold_with_sidecar(
        tmp_path,
        "sub-s03_task-flanker_space-T1w_desc-preproc_bold.nii.gz",
        {"RepetitionTime": 1.49},
    )
    assert resolve_slice_time_ref(bold) == 0.0


def test_raises_when_corrected_but_start_time_missing(tmp_path):
    """Inconsistent metadata must fail loudly, never silently fall back to TR/2."""
    bold = _bold_with_sidecar(
        tmp_path,
        "sub-s03_task-flanker_space-T1w_desc-preproc_bold.nii.gz",
        {"RepetitionTime": 1.49, "SliceTimingCorrected": True},
    )
    with pytest.raises(ValueError, match="StartTime"):
        resolve_slice_time_ref(bold)


def test_raises_when_sidecar_missing(tmp_path):
    """Without a sidecar the reference is unknowable; do not guess."""
    bold = _bold_with_sidecar(
        tmp_path, "sub-s03_task-flanker_space-T1w_desc-preproc_bold.nii.gz", None
    )
    with pytest.raises(FileNotFoundError):
        resolve_slice_time_ref(bold)


@pytest.mark.parametrize(
    "bold_name",
    [
        "sub-s03_task-flanker_space-T1w_desc-preproc_bold.nii.gz",
        "sub-s03_task-flanker_space-fsLR_den-91k_bold.dtseries.nii",
        "sub-s03_task-flanker_hemi-L_space-fsaverage6_bold.func.gii",
    ],
)
def test_finds_sidecar_for_every_output_type(tmp_path, bold_name):
    """Volumetric, CIFTI and GIFTI derivatives all pair with a plain .json."""
    bold = _bold_with_sidecar(
        tmp_path, bold_name, {"StartTime": 0.701, "SliceTimingCorrected": True}
    )
    assert resolve_slice_time_ref(bold) == pytest.approx(0.701)


# --------------------------------------------------------------------------
# create_regressor honours the reference
# --------------------------------------------------------------------------


def _events():
    return pd.DataFrame(
        {
            "onset": [10.0, 20.0, 30.0],
            "duration": [1.0, 1.0, 1.0],
            "trial_type": ["go", "go", "go"],
        }
    )


def _config():
    return {
        "amplitude_column": "constant_1_column",
        "duration_column": "duration",
        "subset": "trial_type == 'go'",
    }


def test_regressor_samples_at_the_given_slice_time_ref():
    """frame_times must be arange(n)*TR + slice_time_ref."""
    tr, n_scans = 1.49, 40
    reg_ref, _ = create_regressor(
        _events(), _config(), n_scans, "go", tr, slice_time_ref=0.701
    )

    from nilearn.glm.first_level import hemodynamic_models

    expected, _ = hemodynamic_models.compute_regressor(
        exp_condition=(np.array([10.0, 20.0, 30.0]), np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])),
        hrf_model="spm",
        frame_times=np.arange(n_scans) * tr + 0.701,
    )
    np.testing.assert_allclose(reg_ref["go"].to_numpy(), expected.flatten(), rtol=1e-10)


def test_start_time_and_half_tr_give_different_regressors():
    """The 44 ms difference is real, not numerically negligible."""
    tr, n_scans = 1.49, 40
    reg_start, _ = create_regressor(
        _events(), _config(), n_scans, "go", tr, slice_time_ref=0.701
    )
    reg_half, _ = create_regressor(
        _events(), _config(), n_scans, "go", tr, slice_time_ref=tr / 2
    )
    assert not np.allclose(reg_start["go"].to_numpy(), reg_half["go"].to_numpy())


def test_defaults_to_no_offset():
    """Default is no shift; the caller must supply the reference explicitly."""
    tr, n_scans = 1.49, 40
    reg_default, _ = create_regressor(_events(), _config(), n_scans, "go", tr)
    reg_zero, _ = create_regressor(
        _events(), _config(), n_scans, "go", tr, slice_time_ref=0.0
    )
    np.testing.assert_allclose(reg_default["go"].to_numpy(), reg_zero["go"].to_numpy())
