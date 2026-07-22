"""``--no-residual-filter`` should turn off the fsLR/CIFTI residual band-pass.

``process_cifti_run`` (network_glm.lev1.runner) hard-codes the band-pass via
``process_cifti_residuals``'s defaults (low_pass=0.1, high_pass=0.01). When the
caller sets ``args.no_residual_filter = True`` we want it to forward
``low_pass=None, high_pass=None`` instead so residuals can be emitted
unfiltered and band-passed downstream (e.g. by XCP-D). Default behavior
(flag unset / False) must be unchanged.
"""

import argparse
from unittest.mock import patch

import numpy as np
import pandas as pd


def _base_args(**overrides):
    defaults = dict(
        residuals=True,
        space="fsLR",
        task_name="stopSignal",
        subj_id="sub-x",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _run_process_cifti_run(args):
    """Drive process_cifti_run with load_dtseries/validate_design_matrix/SurfaceGLM
    mocked out so it reaches the process_cifti_residuals call, and return the
    mock used to intercept that call.
    """
    fake_data = np.zeros((2, 5))
    fake_template = object()
    design_matrix = pd.DataFrame({"reg": [0, 1], "constant": [1, 1]})

    with (
        patch("network_glm.lev1.runner.load_dtseries") as mock_load,
        patch("network_glm.lev1.runner.validate_design_matrix") as mock_validate,
        patch("network_glm.lev1.runner.SurfaceGLM") as mock_surface_glm_cls,
        patch("network_glm.lev1.runner.process_cifti_residuals") as mock_process_residuals,
    ):
        mock_load.return_value = (fake_data, fake_template)
        mock_validate.return_value = {"is_valid": True, "errors": []}
        mock_glm_instance = mock_surface_glm_cls.return_value
        mock_glm_instance.fit.return_value = mock_glm_instance
        mock_process_residuals.return_value = {"success": True, "saved_path": None, "errors": []}

        from network_glm.lev1.runner import process_cifti_run

        run_files = {"cifti_bold": "fake_cifti_bold.dtseries.nii"}
        process_cifti_run(
            run_files, design_matrix, args, {"task_residuals": "out"}, "sub-x_ses-01_task-t_run-1",
            1.49, fc_confounds=None,
        )

    return mock_process_residuals


def test_no_residual_filter_forwards_none_lowpass_highpass():
    args = _base_args(no_residual_filter=True)
    mock_process_residuals = _run_process_cifti_run(args)

    mock_process_residuals.assert_called_once()
    _, kwargs = mock_process_residuals.call_args
    assert kwargs["low_pass"] is None
    assert kwargs["high_pass"] is None


def test_default_still_applies_bandpass():
    args = _base_args(no_residual_filter=False)
    mock_process_residuals = _run_process_cifti_run(args)

    mock_process_residuals.assert_called_once()
    _, kwargs = mock_process_residuals.call_args
    assert kwargs["low_pass"] == 0.1
    assert kwargs["high_pass"] == 0.01


def test_missing_no_residual_filter_attr_defaults_to_bandpass():
    """Namespaces built before this flag existed (no attribute at all) must
    still get the original band-passed behavior via getattr(..., False)."""
    args = _base_args()  # no_residual_filter intentionally omitted
    mock_process_residuals = _run_process_cifti_run(args)

    mock_process_residuals.assert_called_once()
    _, kwargs = mock_process_residuals.call_args
    assert kwargs["low_pass"] == 0.1
    assert kwargs["high_pass"] == 0.01
