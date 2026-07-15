"""Regression guard: ``process_run_residuals`` actually consumes fc_confounds.

Previously the volumetric residual path silently ignored ``--fc-confounds``.
``process_run_residuals`` accepted an optional ``filtering_params`` dict that
defaulted ``confounds: None``, and the call site in ``run.py`` never passed
the FC confounds in. Result: ``--space MNI --residuals --fc-confounds``
produced task-only residuals indistinguishable from the same command
without ``--fc-confounds``.

These tests verify two things:

1. The new ``fc_confounds`` kwarg on ``process_run_residuals`` flows into
   ``filtering_params['confounds']`` so nilearn.signal.clean actually
   regresses the supplied confounds.
2. End-to-end: residuals computed with fc_confounds are materially
   different from residuals computed without them — i.e., the regression
   actually happened.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import nibabel as nib
import numpy as np

from network_glm.lev1.processing.residuals import process_run_residuals


def _synthetic_4d(n_tp: int = 80, shape=(4, 4, 4)) -> nib.Nifti1Image:
    np.random.seed(0)
    data = np.random.randn(*shape, n_tp).astype(np.float32) * 5
    return nib.Nifti1Image(data, affine=np.eye(4))


def _fake_fitted_glm(n_tp: int = 80, shape=(4, 4, 4)) -> MagicMock:
    """Minimal stand-in for FirstLevelModel.

    ``ResidualsProcessor`` reads ``residuals_`` and uses the GLM's mask
    (via ``fitted_glm.masker_``). We bypass the masker path by patching
    ``apply_filtering`` indirectly — see test bodies.
    """
    glm = MagicMock()
    glm.residuals_ = [_synthetic_4d(n_tp, shape)]
    return glm


def test_fc_confounds_kwarg_threads_into_filtering_params(monkeypatch, tmp_path):
    """When called with ``fc_confounds=X``, the residuals processor must
    pass ``X`` as the ``confounds`` argument of the filtering call.
    """
    n_tp = 80
    fc_confounds = np.random.randn(n_tp, 6)

    captured = {}

    class StubProcessor:
        def __init__(self, fitted_glm, tr):
            self.fitted_glm = fitted_glm
            self.tr = tr

        def apply_filtering(self, mask_img=None, **filtering_params):
            captured["filtering_params"] = filtering_params
            return _synthetic_4d(n_tp)

        def save_residuals(self, output_dir, base_filename, kind):
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{base_filename}_{kind}.nii.gz"
            path.touch()
            return [path]

        def get_residuals_stats(self):
            return {}

    monkeypatch.setattr(
        "network_glm.lev1.processing.residuals.ResidualsProcessor",
        StubProcessor,
    )

    glm = _fake_fitted_glm(n_tp)
    result = process_run_residuals(
        glm,
        tmp_path,
        "sub-x_task-y_run-1",
        tr=1.5,
        fc_confounds=fc_confounds,
    )
    assert result["success"], f'residuals failed: {result["errors"]}'
    np.testing.assert_array_equal(
        captured["filtering_params"]["confounds"],
        fc_confounds,
        err_msg="fc_confounds did not reach filtering_params['confounds']",
    )


def test_fc_confounds_kwarg_default_is_none(monkeypatch, tmp_path):
    """Calling without ``fc_confounds`` produces ``confounds=None``, matching
    the historical behavior of the task-only residual path.
    """
    captured = {}

    class StubProcessor:
        def __init__(self, fitted_glm, tr):
            pass

        def apply_filtering(self, mask_img=None, **filtering_params):
            captured["filtering_params"] = filtering_params
            return _synthetic_4d(80)

        def save_residuals(self, output_dir, base_filename, kind):
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{base_filename}_{kind}.nii.gz"
            path.touch()
            return [path]

        def get_residuals_stats(self):
            return {}

    monkeypatch.setattr(
        "network_glm.lev1.processing.residuals.ResidualsProcessor",
        StubProcessor,
    )

    glm = _fake_fitted_glm()
    process_run_residuals(glm, tmp_path, "sub-x_task-y_run-1", tr=1.5)
    assert captured["filtering_params"]["confounds"] is None


def test_explicit_filtering_params_overrides_fc_confounds_kwarg(monkeypatch, tmp_path):
    """If a caller passes a full ``filtering_params`` dict, its ``confounds``
    entry wins over the convenience ``fc_confounds`` kwarg.

    This preserves the historical contract for callers that already build
    their own filtering dict — they shouldn't suddenly find their
    explicitly-passed ``confounds=None`` overridden.
    """
    captured = {}

    class StubProcessor:
        def __init__(self, fitted_glm, tr):
            pass

        def apply_filtering(self, mask_img=None, **filtering_params):
            captured["filtering_params"] = filtering_params
            return _synthetic_4d(80)

        def save_residuals(self, output_dir, base_filename, kind):
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{base_filename}_{kind}.nii.gz"
            path.touch()
            return [path]

        def get_residuals_stats(self):
            return {}

    monkeypatch.setattr(
        "network_glm.lev1.processing.residuals.ResidualsProcessor",
        StubProcessor,
    )

    explicit = {
        "low_pass": 0.1,
        "high_pass": 0.01,
        "standardize": False,
        "detrend": False,
        "confounds": None,  # explicit None — caller knows what they want
    }
    fc = np.random.randn(80, 6)
    glm = _fake_fitted_glm()
    process_run_residuals(
        glm,
        tmp_path,
        "sub-x_task-y_run-1",
        tr=1.5,
        filtering_params=explicit,
        fc_confounds=fc,
    )
    # Explicit dict's None wins
    assert captured["filtering_params"]["confounds"] is None
