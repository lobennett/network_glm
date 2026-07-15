"""Regression guard: surface GLM path must validate the design matrix.

Before this commit, the volumetric branch in ``process_volumetric_run`` called
``validate_glm_inputs`` before fitting; the surface branch had no equivalent.
A NaN-bearing or row-count-mismatched design matrix would have flowed straight
into nilearn's ``run_glm`` and produced silently corrupt contrast maps.

These tests cover ``validate_design_matrix`` directly (the new helper that
both paths now share) plus end-to-end behavior of ``SurfaceGLM.fit`` when
inputs are degenerate.

Specifically:

1. **NaN design matrix** — validation flags the offending column and raises
   ``is_valid=False`` rather than silently passing.
2. **Row-count mismatch** — design matrix rows != BOLD timepoints flags
   the dimensional bug.  Without this catch, nilearn would broadcast or
   raise an opaque numpy error far downstream.
3. **Infinite values** — flagged identically to NaN.
4. **Empty design matrix** — flagged with an empty-matrix error.
5. **Clean inputs (positive control)** — validation returns ``is_valid=True``
   so the surface fit proceeds normally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from network_glm.lev1.processing.glm import validate_design_matrix


def _clean_dm(n_tp: int = 100, n_reg: int = 4) -> pd.DataFrame:
    """Build a non-degenerate design matrix with an explicit intercept."""
    rng = np.random.default_rng(0)
    cols = {f"r{i}": rng.normal(size=n_tp) for i in range(n_reg)}
    cols["constant"] = np.ones(n_tp)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_validate_flags_nan_in_design_matrix():
    """A single NaN cell triggers an explicit error naming the column."""
    dm = _clean_dm()
    dm.loc[5, "r1"] = np.nan
    result = validate_design_matrix(dm, n_scans=dm.shape[0])
    assert result["is_valid"] is False, "NaN in design matrix should fail validation"
    assert any("NaN" in e for e in result["errors"]), result
    assert any(
        "'r1'" in e for e in result["errors"]
    ), f'Error should name the offending column; got {result["errors"]}'


def test_validate_flags_infinite_values():
    """Inf cells trigger an explicit error naming the column."""
    dm = _clean_dm()
    dm.loc[10, "r2"] = np.inf
    result = validate_design_matrix(dm, n_scans=dm.shape[0])
    assert result["is_valid"] is False
    assert any("infinite" in e.lower() for e in result["errors"]), result
    assert any("'r2'" in e for e in result["errors"])


def test_validate_flags_row_count_mismatch():
    """Design matrix rows != BOLD timepoints fails with an explicit message.

    This is the failure mode where someone trims BOLD but not the events
    that build the design matrix, and the resulting size mismatch silently
    propagates into nilearn's GLM with garbled output.
    """
    dm = _clean_dm(n_tp=100)
    result = validate_design_matrix(dm, n_scans=80)
    assert result["is_valid"] is False
    msg = " ".join(result["errors"])
    assert (
        "100" in msg and "80" in msg
    ), f'Error should name both dimensions; got {result["errors"]}'


def test_validate_flags_empty_design_matrix():
    """An empty design matrix is caught before any other check."""
    dm = pd.DataFrame()
    result = validate_design_matrix(dm, n_scans=100)
    assert result["is_valid"] is False
    assert any("empty" in e.lower() for e in result["errors"])


# ---------------------------------------------------------------------------
# Positive control
# ---------------------------------------------------------------------------


def test_validate_passes_for_clean_inputs():
    """Clean design matrix + matching n_scans returns is_valid=True."""
    dm = _clean_dm(n_tp=100)
    result = validate_design_matrix(dm, n_scans=100)
    assert result["is_valid"] is True, result
    assert result["errors"] == []


def test_validate_recognizes_non_named_intercept():
    """A constant column that isn't named 'constant' should still be
    recognised as an intercept.

    fMRIPrep's cosine00 column is constant on very short scans, and other
    pipelines name their intercepts differently (e.g. ``baseline``).  The
    'no intercept' warning should only fire when there's actually no
    constant-valued column in the matrix.
    """
    rng = np.random.default_rng(0)
    n_tp = 100
    dm = pd.DataFrame(
        {
            "r0": rng.normal(size=n_tp),
            "r1": rng.normal(size=n_tp),
            "cosine00": np.full(n_tp, 0.1),  # constant non-zero
        }
    )
    result = validate_design_matrix(dm, n_scans=n_tp)
    assert result["is_valid"] is True, result
    no_intercept_warning = [w for w in result["warnings"] if "intercept" in w.lower()]
    assert not no_intercept_warning, (
        f'Constant non-zero column should suffice as intercept; got '
        f'warnings: {result["warnings"]}'
    )


# ---------------------------------------------------------------------------
# Surface end-to-end: process_surface_run raises on NaN
# ---------------------------------------------------------------------------


def test_surface_run_raises_on_nan_design_matrix(tmp_path, monkeypatch):
    """End-to-end: process_surface_run must raise rather than silently
    feeding NaN to nilearn's run_glm.

    We stub out the file-loading and the GLM fit so the test isolates the
    validation gate.  Before this fix the function would have fallen
    through into ``SurfaceGLM.fit`` with a NaN design matrix.
    """
    from network_glm.lev1 import runner as run_module

    # Stub surface loading to return a deterministic ndarray
    n_tp = 100

    def fake_load(_path, dummy_scans=0):
        return np.random.randn(n_tp, 50).astype(np.float32)

    monkeypatch.setattr(run_module, "load_surface_data", fake_load)

    # Stub SurfaceGLM to assert it never gets called when validation fails
    fit_calls = []

    class FailIfCalled:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, data, dm):
            fit_calls.append((data, dm))
            return self

    monkeypatch.setattr(run_module, "SurfaceGLM", FailIfCalled)

    # NaN-bearing design matrix
    dm = _clean_dm(n_tp=n_tp)
    dm.loc[3, "r0"] = np.nan

    from argparse import Namespace

    args = Namespace(
        fmriprep_dir=str(tmp_path),
        subj_id="sub-test",
        task_name="flanker",
        smoothing_fwhm=None,
        space="fsaverage6",
    )

    run_files = {"left_surface": "L.func.gii", "right_surface": "R.func.gii"}
    dirs = {"indiv_contrasts": tmp_path, "quality_control": tmp_path, "task_residuals": tmp_path}

    with pytest.raises(ValueError, match="validation failed"):
        run_module.process_surface_run(
            run_files=run_files,
            design_matrix=dm,
            contrasts={},
            args=args,
            dirs=dirs,
            base_filename="sub-test_task-flanker_run-1",
            tr=1.5,
            dummy_scans=0,
            compute_residuals=False,
            surface_space="fsaverage6",
        )

    assert fit_calls == [], (
        "SurfaceGLM.fit should never run when validation fails; got " f"{len(fit_calls)} call(s)."
    )
