"""Tests for frame-level motion censoring via ``motion_outlier_NN`` columns.

Coverage (per the methodology check that triggered this change):

1. The base confound regex must capture every ``motion_outlier_NN`` column
   fmriprep emits for a scan. Naming follows fmriprep convention
   ``motion_outlier00, motion_outlier01, ...``.

2. Bare motion-outlier-adjacent column names (``motion_outlier``,
   ``rmsd``, ``framewise_displacement``) must NOT be picked up — only the
   indexed ``motion_outlier\\d+`` columns. ``framewise_displacement`` is a
   continuous metric; pulling it in would double-count motion variance the
   24-parameter model already handles.

3. End-to-end: a synthetic confounds.tsv with motion_outlier spike columns
   should round-trip through ``load_and_process_confounds`` with the spike
   columns present and the continuous motion params intact.

4. Edge case: a scan with zero motion outliers (no ``motion_outlier_*``
   columns at all) must still work — the regex matches nothing, and the
   pipeline gracefully proceeds with only the 24 motion + drift
   regressors. This was the historical behavior; verify it survives.

The change matters because the 24-parameter Friston motion model absorbs
*continuous* motion variance but doesn't cleanly handle isolated spikes —
a 0.8 mm jolt for one TR leaks into task betas and into the residuals
that prep-mshbm consumes. The one-hot spike regressors effectively delete
those TRs from the fit, the same idea XCP-D applies as a separate
frame-censoring step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from network_glm.lev1.processing.confounds import (
    _get_base_confound_pattern,
    load_and_process_confounds,
)


def _make_confounds_tsv(
    tmp_path, n_tp: int = 200, n_outliers: int = 5, discovery_nback: bool = False
) -> str:
    """Write a synthetic fMRIPrep-style confounds.tsv to disk.

    Includes the 24-param motion model, DCT cosines, a handful of
    motion_outlier_NN spike columns, and a couple of decoy columns that
    must NOT be picked up by the regex (``rmsd``, ``framewise_displacement``,
    ``a_comp_cor_00``, etc.).
    """
    rng = np.random.default_rng(0)
    base = {
        "trans_x": rng.normal(size=n_tp),
        "trans_y": rng.normal(size=n_tp),
        "trans_z": rng.normal(size=n_tp),
        "rot_x": rng.normal(size=n_tp),
        "rot_y": rng.normal(size=n_tp),
        "rot_z": rng.normal(size=n_tp),
        "trans_x_derivative1": rng.normal(size=n_tp),
        "trans_y_derivative1": rng.normal(size=n_tp),
        "trans_z_derivative1": rng.normal(size=n_tp),
        "rot_x_derivative1": rng.normal(size=n_tp),
        "rot_y_derivative1": rng.normal(size=n_tp),
        "rot_z_derivative1": rng.normal(size=n_tp),
        "trans_x_power2": rng.normal(size=n_tp),
        "trans_y_power2": rng.normal(size=n_tp),
        "trans_z_power2": rng.normal(size=n_tp),
        "rot_x_power2": rng.normal(size=n_tp),
        "rot_y_power2": rng.normal(size=n_tp),
        "rot_z_power2": rng.normal(size=n_tp),
        "trans_x_derivative1_power2": rng.normal(size=n_tp),
        "trans_y_derivative1_power2": rng.normal(size=n_tp),
        "trans_z_derivative1_power2": rng.normal(size=n_tp),
        "rot_x_derivative1_power2": rng.normal(size=n_tp),
        "rot_y_derivative1_power2": rng.normal(size=n_tp),
        "rot_z_derivative1_power2": rng.normal(size=n_tp),
    }
    # DCT drift cosines — fmriprep emits cosine00..cosineNN
    for i in range(10):
        base[f"cosine{i:02d}"] = rng.normal(size=n_tp)
    # Tissue confounds (FC denoising) — must NOT enter the task design matrix
    base["global_signal"] = rng.normal(size=n_tp)
    base["csf"] = rng.normal(size=n_tp)
    base["white_matter"] = rng.normal(size=n_tp)
    # aCompCor — should also be excluded from the task design matrix
    for i in range(5):
        base[f"a_comp_cor_{i:02d}"] = rng.normal(size=n_tp)
    # Decoys that must NOT be picked up by ``motion_outlier\d+``.
    base["framewise_displacement"] = rng.normal(size=n_tp).astype(float)
    base["rmsd"] = rng.normal(size=n_tp).astype(float)
    # One-hot motion-outlier spikes: each fires at a single TR.
    for i in range(n_outliers):
        col = np.zeros(n_tp)
        col[10 + i * 7] = 1.0
        base[f"motion_outlier{i:02d}"] = col

    df = pd.DataFrame(base)
    path = tmp_path / "confounds.tsv"
    df.to_csv(path, sep="\t", index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Regex selection
# ---------------------------------------------------------------------------


def test_base_pattern_matches_motion_outlier_columns():
    """``motion_outlier_NN`` columns are captured by the base regex."""
    pattern = _get_base_confound_pattern("flanker", "validation")
    for i in range(10):
        col = f"motion_outlier{i:02d}"
        # The pattern is used with pandas.filter(regex=...); .filter does a
        # search, not a fullmatch, so use re.search to mirror behavior.
        import re

        assert re.search(pattern, col), (
            f"Base confound regex {pattern!r} did not match expected spike " f"column {col!r}."
        )


def test_base_pattern_excludes_continuous_motion_metrics():
    """``framewise_displacement`` and ``rmsd`` must NOT match the base regex.

    Pulling them in would double-count motion variance the 24-parameter
    model already handles, and could destabilize the GLM by introducing
    near-collinear columns.
    """
    import re

    pattern = _get_base_confound_pattern("flanker", "validation")
    for decoy in ("framewise_displacement", "rmsd", "dvars", "std_dvars"):
        assert not re.search(pattern, decoy), (
            f"Decoy column {decoy!r} should NOT be selected by the base "
            f"regex; pattern={pattern!r}."
        )


# ---------------------------------------------------------------------------
# End-to-end through load_and_process_confounds
# ---------------------------------------------------------------------------


def test_load_and_process_confounds_includes_motion_outlier_columns(tmp_path):
    """A synthetic confounds.tsv with 5 spike columns should round-trip
    through ``load_and_process_confounds`` with those 5 columns present.
    """
    path = _make_confounds_tsv(tmp_path, n_tp=200, n_outliers=5)
    df = load_and_process_confounds(path, "flanker", sample_type="validation")

    outlier_cols = [c for c in df.columns if c.startswith("motion_outlier")]
    assert len(outlier_cols) == 5, f"Expected 5 motion_outlier columns; got {outlier_cols}"

    # Each spike column is one-hot — exactly one nonzero TR, value 1.0.
    for col in outlier_cols:
        nonzero = (df[col] != 0).sum()
        assert nonzero == 1, (
            f"{col} should be one-hot (single 1, rest zeros); " f"got {nonzero} nonzero rows"
        )


def test_load_and_process_confounds_handles_zero_outlier_scan(tmp_path):
    """A scan with no motion_outlier columns at all must still load.

    Subjects with very clean motion across the scan will produce a
    confounds.tsv without any motion_outlier_NN columns. The regex matches
    nothing, so the design matrix gets only the 24 motion + drift terms,
    which is the historical behavior. This must not crash or warn.
    """
    path = _make_confounds_tsv(tmp_path, n_tp=200, n_outliers=0)
    df = load_and_process_confounds(path, "flanker", sample_type="validation")

    outlier_cols = [c for c in df.columns if c.startswith("motion_outlier")]
    assert outlier_cols == []
    # The 24-parameter model + cosines should still be intact.
    motion_cols = [c for c in df.columns if c.startswith(("trans_", "rot_"))]
    assert len(motion_cols) == 24, (
        f"24-parameter motion model should still produce 24 columns; "
        f"got {len(motion_cols)}: {motion_cols}"
    )


def test_load_and_process_confounds_excludes_tissue_signals(tmp_path):
    """``global_signal``, ``csf``, ``white_matter``, and ``a_comp_cor_*``
    must NOT enter the task-GLM confound set.

    Tissue signals correlate with task activity in some regions; pulling
    them into the task GLM would partial out real BOLD effects. They're
    reserved for the post-residual FC-denoising step (gated by
    ``--fc-confounds``).
    """
    path = _make_confounds_tsv(tmp_path, n_tp=200, n_outliers=2)
    df = load_and_process_confounds(path, "flanker", sample_type="validation")

    for tissue_col in ("global_signal", "csf", "white_matter"):
        assert tissue_col not in df.columns, (
            f"{tissue_col} must not be in the task-GLM confound set; "
            f"it belongs in the FC-confounds path (--fc-confounds)."
        )
    a_comp = [c for c in df.columns if c.startswith("a_comp_cor")]
    assert a_comp == [], f"aCompCor columns leaked into task confounds: {a_comp}"


def test_load_and_process_confounds_excludes_decoy_motion_metrics(tmp_path):
    """``framewise_displacement`` and ``rmsd`` must NOT enter the design.

    They duplicate variance the 24-parameter model already handles, and
    would risk near-collinearity in the design matrix.
    """
    path = _make_confounds_tsv(tmp_path, n_tp=200, n_outliers=2)
    df = load_and_process_confounds(path, "flanker", sample_type="validation")
    for decoy in ("framewise_displacement", "rmsd"):
        assert decoy not in df.columns, (
            f"{decoy} must not be in the task-GLM confound set; "
            f"it duplicates variance the 24-parameter motion model handles."
        )


# ---------------------------------------------------------------------------
# Spike regressors are one-hot orthogonal: VIF == 1
# ---------------------------------------------------------------------------


def test_motion_outlier_columns_are_orthogonal_to_continuous_motion(tmp_path):
    """One-hot motion-outlier columns are by construction orthogonal to the
    continuous 24-parameter motion regressors (a single 1 at a TR vs.
    smoothly varying continuous values).  Confirm correlations are <0.1.
    Avoids the failure mode where adding spikes inflates the VIF of the
    Friston regressors.
    """
    path = _make_confounds_tsv(tmp_path, n_tp=400, n_outliers=8)
    df = load_and_process_confounds(path, "flanker", sample_type="validation")

    spike_cols = [c for c in df.columns if c.startswith("motion_outlier")]
    motion_cols = [c for c in df.columns if c.startswith(("trans_", "rot_"))]
    assert spike_cols and motion_cols, "fixture setup error"

    corrs = df[spike_cols + motion_cols].corr().loc[spike_cols, motion_cols]
    max_abs = corrs.abs().values.max()
    assert max_abs < 0.25, (
        f"A motion_outlier spike column is unexpectedly correlated with a "
        f"continuous motion regressor (max |r| = {max_abs:.3f}). One-hot "
        f"spikes should be near-orthogonal to continuous motion."
    )
