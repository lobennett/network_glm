"""Design matrix creation for GLM analysis."""

import logging
import re

import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor

from network_glm.task_config.loader import get_regressor_config

logger = logging.getLogger(__name__)

_CONSTANT_SENTINEL_RE = re.compile(r"^constant_(-?\d+(?:\.\d+)?)_column$")


def _resolve_column_or_constant(events: pd.DataFrame, col_spec: str) -> np.ndarray:
    """Resolve a regressor's amplitude/duration spec to a per-event vector.

    The loader encodes YAML literals as ``constant_{N}_column`` sentinels so
    the numeric value survives round-tripping (a previous bug collapsed every
    numeric duration to 1.0, silently making ``duration: 10`` model a
    one-second epoch). This function:

      * Parses the sentinel back to its float value and returns a constant
        vector at that value.
      * Otherwise treats ``col_spec`` as an events.tsv column name and pulls
        the per-event values from it.
    """
    match = _CONSTANT_SENTINEL_RE.match(col_spec)
    if match:
        value = float(match.group(1))
        return np.full(len(events), value, dtype=float)
    return events[col_spec].values


def create_regressor(
    events_df: pd.DataFrame,
    regressor_config: dict[str, str],
    n_scans: int,
    regressor_name: str,
    tr: float = 1.49,
) -> tuple[pd.DataFrame, tuple]:
    """Create a single regressor from events data.

    Args:
        events_df: Events dataframe with trial information
        regressor_config: Configuration dictionary for this regressor
        n_scans: Number of scans in the run
        regressor_name: Name of the regressor
        tr: Repetition time in seconds

    Returns:
        Tuple of (regressor DataFrame, 3-column format tuple)

    Examples:
        >>> events = pd.DataFrame({
        ...     'onset': [1.0, 3.0], 'duration': [1.0, 1.0],
        ...     'trial_type': ['go', 'go'], 'response_time': [0.5, 0.6]
        ... })
        >>> config = {'amplitude_column': 'constant_1_column', 'duration_column': 'duration', 'subset': "trial_type == 'go'"}
        >>> reg_df, reg_3col = create_regressor(events, config, 100, 'go_trials')
    """
    try:
        # Apply subset filter
        subset_query = regressor_config["subset"]
        subset_events = events_df.query(subset_query) if subset_query else events_df

        if subset_events.empty:
            # Create a zero regressor when no events match the subset condition
            logger.warning("No events for regressor '%s', creating zero regressor", regressor_name)
            regressor_values = np.zeros(n_scans)
            regressor_df = pd.DataFrame({regressor_name: regressor_values})
            regressor_3col = ([], [], [])  # Empty 3-column format
            return regressor_df, regressor_3col

        # Get amplitude and duration columns
        amp_col = regressor_config["amplitude_column"]
        dur_col = regressor_config["duration_column"]

        # Create 3-column format (onset, duration, amplitude)
        onsets = subset_events["onset"].values
        durations = _resolve_column_or_constant(subset_events, dur_col)
        amplitudes = _resolve_column_or_constant(subset_events, amp_col)

        # Drop rows with NaN in any of the 3 columns — nilearn's compute_regressor
        # silently produces garbled output when passed NaN values. This protects
        # against "n/a" strings in event columns (e.g., response_time for omissions).
        valid = (~pd.isna(onsets)) & (~pd.isna(durations)) & (~pd.isna(amplitudes))
        if not valid.all():
            n_dropped = (~valid).sum()
            logger.warning(
                "Dropped %d rows with NaN in onset/duration/amplitude for regressor '%s'",
                n_dropped,
                regressor_name,
            )
            onsets = onsets[valid]
            durations = durations[valid]
            amplitudes = amplitudes[valid]

        if len(onsets) == 0:
            logger.warning("No valid events for regressor '%s' after NaN filter", regressor_name)
            regressor_values = np.zeros(n_scans)
            regressor_df = pd.DataFrame({regressor_name: regressor_values})
            regressor_3col = ([], [], [])
            return regressor_df, regressor_3col

        # Shift frame_times by +TR/2 to align with fMRIPrep's slice timing
        # correction, which references the middle slice (Poldrack & Mumford,
        # 2021; https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/).
        frame_times = np.arange(n_scans) * tr + tr / 2
        regressor_values, _ = compute_regressor(
            exp_condition=(onsets, durations, amplitudes),
            hrf_model="spm",
            frame_times=frame_times,
        )

        regressor_df = pd.DataFrame({regressor_name: regressor_values.flatten()})
        regressor_3col = (onsets.tolist(), durations.tolist(), amplitudes.tolist())

        return regressor_df, regressor_3col

    except Exception as e:
        raise ValueError(f"Failed to create regressor {regressor_name}: {e}") from e


def create_design_matrix(
    events_df: pd.DataFrame,
    confounds_df: pd.DataFrame,
    task_name: str,
    n_scans: int,
    tr: float = 1.49,
) -> tuple[pd.DataFrame, list[tuple]]:
    """Create complete design matrix from events and confounds.

    Args:
        events_df: Events dataframe from BIDS events.tsv
        confounds_df: Confounds dataframe from fMRIPrep
        task_name: Name of the task
        n_scans: Number of scans in the run
        tr: Repetition time in seconds

    Returns:
        Tuple of (design_matrix DataFrame, list of regressor 3-column tuples)

    Examples:
        >>> events = pd.DataFrame({
        ...     'onset': [1.0, 3.0], 'duration': [1.0, 1.0],
        ...     'trial_type': ['go', 'stop_success'], 'key_press': [1, -1],
        ...     'correct_response': [1, -1], 'response_time': [0.5, -1]
        ... })
        >>> confounds = pd.DataFrame({'trans_x': [0.1, 0.2], 'trans_y': [0.0, 0.1]})
        >>> dm, reg_3col = create_design_matrix(events, confounds, 'stopSignal', 100)
    """
    # Get regressor configuration for this task
    task_config = get_regressor_config(task_name)

    # Create regressors
    regressors = {}
    regressor_3cols = []

    for name, config in task_config.items():
        reg_df, reg_3col = create_regressor(events_df, config, n_scans, name, tr)
        regressors[name] = reg_df
        regressor_3cols.append((reg_3col, name))

    # Combine task regressors with fMRIPrep confounds. The confounds include
    # cosine basis regressors (cosine00, cosine01, ...) that implement a
    # discrete cosine transform high-pass filter. Including them in the
    # design matrix is equivalent to explicit high-pass filtering of the
    # BOLD signal, so no separate high_pass parameter is needed in the GLM.
    design_matrices = [*regressors.values(), confounds_df]
    design_matrix = pd.concat(design_matrices, axis=1)

    # Ensure an intercept/constant term exists in the design matrix.
    # fMRIPrep confounds typically include cosine00 (a constant column),
    # but if absent, the GLM needs an explicit intercept.
    has_constant = any(
        design_matrix[col].nunique() == 1 and design_matrix[col].iloc[0] != 0
        for col in design_matrix.columns
    )
    if not has_constant:
        logger.warning("No constant/intercept column detected; adding one")
        design_matrix["constant"] = 1.0

    return design_matrix, regressor_3cols
