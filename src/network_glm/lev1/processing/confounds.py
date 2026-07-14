"""Enhanced confounds processing with task-specific selection and dummy scan handling."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _get_base_confound_pattern(task_name: str, sample_type: str) -> str:
    """Get base confound selection pattern.

    Args:
        task_name: Name of the task
        sample_type: Sample type ('discovery' or 'validation')

    Returns:
        Regex pattern for confound selection
    """
    # Base pattern for motion (24 parameter Friston model), drift (DCT
    # cosines), and per-frame motion-outlier spike regressors.
    #
    # ``motion_outlier_NN`` are one-hot indicator columns fMRIPrep emits for
    # every TR with FD > 0.5 mm. The 24-parameter model absorbs *continuous*
    # motion variance (e.g. drift, slow head movement); isolated frame-level
    # spikes don't get cleanly handled by it and would otherwise leak into
    # task betas and into the residuals consumed by prep-mshbm. The one-hot
    # spike regressors effectively delete those single TRs from the fit, the
    # same idea XCP-D applies as a separate frame-censoring step.
    #
    # Run-level FD exclusion (.bidsignore: drop scans where >20% of TRs
    # exceed FD>0.5 mm) handles whole-scan motion; the spike regressors
    # catch the residual within-scan high-motion frames.
    base_pattern = (
        "cosine|trans_[xyz]$|trans_[xyz]_derivative1$|trans_[xyz]_power2$|"
        "trans_[xyz]_derivative1_power2$|rot_[xyz]$|rot_[xyz]_derivative1$|"
        "rot_[xyz]_power2$|rot_[xyz]_derivative1_power2$|motion_outlier\\d+"
    )

    # Cap DCT cosine high-pass regressors for (sample, task) combinations whose
    # runs are short enough that the full cosine set induces rank deficiency /
    # collinearity with the task design. Config-driven via thresholds.yaml
    # `confounds.cosine_max_index` (previously a hardcoded discovery/nBack special
    # case; lifted to config so the analytic choice is auditable). Behavior is
    # unchanged: discovery/nBack -> cosine0[0-4], everything else -> full set.
    from network_glm.thresholds import confounds_cosine_caps

    max_idx = confounds_cosine_caps().get(sample_type, {}).get(task_name)
    if max_idx is not None:
        return base_pattern.replace("cosine", f"cosine0[0-{int(max_idx)}]")
    return base_pattern


def load_and_process_confounds(
    confounds_file: str | Path,
    task_name: str,
    sample_type: str = "validation",
    dummy_scans: int = 0,
    additional_patterns: list[str] | None = None,
) -> pd.DataFrame:
    """Load and process confounds with task-specific selection.

    Args:
        confounds_file: Path to confounds file
        task_name: Name of the task
        sample_type: Sample type
        dummy_scans: Number of dummy scans to remove
        additional_patterns: Additional regex patterns

    Returns:
        Processed confounds dataframe

    Examples:
        >>> confounds = load_and_process_confounds(
        ...     'confounds.tsv', 'stopSignal', 'validation'
        ... )
    """
    # Load confounds
    confounds_df = pd.read_csv(confounds_file, sep="\t", na_values=["n/a"]).fillna(0)

    # Remove dummy scans
    if dummy_scans > 0:
        confounds_df = confounds_df.iloc[dummy_scans:].reset_index(drop=True)

    # Get base pattern for confound selection
    pattern = _get_base_confound_pattern(task_name, sample_type)

    # Add additional patterns if provided
    if additional_patterns:
        pattern = "|".join([pattern] + additional_patterns)

    # Filter and return confounds
    selected_confounds = confounds_df.filter(regex=pattern).reset_index(drop=True)
    return selected_confounds


def get_fc_confounds(confounds_df: pd.DataFrame) -> pd.DataFrame:
    """Extract tissue-based confounds for FC analysis.

    Following Du et al. 2025 (Neuron): global_signal, csf, white_matter,
    plus temporal derivatives of each.

    Args:
        confounds_df: Full confounds DataFrame from fMRIPrep TSV

    Returns:
        DataFrame with available FC confound columns. Empty if none found.
    """
    fc_columns = [
        "global_signal",
        "global_signal_derivative1",
        "csf",
        "csf_derivative1",
        "white_matter",
        "white_matter_derivative1",
    ]
    available = [c for c in fc_columns if c in confounds_df.columns]
    if not available:
        logger.warning("No tissue confound columns found in confounds TSV")
        return pd.DataFrame()
    return confounds_df[available].copy()
