"""Task-related utility functions."""

from pathlib import Path

from network_glm.task_config.loader import get_task_parameters


def detect_sample_type(bids_dir: Path) -> str:
    """Auto-detect sample type from directory path.

    Args:
        bids_dir: Path to BIDS directory

    Returns:
        'discovery' if 'discovery' in path, otherwise 'validation'

    Examples:
        >>> detect_sample_type(Path('/data/discovery/bids'))
        'discovery'
        >>> detect_sample_type(Path('/data/validation/bids'))
        'validation'
    """
    return "discovery" if "discovery" in str(bids_dir) else "validation"


def get_expected_sessions(task_name: str) -> int | None:
    """Get expected number of sessions for a task from YAML config.

    Args:
        task_name: Name of the task

    Returns:
        Number of expected sessions as defined in the task's YAML config.
        Base tasks return an int (5); dual tasks return ``None`` because a
        dual pairing appears in a variable, not fixed, number of sessions
        (their YAML sets ``expected_sessions: null``).

    Examples:
        >>> get_expected_sessions('flanker')
        5
        >>> get_expected_sessions('stopSignalWDirectedForgetting') is None
        True
    """
    params = get_task_parameters(task_name)
    return params["expected_sessions"]
