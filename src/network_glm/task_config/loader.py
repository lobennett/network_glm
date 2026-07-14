"""Load and validate task configurations from YAML files.

Each task has a YAML file in task_config/tasks/ defining its regressors,
contrasts, and parameters. This module provides the same API as the old
tasks.py so callers do not need to change.

YAML regressor format:
    amplitude: 1              -> constant amplitude of 1
    amplitude: omission       -> use the 'omission' column from events
    duration: 1               -> constant duration of 1 second
    duration: response_time   -> use the 'response_time' column from events
    subset: null              -> use all rows (no filter)
    subset: "trial_type == 'go'"  -> pandas query string
"""

import copy
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from network_glm.acquisition import (
    N_DUMMY as DEFAULT_DUMMY_SCANS,
)
from network_glm.acquisition import (
    TR_SECONDS as DEFAULT_TR,
)

# Path to the task battery YAML (base + dual task lists)
_BATTERY_YAML = Path(__file__).parent / "battery.yaml"

logger = logging.getLogger(__name__)

# Default analysis parameters (TR/dummy-scans come from the single source of
# truth in core.acquisition, RF-6; can be overridden per task in YAML).
DEFAULT_MIN_RT = 0.2

# Directory containing per-task YAML files
_TASKS_DIR = Path(__file__).parent / "tasks"

# Required fields in each YAML file
_REQUIRED_FIELDS = {"regressors", "contrasts"}
_REQUIRED_REGRESSOR_FIELDS = {"amplitude", "duration", "subset"}


class TaskNotConfiguredError(ValueError):
    """Raised when a task's YAML exists but has `regressors: null` (placeholder)."""


class ContrastFormulaError(ValueError):
    """Raised when a contrast formula references an undeclared regressor name.

    Caught by ``except ValueError`` callers for backward compatibility.
    """


# Regex to extract identifier tokens from a contrast formula string.
# Matches bare Python identifiers (letters/underscores, then alphanumerics).
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


def _validate_contrasts(
    task_name: str,
    regressors: dict[str, Any],
    contrasts: dict[str, str],
) -> None:
    """Validate that every contrast formula only references declared regressor names.

    Args:
        task_name: Name of the task (for error messages).
        regressors: Mapping of declared regressor names to their config dicts.
        contrasts: Mapping of contrast names to formula strings.

    Raises:
        ContrastFormulaError: If any formula token is not a declared regressor name.
    """
    declared = set(regressors.keys())
    for cname, formula in contrasts.items():
        tokens = set(_IDENT_RE.findall(formula))
        unknown = tokens - declared
        if unknown:
            raise ContrastFormulaError(
                f"task '{task_name}' contrast '{cname}': "
                f"unknown regressor(s) {sorted(unknown)} "
                f"(declared: {sorted(declared)})"
            )


def _load_yaml(task_name: str) -> dict[str, Any]:
    """Load a single task YAML file.

    Args:
        task_name: Name of the task (matches filename without extension).

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If required fields are missing.
    """
    yaml_path = _TASKS_DIR / f"{task_name}.yaml"
    if not yaml_path.exists():
        available = list_available_tasks()
        raise FileNotFoundError(
            f"No config file for task '{task_name}'. " f"Available tasks: {available}"
        )

    with open(yaml_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate top-level fields
    missing = _REQUIRED_FIELDS - set(config.keys())
    if missing:
        raise ValueError(f"Task config '{task_name}' is missing required fields: {missing}")

    # Skip per-regressor validation for placeholder YAMLs (regressors: null).
    # get_regressor_config raises TaskNotConfiguredError on these.
    if config.get("regressors") is None:
        return config

    # Validate each regressor has required fields
    for reg_name, reg_config in config.get("regressors", {}).items():
        if not isinstance(reg_config, dict):
            raise ValueError(
                f"Regressor '{reg_name}' in task '{task_name}' must be a dict, "
                f"got {type(reg_config).__name__}"
            )
        missing_reg = _REQUIRED_REGRESSOR_FIELDS - set(reg_config.keys())
        if missing_reg:
            raise ValueError(
                f"Regressor '{reg_name}' in task '{task_name}' is missing "
                f"required fields: {missing_reg}"
            )

    # Validate contrast formulas reference only declared regressor names.
    # Skip tasks whose contrasts dict is empty or None (e.g. stopSignalWDirectedForgetting).
    regressors = config.get("regressors") or {}
    contrasts = config.get("contrasts") or {}
    if regressors and contrasts:
        _validate_contrasts(task_name, regressors, contrasts)

    return config


@lru_cache(maxsize=32)
def _get_task_config(task_name: str) -> dict[str, Any]:
    """Cached wrapper around _load_yaml."""
    return _load_yaml(task_name)


def _convert_regressor_config(yaml_regressors: dict) -> dict[str, dict[str, str]]:
    """Convert YAML regressor format to the internal format used by design.py.

    YAML format:
        amplitude: 1 | 10 | <column_name>
        duration: 1 | 10 | <column_name>
        subset: null | query_string

    Internal format (used by create_regressor):
        amplitude_column: 'constant_{N}_column' | <column_name>
        duration_column: 'constant_{N}_column' | <column_name>
        subset: None | query_string

    The ``constant_{N}_column`` sentinel encodes the literal numeric value so
    a YAML ``duration: 10`` is preserved end-to-end (instead of collapsing to
    1s). ``create_regressor`` parses the value back out by regex. Plain
    strings are passed through as events-column names.
    """
    converted = {}
    for name, cfg in yaml_regressors.items():
        amp = cfg["amplitude"]
        dur = cfg["duration"]

        amp_col = _encode_numeric_or_column(amp)
        dur_col = _encode_numeric_or_column(dur)

        # subset: null in YAML becomes None in Python
        subset = cfg.get("subset")

        converted[name] = {
            "amplitude_column": amp_col,
            "duration_column": dur_col,
            "subset": subset,
        }

    return converted


def _encode_numeric_or_column(value) -> str:
    """Encode a YAML numeric value as a ``constant_{N}_column`` sentinel, or
    pass a column-name string through unchanged.

    Integer-valued floats render as integers (``1.0 -> 'constant_1_column'``)
    so the sentinel stays canonical across float vs int YAML literals.
    """
    if isinstance(value, bool):
        # Treat bools as numeric — Python's True/False are int subclasses,
        # so explicit guard before the (int, float) check below.
        return f"constant_{int(value)}_column"
    if isinstance(value, int | float):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return f"constant_{value}_column"
    return str(value)


@lru_cache(maxsize=1)
def _load_battery() -> dict[str, list[str]]:
    """Load battery.yaml and return {'base': [...], 'dual': [...]}.

    Cached so the YAML is read at most once per process.
    """
    with open(_BATTERY_YAML, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data


def get_base_tasks() -> list[str]:
    """Return the ordered list of 8 base (single-task) paradigm names."""
    return list(_load_battery()["base"])


def get_dual_tasks() -> list[str]:
    """Return the ordered list of 11 dual-task paradigm names."""
    return list(_load_battery()["dual"])


def get_all_tasks() -> list[str]:
    """Return base + dual tasks in canonical order (19 tasks total)."""
    batt = _load_battery()
    return list(batt["base"]) + list(batt["dual"])


def list_available_tasks() -> list[str]:
    """List all tasks with YAML config files.

    Returns:
        Sorted list of task names.
    """
    if not _TASKS_DIR.exists():
        return []
    return sorted(p.stem for p in _TASKS_DIR.glob("*.yaml"))


def get_regressor_config(task_name: str) -> dict[str, dict[str, str]]:
    """Get regressor configuration for a task.

    Args:
        task_name: Name of the task.

    Returns:
        Dictionary mapping regressor names to their configuration.

    Raises:
        FileNotFoundError: If no YAML file exists for the task.
        TaskNotConfiguredError: If the YAML is a placeholder (regressors: null).
        ValueError: If the regressor config is empty.
    """
    config = _get_task_config(task_name)
    if config.get("regressors") is None:
        raise TaskNotConfiguredError(
            f"task {task_name!r} has no regressors defined "
            f"(placeholder YAML — fill in regressors: and contrasts: before running lev1)"
        )
    regressors = config.get("regressors", {})
    if not regressors:
        raise ValueError(
            f"Regressor config is empty for task '{task_name}'. "
            "Define regressors in the YAML file before running."
        )
    return _convert_regressor_config(regressors)


# Canonical name used by lev1 callers; alias kept for backward compatibility.
get_task_regressors = get_regressor_config


def get_task_contrasts(task_name: str) -> dict[str, str]:
    """Get contrast definitions for a task.

    Args:
        task_name: Name of the task.

    Returns:
        Dictionary mapping contrast names to formula strings.

    Raises:
        FileNotFoundError: If no YAML file exists for the task.
        ValueError: If the contrast config is empty.
    """
    config = _get_task_config(task_name)
    contrasts = config.get("contrasts", {})
    if not contrasts:
        raise ValueError(
            f"Contrast config is empty for task '{task_name}'. "
            "Define contrasts in the YAML file before running."
        )
    return dict(contrasts)


def get_task_parameters(task_name: str) -> dict[str, Any]:
    """Get general parameters for a task.

    Args:
        task_name: Name of the task.

    Returns:
        Dictionary with keys: tr, dummy_scans, min_rt, expected_sessions.
    """
    config = _get_task_config(task_name)
    return {
        "tr": config.get("tr", DEFAULT_TR),
        "dummy_scans": config.get("dummy_scans", DEFAULT_DUMMY_SCANS),
        "min_rt": config.get("min_rt", DEFAULT_MIN_RT),
        "expected_sessions": config.get("expected_sessions", 5),
    }


def get_raw_yaml_config(task_name: str) -> dict[str, Any]:
    """Get the raw YAML config for inspection/debugging.

    Args:
        task_name: Name of the task.

    Returns:
        A deep copy of the full parsed YAML dictionary. The underlying parse is
        ``lru_cache``d and the YAML is nested (regressors/contrasts/parameters),
        so a deepcopy is returned to prevent a caller mutating the cached dict.
    """
    return copy.deepcopy(_get_task_config(task_name))


# Backward-compatible constants (used by events.py, confounds.py, etc.)
TR = DEFAULT_TR
DUMMY_SCANS = DEFAULT_DUMMY_SCANS
MIN_RT = DEFAULT_MIN_RT
