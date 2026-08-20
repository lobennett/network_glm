"""Single source for study-level analysis thresholds (config-as-code).

Loads ``config/thresholds.yaml`` (package data) once at import. The public
threshold constants in the sibling events QC-globals module and the argparse
defaults in the motion / lev1_outlier exclusion generators are bound from these
values, so externalizing them is behavior-preserving.

This module also exposes :func:`config_version`, a short stable hash over the
canonical config files (``thresholds.yaml`` + the task ``battery.yaml``),
consumed by the provenance work in PR4.

Fail-loud policy: a missing or empty config file raises immediately. There is
no silent fallback to baked-in defaults — the YAML is the source of truth.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Package data, so it resolves the same from a checkout and an installed wheel.
THRESHOLDS_PATH = Path(__file__).resolve().parent / "config" / "thresholds.yaml"

# Task battery config (de-hardcoded in PR3b). Folded into config_version so the
# provenance hash also tracks the canonical task list.
_BATTERY_PATH = Path(__file__).resolve().parent / "task_config" / "battery.yaml"


def load_thresholds(path: Path | None = None) -> dict[str, Any]:
    """Load and parse the thresholds YAML.

    Args:
        path: Optional override (used by tests). Defaults to ``THRESHOLDS_PATH``.

    Returns:
        The parsed config as a dict.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError: if the file is empty or does not parse to a non-empty dict.
    """
    cfg_path = THRESHOLDS_PATH if path is None else path
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"thresholds config not found: {cfg_path}. "
            "thresholds.yaml is the single source of truth for analysis "
            "thresholds; it must exist (no silent fallback)."
        )
    with cfg_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"thresholds config is empty or not a mapping: {cfg_path}")
    return data


@lru_cache(maxsize=1)
def _cached_thresholds() -> dict[str, Any]:
    """Process-wide cached load of the canonical thresholds file."""
    return load_thresholds()


def behavioral_qc() -> dict[str, Any]:
    """Behavioral-QC threshold section.

    Returns a fresh shallow copy each call: the sections are flat scalar maps,
    so a caller mutating the result cannot poison the process-wide lru_cache.
    """
    return dict(_cached_thresholds()["behavioral_qc"])


def motion() -> dict[str, Any]:
    """Motion-exclusion threshold section (fresh copy; see :func:`behavioral_qc`)."""
    return dict(_cached_thresholds()["motion"])


def lev1_outlier() -> dict[str, Any]:
    """Lev1-outlier (VIF) threshold section (fresh copy; see :func:`behavioral_qc`)."""
    return dict(_cached_thresholds()["lev1_outlier"])


def junk_fraction_max() -> float:
    """Behavioral junk-trial fraction above which a run's design fails lev1 QA.

    Lifted from the hardcoded ``percent_junk > 0.30`` literal in
    :func:`network_glm.lev1.processing.quality_control.run_quality_control`
    so the cutoff is auditable and folded into :func:`config_version`. Value
    unchanged (0.30).
    """
    return float(_cached_thresholds()["lev1"]["junk_fraction_max"])


def min_runs_floor(is_dual: bool) -> int:
    """Minimum retained runs required to emit a non-tagged fixed-effects map.

    Below this floor the fixed-effects map is tagged ``_desc-belowMinRuns`` and
    lev2 filters it out. Base single-task sessions require 2 runs; dual-task
    sessions require 1. Lifted from the ``--min-runs`` argparse default in
    ``analysis/lev1/run.py`` so the floors are auditable and folded into
    :func:`config_version`. Values unchanged (base=2, dual=1).
    """
    section = _cached_thresholds()["lev1"]["min_runs"]
    return int(section["dual"] if is_dual else section["base"])


def confounds_cosine_caps() -> dict[str, dict[str, int]]:
    """Per-(sample, task) caps on the number of DCT cosine high-pass regressors.

    Shape ``{sample_type: {task_name: max_cosine_index}}``; empty when the
    optional ``confounds.cosine_max_index`` block is absent. A cap ``N`` keeps
    ``cosine00..cosine0N`` and drops the rest — used where a run is short enough
    that the full fMRIPrep cosine set induces rank deficiency / collinearity with
    the task design (see
    :func:`network_glm.lev1.processing.confounds._get_base_confound_pattern`).
    """
    return dict(_cached_thresholds().get("confounds", {}).get("cosine_max_index", {}))


@lru_cache(maxsize=1)
def config_version() -> str:
    """Return a short, stable hash of the canonical config files.

    Hashes the raw bytes of ``thresholds.yaml`` and ``battery.yaml`` (in a fixed
    order) so that any edit to a study-level config produces a new version
    string. Consumed by the provenance work (PR4).

    If a config file is absent, returns the literal ``"unknown"`` rather than
    propagating an opaque ``FileNotFoundError`` -- provenance recording must not
    crash the pipeline just because a config file is missing.
    """
    h = hashlib.sha256()
    for p in (THRESHOLDS_PATH, _BATTERY_PATH):
        try:
            h.update(p.read_bytes())
        except OSError:
            return "unknown"
    return h.hexdigest()[:12]
