"""Task 1 (provenance finalization): version the junk-% and min-runs floors.

These two values used to be hardcoded literals:
- ``percent_junk > 0.30`` in lev1 quality_control.py (behavioral junk QA fail).
- ``--min-runs`` argparse default in lev1/run.py (base tasks 2, dual tasks 1).

This task lifts them into the canonical thresholds config (config-as-code) so
they are auditable and folded into ``config_version()`` for provenance. Values
are UNCHANGED from prior behavior (0.30 / base=2 / dual=1). This task only adds
the config + typed accessors; the read-sites are intentionally NOT rewired here.
"""

from __future__ import annotations

# The config_version() over thresholds.yaml + battery.yaml BEFORE this task's
# edit. Recorded so we can prove the new fields are folded into the provenance
# hash (a fresh process reads the file bytes at call time; captured pre-change).
_PRE_CHANGE_CONFIG_VERSION = "2891bffa6044"


def test_junk_fraction_max_is_versioned_and_unchanged():
    """junk_fraction_max is exposed via the config accessor at 0.30 (unchanged)."""
    from network_glm.thresholds import junk_fraction_max

    assert junk_fraction_max() == 0.30


def test_min_runs_floor_base_and_dual_unchanged():
    """min_runs floors: base tasks require 2 runs, dual tasks require 1."""
    from network_glm.thresholds import min_runs_floor

    assert min_runs_floor(is_dual=False) == 2
    assert min_runs_floor(is_dual=True) == 1


def test_new_fields_present_in_config_dict():
    """The raw config exposes the new lev1 block so provenance stamps see it."""
    from network_glm.thresholds import load_thresholds

    lev1 = load_thresholds()["lev1"]
    assert lev1["junk_fraction_max"] == 0.30
    assert lev1["min_runs"]["base"] == 2
    assert lev1["min_runs"]["dual"] == 1


def test_config_version_folds_in_new_fields():
    """Adding the fields bumps config_version() (they are hashed in)."""
    from network_glm.thresholds import config_version

    assert config_version() != _PRE_CHANGE_CONFIG_VERSION


def test_config_version_stable_across_calls():
    """config_version() is deterministic within a process."""
    from network_glm.thresholds import config_version

    assert config_version() == config_version()
