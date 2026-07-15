"""Loader raises TaskNotConfiguredError on placeholder YAMLs.

All 11 real dual tasks are now configured, so this uses a synthetic placeholder
YAML (via a monkeypatched tasks dir) to exercise the sentinel behavior without
depending on any real task staying unconfigured.
"""

import pytest

from network_glm.task_config import loader
from network_glm.task_config.loader import (
    TaskNotConfiguredError,
    get_task_parameters,
    get_task_regressors,
)

_PLACEHOLDER = """\
tr: 1.49
dummy_scans: 7
min_rt: 0.2
expected_sessions: null
regressors: null
contrasts: null
"""


@pytest.fixture
def placeholder_tasks_dir(tmp_path, monkeypatch):
    """A tasks dir containing a single placeholder YAML."""
    (tmp_path / "sentinelPlaceholder.yaml").write_text(_PLACEHOLDER)
    monkeypatch.setattr(loader, "_TASKS_DIR", tmp_path)
    loader._get_task_config.cache_clear()
    yield tmp_path
    loader._get_task_config.cache_clear()


def test_placeholder_raises_on_regressors(placeholder_tasks_dir):
    with pytest.raises(TaskNotConfiguredError, match="regressors"):
        get_task_regressors("sentinelPlaceholder")


def test_placeholder_parameters_still_load(placeholder_tasks_dir):
    # Parameters (tr, dummy_scans, ...) are intentionally still readable so callers
    # can introspect a placeholder before deciding whether to skip.
    params = get_task_parameters("sentinelPlaceholder")
    assert params["tr"] == 1.49


def test_base_task_loads_normally():
    params = get_task_parameters("stopSignal")
    assert params["tr"] == 1.49


def test_configured_dual_task_loads_normally():
    # cuedTSWFlanker is now a fully configured dual task (no longer a placeholder).
    regressors = get_task_regressors("cuedTSWFlanker")
    assert "cstay_tstay_congruent" in regressors
