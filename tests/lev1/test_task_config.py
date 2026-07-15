"""Tests for the YAML-based task configuration system."""

import pytest

from network_glm.task_config.loader import (
    get_raw_yaml_config,
    get_regressor_config,
    get_task_contrasts,
    get_task_parameters,
    list_available_tasks,
)

# All base tasks that should have YAML files
BASE_TASKS = [
    "cuedTS",
    "directedForgetting",
    "flanker",
    "goNogo",
    "nBack",
    "shapeMatching",
    "spatialTS",
    "stopSignal",
]

DUAL_TASKS = [
    "stopSignalWDirectedForgetting",
]


class TestListAvailableTasks:
    """Tests for listing available task configs."""

    def test_all_base_tasks_available(self):
        """All 8 base tasks should have YAML config files."""
        available = list_available_tasks()
        for task in BASE_TASKS:
            assert task in available, f"Missing YAML config for base task: {task}"

    def test_returns_sorted_list(self):
        """Available tasks should be sorted alphabetically."""
        available = list_available_tasks()
        assert available == sorted(available)


class TestGetRawYamlConfig:
    """Tests for raw YAML config loading."""

    @pytest.mark.parametrize("task_name", BASE_TASKS)
    def test_load_each_task(self, task_name):
        """Each base task YAML should load without errors."""
        config = get_raw_yaml_config(task_name)
        assert isinstance(config, dict)

    @pytest.mark.parametrize("task_name", BASE_TASKS)
    def test_required_top_level_fields(self, task_name):
        """Each task config must have regressors and contrasts."""
        config = get_raw_yaml_config(task_name)
        assert "regressors" in config
        assert "contrasts" in config

    @pytest.mark.parametrize("task_name", BASE_TASKS)
    def test_regressor_fields(self, task_name):
        """Each regressor must have amplitude, duration, and subset."""
        config = get_raw_yaml_config(task_name)
        for reg_name, reg_config in config["regressors"].items():
            assert "amplitude" in reg_config, f"{task_name}/{reg_name} missing amplitude"
            assert "duration" in reg_config, f"{task_name}/{reg_name} missing duration"
            assert "subset" in reg_config, f"{task_name}/{reg_name} missing subset"

    def test_nonexistent_task_raises(self):
        """Loading a nonexistent task should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_raw_yaml_config("nonExistentTask")


class TestGetRegressorConfig:
    """Tests for regressor config conversion."""

    @pytest.mark.parametrize("task_name", BASE_TASKS)
    def test_returns_converted_format(self, task_name):
        """Converted regressors should have amplitude_column and duration_column."""
        regressors = get_regressor_config(task_name)
        assert isinstance(regressors, dict)
        assert len(regressors) > 0

        for reg_name, reg_config in regressors.items():
            assert (
                "amplitude_column" in reg_config
            ), f"{task_name}/{reg_name} missing amplitude_column"
            assert (
                "duration_column" in reg_config
            ), f"{task_name}/{reg_name} missing duration_column"
            assert "subset" in reg_config, f"{task_name}/{reg_name} missing subset"

    def test_constant_amplitude_conversion(self):
        """YAML amplitude: 1 should convert to 'constant_1_column'."""
        regressors = get_regressor_config("flanker")
        assert regressors["congruent"]["amplitude_column"] == "constant_1_column"

    def test_column_amplitude_conversion(self):
        """YAML amplitude: omission should convert to 'omission'."""
        regressors = get_regressor_config("flanker")
        assert regressors["omission"]["amplitude_column"] == "omission"

    def test_constant_duration_conversion(self):
        """YAML duration: 1 should convert to 'constant_1_column'."""
        regressors = get_regressor_config("flanker")
        assert regressors["congruent"]["duration_column"] == "constant_1_column"

    def test_column_duration_conversion(self):
        """YAML duration: response_time should convert to 'response_time'."""
        regressors = get_regressor_config("flanker")
        assert regressors["response_time"]["duration_column"] == "response_time"

    def test_null_subset_conversion(self):
        """YAML subset: null should convert to None."""
        regressors = get_regressor_config("flanker")
        assert regressors["omission"]["subset"] is None

    def test_string_subset_preserved(self):
        """YAML subset string should be preserved as-is."""
        regressors = get_regressor_config("stopSignal")
        assert regressors["stop_success"]["subset"] is not None
        assert "stop_success" in regressors["stop_success"]["subset"]

    def test_dual_task_regressor_config_populated(self):
        """Dual tasks are configured (PR #42): regressor config is non-empty.

        (The empty-config ValueError guard is still exercised by the
        placeholder-sentinel loader test; no task ships an empty config now.)
        """
        regressors = get_regressor_config("stopSignalWDirectedForgetting")
        assert isinstance(regressors, dict) and len(regressors) > 0


class TestGetTaskContrasts:
    """Tests for contrast loading."""

    @pytest.mark.parametrize("task_name", BASE_TASKS)
    def test_returns_contrasts(self, task_name):
        """Each base task should have at least one contrast."""
        contrasts = get_task_contrasts(task_name)
        assert isinstance(contrasts, dict)
        assert len(contrasts) > 0

    def test_flanker_contrasts(self):
        """Verify specific flanker contrasts."""
        contrasts = get_task_contrasts("flanker")
        assert "incongruent-congruent" in contrasts
        assert "task-baseline" in contrasts
        assert "response_time" in contrasts

    def test_stop_signal_contrasts(self):
        """Verify specific stopSignal contrasts."""
        contrasts = get_task_contrasts("stopSignal")
        assert "go" in contrasts
        assert "stop_success-go" in contrasts
        assert "task-baseline" in contrasts

    def test_contrast_formulas_are_strings(self):
        """All contrast formulas should be strings."""
        for task_name in BASE_TASKS:
            contrasts = get_task_contrasts(task_name)
            for name, formula in contrasts.items():
                assert isinstance(formula, str), f"{task_name}/{name} formula is not a string"

    def test_dual_task_contrasts_populated(self):
        """Dual tasks are configured (PR #42): contrasts are non-empty."""
        contrasts = get_task_contrasts("stopSignalWDirectedForgetting")
        assert isinstance(contrasts, dict) and len(contrasts) > 0


class TestGetTaskParameters:
    """Tests for task parameter loading."""

    @pytest.mark.parametrize("task_name", BASE_TASKS)
    def test_returns_parameters(self, task_name):
        """Each task should return valid parameters."""
        params = get_task_parameters(task_name)
        assert "tr" in params
        assert "dummy_scans" in params
        assert "min_rt" in params
        assert "expected_sessions" in params

    def test_default_values(self):
        """Base tasks should have standard parameter values."""
        params = get_task_parameters("flanker")
        assert params["tr"] == 1.49
        assert params["dummy_scans"] == 7
        assert params["min_rt"] == 0.2
        assert params["expected_sessions"] == 5

    def test_dual_task_sessions(self):
        """Dual tasks have no fixed expected-session count (YAML expected_sessions: null)."""
        params = get_task_parameters("stopSignalWDirectedForgetting")
        assert params["expected_sessions"] is None
