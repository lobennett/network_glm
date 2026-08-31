"""The noRT arm drops the RT regressor so its variance stays in the residual.

Mirrors test_confounds_modes.py: these are experiment arms, and the arm is recorded in the
``rtmodel-`` entity of every output filename so two arms can never be confused for one
another.
"""
import pytest

from network_glm.task_config.loader import (
    RT_REGRESSOR,
    get_base_tasks,
    get_regressor_config,
    get_task_contrasts,
)


class TestRegressors:
    def test_rtdur_keeps_the_rt_regressor(self):
        assert RT_REGRESSOR in get_regressor_config("flanker", "RTDur")

    def test_nort_drops_it(self):
        assert RT_REGRESSOR not in get_regressor_config("flanker", "noRT")

    def test_nothing_else_is_dropped(self):
        keep = set(get_regressor_config("flanker", "RTDur")) - {RT_REGRESSOR}
        assert set(get_regressor_config("flanker", "noRT")) == keep

    def test_default_is_rtdur(self):
        assert get_regressor_config("flanker") == get_regressor_config("flanker", "RTDur")


class TestContrasts:
    def test_nort_drops_contrasts_referencing_rt(self):
        """Dropped by formula, not by name: the column is gone from the design."""
        out = get_task_contrasts("flanker", "noRT")
        assert all(RT_REGRESSOR not in f for f in out.values())

    def test_the_contrasts_of_interest_survive(self):
        out = get_task_contrasts("flanker", "noRT")
        assert "incongruent-congruent" in out and "task-baseline" in out

    def test_default_is_rtdur(self):
        assert "response_time" in get_task_contrasts("flanker")

    def test_every_base_task_keeps_some_contrast(self):
        """A task whose every contrast referenced RT would silently produce nothing."""
        for task in get_base_tasks():
            assert get_task_contrasts(task, "noRT"), task


class TestValidation:
    @pytest.mark.parametrize("fn", [get_regressor_config, get_task_contrasts])
    def test_unknown_arm_raises(self, fn):
        with pytest.raises(ValueError, match="rt_model"):
            fn("flanker", "RTonset")


class TestFilenames:
    def test_fixed_effects_filename_records_the_arm(self):
        from network_glm.lev1.processing.fixed_effects import FixedEffectsAnalyzer

        for arm in ("RTDur", "noRT"):
            a = FixedEffectsAnalyzer("sub-s03", "flanker", rt_model=arm)
            a.contrast_results = {"incongruent-congruent": {"n_runs": 5}}
            assert f"_rtmodel-{arm}" in a._build_base_filename("incongruent-congruent")


class TestRTepoch:
    """Grinband variable epoch: no RT regressor, conditions carry duration = RT."""

    def test_rt_regressor_is_dropped(self):
        assert RT_REGRESSOR not in get_regressor_config("flanker", "RTepoch")

    def test_conditions_carry_rt_duration(self):
        r = get_regressor_config("flanker", "RTepoch")
        for cond in ("congruent", "incongruent"):
            assert r[cond]["duration_column"] == RT_REGRESSOR

    def test_non_condition_durations_are_untouched(self):
        """Nuisance epochs keep their own durations; only constant-duration conditions move."""
        base = get_regressor_config("flanker", "RTDur")
        out = get_regressor_config("flanker", "RTepoch")
        for name in ("omission", "commission", "rt_fast",
                     "break_with_performance_feedback"):
            assert out[name]["duration_column"] == base[name]["duration_column"], name

    def test_contrasts_referencing_rt_are_dropped(self):
        out = get_task_contrasts("flanker", "RTepoch")
        assert all(RT_REGRESSOR not in f for f in out.values())
        assert "incongruent-congruent" in out and "task-baseline" in out

    @pytest.mark.parametrize(
        "task", ["goNogo", "stopSignal", "stopSignalWFlanker",
                 "stopSignalWDirectedForgetting"])
    def test_refuses_the_inhibition_tasks(self, task):
        """stop/nogo trials have no response, so they cannot carry duration = RT.
        Converting only the responded conditions would contrast an RT-length epoch
        against a one-second epoch."""
        with pytest.raises(ValueError, match="RTepoch"):
            get_regressor_config(task, "RTepoch")

    def test_every_other_base_task_converts(self):
        from network_glm.task_config.loader import get_base_tasks
        inhibition = {"goNogo", "stopSignal"}
        for task in get_base_tasks():
            if task in inhibition:
                continue
            r = get_regressor_config(task, "RTepoch")
            assert RT_REGRESSOR not in r, task
            assert any(v["duration_column"] == RT_REGRESSOR for v in r.values()), task
