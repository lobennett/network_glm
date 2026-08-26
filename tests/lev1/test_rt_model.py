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
