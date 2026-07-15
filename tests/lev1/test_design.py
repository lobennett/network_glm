"""Tests for the design matrix creation module."""

import numpy as np
import pandas as pd
import pytest

from network_glm.lev1.processing.design import (
    _resolve_column_or_constant,
    create_design_matrix,
    create_regressor,
)
from network_glm.task_config.loader import _convert_regressor_config


class TestSentinelDurationRoundTrip:
    """Regression guard for the loader/design sentinel bug.

    Previously the loader collapsed every numeric duration to the literal
    sentinel ``'constant_1_column'``, ignoring the actual number. A YAML
    ``duration: 10`` would silently produce a 1-second epoch, leaving 9
    seconds of every break_with_performance_feedback period as implicit
    baseline. Tests below verify that any numeric duration in the YAML
    round-trips through the loader and design path with the literal value
    preserved.
    """

    def test_constant_sentinel_decoder_resolves_value(self):
        """``_resolve_column_or_constant`` reads the literal numeric value
        from a sentinel and returns a constant vector."""
        events = pd.DataFrame({"onset": [1.0, 2.0, 3.0]})
        result = _resolve_column_or_constant(events, "constant_10_column")
        np.testing.assert_array_equal(result, [10.0, 10.0, 10.0])
        assert result.dtype == np.float64

    def test_constant_sentinel_decoder_falls_through_to_column_lookup(self):
        """A non-sentinel column spec is looked up in the events frame as-is."""
        events = pd.DataFrame({"onset": [1.0, 2.0], "duration": [0.5, 0.8]})
        result = _resolve_column_or_constant(events, "duration")
        np.testing.assert_array_equal(result, [0.5, 0.8])

    @pytest.mark.parametrize("yaml_value", [1, 10, 1.5, 0.25, 1.0])
    def test_loader_preserves_numeric_duration_through_sentinel(self, yaml_value):
        """The loader must encode any YAML numeric duration so the design
        path produces a regressor at that value — not 1.0."""
        yaml_regressors = {"block": {"amplitude": 1, "duration": yaml_value, "subset": None}}
        converted = _convert_regressor_config(yaml_regressors)
        sentinel = converted["block"]["duration_column"]

        events = pd.DataFrame({"onset": [10.0, 20.0]})
        result = _resolve_column_or_constant(events, sentinel)
        expected = float(yaml_value)
        np.testing.assert_array_equal(result, [expected, expected])

    def test_break_with_performance_feedback_models_a_full_10s_epoch(self):
        """End-to-end check on the canonical break regressor.

        Synthetic events with two 10-second breaks. ``create_regressor`` must
        produce a regressor whose convolved HRF reflects 10s of stimulus —
        not 1s. Concrete check: the area-under-curve of the regressor scales
        with the duration. A 10s epoch produces a substantially larger AUC
        than a 1s epoch.
        """
        from network_glm.task_config.loader import (
            _convert_regressor_config,
        )

        events = pd.DataFrame(
            {
                "onset": [30.0, 90.0],
                "duration": [10.0, 10.0],
                "trial_id": ["break_with_performance_feedback"] * 2,
            }
        )

        # Convert 10s-duration YAML through the same path the production
        # loader uses, then materialise the design column.
        cfg_10s = _convert_regressor_config(
            {
                "break": {
                    "amplitude": 1,
                    "duration": 10,
                    "subset": "trial_id == 'break_with_performance_feedback'",
                }
            }
        )["break"]
        cfg_1s = _convert_regressor_config(
            {
                "break": {
                    "amplitude": 1,
                    "duration": 1,
                    "subset": "trial_id == 'break_with_performance_feedback'",
                }
            }
        )["break"]

        n_scans = 120
        reg_10s, _ = create_regressor(events, cfg_10s, n_scans, "break", tr=1.0)
        reg_1s, _ = create_regressor(events, cfg_1s, n_scans, "break", tr=1.0)

        auc_10s = reg_10s["break"].abs().sum()
        auc_1s = reg_1s["break"].abs().sum()
        # Ten-second blocks integrate to ~5-10x the AUC of one-second
        # impulses for the same number of events. The sentinel bug used to
        # make these identical.
        assert auc_10s > 4 * auc_1s, (
            "The 10s regressor should integrate to substantially more than "
            "the 1s regressor. If they are equal, the loader/design sentinel "
            "bug has regressed and `duration: 10` is silently producing 1s "
            f"epochs. auc_10s={auc_10s}, auc_1s={auc_1s}"
        )


class TestCreateRegressor:
    """Tests for create_regressor function."""

    @pytest.fixture
    def simple_events(self):
        """Simple events DataFrame for testing."""
        return pd.DataFrame(
            {
                "onset": [10.0, 20.0, 30.0],
                "duration": [1.0, 1.0, 1.0],
                "trial_type": ["go", "go", "go"],
                "response_time": [0.5, 0.6, 0.7],
            }
        )

    def test_constant_amplitude_and_duration(self, simple_events):
        """Test regressor with constant amplitude and duration."""
        config = {
            "amplitude_column": "constant_1_column",
            "duration_column": "constant_1_column",
            "subset": "trial_type == 'go'",
        }
        reg_df, reg_3col = create_regressor(simple_events, config, 50, "go")

        assert "go" in reg_df.columns
        assert len(reg_df) == 50
        # HRF convolution should produce non-zero values
        assert reg_df["go"].abs().sum() > 0

        # 3-column format should have 3 events
        onsets, durations, amplitudes = reg_3col
        assert len(onsets) == 3
        assert all(d == 1.0 for d in durations)
        assert all(a == 1.0 for a in amplitudes)

    def test_column_duration(self, simple_events):
        """Test regressor using column for duration (RT-as-duration)."""
        config = {
            "amplitude_column": "constant_1_column",
            "duration_column": "response_time",
            "subset": "trial_type == 'go'",
        }
        reg_df, reg_3col = create_regressor(simple_events, config, 50, "rt_reg")

        assert len(reg_df) == 50
        _, durations, _ = reg_3col
        assert durations == [0.5, 0.6, 0.7]

    def test_column_amplitude(self):
        """Test regressor using a column for amplitude."""
        events = pd.DataFrame(
            {
                "onset": [10.0, 20.0, 30.0],
                "weight": [0.5, 1.0, 0.0],
                "trial_type": ["go", "go", "go"],
            }
        )
        config = {
            "amplitude_column": "weight",
            "duration_column": "constant_1_column",
            "subset": "trial_type == 'go'",
        }
        reg_df, reg_3col = create_regressor(events, config, 50, "weighted")

        _, _, amplitudes = reg_3col
        assert amplitudes == [0.5, 1.0, 0.0]

    def test_empty_subset_returns_zero_regressor(self, simple_events):
        """Test that an empty subset produces a zero regressor."""
        config = {
            "amplitude_column": "constant_1_column",
            "duration_column": "constant_1_column",
            "subset": "trial_type == 'nonexistent'",
        }
        reg_df, reg_3col = create_regressor(simple_events, config, 50, "empty")

        assert len(reg_df) == 50
        assert reg_df["empty"].abs().sum() == 0.0
        assert reg_3col == ([], [], [])

    def test_null_subset_uses_all_rows(self, simple_events):
        """Test that a null/empty subset uses all rows."""
        config = {
            "amplitude_column": "constant_1_column",
            "duration_column": "constant_1_column",
            "subset": "",
        }
        reg_df, reg_3col = create_regressor(simple_events, config, 50, "all")

        onsets, _, _ = reg_3col
        assert len(onsets) == 3

    def test_invalid_column_raises(self, simple_events):
        """Test that referencing a nonexistent column raises ValueError."""
        config = {
            "amplitude_column": "nonexistent_column",
            "duration_column": "constant_1_column",
            "subset": "trial_type == 'go'",
        }
        with pytest.raises(ValueError, match="Failed to create regressor"):
            create_regressor(simple_events, config, 50, "bad")


class TestCreateDesignMatrix:
    """Tests for create_design_matrix and intercept detection."""

    @pytest.fixture
    def flanker_events(self):
        """Events matching flanker task regressors."""
        return pd.DataFrame(
            {
                "onset": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
                "duration": [1.0] * 6,
                "trial_type": [
                    "congruent",
                    "incongruent",
                    "congruent",
                    "incongruent",
                    "congruent",
                    "incongruent",
                ],
                "trial_id": ["test_trial"] * 6,
                "key_press": [1, 2, 1, 2, 1, 2],
                "correct_response": [1, 2, 1, 2, 1, 2],
                "response_time": [0.5, 0.6, 0.5, 0.6, 0.5, 0.6],
                "omission": [0, 0, 0, 0, 0, 0],
                "commission": [0, 0, 0, 0, 0, 0],
                "rt_too_fast": [0, 0, 0, 0, 0, 0],
            }
        )

    def test_intercept_added_when_missing(self, flanker_events):
        """Test that a constant column is added when confounds lack one."""
        n_scans = 50
        confounds = pd.DataFrame(
            {
                "trans_x": np.random.randn(n_scans),
                "trans_y": np.random.randn(n_scans),
            }
        )

        dm, _ = create_design_matrix(flanker_events, confounds, "flanker", n_scans)

        # Should have added a 'constant' column
        assert "constant" in dm.columns
        assert (dm["constant"] == 1.0).all()

    def test_intercept_not_added_when_present(self, flanker_events):
        """Test that no constant is added when confounds already have one."""
        n_scans = 50
        confounds = pd.DataFrame(
            {
                "trans_x": np.random.randn(n_scans),
                "cosine00": np.ones(n_scans),  # Constant column from fMRIPrep
            }
        )

        dm, _ = create_design_matrix(flanker_events, confounds, "flanker", n_scans)

        # Should NOT add a redundant 'constant' column
        assert "constant" not in dm.columns
        # cosine00 should still be present
        assert "cosine00" in dm.columns

    def test_design_matrix_includes_task_regressors(self, flanker_events):
        """Test that design matrix includes task regressors from YAML config."""
        n_scans = 50
        confounds = pd.DataFrame(
            {
                "cosine00": np.ones(n_scans),
                "trans_x": np.random.randn(n_scans),
            }
        )

        dm, reg_3cols = create_design_matrix(flanker_events, confounds, "flanker", n_scans)

        # Flanker should have these regressors
        expected_regressors = [
            "congruent",
            "incongruent",
            "response_time",
            "omission",
            "commission",
            "rt_fast",
            "break_with_performance_feedback",
        ]
        for reg_name in expected_regressors:
            assert reg_name in dm.columns, f"Missing regressor: {reg_name}"

        # Confounds should also be in the design matrix
        assert "cosine00" in dm.columns
        assert "trans_x" in dm.columns

    def test_design_matrix_row_count(self, flanker_events):
        """Test that design matrix has correct number of rows."""
        n_scans = 50
        confounds = pd.DataFrame(
            {
                "cosine00": np.ones(n_scans),
            }
        )

        dm, _ = create_design_matrix(flanker_events, confounds, "flanker", n_scans)
        assert len(dm) == n_scans

    def test_regressor_3col_format(self, flanker_events):
        """Test that regressor 3-column tuples are returned."""
        n_scans = 50
        confounds = pd.DataFrame({"cosine00": np.ones(n_scans)})

        _, reg_3cols = create_design_matrix(flanker_events, confounds, "flanker", n_scans)

        # Should have one entry per regressor
        assert len(reg_3cols) > 0
        for reg_3col, name in reg_3cols:
            assert isinstance(name, str)
            assert isinstance(reg_3col, tuple)
            assert len(reg_3col) == 3  # onsets, durations, amplitudes

    def test_zero_column_not_treated_as_intercept(self, flanker_events):
        """Test that a column of zeros is not treated as a constant."""
        n_scans = 50
        confounds = pd.DataFrame(
            {
                "zero_col": np.zeros(n_scans),  # All zeros, NOT a valid intercept
                "trans_x": np.random.randn(n_scans),
            }
        )

        dm, _ = create_design_matrix(flanker_events, confounds, "flanker", n_scans)

        # A column of zeros is not a valid intercept, so 'constant' should be added
        assert "constant" in dm.columns
