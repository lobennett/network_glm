"""Go/no-go regressors must distinguish behavioral trials from labeled breaks.

All rows are synthetic. Breaks mirror the observed event structure: a retained
condition label, ten-second duration, no key press, and missing response data.
"""

import numpy as np
import pandas as pd
import pytest

from network_glm.lev1.processing.contrasts import filter_contrasts_for_dropped_columns
from network_glm.lev1.processing.design import create_design_matrix
from network_glm.lev1.processing.events import (
    add_junk_trials,
    preprocess_events,
    save_simplified_events,
)
from network_glm.lev1.processing.glm import (
    handle_zero_variance_columns,
    validate_design_matrix,
)
from network_glm.task_config.loader import get_task_contrasts


@pytest.fixture
def gonogo_events():
    trials = pd.DataFrame(
        {
            "onset": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "duration": [1.5] * 8,
            "trial_id": ["test_trial"] * 8,
            "trial_type": ["go"] * 6 + ["nogo_success", "nogo_failure"],
            "key_press": [1, -1, 2, 1, 1, 1, -1, 1],
            "correct_response": [1] * 6 + [-1, -1],
            "response_time": [0.5, "n/a", 0.4, 0.1, 0.6, -1.0, "n/a", 0.4],
            "junk": [0, 0, 0, 0, 1, 0, 0, 0],
        }
    )
    breaks = pd.DataFrame(
        {
            "onset": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0],
            "duration": [10.0] * 8,
            "trial_id": ["break"] * 4 + ["break_with_performance_feedback"] * 4,
            "trial_type": ["go", "nogo_success", "nogo_failure", "n/a"] * 2,
            "key_press": [-1] * 8,
            "correct_response": ["n/a"] * 8,
            "response_time": ["n/a"] * 8,
            "junk": [0] * 8,
        }
    )
    return pd.concat([trials, breaks], ignore_index=True)


def test_gonogo_junk_rate_counts_genuine_go_trials_only(gonogo_events):
    processed, fraction = add_junk_trials(
        preprocess_events(gonogo_events, "goNogo"), "goNogo"
    )
    # Both break row types remain available for their existing event modeling.
    assert len(processed) == 16
    expected = {
        "omission": [0, 1, 0, 0, 0, 0, 0, 0],
        "commission": [0, 0, 1, 0, 0, 0, 0, 0],
        "rt_too_fast": [0, 0, 0, 1, 0, 0, 0, 0],
        "junk_trials": [0, 1, 1, 1, 1, 1, 0, 0],
    }
    for field, values in expected.items():
        assert processed[field].tolist() == values + [0] * 8
    # Neither successful no-go trials nor breaks belong in the go denominator.
    assert fraction == pytest.approx(5 / 6)


@pytest.mark.parametrize("rt_model", ["RTDur", "noRT"])
def test_gonogo_trial_design_matches_input_without_breaks(gonogo_events, rt_model, tmp_path):
    def build(events):
        processed, _ = add_junk_trials(preprocess_events(events, "goNogo"), "goNogo")
        return create_design_matrix(
            processed,
            pd.DataFrame(index=range(150)),
            "goNogo",
            150,
            tr=1.49,
            slice_time_ref=0.701,
            rt_model=rt_model,
        )

    actual, event_columns = build(gonogo_events)
    clean, _ = build(gonogo_events.iloc[:8].copy())
    trial_columns = [
        "go", "nogo_success", "nogo_failure", "go_omission", "go_commission", "go_rt_fast",
    ]
    if rt_model == "RTDur":
        trial_columns.append("response_time")
    else:
        assert "response_time" not in actual
    np.testing.assert_allclose(actual[trial_columns], clean[trial_columns], rtol=0, atol=1e-12)

    # Inspect the saved event model too: genuine omission and no-go events must
    # survive, while neither kind of break can contribute to trial regressors.
    saved_path = save_simplified_events(event_columns, tmp_path / "events.csv")
    saved = pd.read_csv(saved_path)
    expected_onsets = {
        "go": [10.0, 50.0],  # Preserve the existing handling of premarked junk.
        "nogo_success": [70.0],
        "nogo_failure": [80.0],
        "go_omission": [20.0],
        "go_commission": [30.0],
        "go_rt_fast": [40.0],
        "break_with_performance_feedback": [140.0, 150.0, 160.0, 170.0],
    }
    if rt_model == "RTDur":
        expected_onsets["response_time"] = [10.0, 50.0]
    for name, onsets in expected_onsets.items():
        assert saved.loc[saved.regressor.eq(name), "onset"].tolist() == onsets
    assert saved.loc[saved.regressor.eq("go_omission"), "duration"].tolist() == [1.5]
    assert saved.loc[saved.regressor.eq("nogo_success"), "duration"].tolist() == [1.0]
    feedback = saved.loc[saved.regressor.eq("break_with_performance_feedback")]
    assert feedback["duration"].tolist() == [10.0] * 4


def test_gonogo_run_without_genuine_omissions_still_fits():
    """A run whose only go-labeled omission is a break must remain fittable."""
    events = pd.DataFrame(
        {
            "onset": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 100.0, 140.0],
            "duration": [1.5] * 6 + [10.0, 10.0],
            "trial_id": ["test_trial"] * 6
            + ["break", "break_with_performance_feedback"],
            "trial_type": ["go"] * 4 + ["nogo_success", "nogo_failure", "go", "n/a"],
            "key_press": [1, 1, 1, 1, -1, 1, -1, -1],
            "correct_response": [1] * 4 + [-1, -1, "n/a", "n/a"],
            "response_time": [0.5, 0.62, 0.44, 0.71, "n/a", 0.38, "n/a", "n/a"],
            "junk": [0] * 8,
        }
    )
    n_scans = 150
    processed, fraction = add_junk_trials(preprocess_events(events, "goNogo"), "goNogo")
    # No genuine go omission survives; the sole omission-shaped row is a break.
    assert processed["omission"].sum() == 0
    assert fraction == pytest.approx(0.0)

    design_matrix, _ = create_design_matrix(
        processed,
        pd.DataFrame(index=range(n_scans)),
        "goNogo",
        n_scans,
        tr=1.49,
        slice_time_ref=0.701,
    )
    assert (design_matrix["go_omission"] == 0).all()

    # This is the runner's own ordering: prune, filter contrasts, then validate.
    design_matrix, dropped = handle_zero_variance_columns(design_matrix)
    assert "go_omission" in dropped
    contrasts, skipped = filter_contrasts_for_dropped_columns(
        get_task_contrasts("goNogo", "RTDur"), dropped
    )
    assert skipped == {}
    assert "task-baseline" in contrasts

    validation = validate_design_matrix(design_matrix, n_scans=n_scans)
    assert validation["is_valid"], validation["errors"]
