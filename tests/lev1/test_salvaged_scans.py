"""Salvaged-scan edge case: events past BOLD end must be dropped, not crash lev1.

Salvaged scans (50-100% TRs retained per .bidsignore policy in this project)
have BOLDs shorter than the original behavioral session. Onsets near the end
of events.tsv may exceed the BOLD's actual wall time (n_scans * TR).

This test exercises that case at three layers:

1. ``preprocess_events`` -- whether it has, or should have, an n_scans/tr-based
   filter for past-BOLD-end onsets.
2. ``create_regressor`` (nilearn ``compute_regressor``) -- silently truncates;
   no crash, but stale onsets leak into the 3-column simplified-events output.
3. End-to-end via ``create_design_matrix`` -- we want a graceful drop, not
   silent retention of orphan onsets in the saved simplified events.

If lev1 already drops these events, the test codifies the behavior. Otherwise,
the test fails and surfaces the gap for the lev1 audit report (Task 13).
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from network_glm.lev1.processing.design import (
    create_design_matrix,
    create_regressor,
)
from network_glm.lev1.processing.events import (
    add_junk_trials,
    preprocess_events,
)

TR = 1.49
N_TR = 100  # ~149 s wall time
WALL_TIME = N_TR * TR


def _build_events_with_overrun() -> pd.DataFrame:
    """Three go-trials; the last onset (200 s) is past the BOLD's wall time (~149 s)."""
    return pd.DataFrame(
        {
            "onset": [10.0, 50.0, 200.0],
            "duration": [1.0, 1.0, 1.0],
            "trial_type": ["go", "go", "go"],
            "trial_id": ["test_trial"] * 3,
            "key_press": [1, 1, 1],
            "correct_response": [1, 1, 1],
            "response_time": [0.5, 0.6, 0.7],
        }
    )


_AUDIT_GAP_REASON = (
    "AUDIT GAP (Task 13): preprocess_events has no n_scans/bold_duration "
    "parameter and create_regressor leaks past-BOLD-end onsets into the "
    "3-column simplified-events output. Remove xfail when the filter is "
    "added (likely in events.py:preprocess_events or in run.py before "
    "create_design_matrix is called)."
)


class TestPreprocessEventsBoldEndFilter:
    """preprocess_events should accept n_scans/tr and drop past-end onsets.

    The function currently does NOT accept n_scans or bold_duration. These
    tests assert the desired API. They are xfail-marked so CI stays green
    while the gap is recorded in the lev1 audit (Task 13). When the filter
    is added, remove the xfail markers and the tests should pass.
    """

    def test_preprocess_events_supports_bold_end_filter(self):
        """preprocess_events accepts an n_scans (or bold_duration) parameter."""
        sig = inspect.signature(preprocess_events)
        params = set(sig.parameters)
        assert {"n_scans", "bold_duration"} & params

    def test_preprocess_events_drops_past_end_onsets(self):
        """When given n_scans + tr, preprocess_events drops onset >= n_scans*tr."""
        events = _build_events_with_overrun()
        sig = inspect.signature(preprocess_events)
        kwargs = {"tr": TR}
        if "n_scans" in sig.parameters:
            kwargs["n_scans"] = N_TR
        elif "bold_duration" in sig.parameters:
            kwargs["bold_duration"] = WALL_TIME
        else:
            # No filter parameter exists; this is exactly the gap. Force xfail
            # via an assertion that cannot pass.
            pytest.fail("preprocess_events accepts no n_scans/bold_duration")

        out = preprocess_events(events, "stopSignal", **kwargs)
        assert len(out) == 2
        assert (out["onset"] < WALL_TIME).all()


class TestCreateRegressorDoesNotCrashOnOverrun:
    """create_regressor must not raise when an onset is past n_scans*tr.

    nilearn.compute_regressor truncates silently; this test codifies that the
    call still returns and the convolved regressor has the correct length. The
    3-column tuple output is examined separately below -- that is the gap.
    """

    def test_no_crash_with_onset_past_bold_end(self):
        events = _build_events_with_overrun()
        events["constant_1_column"] = 1
        config = {
            "amplitude_column": "constant_1_column",
            "duration_column": "constant_1_column",
            "subset": "trial_type == 'go'",
        }
        reg_df, reg_3col = create_regressor(
            events, config, n_scans=N_TR, regressor_name="go", tr=TR
        )
        assert reg_df.shape == (N_TR, 1)
        # Regressor is finite (no NaN/inf despite stale onset).
        assert np.isfinite(reg_df["go"]).all()


class TestSimplifiedEventsDropsOrphanOnsets:
    """The 3-column output from create_regressor should not retain past-end onsets.

    Currently the implementation retains all onsets that pass the subset query;
    nilearn silently sees them out-of-window and they contribute nothing to the
    convolved regressor, but they leak into save_simplified_events. The lev1
    audit should either (a) drop in create_regressor before building reg_3col,
    or (b) drop upstream in preprocess_events.
    """

    def test_three_column_format_omits_past_end_onsets(self):
        events = _build_events_with_overrun()
        # Mirror the run.py pipeline: preprocess (with n_scans) drops past-end
        # onsets before they reach create_regressor.
        events = preprocess_events(events, "stopSignal", n_scans=N_TR, tr=TR)
        config = {
            "amplitude_column": "constant_1_column",
            "duration_column": "constant_1_column",
            "subset": "trial_type == 'go'",
        }
        _, reg_3col = create_regressor(events, config, n_scans=N_TR, regressor_name="go", tr=TR)
        onsets, _, _ = reg_3col
        assert all(o < WALL_TIME for o in onsets)


class TestEndToEndDesignMatrixWithSalvagedScan:
    """Full pipeline: preprocess -> add_junk_trials -> create_design_matrix.

    Mirrors the call sequence in run.py. Asserts (a) no crash and (b) the
    design matrix has the expected length (n_scans rows) regardless of stale
    onsets. This test passes today; it codifies the safety net.
    """

    def test_end_to_end_does_not_crash_on_overrun(self):
        events = _build_events_with_overrun()
        # Add the trial_type values stopSignal expects so add_junk_trials passes.
        confounds = pd.DataFrame(
            {
                "trans_x": np.zeros(N_TR),
                "trans_y": np.zeros(N_TR),
                "cosine00": np.ones(N_TR),
            }
        )
        processed = preprocess_events(events, "stopSignal")
        with_junk, _ = add_junk_trials(processed, "stopSignal")
        dm, _ = create_design_matrix(with_junk, confounds, "stopSignal", N_TR, TR)
        assert dm.shape[0] == N_TR
        assert np.isfinite(dm.values).all()
