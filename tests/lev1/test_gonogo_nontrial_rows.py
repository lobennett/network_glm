"""Go/no-go regressors must distinguish behavioral trials from labeled breaks.

All rows are synthetic. Breaks mirror the observed event structure: a retained
condition label, ten-second duration, no key press, and missing response data.
"""

import json
from argparse import Namespace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from network_glm.lev1 import runner
from network_glm.lev1.processing.design import create_design_matrix
from network_glm.lev1.processing.events import (
    add_junk_trials,
    preprocess_events,
    save_simplified_events,
)


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


@pytest.mark.parametrize("rt_model", ["RTDur", "noRT"])
def test_gonogo_run_without_genuine_omissions_still_fits(rt_model, tmp_path):
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
    rng = np.random.default_rng(42)
    confounds = pd.DataFrame(
        {
            "cosine00": np.cos(np.pi * (np.arange(n_scans) + 0.5) / n_scans),
            "trans_x": rng.normal(scale=0.01, size=n_scans),
        }
    )
    processed, fraction = add_junk_trials(preprocess_events(events, "goNogo"), "goNogo")
    # No genuine go omission survives; the sole omission-shaped row is a break.
    assert processed["omission"].sum() == 0
    assert fraction == pytest.approx(0.0)

    design_matrix, _ = create_design_matrix(
        processed,
        confounds,
        "goNogo",
        n_scans,
        tr=1.49,
        slice_time_ref=0.701,
        rt_model=rt_model,
    )
    assert (design_matrix["go_omission"] == 0).all()

    run_files = {
        "events": tmp_path / "events.tsv",
        "confounds": tmp_path / "confounds.tsv",
        "mni_data": tmp_path / "bold.nii.gz",
        "mni_brain_mask": tmp_path / "mask.nii.gz",
    }
    events.to_csv(run_files["events"], sep="\t", index=False)
    confounds.to_csv(run_files["confounds"], sep="\t", index=False)
    signal = design_matrix.to_numpy() @ rng.normal(
        scale=5, size=(design_matrix.shape[1], 27)
    )
    data = 1000 + signal + rng.normal(size=(n_scans, 27))
    nib.save(
        nib.Nifti1Image(data.T.reshape(3, 3, 3, n_scans), np.eye(4)),
        run_files["mni_data"],
    )
    nib.save(
        nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), np.eye(4)),
        run_files["mni_brain_mask"],
    )
    (tmp_path / "bold.json").write_text(
        json.dumps({"SliceTimingCorrected": True, "StartTime": 0.701})
    )
    dirs = {
        name: tmp_path / name
        for name in (
            "indiv_contrasts", "quality_control", "task_residuals", "simplified_events"
        )
    }
    for directory in dirs.values():
        directory.mkdir()
    args = Namespace(
        subj_id="sub-test",
        task_name="goNogo",
        space="MNI",
        rt_model=rt_model,
        smoothing_fwhm=None,
        residuals=False,
        fc_confounds=False,
        skip_existing=False,
    )

    assert runner.process_single_run(
        "ses-1", "run-1", run_files, args, "discovery", dirs, {"tr": 1.49}, set()
    )

    base = "sub-test_ses-1_task-goNogo_run-1"
    saved_design = pd.read_csv(
        dirs["quality_control"] / f"{base}_desc-designMatrix.csv"
    )
    assert len(saved_design) == n_scans
    assert "go_omission" not in saved_design
    assert ("response_time" in saved_design) == (rt_model == "RTDur")
    for name in ("go", "nogo_success", "nogo_failure", "break_with_performance_feedback"):
        np.testing.assert_allclose(saved_design[name], design_matrix[name], atol=1e-12)

    expected_contrasts = ["go", "nogo_success", "nogo_success-go", "task-baseline"]
    if rt_model == "RTDur":
        expected_contrasts.append("response_time")
    expected_paths = set()
    for contrast in expected_contrasts:
        for stat in ("effect-size", "variance", "z_score"):
            path = dirs["indiv_contrasts"] / (
                f"{base}_contrast-{contrast}_rtmodel-{rt_model}_stat-{stat}.nii.gz"
            )
            expected_paths.add(path)
            values = nib.load(path).get_fdata()
            assert values.shape == (3, 3, 3)
            assert np.isfinite(values).all()
            if stat == "variance":
                assert (values > 0).all()
            else:
                assert np.any(values != 0)
    assert set(dirs["indiv_contrasts"].glob("*.nii.gz")) == expected_paths
