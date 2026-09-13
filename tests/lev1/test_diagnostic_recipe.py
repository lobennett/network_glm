"""Execute the documented opt-in recipe against a synthetic fitted model.

Catch wrong coefficient selection, implicit default-map export, absent-column
errors, RT maps in non-RT arms, and overwriting an existing output directory.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from network_glm.lev1.processing.surface_data import SurfaceGLM
from network_glm.task_config.loader import get_regressor_config, get_task_contrasts


TASK_COEFFICIENTS = {
    "flanker": {
        "congruent-baseline": "congruent",
        "incongruent-baseline": "incongruent",
    },
    "cuedTS": {
        "task_stay_cue_switch-baseline": "task_stay_cue_switch",
        "task_switch_cue_switch-baseline": "task_switch_cue_switch",
    },
    "directedForgettingWFlanker": {"response_time": "response_time"},
    "stopSignalWDirectedForgetting": {"response_time": "response_time"},
    "stopSignalWFlanker": {"response_time": "response_time"},
}


def _run_recipe(namespace):
    document = Path(__file__).resolve().parents[2] / "docs/DIAGNOSTIC-CONTRASTS.md"
    assert document.is_file(), "The executable diagnostic recipe must be published"
    source = document.read_text().split("```python\n", 1)[1].split("```", 1)[0]
    exec(compile(source, str(document), "exec"), namespace)
    return namespace["saved"]


def _example(tmp_path, task, arm, drop=()):
    columns = [c for c in get_regressor_config(task, arm) if c not in drop]
    columns.append("constant")
    rng = np.random.default_rng(12)
    design = pd.DataFrame(rng.normal(size=(80, len(columns))), columns=columns)
    design["constant"] = 1.0
    coefficients = rng.normal(size=(len(columns), 6))
    # Orthogonal noise leaves the known OLS coefficient values unchanged.
    noise = rng.normal(size=(80, 6))
    noise -= design.values @ np.linalg.lstsq(design.values, noise, rcond=None)[0]
    data = design.values @ coefficients + noise
    model = SurfaceGLM(t_r=1.49, noise_model="ols").fit(data, design)
    return {
        "fitted_glm": model,
        "design_columns": design.columns,
        "task_name": task,
        "rt_model": arm,
        "diagnostic_dir": tmp_path / "diagnostic",
        "base_filename": f"sub-synthetic_ses-01_task-{task}_run-1",
        "hemisphere": "L",
        "surface_space": "fsaverage6",
    }, coefficients


@pytest.mark.parametrize("task", TASK_COEFFICIENTS)
@pytest.mark.parametrize("arm", ["RTDur", "noRT"])
def test_recipe_exports_only_requested_coefficients_without_changing_fit(
    tmp_path, task, arm
):
    namespace, coefficients = _example(tmp_path, task, arm)
    defaults = get_task_contrasts(task, arm)
    regressors = get_regressor_config(task, arm)
    model = namespace["fitted_glm"]
    before = model.get_residuals().copy()
    formula = next(iter(defaults.values()))
    original_effect = model.compute_contrast(formula)["effect_size"].data.copy()
    saved = _run_recipe(namespace)

    expected = TASK_COEFFICIENTS[task]
    if arm == "noRT" and "response_time" in expected:
        expected = {}
    assert saved.keys() == expected.keys()
    if not expected:
        assert not namespace["diagnostic_dir"].exists()
    for name, coefficient in expected.items():
        assert set(saved[name]) == {"effect_size", "effect_variance", "z_score"}
        for path in saved[name].values():
            assert path.is_file()
            assert path.parent == namespace["diagnostic_dir"]
            assert f"_rtmodel-{arm}_" in path.name
        effect = nib.load(saved[name]["effect_size"]).darrays[0].data
        index = list(namespace["design_columns"]).index(coefficient)
        np.testing.assert_allclose(effect, coefficients[index], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(model.get_residuals(), before)
    np.testing.assert_array_equal(
        model.compute_contrast(formula)["effect_size"].data, original_effect
    )
    assert get_task_contrasts(task, arm) == defaults
    assert get_regressor_config(task, arm) == regressors


@pytest.mark.parametrize(
    "task", ["flanker", "cuedTS", "directedForgettingWFlanker"]
)
def test_recipe_rtepoch_has_no_pooled_rt_map(tmp_path, task):
    namespace, _ = _example(tmp_path, task, "RTepoch")
    saved = _run_recipe(namespace)
    expected = TASK_COEFFICIENTS[task] if task != "directedForgettingWFlanker" else {}
    assert saved.keys() == expected.keys()


def test_recipe_skips_dropped_columns(tmp_path):
    namespace, _ = _example(tmp_path, "flanker", "RTDur", drop=("incongruent",))
    saved = _run_recipe(namespace)
    assert set(saved) == {"congruent-baseline"}
    assert set(namespace["skipped"]) == {"incongruent-baseline"}


def test_recipe_does_not_overwrite_existing_output(tmp_path):
    namespace, _ = _example(tmp_path, "flanker", "RTDur")
    directory = namespace["diagnostic_dir"]
    directory.mkdir()
    sentinel = directory / "existing.txt"
    sentinel.write_text("preserved")
    with pytest.raises(FileExistsError):
        _run_recipe(namespace)
    assert list(directory.iterdir()) == [sentinel]
    assert sentinel.read_text() == "preserved"


def test_recipe_rejects_unknown_arm_before_writing(tmp_path):
    namespace, _ = _example(tmp_path, "flanker", "RTDur")
    namespace["rt_model"] = "typo"
    with pytest.raises(ValueError, match="rt_model"):
        _run_recipe(namespace)
    assert not namespace["diagnostic_dir"].exists()
