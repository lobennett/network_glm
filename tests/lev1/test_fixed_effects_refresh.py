"""Fixed-effects refresh must fail visibly and retire obsolete group inputs."""

import numpy as np
import pytest

from network_glm.lev1.processing.fixed_effects import FixedEffectsAnalyzer
from network_glm.lev1.processing.surface_data import SurfaceResult


def _run(directory, run, *, variance=True):
    prefix = (
        f"sub-test_ses-1_task-flanker_run-{run}_hemi-L_space-fsaverage6"
        "_contrast-task_rtmodel-RTDur"
    )
    for stat in ("effect-size", "variance", "z_score"):
        if stat == "variance" and not variance:
            continue
        SurfaceResult(np.array([1.0, 2.0, 3.0])).to_filename(
            directory / f"{prefix}_stat-{stat}.func.gii"
        )


def _analyzer():
    return FixedEffectsAnalyzer(
        "sub-test", "flanker", hemisphere="L", surface_space="fsaverage6"
    )


def test_real_aggregation_error_propagates(tmp_path):
    _run(tmp_path, 1)
    analyzer = _analyzer()
    effects, variances = analyzer.find_contrast_files(tmp_path, "task")
    effects[0].write_text("corrupted GIFTI")
    with pytest.raises(RuntimeError, match="Fixed effects failed for task"):
        analyzer.compute_fixed_effects_contrast("task", effects, variances)


def test_missing_variance_cannot_silently_skip_aggregation(tmp_path):
    _run(tmp_path, 1, variance=False)
    with pytest.raises(ValueError, match="File count mismatch"):
        _analyzer().compute_all_task_fixed_effects(
            tmp_path, tmp_path / "out", contrasts={"task": "constant"}
        )


def test_exclusion_refresh_retires_normal_and_below_minimum_maps(tmp_path):
    _run(tmp_path, 1)
    _run(tmp_path, 2)
    output = tmp_path / "out"
    contrasts = {"task": "constant"}
    _analyzer().compute_all_task_fixed_effects(tmp_path, output, contrasts=contrasts)
    old = list(output.glob("*.func.gii"))
    original = {path: path.read_bytes() for path in old}
    assert len(old) == 4
    unrelated = output / "unrelated_rtmodel-noRT_stat-fixed-effects.func.gii"
    unrelated.write_text("other arm")

    exclusions = {"sub-test_ses-1_task-flanker_run-1"}
    _analyzer().compute_all_task_fixed_effects(
        tmp_path, output, exclusions=exclusions, contrasts=contrasts
    )
    assert all(not path.exists() for path in old)
    assert len(list(output.glob("*desc-belowMinRuns*.func.gii"))) == 4
    for path, contents in original.items():
        archived = list(output.glob(path.name + ".superseded-*"))
        assert len(archived) == 1
        assert archived[0].read_bytes() == contents

    exclusions.add("sub-test_ses-1_task-flanker_run-2")
    _analyzer().compute_all_task_fixed_effects(
        tmp_path, output, exclusions=exclusions, contrasts=contrasts
    )
    assert list(output.glob("*.func.gii")) == [unrelated]
    assert unrelated.read_text() == "other arm"
