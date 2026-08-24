"""Focused tests for meanRunZ aggregation, metadata, and surface naming."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from network_glm.lev1.processing.contrasts import compute_run_contrasts
from network_glm.lev1.processing.fixed_effects import (
    FixedEffectsAnalyzer,
    compute_mean_run_z,
)
from network_glm.lev1.processing.surface_data import SurfaceResult


def _write_nifti(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data.astype(np.float32), np.eye(4)), path)


def _write_gifti(path: Path, data: np.ndarray) -> None:
    SurfaceResult(data.astype(np.float32)).to_filename(path)


def test_compute_mean_run_z_volume_is_arithmetic_and_nan_safe(tmp_path):
    z1 = tmp_path / "run-1_stat-z_score.nii.gz"
    z2 = tmp_path / "run-2_stat-z_score.nii.gz"
    _write_nifti(z1, np.array([[[1.0, np.nan, np.nan]]]))
    _write_nifti(z2, np.array([[[3.0, 5.0, np.nan]]]))

    result = compute_mean_run_z([z1, z2])

    np.testing.assert_allclose(result.get_fdata()[0, 0, :2], [2.0, 5.0])
    assert np.isnan(result.get_fdata()[0, 0, 2])
    assert result.get_data_dtype() == np.dtype("float32")


def test_compute_mean_run_z_surface_is_arithmetic_and_nan_safe(tmp_path):
    z1 = tmp_path / "run-1_stat-z_score.func.gii"
    z2 = tmp_path / "run-2_stat-z_score.func.gii"
    _write_gifti(z1, np.array([2.0, np.nan, np.nan]))
    _write_gifti(z2, np.array([4.0, 8.0, np.nan]))

    result = compute_mean_run_z([z1, z2], is_surface=True)

    np.testing.assert_allclose(result.data[:2], [3.0, 8.0])
    assert np.isnan(result.data[2])


def test_save_outputs_mean_run_z_and_complete_metadata(tmp_path):
    contrast_dir = tmp_path / "indiv_contrasts"
    output_dir = tmp_path / "fixed_effects"
    contrast_dir.mkdir()
    sources = []
    for run, z_value in (("01", 1.0), ("02", 3.0)):
        base = (
            f"sub-s10_ses-02_task-flanker_run-{run}"
            "_contrast-incongruent-congruent_rtmodel-RTDur"
        )
        effect = contrast_dir / f"{base}_stat-effect-size.nii.gz"
        variance = contrast_dir / f"{base}_stat-variance.nii.gz"
        z_score = contrast_dir / f"{base}_stat-z_score.nii.gz"
        _write_nifti(effect, np.full((2, 1, 1), z_value))
        _write_nifti(variance, np.ones((2, 1, 1)))
        _write_nifti(z_score, np.array([[[z_value]], [[np.nan if run == '01' else 7.0]]]))
        sources.append((effect, variance))

    analyzer = FixedEffectsAnalyzer(
        "sub-s10",
        "flanker",
        analysis_space="MNI",
        smoothing_fwhm=5.0,
    )
    fixed = nib.Nifti1Image(np.ones((2, 1, 1), dtype=np.float32), np.eye(4))
    analyzer.contrast_results["incongruent-congruent"] = {
        "fixed_effect": fixed,
        "fixed_variance": fixed,
        "fixed_stat": fixed,
        "n_runs": 2,
        "input_files": {
            "effects": [item[0] for item in sources],
            "variances": [item[1] for item in sources],
        },
    }
    scan_exclusion = "sub-s10_ses-03_task-flanker_run-1"
    contrast_exclusion = "sub-s10_ses-04_task-flanker_run-2"
    saved = analyzer.save_fixed_effects_maps(
        "incongruent-congruent",
        output_dir,
        contrast_formula="incongruent - congruent",
        exclusions={scan_exclusion, "sub-other_ses-01_task-flanker_run-1"},
        contrast_exclusions={
            (contrast_exclusion, "incongruent-congruent"),
            ("sub-s10_ses-05_task-flanker_run-1", "response_time"),
        },
    )

    assert "_stat-meanRunZ.nii.gz" in saved["mean_run_z"].name
    mean = nib.load(saved["mean_run_z"]).get_fdata().ravel()
    np.testing.assert_allclose(mean, [2.0, 7.0])

    metadata = json.loads(saved["metadata"].read_text())
    assert metadata["ContrastFormula"] == "incongruent - congruent"
    assert metadata["AnalysisSpace"] == "MNI"
    assert metadata["SmoothingFWHM"] == 5.0
    assert metadata["AggregationMethod"] == "unweighted fixed effects across runs"
    assert metadata["NumberOfIncludedRuns"] == 2
    assert metadata["IncludedRuns"] == [
        "sub-s10_ses-02_task-flanker_run-1",
        "sub-s10_ses-02_task-flanker_run-2",
    ]
    assert len(metadata["ContributingFiles"]["ZScore"]) == 2
    assert metadata["Exclusions"] == {
        "ScanLevel": [scan_exclusion],
        "ContrastLevel": [contrast_exclusion],
    }


class _FakeSurfaceGLM:
    def compute_contrast(self, formula, output_type="all"):
        assert formula == "a"
        assert output_type == "all"
        return {
            "effect_size": SurfaceResult(np.array([1.0, 2.0])),
            "effect_variance": SurfaceResult(np.array([0.5, 0.5])),
            "z_score": SurfaceResult(np.array([1.5, 2.5])),
        }


def test_per_run_surface_contrast_filename_has_requested_space(tmp_path):
    saved = compute_run_contrasts(
        _FakeSurfaceGLM(),
        "flanker",
        tmp_path,
        "sub-s10_ses-02_task-flanker_run-01",
        contrasts={"a": "a"},
        hemisphere="L",
        surface_space="fsaverage6",
    )

    for path in saved["a"].values():
        assert "_hemi-L_space-fsaverage6_contrast-a_" in path.name


def test_per_run_surface_contrast_defaults_to_fsnative(tmp_path):
    saved = compute_run_contrasts(
        _FakeSurfaceGLM(),
        "flanker",
        tmp_path,
        "sub-s10_ses-02_task-flanker_run-01",
        contrasts={"a": "a"},
        hemisphere="R",
    )

    assert all("_hemi-R_space-fsnative_contrast-a_" in path.name for path in saved["a"].values())


def test_surface_fixed_effects_discovery_accepts_current_and_legacy_names(tmp_path):
    current = (
        tmp_path
        / "sub-s10_hemi-L_space-fsaverage6_task-flanker_run-01"
        "_contrast-a_rtmodel-RTDur_stat-effect-size.func.gii"
    )
    legacy = (
        tmp_path
        / "sub-s10_hemi-L_task-flanker_run-02"
        "_contrast-a_rtmodel-RTDur_stat-effect-size.func.gii"
    )
    for effect in (current, legacy):
        effect.write_bytes(b"")
        effect.with_name(effect.name.replace("stat-effect-size", "stat-variance")).write_bytes(b"")

    analyzer = FixedEffectsAnalyzer("sub-s10", "flanker", hemisphere="L")
    effects, variances = analyzer.find_contrast_files(tmp_path, "a")

    assert effects == sorted([current, legacy])
    assert len(variances) == 2
