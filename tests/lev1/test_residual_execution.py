"""Residual requests must write the promised data or fail the run."""

from argparse import Namespace

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import FirstLevelModel

from network_glm.lev1 import runner
from network_glm.lev1.processing.residuals import (
    ResidualsProcessor,
    process_run_residuals,
)
from network_glm.lev1.processing.surface_data import SurfaceGLM
from network_glm.lev1.processing.glm import fit_run_glm
from network_glm.lev1.processing.contrasts import compute_run_contrasts


@pytest.fixture
def surface_run(tmp_path):
    rng = np.random.default_rng(12)
    n_scans = 100
    design = pd.DataFrame({"task": rng.normal(size=n_scans), "constant": 1.0})
    files = {}
    for hemi, key in (("L", "left_surface"), ("R", "right_surface")):
        data = (
            1000
            + design.to_numpy() @ rng.normal(size=(2, 5))
            + rng.normal(size=(n_scans, 5))
        ).astype(np.float32)
        path = tmp_path / f"{hemi}.func.gii"
        nib.save(
            nib.gifti.GiftiImage(
                darrays=[nib.gifti.GiftiDataArray(row) for row in data]
            ),
            path,
        )
        files[key] = path
    args = Namespace(
        subj_id="sub-test",
        task_name="flanker",
        fmriprep_dir=str(tmp_path),
        smoothing_fwhm=None,
        skip_qc_plots=True,
        no_residual_filter=False,
    )
    dirs = {
        name: tmp_path / name
        for name in ("indiv_contrasts", "quality_control", "task_residuals")
    }
    for directory in dirs.values():
        directory.mkdir()
    return dict(
        run_files=files,
        design_matrix=design,
        contrasts={"task": "task"},
        args=args,
        dirs=dirs,
        base_filename="sub-test_ses-1_task-flanker_run-1",
        tr=1.49,
        dummy_scans=0,
        compute_residuals=True,
        surface_space="fsaverage6",
    )


def test_skip_plots_still_writes_both_hemisphere_residuals(surface_run):
    result = runner.process_surface_run(**surface_run)
    assert set(result) == {"L", "R"}
    outputs = sorted(surface_run["dirs"]["task_residuals"].glob("*.func.gii"))
    assert len(outputs) == 2
    for path in outputs:
        data = np.stack([array.data for array in nib.load(path).darrays])
        assert data.shape == (100, 5)
        assert np.isfinite(data).all()
    assert not list(surface_run["dirs"]["quality_control"].glob("*.png"))


def test_surface_no_filter_writes_unfiltered_model_residuals(surface_run):
    surface_run["args"].no_residual_filter = True
    runner.process_surface_run(**surface_run)
    path = next(surface_run["dirs"]["task_residuals"].glob("*hemi-L*.func.gii"))
    actual = np.stack([array.data for array in nib.load(path).darrays])
    data = np.stack(
        [
            array.data
            for array in nib.load(surface_run["run_files"]["left_surface"]).darrays
        ]
    )
    model = SurfaceGLM(t_r=1.49).fit(data, surface_run["design_matrix"])
    np.testing.assert_allclose(actual, model.get_residuals(), atol=1e-6)


def test_surface_filtering_failure_fails_the_run(surface_run, monkeypatch):
    surface_run["args"].skip_qc_plots = False
    # Rendering is unrelated to this failure; exercise real fitting/filtering/IO.
    monkeypatch.setattr(runner, "plot_surface_stat_map", lambda *a, **kw: None)
    surface_run["fc_confounds"] = np.ones((99, 1))
    with pytest.raises(RuntimeError, match="residual"):
        runner.process_surface_run(**surface_run)


@pytest.fixture
def volume_model():
    rng = np.random.default_rng(31)
    n_scans = 100
    design = pd.DataFrame({"task": rng.normal(size=n_scans), "constant": 1.0})
    noise = rng.normal(size=(n_scans, 8))
    for t in range(1, n_scans):
        noise[t] += 0.8 * noise[t - 1]
    data = 1000 + design.to_numpy() @ rng.normal(size=(2, 8)) + noise
    image = nib.Nifti1Image(data.T.reshape(2, 2, 2, n_scans), np.eye(4))
    mask = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    return FirstLevelModel(
        mask_img=mask, noise_model="ar1", signal_scaling=False, minimize_memory=False
    ).fit(image, design_matrices=design)


def test_volume_filter_failure_does_not_save_raw_data_as_filtered(
    tmp_path, volume_model
):

    result = process_run_residuals(
        volume_model, tmp_path, "sub-test", fc_confounds=np.ones((99, 1))
    )
    assert result["success"] is False
    assert not list(tmp_path.glob("*.nii.gz"))


def test_volume_ar1_residuals_use_unwhitened_prediction(volume_model):
    image = ResidualsProcessor(volume_model).get_raw_residuals()[0]
    actual = volume_model.masker_.transform(image)
    design = volume_model.design_matrices_[0].to_numpy()
    for label, result in volume_model.results_[0].items():
        selected = volume_model.labels_[0] == label
        expected = result.Y - design @ result.theta
        np.testing.assert_allclose(actual[:, selected], expected, atol=1e-9)


def test_cifti_filter_failure_fails_run(tmp_path):
    rng = np.random.default_rng(17)
    data = rng.normal(size=(100, 5)).astype(np.float32)
    axes = (
        nib.cifti2.SeriesAxis(start=0, step=1.49, size=100),
        nib.cifti2.BrainModelAxis.from_mask(
            np.ones(5, dtype=bool), name="CIFTI_STRUCTURE_CORTEX_LEFT"
        ),
    )
    path = tmp_path / "input.dtseries.nii"
    nib.Cifti2Image(data, header=nib.Cifti2Header.from_axes(axes)).to_filename(path)
    design = pd.DataFrame({"constant": np.ones(100)})
    with pytest.raises(RuntimeError, match="residual"):
        runner.process_cifti_run(
            {"cifti_bold": path},
            design,
            Namespace(residuals=True),
            {"task_residuals": tmp_path / "out"},
            "sub-test",
            1.49,
            fc_confounds=np.ones((99, 1)),
        )


def test_surface_residual_subtraction_preserves_small_fluctuations():
    rng = np.random.default_rng(42)
    data = (10000 + rng.normal(size=(100, 3))).astype(np.float32)
    design = pd.DataFrame({"constant": np.ones(100)})
    model = SurfaceGLM(t_r=1.49, noise_model="ols").fit(data, design)
    expected = data.astype(np.float64) - data.mean(axis=0, dtype=np.float64)
    np.testing.assert_allclose(model.get_residuals(), expected, atol=1e-9)


def test_integer_storage_does_not_quantize_smoothed_glm_inputs():
    """The same values stored as int16 or float64 must produce the same fit."""
    rng = np.random.default_rng(23)
    n_scans = 60
    design = pd.DataFrame({"task": rng.normal(size=n_scans), "constant": 1.0})
    signal = 1000 + design.to_numpy() @ rng.normal(size=(2, 27)) * 5
    data = (signal + rng.normal(size=signal.shape)).T.reshape(3, 3, 3, n_scans)
    integer_data = data.astype(np.int16)
    mask = nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), np.eye(4))
    estimates = []
    for values in (integer_data, integer_data.astype(np.float64)):
        model = fit_run_glm(
            nib.Nifti1Image(values, np.eye(4)),
            design,
            smoothing_fwhm=1.5,
            mask_img=mask,
        )
        estimates.append(
            model.compute_contrast("task", output_type="effect_size").get_fdata()
        )
    # Nilearn smooths int16 images in float32, but retains float64 for the
    # reference. Allow that rounding, not the ~0.008 quantization error in 0.14.0.
    np.testing.assert_allclose(estimates[0], estimates[1], rtol=0, atol=1e-6)


def test_missing_requested_contrast_fails_instead_of_returning_partial_success(
    tmp_path,
):
    design = pd.DataFrame({"constant": np.ones(100)})
    data = np.random.default_rng(0).normal(size=(100, 3))
    model = SurfaceGLM(t_r=1.49).fit(data, design)
    with pytest.raises(RuntimeError, match="contrast"):
        compute_run_contrasts(
            model,
            "flanker",
            tmp_path,
            "sub-test",
            contrasts={"missing": "missing"},
            hemisphere="L",
        )


def test_fixed_effect_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    def failed_aggregation(*args, **kwargs):
        raise RuntimeError("aggregation failed")

    monkeypatch.setattr(runner, "compute_subject_fixed_effects", failed_aggregation)
    monkeypatch.setattr(runner, "load_contrast_exclusions", lambda path: {})
    args = Namespace(
        space="MNI",
        subj_id="sub-test",
        task_name="flanker",
        exclusions_file="unused",
        min_runs=2,
        smoothing_fwhm=None,
    )
    with pytest.raises(RuntimeError, match="aggregation failed"):
        runner.compute_fixed_effects_all(
            args,
            {"indiv_contrasts": tmp_path, "fixed_effects": tmp_path},
            set(),
            None,
            [],
            2,
        )


@pytest.mark.parametrize("run_count", [1, 2])
def test_failed_runs_cannot_contribute_stale_maps_to_fixed_effects(
    tmp_path, monkeypatch, run_count
):
    included_exclusions = set()

    def aggregate(*args, **kwargs):
        included_exclusions.update(kwargs["exclusions"])
        return {}

    monkeypatch.setattr(runner, "compute_subject_fixed_effects", aggregate)
    monkeypatch.setattr(runner, "load_contrast_exclusions", lambda path: {})
    args = Namespace(
        space="fsaverage6",
        subj_id="sub-test",
        task_name="flanker",
        exclusions_file="unused",
        min_runs=2,
        smoothing_fwhm=None,
    )
    runner.compute_fixed_effects_all(
        args,
        {"indiv_contrasts": tmp_path, "fixed_effects": tmp_path},
        set(),
        None,
        ["ses-1/run-1"],
        run_count,
    )
    assert "sub-test_ses-1_task-flanker_run-1" in included_exclusions


def test_surface_run_completion_controls_resume(surface_run, monkeypatch):
    args = surface_run["args"]
    args.space = "fsaverage6"
    args.residuals = True
    args.fc_confounds = False
    args.skip_existing = False
    args.rt_model = "RTDur"
    files = surface_run["run_files"]
    directory = surface_run["dirs"]["quality_control"]
    files["events"] = directory / "events.tsv"
    files["events"].write_text("onset\tduration\n0\t1\n")
    files["confounds"] = directory / "confounds.tsv"
    files["confounds"].write_text("constant\n" + "1\n" * 100)
    for key in ("left_surface", "right_surface"):
        files[key].with_name(files[key].name.split(".")[0] + ".json").write_text("{}")

    # This test covers orchestration around the fit; event/QA modeling has its
    # own tests. Keep actual GIFTI loading, fitting, residual writing, and hashing.
    monkeypatch.setattr(runner, "preprocess_events", lambda df, *a, **kw: df)
    monkeypatch.setattr(runner, "add_junk_trials", lambda df, *a: (df, 0))
    monkeypatch.setattr(
        runner,
        "load_and_process_confounds",
        lambda *a, **kw: pd.DataFrame(index=range(100)),
    )
    monkeypatch.setattr(
        runner,
        "create_design_matrix",
        lambda *a, **kw: (surface_run["design_matrix"], {}),
    )
    monkeypatch.setattr(runner, "get_task_contrasts", lambda *a: {"task": "task"})
    monkeypatch.setattr(runner, "run_quality_control", lambda *a, **kw: ({}, False))

    def run():
        return runner.process_single_run(
            "ses-1",
            "run-1",
            files,
            args,
            "discovery",
            surface_run["dirs"],
            {"tr": 1.49},
            set(),
        )

    assert run()
    assert (
        len(list(surface_run["dirs"]["task_residuals"].glob("*_completion.json"))) == 1
    )

    def unexpected_refit(*a, **kw):
        raise RuntimeError("refit requested")

    monkeypatch.setattr(runner, "process_surface_run", unexpected_refit)
    args.skip_existing = True
    assert run()
    contrast_dir = surface_run["dirs"]["indiv_contrasts"]
    obsolete = contrast_dir / (
        "sub-test_ses-1_task-flanker_run-1_hemi-L_space-fsaverage6"
        "_contrast-dropped_rtmodel-RTDur_stat-effect-size.func.gii"
    )
    obsolete.write_text("old contrast")
    args.no_residual_filter = True
    with pytest.raises(RuntimeError, match="refit requested"):
        run()
    assert not obsolete.exists()
    archived = list(contrast_dir.glob(obsolete.name + ".superseded-*"))
    assert len(archived) == 1
    assert archived[0].read_text() == "old contrast"
