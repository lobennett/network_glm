"""Tests for the ``--skip-qc-plots`` flag end-to-end.

Surface QC plots are matplotlib renders — ~10 plots × 2 hemispheres × N runs
per subject. At cohort scale (46 subjects × ~5 sessions × ~2 runs each) this
adds many hours of wall time to the surface lev1 rerun for output that isn't
needed during the rerun itself (the underlying ``.func.gii`` files are
persisted and can be re-plotted offline).

This test verifies:

1. The analysis-level flag, when set on ``args``, prevents the QC plot
   loop in ``process_surface_run`` from calling ``plot_surface_stat_map``.
2. Contrast files are still saved when QC plots are skipped — the flag
   must not accidentally short-circuit the contrast-writing code.

NOTE (network_glm lift-and-shift): the monolith also had two tests here
covering the pipeline-level ``neuro-run submit lev1 ... --skip-qc-plots``
sbatch-context propagation (the monolith's ``Lev1Pipeline`` class, in its
``pipelines`` package). That pipeline/submit-context class is out of scope
for network_glm's engine-only extraction (``network_glm.submit`` is a
from-scratch layer, not a port of the monolith's pipeline classes), so
those two tests were dropped.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pandas as pd


def test_surface_smoothing_uses_fsaverage6_when_bold_is_fsaverage6(tmp_path, monkeypatch):
    """When smoothing BOLD in fsaverage6 space, ``mri_surf2surf`` must be
    called with the template (``fsaverage6``) as the FS subject, not the
    per-subject recon. The BOLD has been resampled to the 40962-vertex
    template mesh; passing the subject's fsnative recon (~150k vertices)
    triggers ``ERROR: dimension inconsistency in source data``.

    Regression guard for the failure mode observed during the surface
    lev1 production submission.
    """
    from argparse import Namespace

    import numpy as np
    import pandas as pd

    from network_glm.lev1 import runner as run_module

    # Lay down a SUBJECTS_DIR with both a subject-specific recon and the
    # group template, mimicking what fmriprep deposits.
    fs_dir = tmp_path / "fmriprep" / "sourcedata" / "freesurfer"
    (fs_dir / "sub-s10_ses-09").mkdir(parents=True)
    (fs_dir / "fsaverage6").mkdir(parents=True)

    n_tp, n_verts = 80, 40962
    monkeypatch.setattr(
        run_module,
        "load_surface_data",
        lambda *a, **kw: np.random.randn(n_tp, n_verts).astype(np.float32),
    )
    monkeypatch.setattr(
        run_module,
        "plot_surface_stat_map",
        lambda *a, **kw: None,
    )

    smooth_calls = []
    monkeypatch.setattr(
        run_module,
        "smooth_surface_gifti",
        lambda *args, **kw: smooth_calls.append((args, kw)) or args[1],
    )

    class StubResult:
        def __init__(self):
            self.data = np.zeros(n_verts)

        def to_filename(self, path):
            pass

    class StubGLM:
        def __init__(self, *a, **kw):
            pass

        def fit(self, data, dm):
            return self

        def compute_contrast(self, formula, output_type="all"):
            return {
                "effect_size": StubResult(),
                "effect_variance": StubResult(),
                "z_score": StubResult(),
            }

    monkeypatch.setattr(run_module, "SurfaceGLM", StubGLM)

    dm = pd.DataFrame({"r0": np.random.randn(n_tp), "constant": np.ones(n_tp)})

    args = Namespace(
        fmriprep_dir=str(fs_dir.parent.parent),
        subj_id="sub-s10",
        task_name="flanker",
        smoothing_fwhm=2.0,
        space="fsaverage6",
        skip_qc_plots=True,
    )
    run_files = {"left_surface": "L.func.gii", "right_surface": "R.func.gii"}
    dirs = {"indiv_contrasts": tmp_path, "quality_control": tmp_path, "task_residuals": tmp_path}

    run_module.process_surface_run(
        run_files=run_files,
        design_matrix=dm,
        contrasts={"r0": "r0"},
        args=args,
        dirs=dirs,
        base_filename="sub-s10_task-flanker_run-1",
        tr=1.5,
        dummy_scans=0,
        compute_residuals=False,
        surface_space="fsaverage6",
    )

    # smooth_surface_gifti gets called once per hemisphere = 2 times.
    # Each call's 3rd positional arg is the FS subject. For fsaverage6
    # BOLD this must be 'fsaverage6', NOT 'sub-s10' nor 'sub-s10_ses-09'.
    assert (
        len(smooth_calls) == 2
    ), f"Expected smooth_surface_gifti called twice; got {len(smooth_calls)}"
    for call_args, _ in smooth_calls:
        fs_subject_passed = call_args[2]
        assert fs_subject_passed == "fsaverage6", (
            f"smooth_surface_gifti must receive fsaverage6 (the template) "
            f"as the FS subject when BOLD is in fsaverage6 space; "
            f"got {fs_subject_passed!r}."
        )


def test_analysis_run_skips_plot_loop_when_flag_set(tmp_path, monkeypatch):
    """End-to-end: ``args.skip_qc_plots=True`` prevents
    ``plot_surface_stat_map`` from being called.

    Contrast files must still be written — the flag must only gate the
    matplotlib render loop.
    """
    from network_glm.lev1 import runner as run_module

    # Stubs: surface data, SurfaceGLM, plot function
    n_tp = 100
    monkeypatch.setattr(
        run_module,
        "load_surface_data",
        lambda *a, **kw: np.random.randn(n_tp, 50).astype(np.float32),
    )

    plot_calls = []
    monkeypatch.setattr(
        run_module,
        "plot_surface_stat_map",
        lambda *a, **kw: plot_calls.append((a, kw)),
    )

    written_paths = []

    class StubResult:
        def __init__(self, n_verts=50):
            self.data = np.zeros(n_verts)

        def to_filename(self, path):
            written_paths.append(str(path))

    class StubSurfaceGLM:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, data, dm):
            return self

        def compute_contrast(self, formula, output_type="all"):
            return {
                "effect_size": StubResult(),
                "effect_variance": StubResult(),
                "z_score": StubResult(),
            }

    monkeypatch.setattr(run_module, "SurfaceGLM", StubSurfaceGLM)

    dm = pd.DataFrame(
        {
            "r0": np.random.randn(n_tp),
            "constant": np.ones(n_tp),
        }
    )

    args = Namespace(
        fmriprep_dir=str(tmp_path),
        subj_id="sub-test",
        task_name="flanker",
        smoothing_fwhm=None,
        space="fsaverage6",
        skip_qc_plots=True,
    )

    run_files = {"left_surface": "L.func.gii", "right_surface": "R.func.gii"}
    dirs = {"indiv_contrasts": tmp_path, "quality_control": tmp_path, "task_residuals": tmp_path}

    run_module.process_surface_run(
        run_files=run_files,
        design_matrix=dm,
        contrasts={"incongruent-congruent": "r0"},
        args=args,
        dirs=dirs,
        base_filename="sub-test_task-flanker_run-1",
        tr=1.5,
        dummy_scans=0,
        compute_residuals=False,
        surface_space="fsaverage6",
    )

    assert plot_calls == [], (
        f"plot_surface_stat_map must not be called when "
        f"--skip-qc-plots is on; got {len(plot_calls)} calls."
    )
    # Contrast files (effect-size, variance, z-score) per hemisphere should
    # still be written — verify by inspecting recorded to_filename() calls.
    assert any("stat-effect-size" in p for p in written_paths), (
        f"effect-size files must still be saved when QC plots are skipped; "
        f"written: {written_paths}"
    )
    assert any(
        "stat-z_score" in p for p in written_paths
    ), f"z_score files must still be saved; written: {written_paths}"


def test_analysis_run_keeps_plot_loop_when_flag_unset(tmp_path, monkeypatch):
    """When ``args.skip_qc_plots=False`` (default), plots are generated."""
    from network_glm.lev1 import runner as run_module

    n_tp = 100
    monkeypatch.setattr(
        run_module,
        "load_surface_data",
        lambda *a, **kw: np.random.randn(n_tp, 50).astype(np.float32),
    )

    plot_calls = []
    monkeypatch.setattr(
        run_module,
        "plot_surface_stat_map",
        lambda *a, **kw: plot_calls.append((a, kw)),
    )

    class StubResult:
        def __init__(self, n_verts=50):
            self.data = np.zeros(n_verts)

        def to_filename(self, path):
            pass

    class StubSurfaceGLM:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, data, dm):
            return self

        def compute_contrast(self, formula, output_type="all"):
            return {
                "effect_size": StubResult(),
                "effect_variance": StubResult(),
                "z_score": StubResult(),
            }

    monkeypatch.setattr(run_module, "SurfaceGLM", StubSurfaceGLM)

    dm = pd.DataFrame(
        {
            "r0": np.random.randn(n_tp),
            "constant": np.ones(n_tp),
        }
    )

    args = Namespace(
        fmriprep_dir=str(tmp_path),
        subj_id="sub-test",
        task_name="flanker",
        smoothing_fwhm=None,
        space="fsaverage6",
        skip_qc_plots=False,
    )

    run_files = {"left_surface": "L.func.gii", "right_surface": "R.func.gii"}
    dirs = {"indiv_contrasts": tmp_path, "quality_control": tmp_path, "task_residuals": tmp_path}

    run_module.process_surface_run(
        run_files=run_files,
        design_matrix=dm,
        contrasts={"incongruent-congruent": "r0"},
        args=args,
        dirs=dirs,
        base_filename="sub-test_task-flanker_run-1",
        tr=1.5,
        dummy_scans=0,
        compute_residuals=False,
        surface_space="fsaverage6",
    )

    # Two hemispheres × one contrast = two plot calls
    assert len(plot_calls) == 2, (
        f"Expected 2 plot calls (1 contrast × 2 hemispheres); " f"got {len(plot_calls)}"
    )
