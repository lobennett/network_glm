"""B3 regression: --skip-existing must recognize already-written surface
residuals. The skip-check filename in runner.process_single_run drifted from
the writer in processing/residuals.py (it omitted the `_space-{surface_space}`
segment), so the surface branch of --skip-existing never matched and always
recomputed. Both sites now share surface_residual_filename().
"""

import types

import numpy as np
import pandas as pd

from network_glm.lev1 import runner
from network_glm.lev1.processing.residuals import (
    process_surface_residuals,
    surface_residual_filename,
)
from network_glm.lev1.processing.surface_data import SurfaceGLM


def test_surface_residual_filename_includes_space_segment():
    name = surface_residual_filename("s10_ses-01_task-flanker_run-1", "L", "fsaverage6")
    assert name == (
        "s10_ses-01_task-flanker_run-1_hemi-L_space-fsaverage6" "_task-regressed-residuals.func.gii"
    )


def test_writer_output_basename_matches_helper(tmp_path):
    """The file process_surface_residuals actually writes must be named exactly
    as surface_residual_filename() reports (no drift between the two)."""
    np.random.seed(0)
    n_tp, n_verts = 120, 20  # > filtfilt padlen (default temporal filtering runs)
    X = np.random.randn(n_tp, 3)
    Y = (X @ np.random.randn(3, n_verts)).astype(np.float32)
    glm = SurfaceGLM(t_r=1.49, noise_model="ols")
    glm.fit(Y, pd.DataFrame(X, columns=["a", "b", "c"]))

    base = "s10_ses-01_task-flanker_run-1"
    res = process_surface_residuals(glm, tmp_path, base, "L", tr=1.49, surface_space="fsaverage6")
    assert res["success"], res
    assert res["saved_path"].name == surface_residual_filename(base, "L", "fsaverage6")


def test_skip_existing_skips_when_surface_residuals_present(tmp_path):
    """With both-hemisphere residuals already on disk (writer naming),
    process_single_run must short-circuit to True under --skip-existing."""
    args = types.SimpleNamespace(
        subj_id="s10",
        task_name="flanker",
        space="fsaverage6",
        skip_existing=True,
        residuals=True,
    )
    session, run = "ses-01", "run-1"
    base = f"{args.subj_id}_{session}_task-{args.task_name}_{run}"
    for hemi in ("L", "R"):
        (tmp_path / surface_residual_filename(base, hemi, "fsaverage6")).write_text("x")

    # Empty run_files: if the skip-check fails to match, process_single_run
    # falls through and raises ValueError on missing surface files. A correct
    # skip returns True before touching run_files.
    result = runner.process_single_run(
        session,
        run,
        {},
        args,
        sample_type="validation",
        dirs={"task_residuals": tmp_path},
        task_params={"tr": 1.49},
        exclusions=set(),
    )
    assert result is True
