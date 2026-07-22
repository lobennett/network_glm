import nibabel as nib
import numpy as np
import pandas as pd
from network_glm.lev1.processing.cifti_io import load_dtseries
from network_glm.lev1.processing.residuals import process_cifti_residuals
from network_glm.lev1.processing.surface_data import SurfaceGLM


def _make_dtseries(path, n_t=40, n_gray=500):
    bm = nib.cifti2.BrainModelAxis.from_mask(np.ones(n_gray, dtype=bool),
                                             name="CIFTI_STRUCTURE_CORTEX_LEFT")
    ax = nib.cifti2.SeriesAxis(start=0, step=1.49, size=n_t)
    hdr = nib.Cifti2Header.from_axes((ax, bm))
    rng = np.random.RandomState(1)
    data = rng.randn(n_t, n_gray).astype(np.float32)
    nib.Cifti2Image(data, header=hdr).to_filename(str(path))
    return data


def test_process_cifti_residuals_writes_dtseries(tmp_path):
    p = tmp_path / "in.dtseries.nii"
    data = _make_dtseries(p, n_t=40, n_gray=500)
    _, template = load_dtseries(p)
    dm = pd.DataFrame({"reg": np.linspace(0, 1, 40), "constant": np.ones(40)})
    glm = SurfaceGLM(t_r=1.49).fit(data, dm)
    out = process_cifti_residuals(glm, template, tmp_path, "sub-x_ses-01_task-t_run-1",
                                  tr=1.49, fc_confounds=None, low_pass=None, high_pass=None)
    assert out["success"] is True
    saved = out["saved_path"]
    assert saved.name.endswith("_space-fsLR_den-91k_desc-taskRegressedResiduals_bold.dtseries.nii")
    reloaded, _ = load_dtseries(saved)
    assert reloaded.shape == (40, 500)


def test_cifti_residual_name_matches_postproc_glob():
    import fnmatch

    from network_glm.lev1.processing.residuals import cifti_residual_filename

    name = cifti_residual_filename("sub-x_ses-01_task-rest_run-1")
    assert fnmatch.fnmatch(name, "*_space-fsLR_den-91k_*bold.dtseries.nii")


def test_process_cifti_run_requires_residuals(tmp_path):
    import argparse
    from network_glm.lev1.runner import process_cifti_run
    args = argparse.Namespace(residuals=False, task_name="stopSignal", subj_id="sub-x")
    try:
        process_cifti_run({}, pd.DataFrame(), args, {}, "sub-x_ses-01_task-t_run-1", 1.49, None)
        assert False, "expected ValueError when --residuals not set"
    except ValueError as e:
        assert "residuals" in str(e).lower()
