"""fmriprep convention: framewise_displacement and std_dvars are NaN at row 0
(undefined at t=0). Lev1 must not crash, NaN-out the design matrix, or warn."""

from __future__ import annotations

import numpy as np
import pandas as pd

from network_glm.lev1.processing.confounds import load_and_process_confounds


def test_load_and_process_confounds_handles_nan_first_row(tmp_path):
    """fmriprep emits framewise_displacement and std_dvars as NaN at t=0
    (undefined — no prior frame to compute against). The confounds loader
    must handle this without leaving NaN in the output, which would
    contaminate the design matrix.
    """
    confounds = pd.DataFrame(
        {
            "framewise_displacement": [np.nan, 0.1, 0.1, 0.2],
            "std_dvars": [np.nan, 1.0, 1.0, 1.1],
            "trans_x": [0.0, 0.01, 0.0, 0.0],
            "trans_y": [0.0, 0.0, 0.01, 0.0],
            "trans_z": [0.0, 0.0, 0.0, 0.01],
            "trans_x_derivative1": [np.nan, 0.01, -0.01, 0.0],
            "trans_y_derivative1": [np.nan, 0.0, 0.01, -0.01],
            "trans_z_derivative1": [np.nan, 0.0, 0.0, 0.01],
            "rot_x": [0.0, 0.001, 0.0, 0.0],
            "rot_y": [0.0, 0.0, 0.001, 0.0],
            "rot_z": [0.0, 0.0, 0.0, 0.001],
            "rot_x_derivative1": [np.nan, 0.001, -0.001, 0.0],
            "rot_y_derivative1": [np.nan, 0.0, 0.001, -0.001],
            "rot_z_derivative1": [np.nan, 0.0, 0.0, 0.001],
        }
    )
    fp = tmp_path / "confounds.tsv"
    confounds.to_csv(fp, sep="\t", index=False, na_rep="n/a")

    out = load_and_process_confounds(
        fp,
        task_name="stopSignal",
        sample_type="discovery",
        dummy_scans=0,
    )

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 4, f"expected 4 rows, got {len(out)}"
    # No NaN should remain after preprocessing — the function should fill or drop.
    assert not out.isna().any().any(), (
        "no NaN should remain after load_and_process_confounds; "
        "fmriprep's convention of NaN at t=0 must be handled"
    )
