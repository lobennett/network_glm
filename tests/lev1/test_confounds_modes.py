import pandas as pd
from network_glm.lev1.processing.confounds import load_and_process_confounds


def _write_confounds(tmp_path):
    cols = {
        "cosine00": [1.0, 1.0, 1.0], "cosine01": [0.1, 0.2, 0.3],
        "trans_x": [0.0, 0.1, 0.2], "trans_x_derivative1": [0, 0.1, 0.1],
        "trans_x_power2": [0, 0.01, 0.04], "rot_z": [0.0, 0.01, 0.02],
        "motion_outlier00": [0, 1, 0],
        "csf": [1.0, 1.1, 1.2], "white_matter": [2.0, 2.1, 2.2],
        "global_signal": [3.0, 3.1, 3.2],
    }
    p = tmp_path / "confounds.tsv"
    pd.DataFrame(cols).to_csv(p, sep="\t", index=False)
    return p


def test_full_keeps_motion_cosine_spikes(tmp_path):
    df = load_and_process_confounds(_write_confounds(tmp_path), "stopSignal",
                                    "validation", confounds_mode="full")
    assert any(c.startswith("cosine") for c in df.columns)
    assert "trans_x" in df.columns and "motion_outlier00" in df.columns
    assert "csf" not in df.columns and "global_signal" not in df.columns


def test_no_motion_drops_motion_and_spikes(tmp_path):
    df = load_and_process_confounds(_write_confounds(tmp_path), "stopSignal",
                                    "validation", confounds_mode="no-motion")
    assert any(c.startswith("cosine") for c in df.columns)
    assert not any(c.startswith(("trans_", "rot_")) for c in df.columns)
    assert not any(c.startswith("motion_outlier") for c in df.columns)


def test_no_cosine_drops_cosines_keeps_motion(tmp_path):
    """The fourth cell of the 2x2: motion regressed, drift NOT modelled.

    Asked for by the NSI experiment — `no-motion` (cosines only) did not move the
    score, so this isolates the cosine set as the suspect. Note it leaves the run
    with no high-pass at all: the DCT cosines are the only drift model in the lev1
    design, and the fsLR residual path is run with --no-residual-filter.
    """
    df = load_and_process_confounds(_write_confounds(tmp_path), "stopSignal",
                                    "validation", confounds_mode="no-cosine")
    assert not any(c.startswith("cosine") for c in df.columns)
    assert "trans_x" in df.columns and "trans_x_derivative1" in df.columns
    assert "rot_z" in df.columns and "motion_outlier00" in df.columns
    assert "csf" not in df.columns and "global_signal" not in df.columns


def test_task_only_drops_everything(tmp_path):
    df = load_and_process_confounds(_write_confounds(tmp_path), "stopSignal",
                                    "validation", confounds_mode="task-only")
    assert df.shape[1] == 0
    assert len(df) == 3  # rows preserved for design-matrix concat
