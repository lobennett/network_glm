"""J3 regression: multi-image ResidualsProcessor.save_residuals uses the
unpadded run-N convention (run-1, run-2), not zero-padded run-01, matching
BIDS filenames / exclusion keys / .bidsignore globs everywhere else.
"""

import nibabel as nib
import numpy as np

from network_glm.lev1.processing.residuals import ResidualsProcessor


def test_save_residuals_multi_image_uses_unpadded_run_label(tmp_path):
    proc = ResidualsProcessor.__new__(ResidualsProcessor)  # bypass __init__/GLM
    img = nib.Nifti1Image(np.zeros((2, 2, 2, 3), dtype=np.float32), np.eye(4))
    proc.filtered_residuals = [img, img]  # >1 image -> multi-image branch

    paths = proc.save_residuals(tmp_path, "sub-s10_ses-01_task-flanker", "filtered")
    names = sorted(p.name for p in paths)
    assert names == [
        "sub-s10_ses-01_task-flanker_run-1_task-regressed-residuals.nii.gz",
        "sub-s10_ses-01_task-flanker_run-2_task-regressed-residuals.nii.gz",
    ], names
    assert not any("run-01" in n for n in names)
