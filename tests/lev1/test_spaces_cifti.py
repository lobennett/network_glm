import argparse

from network_glm.lev1.spaces import is_surface_space, is_cifti_space, resolve_surface_space


def test_fslr_is_cifti_not_surface():
    assert is_cifti_space("fsLR") is True
    assert is_surface_space("fsLR") is False


def test_surface_spaces_unchanged():
    assert is_surface_space("surface") is True
    assert is_surface_space("fsaverage6") is True
    assert is_cifti_space("surface") is False
    assert is_cifti_space("MNI") is False
    assert resolve_surface_space("surface") == "fsnative"


def test_setup_masks_skips_for_fslr(tmp_path):
    """setup_masks must not attempt volumetric mask extraction for fsLR (CIFTI):
    there are no brain mask files in the CIFTI derivatives tree, so falling
    through to the volumetric branch raises ValueError("Some masks are
    invalid...") before any per-run work happens. Mirrors the existing
    surface-space skip.
    """
    from network_glm.lev1.prepare import setup_masks

    dirs = {"masks": tmp_path}
    args = argparse.Namespace(space="fsLR", subj_id="sub-x", within_subject_threshold=0.5)
    assert setup_masks({}, args, dirs) is None

    # Same skip behavior as the pre-existing surface space, for comparison.
    args_surface = argparse.Namespace(
        space="fsaverage6", subj_id="sub-x", within_subject_threshold=0.5
    )
    assert setup_masks({}, args_surface, dirs) is None
