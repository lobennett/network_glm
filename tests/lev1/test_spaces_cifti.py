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
