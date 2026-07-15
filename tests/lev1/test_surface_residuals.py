"""Tests for SurfaceGLM residuals computation."""

import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import run_glm

from network_glm.lev1.processing.surface_data import SurfaceGLM


def test_get_residuals_uses_per_vertex_betas():
    """Residuals must use each vertex's own betas, not the first vertex's."""
    np.random.seed(42)
    n_tp, n_verts = 200, 100
    X = np.random.randn(n_tp, 5)
    # Create data with known signal: each vertex has different betas
    true_betas = np.random.randn(5, n_verts)
    Y = X @ true_betas + np.random.randn(n_tp, n_verts) * 0.1

    dm = pd.DataFrame(X, columns=[f"reg_{i}" for i in range(5)])
    glm = SurfaceGLM(t_r=1.5, noise_model="ols")
    glm.fit(Y, dm)

    residuals = glm.get_residuals()
    assert residuals.shape == (n_tp, n_verts)

    # Residuals should be small (signal was well-fit)
    # With the old bug, some vertices got wrong betas -> large residuals
    assert np.std(residuals) < 0.5, (
        f"Residuals std {np.std(residuals):.3f} too large; "
        "per-vertex betas may not be applied correctly"
    )


def test_get_residuals_matches_nilearn_per_label():
    """Residuals should equal Y - X @ theta for each label group."""
    np.random.seed(42)
    n_tp, n_verts = 200, 100
    X = np.random.randn(n_tp, 5)
    true_betas = np.random.randn(5, n_verts)
    Y = X @ true_betas + np.random.randn(n_tp, n_verts) * 0.1

    dm = pd.DataFrame(X, columns=[f"reg_{i}" for i in range(5)])
    glm = SurfaceGLM(t_r=1.5, noise_model="ols")
    glm.fit(Y, dm)

    residuals = glm.get_residuals()

    # Verify against manual correct computation using nilearn's run_glm
    labels, results = run_glm(Y, X, noise_model="ols")
    expected = np.zeros_like(Y)
    for label in np.unique(labels):
        mask = labels == label
        expected[:, mask] = Y[:, mask] - X @ results[label].theta

    np.testing.assert_allclose(residuals, expected, atol=1e-5)


def test_get_residuals_ar1_multi_vertex_labels():
    """AR(1) groups vertices with similar autocorrelation — each must keep own betas."""
    np.random.seed(42)
    n_tp, n_verts = 200, 80
    X = np.random.randn(n_tp, 4)
    true_betas = np.random.randn(4, n_verts) * 2
    Y = (X @ true_betas + np.random.randn(n_tp, n_verts) * 0.5).astype(np.float32)

    dm = pd.DataFrame(X, columns=[f"r{i}" for i in range(4)])
    glm = SurfaceGLM(t_r=1.5, noise_model="ar1")
    glm.fit(Y, dm)

    residuals = glm.get_residuals()

    # Verify per-label computation
    labels, results = run_glm(Y, X, noise_model="ar1")
    expected = np.zeros_like(Y)
    for label in np.unique(labels):
        mask = labels == label
        expected[:, mask] = Y[:, mask] - X @ results[label].theta

    np.testing.assert_allclose(residuals, expected, atol=1e-4)


def test_get_residuals_before_fit_raises():
    """Calling get_residuals before fit should raise ValueError."""
    glm = SurfaceGLM(t_r=1.5)
    with pytest.raises(ValueError, match="Model must be fit"):
        glm.get_residuals()
