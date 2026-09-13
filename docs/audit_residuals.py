"""Offline numerical audit: uv run --frozen python docs/audit_residuals.py."""

import json

import numpy as np
import pandas as pd
from nilearn.signal import clean

from network_glm.lev1.processing.surface_data import SurfaceGLM


def audit():
    rng = np.random.default_rng(20260913)
    n_scans, n_vertices = 250, 8
    design = np.column_stack((np.ones(n_scans), rng.normal(size=(n_scans, 2))))
    noise = rng.normal(size=(n_scans, n_vertices))
    rho = np.linspace(0.1, 0.8, n_vertices)
    for time in range(1, n_scans):
        noise[time] += rho * noise[time - 1]
    data = 1000 + design @ rng.normal(size=(3, n_vertices)) + noise
    model = SurfaceGLM(t_r=1.49).fit(data, pd.DataFrame(design))
    residuals = model.get_residuals()

    # Independently solve the GLS problem for the AR coefficient chosen for
    # each group. Do not use the fitted theta to construct this reference.
    reference = np.empty_like(data)
    orthogonality_error = 0.0
    for label, result in model.results_.items():
        selected = model.labels_ == label
        coefficient = result.model.rho[0]
        whitened_design = design.copy()
        whitened_design[1:] -= coefficient * design[:-1]
        whitened_data = data[:, selected].copy()
        whitened_data[1:] -= coefficient * data[:-1, selected]
        beta = np.linalg.lstsq(whitened_design, whitened_data, rcond=None)[0]
        reference[:, selected] = data[:, selected] - design @ beta
        whitened_residual = residuals[:, selected].copy()
        whitened_residual[1:] -= coefficient * residuals[:-1, selected]
        orthogonality_error = max(
            orthogonality_error,
            float(np.max(np.abs(whitened_design.T @ whitened_residual))),
        )
    error = float(np.max(np.abs(residuals - reference)))
    assert error < 1e-8

    # Isolate the sequential-regression issue with OLS, where the initial
    # residual is exactly orthogonal to the task. This is an illustrative
    # counterexample, not an estimate of the effect in the Network dataset.
    basis, _ = np.linalg.qr(
        np.column_stack((np.ones(n_scans), rng.normal(size=(n_scans, 3))))
    )
    task, independent, other = basis[:, 1], basis[:, 2], basis[:, 3]
    tissue = 0.8 * task + 0.6 * independent
    signal = 3 * task + 2 * tissue + 0.1 * other
    task_design = pd.DataFrame({"task": task, "constant": np.ones(n_scans)})
    raw = (
        SurfaceGLM(t_r=1.49, noise_model="ols")
        .fit(signal[:, None], task_design)
        .get_residuals()
    )
    sequential = clean(raw, confounds=tissue[:, None], detrend=False, standardize=None)
    joint = clean(
        signal[:, None],
        confounds=np.column_stack((task, tissue)),
        detrend=False,
        standardize=None,
    )
    return {
        "ar1_gls_max_absolute_error": error,
        "ar1_whitened_design_residual_inner_product_max": orthogonality_error,
        "ols_task_correlation_before_fc_cleanup": float(
            np.corrcoef(task, raw[:, 0])[0, 1]
        ),
        "ols_task_correlation_after_sequential_fc_cleanup": float(
            np.corrcoef(task, sequential[:, 0])[0, 1]
        ),
        "task_correlation_after_joint_regression": float(
            np.corrcoef(task, joint[:, 0])[0, 1]
        ),
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
