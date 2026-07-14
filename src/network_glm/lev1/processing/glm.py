"""GLM fitting and computation for neuroimaging analysis."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import load_img

from network_glm.task_config.loader import TR

logger = logging.getLogger(__name__)


def handle_zero_variance_columns(
    design_matrix: pd.DataFrame,
    exclude_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Detect and remove zero-variance columns from design matrix.

    Zero-variance columns (constant values across all timepoints) cause
    numerical instability during GLM fitting. This function identifies
    and removes such columns, preserving the intercept/constant term.

    Args:
        design_matrix: Design matrix DataFrame
        exclude_columns: Column names to exclude from variance check
            (e.g., 'constant' for intercept). Defaults to ['constant'].

    Returns:
        Tuple of (cleaned_design_matrix, list_of_dropped_columns)

    Examples:
        >>> dm = pd.DataFrame({'task': [1, 2, 3], 'zero_col': [0, 0, 0], 'constant': [1, 1, 1]})
        >>> cleaned_dm, dropped = handle_zero_variance_columns(dm)
        >>> dropped
        ['zero_col']
    """
    if exclude_columns is None:
        exclude_columns = ["constant"]

    # Calculate variance for each column
    variances = design_matrix.var()

    # Identify zero-variance columns (excluding protected columns like 'constant')
    zero_var_cols = []
    for col in design_matrix.columns:
        if col in exclude_columns:
            continue
        if variances[col] == 0:
            zero_var_cols.append(col)

    if not zero_var_cols:
        logger.debug("Design matrix check: no zero-variance columns")
        return design_matrix, []

    logger.warning("Dropping %d zero-variance column(s): %s", len(zero_var_cols), zero_var_cols)

    # Drop the zero-variance columns
    cleaned_dm = design_matrix.drop(columns=zero_var_cols)

    # Verify the cleaned design matrix is still valid
    if cleaned_dm.shape[1] == 0:
        raise ValueError("All non-constant columns were zero-variance! Cannot fit GLM.")

    return cleaned_dm, zero_var_cols


def fit_run_glm(
    data_img: str | Path,
    design_matrix: pd.DataFrame,
    analysis_type: str = "task",
    subject_label: str | None = None,
    tr: float = TR,
    smoothing_fwhm: float | None = None,
    mask_img: str | Path | None = None,
) -> FirstLevelModel:
    """Fit GLM for a single run.

    Args:
        data_img: Path to 4D BOLD data
        design_matrix: Design matrix for the run
        analysis_type: Type of analysis ('task' or 'residual')
        subject_label: Subject identifier
        tr: Repetition time
        smoothing_fwhm: Optional smoothing kernel FWHM in mm (None for no smoothing)
        mask_img: Brain mask image. When provided (e.g. the per-run
            fMRIPrep mask), the GLM is fit only within this mask.
            None falls back to FirstLevelModel auto-masking.

    Returns:
        Fitted FirstLevelModel

    Examples:
        >>> fitted_glm = fit_run_glm(
        ...     'bold.nii.gz', design_matrix, 'task'
        ... )
    """
    # Handle different input types
    # For paths (including GIFTI), FirstLevelModel will load them
    # For NIfTI images already loaded, pass them directly
    if not isinstance(data_img, str | Path):
        # Assume it's an already-loaded image (NIfTI)
        pass

    # Set GLM parameters
    if analysis_type == "task":
        glm_params = {
            "mask_img": mask_img,
            "noise_model": "ar1",
            "standardize": False,
            "smoothing_fwhm": smoothing_fwhm,
            "minimize_memory": True,
        }
    elif analysis_type == "residual":
        glm_params = {
            "mask_img": mask_img,
            "noise_model": "ar1",
            "standardize": False,
            "smoothing_fwhm": smoothing_fwhm,
            "minimize_memory": False,
        }
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

    # Add subject label if provided
    if subject_label:
        glm_params["subject_label"] = subject_label

    mask_str = "fMRIPrep mask" if mask_img else "auto-masking"
    smoothing_str = f"{smoothing_fwhm}mm smoothing" if smoothing_fwhm else "no smoothing"
    logger.info("Fitting GLM: %s, %s, %s", mask_str, analysis_type, smoothing_str)

    # Initialize and fit model
    model = FirstLevelModel(**glm_params)
    fitted_model = model.fit(data_img, design_matrices=design_matrix)

    return fitted_model


class RankDeficientDesignError(ValueError):
    """Raised when the design matrix is rank-deficient (perfect collinearity)."""


def check_design_matrix_health(design_matrix: pd.DataFrame) -> None:
    """Fail fast on degenerate design matrices.

    Checks rank deficiency only — raises RankDeficientDesignError when
    matrix_rank < n_columns and names the most-correlated column pair.

    Per-column VIFs are intentionally NOT checked here. Nuisance regressors
    (motion + motion**2, cosine drift bases) routinely have inter-column VIFs
    of 100–1500 by design, which doesn't impair contrast estimation. The
    research-relevant signal is the per-contrast VIF, computed in
    `quality_control.py:est_contrast_vifs()` and threshold-checked inside
    `run_quality_control` (default threshold = 5).
    """
    arr = np.asarray(design_matrix.to_numpy(dtype=float, copy=True))
    n_cols = arr.shape[1]
    rank = np.linalg.matrix_rank(arr)
    if rank < n_cols:
        # Find the most-correlated column pair as a hint
        corr = pd.DataFrame(arr, columns=design_matrix.columns).corr().abs()
        corr_vals = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_vals, 0.0)
        corr = pd.DataFrame(corr_vals, index=corr.index, columns=corr.columns)
        worst_pair = corr.stack().idxmax()
        worst_val = corr.stack().max()
        raise RankDeficientDesignError(
            f"design matrix rank {rank} < n_columns {n_cols}; "
            f"most-correlated pair {worst_pair} (|r|={worst_val:.4f})"
        )


def validate_design_matrix(
    design_matrix: pd.DataFrame,
    n_scans: int,
) -> dict[str, Any]:
    """Validate a design matrix independently of the BOLD-data container.

    Used by both the volumetric path (which inlines this inside
    ``validate_glm_inputs``) and the surface path, where the BOLD data is
    already in memory as an ndarray and re-loading a NIfTI just to count
    timepoints is wasteful.  Catches:

      - empty design matrices
      - row-count mismatch with the BOLD timeseries (silent broadcasting
        upstream would otherwise produce nonsense betas)
      - NaN values (nilearn ``run_glm`` accepts them silently and returns
        corrupt results)
      - infinite values (same failure mode)
      - missing intercept (warning only)
      - rank-deficient designs (e.g. perfectly collinear regressors)

    Args:
        design_matrix: Design matrix to check.
        n_scans: Number of BOLD timepoints the design matrix should align with.

    Returns:
        Dict with ``is_valid`` flag, ``errors`` list, ``warnings`` list.
    """
    validation: dict[str, Any] = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
    }

    if design_matrix.empty:
        validation["errors"].append("Design matrix is empty")
        validation["is_valid"] = False
        return validation

    if design_matrix.shape[0] != n_scans:
        validation["errors"].append(
            f"Design matrix rows ({design_matrix.shape[0]}) != " f"BOLD timepoints ({n_scans})"
        )
        validation["is_valid"] = False

    if design_matrix.isnull().any().any():
        bad_cols = design_matrix.columns[design_matrix.isnull().any()].tolist()
        validation["errors"].append(f"Design matrix contains NaN values in columns: {bad_cols}")
        validation["is_valid"] = False

    if np.isinf(design_matrix.values).any():
        bad_cols = design_matrix.columns[np.isinf(design_matrix.values).any(axis=0)].tolist()
        validation["errors"].append(
            f"Design matrix contains infinite values in columns: {bad_cols}"
        )
        validation["is_valid"] = False

    # If the matrix has fundamental data integrity issues (NaN / Inf), skip
    # the rank check — LAPACK SVD will raise an opaque DLASCL error on
    # such matrices, which would mask the real (already-reported) cause.
    if not validation["is_valid"]:
        return validation

    if "constant" not in design_matrix.columns:
        # Look for any column that's a constant-vector intercept under a
        # different name (e.g. cosine00 from the DCT drift basis is often
        # constant when n_scans is small).
        has_constant = any(
            design_matrix[col].nunique() == 1 and design_matrix[col].iloc[0] != 0
            for col in design_matrix.columns
        )
        if not has_constant:
            validation["warnings"].append("No constant/intercept term found in design matrix")

    try:
        check_design_matrix_health(design_matrix)
    except RankDeficientDesignError as exc:
        validation["errors"].append(str(exc))
        validation["is_valid"] = False

    return validation


def validate_glm_inputs(
    data_img: str | Path,
    design_matrix: pd.DataFrame,
    mask_img: str | Path | None = None,
) -> dict[str, Any]:
    """Validate inputs for GLM analysis.

    Args:
        data_img: Path to 4D BOLD data
        design_matrix: Design matrix
        mask_img: Optional path to brain mask

    Returns:
        Dictionary with validation results

    Examples:
        >>> validation = validate_glm_inputs('bold.nii.gz', design_matrix)
        >>> validation['is_valid']
        True
    """
    validation = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
    }

    # Check data image
    try:
        if isinstance(data_img, str | Path):
            data_path = Path(data_img)
            if not data_path.exists():
                validation["errors"].append(f"Data image not found: {data_path}")
                validation["is_valid"] = False
            else:
                # Load and check data
                img = load_img(data_img)
                if len(img.shape) != 4:
                    validation["errors"].append(f"Expected 4D image, got {len(img.shape)}D")
                    validation["is_valid"] = False
                else:
                    n_scans = img.shape[-1]
                    validation["n_scans"] = n_scans

                    # Check design matrix dimensions
                    if design_matrix.shape[0] != n_scans:
                        validation["errors"].append(
                            f"Design matrix length ({design_matrix.shape[0]}) != "
                            f"number of scans ({n_scans})"
                        )
                        validation["is_valid"] = False

    except Exception as e:
        validation["errors"].append(f"Error loading data image: {e}")
        validation["is_valid"] = False

    # Check mask image
    if mask_img:
        try:
            if isinstance(mask_img, str | Path):
                mask_path = Path(mask_img)
                if not mask_path.exists():
                    validation["warnings"].append(f"Mask image not found: {mask_path}")
                else:
                    mask_img_loaded = load_img(mask_img)
                    if len(mask_img_loaded.shape) != 3:
                        validation["warnings"].append(
                            f"Expected 3D mask, got {len(mask_img_loaded.shape)}D"
                        )

        except Exception as e:
            validation["warnings"].append(f"Error loading mask image: {e}")

    # Check design matrix
    if design_matrix.empty:
        validation["errors"].append("Design matrix is empty")
        validation["is_valid"] = False

    if design_matrix.isnull().any().any():
        validation["errors"].append("Design matrix contains NaN values")
        validation["is_valid"] = False

    if np.isinf(design_matrix.values).any():
        validation["errors"].append("Design matrix contains infinite values")
        validation["is_valid"] = False

    # Check for constant regressor (intercept)
    if "constant" not in design_matrix.columns:
        validation["warnings"].append("No constant/intercept term found in design matrix")

    # Inline design-matrix sanity (rank only — contrast VIFs are research-level
    # and live in run_quality_control, saved per-run for cohort-QC review).
    try:
        check_design_matrix_health(design_matrix)
    except RankDeficientDesignError as exc:
        validation["errors"].append(str(exc))
        validation["is_valid"] = False

    return validation
