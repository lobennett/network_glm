"""Quality control and validation for GLM analysis."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.glm.contrasts import expression_to_contrast_vector

logger = logging.getLogger(__name__)


def est_contrast_vifs(desmat, contrasts):
    """
    IMPORTANT: This is only valid to use on design matrices where each regressor represents a condition vs baseline
     or if a parametrically modulated regressor is used the modulator must have more than 2 levels.  If it is a 2 level modulation,
     split the modulation into two regressors instead.

    Calculates VIF for contrasts based on the ratio of the contrast variance estimate using the
    true design to the variance estimate where between condition correaltions are set to 0
    desmat : pandas DataFrame, design matrix
    contrasts : dictionary of contrasts, key=contrast name,  using the desmat column names to express the contrasts
    returns: pandas DataFrame with VIFs for each contrast
    """
    desmat_copy = desmat.copy()
    # Remove constant columns (e.g. intercept) — not needed for VIF
    desmat_copy = desmat_copy.loc[:, desmat_copy.nunique() > 1]
    # Scaling stabilizes the matrix inversion
    nsamp = desmat_copy.shape[0]
    desmat_copy = (desmat_copy - desmat_copy.mean()) / ((nsamp - 1) ** 0.5 * desmat_copy.std())
    vifs_contrasts = {}
    for contrast_name, contrast_string in contrasts.items():
        contrast_cvec = expression_to_contrast_vector(contrast_string, desmat_copy.columns)
        true_var_contrast = (
            contrast_cvec
            @ np.linalg.inv(desmat_copy.transpose() @ desmat_copy)
            @ contrast_cvec.transpose()
        )
        # The following is the "best case" scenario because the between condition regressor correlations are set to 0
        best_var_contrast = (
            contrast_cvec
            @ np.linalg.inv(
                np.multiply(
                    desmat_copy.transpose() @ desmat_copy,
                    np.identity(desmat_copy.shape[1]),
                )
            )
            @ contrast_cvec.transpose()
        )
        vifs_contrasts[contrast_name] = true_var_contrast / best_var_contrast
    return vifs_contrasts


def run_quality_control(
    design_matrix: pd.DataFrame,
    contrasts: dict[str, str],
    percent_junk: float,
    output_dir: Path,
    subject_id: str,
    session: str,
    run: str,
    task_name: str,
) -> tuple[dict[str, float], bool]:
    """Run quality control analysis.

    Computes per-contrast VIFs and writes them to a CSV alongside the design
    matrix. Does NOT fail-fast on high VIFs — those are surfaced for manual
    review by the cohort QC step (`network_glm.cohort.outliers`) which
    aggregates VIFs across subjects and flags >threshold (default 5) entries
    in `lev1_flagged.tsv`.

    QA fails (`any_fail=True`) only on:
        - design matrix < 100 rows
        - regressors used in contrasts that are all-zero
        - junk percentage > 30%

    Args:
        design_matrix: Design matrix dataframe
        contrasts: Dictionary of contrast formulas
        percent_junk: Percentage of junk trials (0-1)
        output_dir: Quality control output directory
        subject_id: Subject identifier (required)
        session: Session identifier (required)
        run: Run identifier (required)
        task_name: Task name (required)

    Returns:
        Tuple of (vifs_dict, any_fail_bool)

    Raises:
        ValueError: If any required identifier is None or empty
    """

    # Check that required identifiers are provided
    required_params = {
        "subject_id": subject_id,
        "session": session,
        "run": run,
        "task_name": task_name,
    }

    for param_name, param_value in required_params.items():
        if param_value is None or param_value == "":
            raise ValueError(f"Required parameter '{param_name}' cannot be None or empty")

    any_fail = False

    # Check if design matrix has fewer than 100 rows
    if design_matrix.shape[0] < 100:
        any_fail = True
        logger.warning("QA FAIL: Design matrix has only %d rows (< 100)", design_matrix.shape[0])

    # Check for regressors used in contrasts that have all zeros
    design_column_names = design_matrix.columns.tolist()
    contrast_matrix = []
    for _key, values in contrasts.items():
        contrast_def = expression_to_contrast_vector(values, design_column_names)
        contrast_matrix.append(np.array(contrast_def))

    if contrast_matrix:
        contrast_matrix = np.array(contrast_matrix)
        columns_to_check = np.where(np.sum(np.abs(contrast_matrix), 0) != 0)[0]
        checked_columns_fail = (design_matrix.iloc[:, columns_to_check] == 0).all()
        any_column_fail = checked_columns_fail.any()

        if any_column_fail:
            any_fail = True
            bad_columns = list(checked_columns_fail.index[checked_columns_fail.values])
            logger.warning("QA FAIL: Regressors with all zeros used in contrasts: %s", bad_columns)

    # Check if percent junk is greater than 30%
    if percent_junk > 0.30:
        any_fail = True
        logger.warning("QA FAIL: High junk percentage: %.1f%% (> 30%%)", percent_junk * 100)

    # Calculate contrast VIFs (saved to CSV; not used to fail QA — cohort QC
    # at network_glm.cohort.outliers handles thresholding for review).
    try:
        vifs = est_contrast_vifs(design_matrix, contrasts)
    except Exception as e:
        logger.warning("VIF calculation failed: %s", e)
        vifs = {name: 0.0 for name in contrasts.keys()}

    # Save design matrix to quality control directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    design_matrix_filename = f"{subject_id}_{session}_task-{task_name}_{run}_desc-designMatrix.csv"
    design_matrix_file = output_dir / design_matrix_filename
    design_matrix.to_csv(design_matrix_file, index=False)
    logger.debug("Design matrix saved: %s", design_matrix_file)

    vifs_filename = f"{subject_id}_{session}_task-{task_name}_{run}_desc-contrastVIFs.csv"
    vifs_file = output_dir / vifs_filename
    vifs_df = pd.DataFrame(list(vifs.items()), columns=["contrast", "VIF"])
    vifs_df.to_csv(vifs_file, index=False)
    logger.debug("Contrast VIFs saved: %s", vifs_file)

    if any_fail:
        logger.warning("Overall QA: FAILED")
    else:
        logger.info("Overall QA: PASSED")

    return vifs, any_fail
