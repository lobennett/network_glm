"""Contrast computation and management for GLM analysis."""

import logging
import re
from pathlib import Path

from nilearn.glm.first_level import FirstLevelModel

from network_glm.lev1.processing.imaging import cast_nifti_to_float32
from network_glm.task_config.loader import get_task_contrasts

logger = logging.getLogger(__name__)


def filter_contrasts_for_dropped_columns(
    contrasts: dict[str, str],
    dropped_columns: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Filter contrasts to exclude those referencing dropped columns.

    When zero-variance columns are dropped from the design matrix,
    contrasts that reference those columns cannot be computed. This
    function identifies which contrasts are affected and returns
    only the computable contrasts.

    Args:
        contrasts: Dictionary mapping contrast names to contrast formulas
        dropped_columns: List of column names that were dropped from design matrix

    Returns:
        Tuple of (computable_contrasts, skipped_contrasts)
        - computable_contrasts: Contrasts that can be computed
        - skipped_contrasts: Contrasts that reference dropped columns (with reason)

    Examples:
        >>> contrasts = {'go_vs_baseline': 'go', 'stop_vs_go': 'stop_success - go'}
        >>> dropped = ['stop_success']
        >>> valid, skipped = filter_contrasts_for_dropped_columns(contrasts, dropped)
        >>> list(valid.keys())
        ['go_vs_baseline']
        >>> 'stop_vs_go' in skipped
        True
    """
    if not dropped_columns:
        return contrasts, {}

    computable = {}
    skipped = {}

    for contrast_name, formula in contrasts.items():
        # Check if any dropped column appears in the formula
        # Use word boundary matching to avoid partial matches
        # (e.g., 'go' should not match 'go_success')
        references_dropped = False
        dropped_refs = []

        for col in dropped_columns:
            # Match the column name as a whole word in the formula
            # This handles formulas like "stop_success - go" or "0.5*go + 0.5*stop"
            pattern = rf"\b{re.escape(col)}\b"
            if re.search(pattern, formula):
                references_dropped = True
                dropped_refs.append(col)

        if references_dropped:
            skipped[contrast_name] = f"References dropped column(s): {dropped_refs}"
            logger.warning(
                'Skipping contrast "%s" - references dropped column(s): %s',
                contrast_name,
                dropped_refs,
            )
        else:
            computable[contrast_name] = formula

    if skipped:
        logger.warning("%d contrast(s) skipped due to dropped columns", len(skipped))

    return computable, skipped


def compute_run_contrasts(
    fitted_glm: FirstLevelModel,
    task_name: str,
    output_dir: Path,
    base_filename: str,
    contrasts: dict[str, str] | None = None,
    hemisphere: str | None = None,
) -> dict[str, dict[str, Path]]:
    """Compute and save all contrasts for a run.

    Args:
        fitted_glm: Fitted GLM model
        task_name: Name of the task
        output_dir: Directory to save contrast maps
        base_filename: Base filename for output files
        contrasts: Optional custom contrast dictionary
        hemisphere: Optional hemisphere indicator ('L' or 'R') for surface data

    Returns:
        Dictionary mapping contrast names to saved file paths

    Examples:
        >>> saved_maps = compute_run_contrasts(
        ...     fitted_glm, 'stopSignal', Path('./contrasts'), 'sub-01_run-01'
        ... )
        >>> saved_maps_L = compute_run_contrasts(
        ...     fitted_glm_L, 'stopSignal', Path('./contrasts'), 'sub-01_run-01', hemisphere='L'
        ... )
    """
    # Use task contrasts if none provided
    if contrasts is None:
        contrasts = get_task_contrasts(task_name)

    all_saved_files = {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for contrast_name, contrast_formula in contrasts.items():
        try:
            # Compute contrast
            contrast_result = fitted_glm.compute_contrast(contrast_formula, output_type="all")

            # Determine file extension based on data type (surface vs volumetric)
            if hemisphere is not None:
                # Surface data - use GIFTI format
                file_ext = ".func.gii"
                hemi_part = f"_hemi-{hemisphere}"
            else:
                # Volumetric data - use NIfTI format
                file_ext = ".nii.gz"
                hemi_part = ""

            # Create filenames
            contrast_base = f"{base_filename}{hemi_part}_contrast-{contrast_name}_rtmodel-RTDur"

            saved_files = {}
            # Save contrast outputs as float32. nilearn's compute_contrast returns
            # Nifti1Images that, on to_filename(), get auto-scaled to the input
            # BOLD's storage dtype — for fmriprep preproc BOLDs this lands as uint8
            # with only 256 quantization levels across cal_min..cal_max. That kills
            # variance maps (most values truncate to 0) and degrades z_score and
            # effect_size precision. Cast to float32 explicitly.
            for stat_key, suffix in [
                ("effect_size", "effect-size"),
                ("effect_variance", "variance"),
                ("z_score", "z_score"),
            ]:
                # Skip surface (GIFTI) — only volumetric NIfTIs need the dtype fix.
                img = cast_nifti_to_float32(
                    contrast_result[stat_key], is_surface=hemisphere is not None
                )
                path = output_dir / f"{contrast_base}_stat-{suffix}{file_ext}"
                img.to_filename(path)
                saved_files[stat_key] = path

            all_saved_files[contrast_name] = saved_files

        except Exception as e:
            logger.error("Failed to compute/save contrast %s: %s", contrast_name, e)

    return all_saved_files
