"""Residuals processing and filtering for precision mapping."""

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import clean_img
from nilearn.signal import clean as clean_signal

logger = logging.getLogger(__name__)


def surface_residual_filename(base_filename: str, hemisphere: str, surface_space: str) -> str:
    """Canonical filename for a per-run surface task-residual GIFTI.

    Single source of truth shared by the writer (:func:`process_surface_residuals`)
    and the ``--skip-existing`` check in :mod:`..runner`, so the two cannot drift
    apart (which previously defeated --skip-existing for surface residuals).
    """
    return (
        f"{base_filename}_hemi-{hemisphere}_space-{surface_space}"
        f"_task-regressed-residuals.func.gii"
    )


class ResidualsProcessor:
    """Processor for GLM residuals with filtering capabilities."""

    def __init__(self, fitted_glm: FirstLevelModel, tr: float = 1.49):
        """Initialize residuals processor.

        Args:
            fitted_glm: Fitted GLM model with residuals
            tr: Repetition time in seconds

        Examples:
            >>> processor = ResidualsProcessor(fitted_glm, tr=1.5)
        """
        self.fitted_glm = fitted_glm
        self.tr = tr
        self.raw_residuals = None
        self.filtered_residuals = None

    def get_raw_residuals(self) -> list:
        """Get raw residuals from fitted GLM.

        Returns:
            List of residual images

        Examples:
            >>> residuals = processor.get_raw_residuals()
        """
        if self.raw_residuals is None:
            self.raw_residuals = self.fitted_glm.residuals

        return self.raw_residuals

    def apply_filtering(
        self,
        low_pass: float | None = 0.1,
        high_pass: float | None = 0.01,
        standardize: bool = False,
        detrend: bool = False,
        confounds: str | np.ndarray | None = None,
        mask_img: str | Path | None = None,
    ) -> list:
        """Apply filtering to residuals.

        Args:
            low_pass: Low-pass filter cutoff in Hz
            high_pass: High-pass filter cutoff in Hz
            standardize: Whether to standardize signals
            detrend: Whether to detrend signals
            confounds: Additional confounds to regress out
            mask_img: Optional brain mask

        Returns:
            List of filtered residual images

        Examples:
            >>> filtered = processor.apply_filtering(low_pass=0.1, high_pass=0.01)
        """
        raw_residuals = self.get_raw_residuals()

        if not raw_residuals:
            raise ValueError("No residuals available from GLM")

        self.filtered_residuals = []

        for i, residual_img in enumerate(raw_residuals):
            try:
                filtered_img = clean_img(
                    residual_img,
                    low_pass=low_pass,
                    high_pass=high_pass,
                    t_r=self.tr,
                    standardize=standardize,
                    detrend=detrend,
                    confounds=confounds,
                    mask_img=mask_img,
                )
                self.filtered_residuals.append(filtered_img)

            except Exception as e:
                logger.warning("Failed to filter residuals for run %d: %s", i + 1, e)
                # Use unfiltered residuals as fallback
                self.filtered_residuals.append(residual_img)

        return self.filtered_residuals

    def save_residuals(
        self, output_dir: Path, base_filename: str, residuals_type: str = "filtered"
    ) -> list[Path]:
        """Save residuals to disk.

        Args:
            output_dir: Directory to save residuals
            base_filename: Base filename for saved residuals
            residuals_type: Type of residuals to save ('raw' or 'filtered')

        Returns:
            List of saved file paths

        Examples:
            >>> saved_paths = processor.save_residuals(
            ...     Path('./residuals'), 'sub-01_run-01', 'filtered'
            ... )
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if residuals_type == "filtered":
            if self.filtered_residuals is None:
                raise ValueError("Filtered residuals not available. Call apply_filtering() first.")
            residuals_to_save = self.filtered_residuals
            suffix = "task-regressed-residuals"
        elif residuals_type == "raw":
            residuals_to_save = self.get_raw_residuals()
            suffix = "raw-residuals"
        else:
            raise ValueError(f"Unknown residuals_type: {residuals_type}")

        saved_paths = []

        for i, residual_img in enumerate(residuals_to_save):
            if len(residuals_to_save) == 1:
                # Single run
                filename = f"{base_filename}_{suffix}.nii.gz"
            else:
                # Multiple residual images from one processor. Use the unpadded
                # run-N convention used everywhere else in the codebase (BIDS
                # files, exclusion keys, .bidsignore globs are all run-1, not
                # run-01). NB: the per-run lev1 flow saves a single residual, so
                # this branch is not exercised in production; keeping it
                # convention-consistent avoids a future run-01/run-1 mismatch.
                filename = f"{base_filename}_run-{i + 1}_{suffix}.nii.gz"

            filepath = output_dir / filename
            residual_img.to_filename(filepath)
            saved_paths.append(filepath)

        return saved_paths

    def get_residuals_stats(self) -> dict:
        """Get statistics about the residuals.

        Returns:
            Dictionary with residuals statistics

        Examples:
            >>> stats = processor.get_residuals_stats()
        """
        stats = {
            "n_runs": 0,
            "raw_available": False,
            "filtered_available": False,
            "raw_stats": {},
            "filtered_stats": {},
        }

        # Raw residuals stats
        raw_residuals = self.get_raw_residuals()
        if raw_residuals:
            stats["n_runs"] = len(raw_residuals)
            stats["raw_available"] = True

            for i, residual_img in enumerate(raw_residuals):
                data = residual_img.get_fdata()
                stats["raw_stats"][f"run_{i + 1}"] = {
                    "shape": data.shape,
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                }

        # Filtered residuals stats
        if self.filtered_residuals:
            stats["filtered_available"] = True

            for i, residual_img in enumerate(self.filtered_residuals):
                data = residual_img.get_fdata()
                stats["filtered_stats"][f"run_{i + 1}"] = {
                    "shape": data.shape,
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                }

        return stats


def cifti_residual_filename(base_filename: str) -> str:
    """Canonical fsLR den-91k residual dtseries name (shared with --skip-existing)."""
    return f"{base_filename}_space-fsLR_den-91k_task-regressed-residuals.dtseries.nii"


def process_cifti_residuals(
    surface_glm,
    template,
    output_dir: Path,
    base_filename: str,
    tr: float = 1.49,
    low_pass: float | None = 0.1,
    high_pass: float | None = 0.01,
    fc_confounds: np.ndarray | None = None,
) -> dict:
    """Compute Y - X*beta from a fitted SurfaceGLM over CIFTI grayordinates,
    optionally band-pass + FC-confound clean, and write an fsLR den-91k dtseries
    reusing the input CIFTI header. Mirrors process_surface_residuals."""
    from network_glm.lev1.processing.cifti_io import write_dtseries

    output_dir = Path(output_dir)
    result = {"success": True, "saved_path": None, "errors": []}
    try:
        residuals = surface_glm.get_residuals()  # (T, n_grayordinates)
        if low_pass is not None or high_pass is not None or fc_confounds is not None:
            residuals = clean_signal(
                residuals, t_r=tr, low_pass=low_pass, high_pass=high_pass,
                confounds=fc_confounds, standardize=False, detrend=False,
            )
        out_path = write_dtseries(
            residuals.astype(np.float32), template,
            output_dir / cifti_residual_filename(base_filename),
        )
        result["saved_path"] = out_path
        logger.info("Saved CIFTI residuals: %s", out_path)
    except Exception as e:
        result["success"] = False
        result["errors"].append(str(e))
        logger.error("Failed to process CIFTI residuals: %s", e)
    return result


def process_run_residuals(
    fitted_glm: FirstLevelModel,
    output_dir: Path,
    base_filename: str,
    tr: float = 1.49,
    filtering_params: dict | None = None,
    mask_img: str | Path | None = None,
    fc_confounds: np.ndarray | None = None,
) -> dict:
    """Process and save residuals for a single run.

    Args:
        fitted_glm: Fitted GLM model
        output_dir: Directory to save residuals
        base_filename: Base filename for saved files
        tr: Repetition time in seconds
        filtering_params: Optional filtering parameters.  If provided, its
            ``confounds`` entry takes precedence over ``fc_confounds`` (callers
            who pass a full dict are responsible for what's in it).
        mask_img: Optional brain mask
        fc_confounds: Optional tissue/global-signal confounds (per ``get_fc_confounds``)
            to regress from the residuals during filtering.  Matches the
            ``process_surface_residuals`` signature so ``--fc-confounds``
            behaves symmetrically for volumetric and surface paths — previously
            this was silently ignored by the volumetric branch, producing
            task-only residuals when the user expected FC-quality residuals.

    Returns:
        Dictionary with processing results and saved paths

    Examples:
        >>> results = process_run_residuals(
        ...     fitted_glm, Path('./residuals'), 'sub-01_run-01'
        ... )
    """
    processor = ResidualsProcessor(fitted_glm, tr)

    # Default filtering parameters.  If the caller didn't override
    # filtering_params and supplied fc_confounds, plumb them through so the
    # tissue/global-signal confounds are regressed during nilearn's
    # clean_signal pass.
    if filtering_params is None:
        filtering_params = {
            "low_pass": 0.1,
            "high_pass": 0.01,
            # -- Correction --
            # FirstLevelModel already removed trends and task confounds.
            # `confounds` here are the FC-specific confounds (CSF, WM,
            # global signal + derivatives) applied to the post-GLM residuals,
            # which is the standard pre-FC denoising step.
            "standardize": False,
            "detrend": False,
            "confounds": fc_confounds,
        }

    results = {
        "processor": processor,
        "saved_paths": {},
        "stats": {},
        "success": True,
        "errors": [],
    }

    try:
        # Apply filtering
        processor.apply_filtering(mask_img=mask_img, **filtering_params)

        # Save filtered residuals
        saved_paths = processor.save_residuals(output_dir, base_filename, "filtered")
        results["saved_paths"]["filtered"] = saved_paths

        # Get statistics
        results["stats"] = processor.get_residuals_stats()

        logger.info("Saved residuals: %s", saved_paths[0])

    except Exception as e:
        results["success"] = False
        results["errors"].append(str(e))
        logger.error("Failed to process residuals: %s", e)

    return results


def process_surface_residuals(
    surface_glm,
    output_dir: Path,
    base_filename: str,
    hemisphere: str,
    tr: float = 1.49,
    low_pass: float | None = 0.1,
    high_pass: float | None = 0.01,
    fc_confounds: np.ndarray | None = None,
    surface_space: str = "fsnative",
) -> dict:
    """Process and save residuals for surface GLM.

    Computes Y - X*beta from the fitted SurfaceGLM, applies temporal
    filtering via nilearn's signal.clean, and saves as GIFTI files.

    Args:
        surface_glm: Fitted SurfaceGLM instance
        output_dir: Directory to save residuals
        base_filename: Base filename (without hemisphere)
        hemisphere: 'L' or 'R'
        tr: Repetition time in seconds
        low_pass: Low-pass filter cutoff in Hz (None to skip)
        high_pass: High-pass filter cutoff in Hz (None to skip)
        fc_confounds: Optional tissue confounds to regress (global signal,
            WM, CSF) from residuals for FC analysis.
        surface_space: Surface space name for output filename (default 'fsnative')

    Returns:
        Dictionary with 'success', 'saved_path', and 'errors'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {"success": True, "saved_path": None, "errors": []}

    try:
        residuals = surface_glm.get_residuals()  # (n_timepoints, n_vertices)

        # Apply temporal filtering (and optional FC confound regression)
        if low_pass is not None or high_pass is not None or fc_confounds is not None:
            residuals = clean_signal(
                residuals,
                t_r=tr,
                low_pass=low_pass,
                high_pass=high_pass,
                confounds=fc_confounds,
                standardize=False,
                detrend=False,
            )

        # Save as GIFTI (one darray per timepoint)
        darrays = [
            nib.gifti.GiftiDataArray(
                data=residuals[t].astype(np.float32),
                intent="NIFTI_INTENT_NONE",
                datatype="NIFTI_TYPE_FLOAT32",
            )
            for t in range(residuals.shape[0])
        ]
        gii_img = nib.GiftiImage(darrays=darrays)

        out_path = output_dir / surface_residual_filename(base_filename, hemisphere, surface_space)
        nib.save(gii_img, out_path)
        result["saved_path"] = out_path
        logger.info("Saved surface residuals: %s", out_path)

    except Exception as e:
        result["success"] = False
        result["errors"].append(str(e))
        logger.error("Failed to process surface residuals (hemi-%s): %s", hemisphere, e)

    return result
