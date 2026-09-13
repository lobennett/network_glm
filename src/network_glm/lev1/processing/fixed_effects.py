"""Fixed effects analysis for combining results across runs."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nilearn.glm.contrasts import compute_fixed_effects

from network_glm.lev1 import cache
from network_glm.lev1.processing.imaging import cast_nifti_to_float32
from network_glm.lev1.processing.surface_data import (
    SurfaceResult,
    compute_surface_fixed_effects,
    load_surface_stat_map,
)
from network_glm.task_config.loader import get_task_contrasts

logger = logging.getLogger(__name__)


def compute_mean_run_z(z_files: list[Path], is_surface: bool = False) -> Any:
    """Return the arithmetic, voxel/vertex-wise NaN-safe mean of run Z maps.

    A location that is NaN in some runs uses the remaining values. A location
    that is NaN in every run remains NaN, rather than being converted to zero.
    """
    if not z_files:
        raise ValueError("At least one per-run Z map is required")

    if is_surface:
        arrays = [
            np.asarray(load_surface_stat_map(path), dtype=np.float64)
            for path in z_files
        ]
        template = None
    else:
        images = [nib.load(str(path)) for path in z_files]
        arrays = [np.asarray(image.get_fdata(), dtype=np.float64) for image in images]
        template = images[0]

    stack = np.stack(arrays, axis=0)
    valid_counts = np.sum(~np.isnan(stack), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(stack, axis=0) / valid_counts
    mean = np.where(valid_counts > 0, mean, np.nan).astype(np.float32)

    if is_surface:
        return SurfaceResult(mean)

    header = template.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(mean, template.affine, header)


class FixedEffectsAnalyzer:
    """Analyzer for computing fixed effects across runs."""

    def __init__(
        self,
        subject_id: str,
        task_name: str,
        mask_img: str | Path | None = None,
        min_runs: int = 2,
        hemisphere: str | None = None,
        surface_space: str = "fsnative",
        analysis_space: str | None = None,
        smoothing_fwhm: float | None = None,
        rt_model: str = "RTDur",
    ):
        """Initialize fixed effects analyzer.

        Args:
            subject_id: Subject identifier
            task_name: Task name
            mask_img: Optional brain mask image
            min_runs: Minimum runs required to compute a non-tagged fixed-effects map (default: 2).
            hemisphere: Optional hemisphere ('L' or 'R') for surface data
            surface_space: Surface space name for output filenames (default 'fsnative')
            analysis_space: Analysis space recorded in contrast metadata
            smoothing_fwhm: Smoothing FWHM in millimeters, or None for no smoothing

        Examples:
            >>> analyzer = FixedEffectsAnalyzer('sub-01', 'stopSignal')
            >>> analyzer_L = FixedEffectsAnalyzer('sub-01', 'stopSignal', hemisphere='L')
        """
        self.subject_id = subject_id
        self.task_name = task_name
        self.mask_img = mask_img
        self.min_runs = min_runs
        self.hemisphere = hemisphere
        self.surface_space = surface_space
        self.analysis_space = analysis_space or (
            surface_space if hemisphere is not None else "volume"
        )
        self.smoothing_fwhm = smoothing_fwhm
        self.rt_model = rt_model
        self.contrast_results = {}

    def find_contrast_files(
        self,
        contrast_dir: Path,
        contrast_name: str,
        exclusions: set[str] | None = None,
        contrast_exclusions: set[tuple[str, str]] | None = None,
    ) -> tuple[list[Path], list[Path]]:
        """Find effect size and variance files for a contrast.

        Args:
            contrast_dir: Directory containing contrast files
            contrast_name: Name of the contrast
            exclusions: Scan-level exclusion keys to skip (drops the whole run).
            contrast_exclusions: ``(scan_key, contrast)`` pairs to skip — drops a
                single run's contribution to THIS contrast only (lev1_outlier
                per-contrast VIF exclusions), leaving the run's other contrasts.

        Returns:
            Tuple of (effect_files, variance_files)

        Examples:
            >>> effects, variances = analyzer.find_contrast_files(
            ...     Path('./contrasts'), 'inhibition', {'sub-01_ses-01_task-stop_run-1'}
            ... )
        """
        if exclusions is None:
            exclusions = set()
        if contrast_exclusions is None:
            contrast_exclusions = set()

        # Determine file extension based on hemisphere (surface vs volumetric)
        if self.hemisphere is not None:
            file_ext = ".func.gii"
            # The wildcard after hemi-H_ accepts current names with a space
            # entity and legacy names without one.
            effect_pattern = f"*hemi-{self.hemisphere}_*contrast-{contrast_name}_*stat-effect-size{file_ext}"
            variance_pattern = f"*hemi-{self.hemisphere}_*contrast-{contrast_name}_*stat-variance{file_ext}"
        else:
            file_ext = ".nii.gz"
            # Pattern for volumetric files
            effect_pattern = f"*contrast-{contrast_name}_*stat-effect-size{file_ext}"
            variance_pattern = f"*contrast-{contrast_name}_*stat-variance{file_ext}"

        effect_files = []
        variance_files = []

        # Find all matching files
        all_effect_files = list(contrast_dir.glob(effect_pattern))
        all_variance_files = list(contrast_dir.glob(variance_pattern))

        # Filter out excluded runs (scan-level) and excluded (run, contrast) pairs.
        for effect_file in all_effect_files:
            # Parse filename to check for exclusions
            exclusion_key = self._parse_exclusion_key(effect_file)
            if exclusion_key in exclusions:
                continue
            if (exclusion_key, contrast_name) in contrast_exclusions:
                logger.info(
                    "Dropping %s contrast %s for run %s (per-contrast VIF exclusion)",
                    self.subject_id,
                    contrast_name,
                    exclusion_key,
                )
                continue

            effect_files.append(effect_file)
            # Find corresponding variance file
            variance_file = effect_file.with_name(
                effect_file.name.replace("stat-effect-size", "stat-variance")
            )
            if variance_file in all_variance_files:
                variance_files.append(variance_file)
            else:
                logger.warning("Missing variance file for %s", effect_file)

        # Sort files to ensure consistent ordering
        effect_files.sort()
        variance_files.sort()

        return effect_files, variance_files

    def _parse_exclusion_key(self, filepath: Path) -> str:
        """Parse exclusion key from contrast filename.

        The key format must match what run_lev1.py uses:
            '{subject}_{session}_task-{task_name}_{run}'

        Args:
            filepath: Path to contrast file

        Returns:
            Exclusion key in format 'sub-X_ses-Y_task-TASK_run-Z'

        Examples:
            >>> key = analyzer._parse_exclusion_key(
            ...     Path('sub-s03_ses-01_task-stopSignal_run-01_contrast-go.nii.gz')
            ... )
            >>> key
            'sub-s03_ses-01_task-stopSignal_run-1'
        """
        filename = filepath.name

        # Use [^_]+ (non-underscore) to avoid over-matching across BIDS entities
        patterns = {
            "subject": r"(sub-[^_]+)",
            "session": r"(ses-[^_]+)",
            "run": r"run-(\d+)",
        }

        components = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, filename)
            if match:
                if key == "run":
                    run_num = match.group(1).lstrip("0") or "0"
                    components[key] = f"run-{run_num}"
                else:
                    components[key] = match.group(1)

        # Create exclusion key with task- prefix (matching run_lev1.py format)
        if all(k in components for k in ["subject", "session", "run"]):
            return f"{components['subject']}_{components['session']}_task-{self.task_name}_{components['run']}"

        return filename  # Fallback to filename if parsing fails

    def compute_fixed_effects_contrast(
        self,
        contrast_name: str,
        effect_files: list[Path],
        variance_files: list[Path],
        precision_weighted: bool = False,
    ) -> tuple[Any | None, Any | None, Any | None]:
        """Compute fixed effects for a single contrast.

        Args:
            contrast_name: Name of the contrast
            effect_files: List of effect size image files
            variance_files: List of variance image files
            precision_weighted: Whether to use precision weighting

        Returns:
            Tuple of (fixed_effect_img, fixed_variance_img, fixed_stat_img)

        Examples:
            >>> effect_img, var_img, stat_img = analyzer.compute_fixed_effects_contrast(
            ...     'inhibition', effect_files, variance_files
            ... )
        """
        if len(effect_files) != len(variance_files):
            raise ValueError(
                f"File count mismatch for {contrast_name}: "
                f"{len(effect_files)} effects, {len(variance_files)} variances"
            )

        if not effect_files:
            logger.warning("No files found for contrast %s", contrast_name)
            return None, None, None

        try:
            # Use surface-specific fixed effects for GIFTI files
            if self.hemisphere is not None:
                # Surface data - use custom implementation
                fixed_effect_result, fixed_variance_result, fixed_stat_result = (
                    compute_surface_fixed_effects(
                        effect_files,
                        variance_files,
                        precision_weighted=precision_weighted,
                    )
                )
                # These are SurfaceResult objects, store them directly
                fixed_effect_img = fixed_effect_result
                fixed_variance_img = fixed_variance_result
                fixed_stat_img = fixed_stat_result
            else:
                # Volumetric data - use nilearn's implementation.
                # nilearn >=0.10 returns 4 values: (effect, variance, stat, z_score).
                # We use the z_score (4th) — nilearn's stat (3rd) is effect/sqrt(variance)
                # and blows up to +/- 1e10 at out-of-mask voxels where variance == 0.
                # The z_score uses a stabler formulation. Files saved under the
                # `-z_score` filename suffix correctly hold z-scores.
                _result = compute_fixed_effects(
                    effect_files,
                    variance_files,
                    mask=self.mask_img,
                    precision_weighted=precision_weighted,
                )
                fixed_effect_img = _result[0]
                fixed_variance_img = _result[1]
                fixed_stat_img = _result[3] if len(_result) >= 4 else _result[2]

            logger.info(
                "Fixed effects for %s: %d runs included",
                contrast_name,
                len(effect_files),
            )

            # Store results
            self.contrast_results[contrast_name] = {
                "fixed_effect": fixed_effect_img,
                "fixed_variance": fixed_variance_img,
                "fixed_stat": fixed_stat_img,
                "n_runs": len(effect_files),
                "input_files": {"effects": effect_files, "variances": variance_files},
            }

            return fixed_effect_img, fixed_variance_img, fixed_stat_img

        except Exception as e:
            raise RuntimeError(f"Fixed effects failed for {contrast_name}: {e}") from e

    def _build_base_filename(
        self, contrast_name: str, *, below_min: bool | None = None
    ) -> str:
        """Construct the BIDS-style base filename for this contrast's saved maps.

        Applies the `_desc-belowMinRuns` tag when this contrast's n_runs is
        below `self.min_runs`. The tag substring includes the trailing
        underscore so downstream lev2 filtering can use a substring match.
        """
        if self.hemisphere is not None:
            hemi_tag = f"_hemi-{self.hemisphere}"
            space_tag = f"_space-{self.surface_space}"
        else:
            hemi_tag = ""
            space_tag = ""

        if below_min is None:
            below_min = self.contrast_results[contrast_name]["n_runs"] < self.min_runs
        below_min_tag = "_desc-belowMinRuns" if below_min else ""

        return (
            f"{self.subject_id}{hemi_tag}{space_tag}"
            f"_task-{self.task_name}"
            f"_contrast-{contrast_name}"
            f"_rtmodel-{self.rt_model}{below_min_tag}"
            f"_stat-fixed-effects"
        )

    def retire_contrast_outputs(self, contrast_name: str, output_dir: Path) -> None:
        """Retire both eligibility variants before refreshing this contrast."""
        extension = ".func.gii" if self.hemisphere is not None else ".nii.gz"
        paths = []
        for below_min in (False, True):
            base = self._build_base_filename(contrast_name, below_min=below_min)
            paths.extend(
                Path(output_dir) / (base + suffix)
                for suffix in (
                    extension,
                    "-variance" + extension,
                    "-z_score" + extension,
                    ".json",
                )
            )
            paths.append(
                Path(output_dir)
                / (base.replace("_stat-fixed-effects", "_stat-meanRunZ") + extension)
            )
        cache.retire_outputs(paths)

    def _metadata(
        self,
        contrast_name: str,
        contrast_formula: str,
        exclusions: set[str],
        contrast_exclusions: set[tuple[str, str]],
    ) -> dict[str, Any]:
        """Build the JSON-safe audit record for one fixed-effects contrast."""
        results = self.contrast_results[contrast_name]
        effect_files = [Path(path) for path in results["input_files"]["effects"]]
        variance_files = [Path(path) for path in results["input_files"]["variances"]]
        z_files = [
            path.with_name(path.name.replace("stat-effect-size", "stat-z_score"))
            for path in effect_files
        ]
        included_runs = [self._parse_exclusion_key(path) for path in effect_files]
        relevant_prefix = f"{self.subject_id}_"
        relevant_task = f"_task-{self.task_name}_"
        scan_exclusions = sorted(
            key
            for key in exclusions
            if key.startswith(relevant_prefix) and relevant_task in key
        )
        per_contrast_exclusions = sorted(
            key
            for key, excluded_contrast in contrast_exclusions
            if excluded_contrast == contrast_name
            and key.startswith(relevant_prefix)
            and relevant_task in key
        )
        return {
            "ContrastName": contrast_name,
            "ContrastFormula": contrast_formula,
            "AnalysisSpace": self.analysis_space,
            "SmoothingFWHM": self.smoothing_fwhm,
            "AggregationMethod": "unweighted fixed effects across runs",
            "MeanRunZAggregationMethod": (
                "arithmetic mean across contributing per-run Z maps, ignoring NaN values"
            ),
            "NumberOfIncludedRuns": len(included_runs),
            "IncludedRuns": included_runs,
            "ContributingFiles": {
                "EffectSize": [str(path) for path in effect_files],
                "Variance": [str(path) for path in variance_files],
                "ZScore": [str(path) for path in z_files],
            },
            "Exclusions": {
                "ScanLevel": scan_exclusions,
                "ContrastLevel": per_contrast_exclusions,
            },
        }

    def save_fixed_effects_maps(
        self,
        contrast_name: str,
        output_dir: Path,
        base_filename: str | None = None,
        contrast_formula: str = "",
        exclusions: set[str] | None = None,
        contrast_exclusions: set[tuple[str, str]] | None = None,
    ) -> dict[str, Path]:
        """Save fixed effects maps for a contrast.

        Args:
            contrast_name: Name of the contrast
            output_dir: Directory to save maps
            base_filename: Optional base filename

        Returns:
            Dictionary mapping map types to saved paths

        Examples:
            >>> saved_files = analyzer.save_fixed_effects_maps(
            ...     'inhibition', Path('./fixed_effects')
            ... )
        """
        if contrast_name not in self.contrast_results:
            raise ValueError(
                f"Fixed effects for {contrast_name} have not been computed"
            )

        results = self.contrast_results[contrast_name]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension (hemisphere/space tags are now in _build_base_filename)
        file_ext = ".func.gii" if self.hemisphere is not None else ".nii.gz"

        if base_filename is None:
            base_filename = self._build_base_filename(contrast_name)

        # Warn whenever this contrast is below the floor, regardless of whether
        # the caller supplied base_filename. The warning is the audit trail.
        n_runs = self.contrast_results[contrast_name]["n_runs"]
        if n_runs < self.min_runs:
            logger.warning(
                "tagged %s/task-%s/contrast-%s as _desc-belowMinRuns: "
                "n_runs=%d (min_runs=%d)",
                self.subject_id,
                self.task_name,
                contrast_name,
                n_runs,
                self.min_runs,
            )

        saved_files = {}

        # Save fixed effects maps. Cast volumetric outputs to float32 to avoid
        # the uint8 auto-scaling on to_filename() that nibabel applies when the
        # input BOLD's header marks integer storage (see imaging.cast_nifti_to_float32).
        is_surface = self.hemisphere is not None

        if results["fixed_effect"] is not None:
            effect_path = output_dir / f"{base_filename}{file_ext}"
            cast_nifti_to_float32(
                results["fixed_effect"], is_surface=is_surface
            ).to_filename(effect_path)
            saved_files["fixed_effect"] = effect_path

        if results["fixed_variance"] is not None:
            variance_path = output_dir / f"{base_filename}-variance{file_ext}"
            cast_nifti_to_float32(
                results["fixed_variance"], is_surface=is_surface
            ).to_filename(variance_path)
            saved_files["fixed_variance"] = variance_path

        if results["fixed_stat"] is not None:
            stat_path = output_dir / f"{base_filename}-z_score{file_ext}"
            cast_nifti_to_float32(
                results["fixed_stat"], is_surface=is_surface
            ).to_filename(stat_path)
            saved_files["fixed_stat"] = stat_path

        z_files = [
            Path(path).with_name(
                Path(path).name.replace("stat-effect-size", "stat-z_score")
            )
            for path in results["input_files"]["effects"]
        ]
        missing_z = [path for path in z_files if not path.is_file()]
        if missing_z:
            raise FileNotFoundError(
                "Cannot compute meanRunZ; missing contributing per-run Z map(s): "
                + ", ".join(str(path) for path in missing_z)
            )
        mean_run_z = compute_mean_run_z(z_files, is_surface=is_surface)
        mean_base = base_filename.replace("_stat-fixed-effects", "_stat-meanRunZ")
        mean_path = output_dir / f"{mean_base}{file_ext}"
        cast_nifti_to_float32(mean_run_z, is_surface=is_surface).to_filename(mean_path)
        saved_files["mean_run_z"] = mean_path

        metadata = self._metadata(
            contrast_name,
            contrast_formula,
            exclusions or set(),
            contrast_exclusions or set(),
        )
        metadata_path = output_dir / f"{base_filename}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        saved_files["metadata"] = metadata_path

        return saved_files

    def compute_all_task_fixed_effects(
        self,
        contrast_dir: Path,
        output_dir: Path,
        exclusions: set[str] | None = None,
        contrasts: dict[str, str] | None = None,
        contrast_exclusions: set[tuple[str, str]] | None = None,
    ) -> dict[str, dict[str, Path]]:
        """Compute fixed effects for all task contrasts.

        Args:
            contrast_dir: Directory containing individual contrast files
            output_dir: Directory to save fixed effects results
            exclusions: Set of runs to exclude
            contrasts: Optional custom contrasts dictionary

        Returns:
            Dictionary mapping contrast names to saved file paths

        Examples:
            >>> results = analyzer.compute_all_task_fixed_effects(
            ...     Path('./contrasts'), Path('./fixed_effects'), exclusions
            ... )
        """
        if contrasts is None:
            contrasts = get_task_contrasts(self.task_name, self.rt_model)

        all_saved_files = {}
        exclusions = exclusions or set()
        contrast_exclusions = contrast_exclusions or set()

        for contrast_name in contrasts:
            self.retire_contrast_outputs(contrast_name, output_dir)

        for contrast_name, contrast_formula in contrasts.items():
            # Find files for this contrast
            effect_files, variance_files = self.find_contrast_files(
                contrast_dir, contrast_name, exclusions, contrast_exclusions
            )

            if effect_files or variance_files:
                # Compute fixed effects
                fixed_effect, fixed_variance, fixed_stat = (
                    self.compute_fixed_effects_contrast(
                        contrast_name, effect_files, variance_files
                    )
                )

                if fixed_effect is not None:
                    # Save results
                    saved_files = self.save_fixed_effects_maps(
                        contrast_name,
                        output_dir,
                        contrast_formula=contrast_formula,
                        exclusions=exclusions,
                        contrast_exclusions=contrast_exclusions,
                    )
                    all_saved_files[contrast_name] = saved_files

        # Surface silent-loss: contrasts whose per-run files never got written
        # (e.g. all runs hit a zero-variance regressor and skipped the save in
        # filter_contrasts_for_dropped_columns) silently disappear from the
        # subject's output. lev2 then has to discover their absence from
        # a glob.  Log a warning here so the loss is visible at the per-subject
        # log level rather than buried in an absent file.
        expected = set(contrasts.keys())
        produced = set(all_saved_files.keys())
        missing = expected - produced
        if missing:
            logger.warning(
                "%s/%s/%s: %d of %d expected contrasts have no fixed-effects "
                "output; missing: %s. Likely cause: every run dropped these "
                "contrasts at write time because the contributing regressor "
                "had zero variance in that run (rare event type or sparse "
                "subset).",
                getattr(self, "subject_id", "?"),
                getattr(self, "task_name", "?"),
                getattr(self, "hemisphere", None) or "volumetric",
                len(missing),
                len(expected),
                sorted(missing),
            )

        return all_saved_files

    def get_contrast_summary(self) -> dict[str, dict]:
        """Get summary of computed fixed effects contrasts.

        Returns:
            Dictionary with contrast summaries

        Examples:
            >>> summary = analyzer.get_contrast_summary()
        """
        summary = {}

        for contrast_name, results in self.contrast_results.items():
            summary[contrast_name] = {
                "n_runs_included": results["n_runs"],
                "has_fixed_effect": results["fixed_effect"] is not None,
                "has_fixed_variance": results["fixed_variance"] is not None,
                "has_fixed_stat": results["fixed_stat"] is not None,
                "input_files": {
                    "n_effect_files": len(results["input_files"]["effects"]),
                    "n_variance_files": len(results["input_files"]["variances"]),
                },
            }

        return summary


def compute_subject_fixed_effects(
    subject_id: str,
    task_name: str,
    contrast_dir: Path,
    output_dir: Path,
    mask_img: str | Path | None = None,
    exclusions: set[str] | None = None,
    min_runs: int = 2,
    hemisphere: str | None = None,
    surface_space: str = "fsnative",
    contrast_exclusions: set[tuple[str, str]] | None = None,
    analysis_space: str | None = None,
    smoothing_fwhm: float | None = None,
    rt_model: str = "RTDur",
) -> dict[str, dict[str, Path]]:
    """Compute fixed effects for all contrasts for a subject.

    Args:
        subject_id: Subject identifier
        task_name: Task name
        contrast_dir: Directory with individual contrast files
        output_dir: Directory to save fixed effects
        mask_img: Optional brain mask
        exclusions: Optional set of runs to exclude
        min_runs: Minimum runs threshold passed to the analyzer (default 2).
        hemisphere: Optional hemisphere ('L' or 'R') for surface data
        surface_space: Surface space name for output filenames (default 'fsnative')
        analysis_space: Analysis space recorded in contrast metadata
        smoothing_fwhm: Smoothing FWHM in millimeters, or None for no smoothing

    Returns:
        Dictionary mapping contrast names to saved file paths

    Examples:
        >>> results = compute_subject_fixed_effects(
        ...     'sub-01', 'stopSignal', Path('./contrasts'), Path('./fixed_effects')
        ... )
        >>> results_L = compute_subject_fixed_effects(
        ...     'sub-01', 'stopSignal', Path('./contrasts'), Path('./fixed_effects'),
        ...     hemisphere='L'
        ... )
    """
    analyzer = FixedEffectsAnalyzer(
        subject_id,
        task_name,
        mask_img,
        min_runs=min_runs,
        hemisphere=hemisphere,
        surface_space=surface_space,
        analysis_space=analysis_space,
        smoothing_fwhm=smoothing_fwhm,
        rt_model=rt_model,
    )

    return analyzer.compute_all_task_fixed_effects(
        contrast_dir, output_dir, exclusions, contrast_exclusions=contrast_exclusions
    )
