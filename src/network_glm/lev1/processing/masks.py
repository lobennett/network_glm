"""Mask processing utilities for neuroimaging analysis."""

import logging
from pathlib import Path

from nilearn.image import load_img
from nilearn.masking import intersect_masks

logger = logging.getLogger(__name__)


class MaskProcessor:
    """Process and combine brain masks."""

    @staticmethod
    def combine_masks_with_threshold(mask_files: list[Path], threshold: float = 1.0):
        """Combine masks using specified threshold.

        Args:
            mask_files: List of paths to mask files
            threshold: Intersection threshold (0.0-1.0)
                      1.0 = all masks must overlap
                      0.5 = majority of masks must overlap

        Returns:
            Combined mask as nibabel image

        Examples:
            >>> mask_files = [Path('mask1.nii.gz'), Path('mask2.nii.gz')]
            >>> combined = MaskProcessor.combine_masks_with_threshold(mask_files, 0.8)
            >>> combined.to_filename('combined_mask.nii.gz')
        """
        if len(mask_files) == 1:
            return load_img(mask_files[0])

        return intersect_masks(mask_files, threshold=threshold, connected=True)

    @staticmethod
    def extract_mask_files(files: dict, analysis_space: str = "MNI") -> list[Path]:
        """Extract mask files from files dictionary for specified space.

        Args:
            files: Files dictionary from FileFinder.get_files()
            analysis_space: Analysis space ('T1w' or 'MNI')

        Returns:
            List of mask file paths

        Examples:
            >>> files = {'ses-01': {'run-01': {'mni_brain_mask': Path('mask.nii.gz')}}}
            >>> masks = MaskProcessor.extract_mask_files(files, 'MNI')
            >>> print(f"Found {len(masks)} mask files")
        """
        mask_key = "t1w_brain_mask" if analysis_space == "T1w" else "mni_brain_mask"
        mask_files = []

        for session_runs in files.values():
            for run_files in session_runs.values():
                if mask_key in run_files:
                    mask_files.append(run_files[mask_key])

        return mask_files

    @staticmethod
    def create_combined_mask(
        files: dict,
        analysis_space: str = "MNI",
        threshold: float = 1.0,
        output_path: Path | str | None = None,
    ):
        """Create and optionally save combined mask.

        Args:
            files: Files dictionary from FileFinder.get_files()
            analysis_space: Analysis space ('T1w' or 'MNI')
            threshold: Intersection threshold (0.0-1.0)
            output_path: Optional path to save combined mask

        Returns:
            Combined mask as nibabel image

        Examples:
            >>> files = get_subject_files()  # From FileFinder
            >>> mask = MaskProcessor.create_combined_mask(
            ...     files, 'MNI', 0.9, 'combined_mask.nii.gz'
            ... )
            >>> print(f"Created mask with shape: {mask.shape}")
        """
        mask_files = MaskProcessor.extract_mask_files(files, analysis_space)

        if not mask_files:
            raise ValueError(f"No {analysis_space} mask files found in the provided files")

        combined_mask = MaskProcessor.combine_masks_with_threshold(mask_files, threshold)

        if output_path:
            combined_mask.to_filename(output_path)

        return combined_mask

    @staticmethod
    def validate_masks(mask_files: list[Path]) -> bool:
        """Validate that mask files exist and can be loaded.

        Args:
            mask_files: List of mask file paths

        Returns:
            True if all masks are valid, False otherwise

        Examples:
            >>> mask_files = [Path('mask1.nii.gz'), Path('mask2.nii.gz')]
            >>> valid = MaskProcessor.validate_masks(mask_files)
            >>> print(f"Masks valid: {valid}")
        """
        if not mask_files:
            return False

        for mask_file in mask_files:
            if not mask_file.exists():
                logger.warning("Mask file not found: %s", mask_file)
                return False

            try:
                load_img(mask_file)
            except Exception as e:
                logger.warning("Cannot load mask file %s: %s", mask_file, e)
                return False

        return True

    @staticmethod
    def get_mask_info(files: dict, analysis_space: str = "MNI") -> dict:
        """Get information about available masks.

        Args:
            files: Files dictionary from FileFinder.get_files()
            analysis_space: Analysis space ('T1w' or 'MNI')

        Returns:
            Dictionary with mask information

        Examples:
            >>> files = get_subject_files()
            >>> info = MaskProcessor.get_mask_info(files, 'MNI')
            >>> print(f"Total masks: {info['total_masks']}")
        """
        mask_files = MaskProcessor.extract_mask_files(files, analysis_space)
        valid = MaskProcessor.validate_masks(mask_files)

        return {
            "total_masks": len(mask_files),
            "analysis_space": analysis_space,
            "mask_files": mask_files,
            "all_valid": valid,
            "mask_key": "t1w_brain_mask" if analysis_space == "T1w" else "mni_brain_mask",
        }
