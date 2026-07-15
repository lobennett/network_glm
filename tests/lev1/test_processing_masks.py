"""Tests for the masks processing module."""

from pathlib import Path

import numpy as np
import pytest
from nilearn.image import load_img

from network_glm.lev1.processing.masks import MaskProcessor


class TestMaskProcessor:
    """Tests for MaskProcessor class methods."""

    def test_combine_masks_with_threshold_single_mask(self, multiple_brain_masks):
        """Test mask combination with single mask."""
        single_mask = [multiple_brain_masks[0]]

        combined = MaskProcessor.combine_masks_with_threshold(single_mask, 1.0)
        original = load_img(multiple_brain_masks[0])

        # Should return the original mask unchanged
        np.testing.assert_array_equal(combined.get_fdata(), original.get_fdata())

    def test_combine_masks_with_threshold_multiple_masks(self, multiple_brain_masks):
        """Test mask combination with multiple masks."""
        combined = MaskProcessor.combine_masks_with_threshold(multiple_brain_masks, threshold=1.0)

        # Load individual masks to verify intersection
        mask1_data = load_img(multiple_brain_masks[0]).get_fdata()
        mask2_data = load_img(multiple_brain_masks[1]).get_fdata()
        mask3_data = load_img(multiple_brain_masks[2]).get_fdata()

        combined_data = combined.get_fdata()

        # With threshold=1.0, only voxels present in ALL masks should be included
        expected_intersection = (mask1_data > 0) & (mask2_data > 0) & (mask3_data > 0)

        # Combined mask should have intersection pattern
        assert np.sum(combined_data > 0) <= np.sum(expected_intersection)
        assert combined_data.shape == mask1_data.shape

    def test_combine_masks_with_threshold_partial_overlap(self, multiple_brain_masks):
        """Test mask combination with partial threshold."""
        combined = MaskProcessor.combine_masks_with_threshold(multiple_brain_masks, threshold=0.5)

        combined_data = combined.get_fdata()

        # With threshold=0.5, more voxels should be included than with 1.0
        combined_strict = MaskProcessor.combine_masks_with_threshold(
            multiple_brain_masks, threshold=1.0
        )
        combined_strict_data = combined_strict.get_fdata()

        assert np.sum(combined_data > 0) >= np.sum(combined_strict_data > 0)

    def test_extract_mask_files_mni_space(self, files_dict_with_masks):
        """Test extracting MNI mask files from files dictionary."""
        mask_files = MaskProcessor.extract_mask_files(files_dict_with_masks, "MNI")

        # Should have 3 mask files (2 from ses-01, 1 from ses-02)
        assert len(mask_files) == 3

        # All should be Path objects
        assert all(isinstance(f, Path) for f in mask_files)

        # All should exist (created by fixtures)
        assert all(f.exists() for f in mask_files)

    def test_extract_mask_files_t1w_space(self, files_dict_with_masks):
        """Test extracting T1w mask files from files dictionary."""
        mask_files = MaskProcessor.extract_mask_files(files_dict_with_masks, "T1w")

        # Should have 3 mask files
        assert len(mask_files) == 3

        # All should exist
        assert all(f.exists() for f in mask_files)

    def test_extract_mask_files_no_masks(self):
        """Test extracting mask files when none exist."""
        files_dict = {"ses-01": {"run-01": {"other_file": Path("some_file.txt")}}}

        mask_files = MaskProcessor.extract_mask_files(files_dict, "MNI")
        assert len(mask_files) == 0

    def test_extract_mask_files_empty_dict(self):
        """Test extracting mask files from empty dictionary."""
        mask_files = MaskProcessor.extract_mask_files({}, "MNI")
        assert len(mask_files) == 0

    def test_create_combined_mask_success(self, files_dict_with_masks, temp_dir):
        """Test successful combined mask creation."""
        output_path = temp_dir / "combined_test_mask.nii.gz"

        combined_mask = MaskProcessor.create_combined_mask(
            files_dict_with_masks,
            analysis_space="MNI",
            threshold=0.8,
            output_path=output_path,
        )

        # Should return a nibabel image
        assert hasattr(combined_mask, "get_fdata")
        assert hasattr(combined_mask, "shape")

        # Output file should be created
        assert output_path.exists()

        # Should be able to load saved file
        loaded_mask = load_img(output_path)
        np.testing.assert_array_equal(combined_mask.get_fdata(), loaded_mask.get_fdata())

    def test_create_combined_mask_no_output_path(self, files_dict_with_masks):
        """Test combined mask creation without saving."""
        combined_mask = MaskProcessor.create_combined_mask(
            files_dict_with_masks, analysis_space="MNI", threshold=1.0
        )

        # Should still return a valid mask
        assert hasattr(combined_mask, "get_fdata")
        assert combined_mask.get_fdata().ndim == 3

    def test_create_combined_mask_no_mask_files(self):
        """Test combined mask creation with no mask files."""
        files_dict = {"ses-01": {"run-01": {}}}

        with pytest.raises(ValueError, match="No MNI mask files found"):
            MaskProcessor.create_combined_mask(files_dict, "MNI")

    def test_validate_masks_all_valid(self, multiple_brain_masks):
        """Test mask validation with all valid masks."""
        is_valid = MaskProcessor.validate_masks(multiple_brain_masks)
        assert is_valid is True

    def test_validate_masks_nonexistent_file(self, multiple_brain_masks, temp_dir):
        """Test mask validation with non-existent file."""
        invalid_masks = multiple_brain_masks + [temp_dir / "nonexistent.nii.gz"]

        is_valid = MaskProcessor.validate_masks(invalid_masks)
        assert is_valid is False

    def test_validate_masks_empty_list(self):
        """Test mask validation with empty list."""
        is_valid = MaskProcessor.validate_masks([])
        assert is_valid is False

    def test_validate_masks_invalid_nifti(self, multiple_brain_masks, temp_dir):
        """Test mask validation with invalid NIfTI file."""
        # Create invalid file
        invalid_file = temp_dir / "invalid.nii.gz"
        invalid_file.write_text("Not a valid NIfTI file")

        invalid_masks = multiple_brain_masks + [invalid_file]

        is_valid = MaskProcessor.validate_masks(invalid_masks)
        assert is_valid is False

    def test_get_mask_info_basic(self, files_dict_with_masks):
        """Test getting mask information."""
        info = MaskProcessor.get_mask_info(files_dict_with_masks, "MNI")

        expected_keys = {
            "total_masks",
            "analysis_space",
            "mask_files",
            "all_valid",
            "mask_key",
        }
        assert set(info.keys()) == expected_keys

        assert info["total_masks"] == 3
        assert info["analysis_space"] == "MNI"
        assert info["mask_key"] == "mni_brain_mask"
        assert info["all_valid"] is True
        assert len(info["mask_files"]) == 3

    def test_get_mask_info_t1w(self, files_dict_with_masks):
        """Test getting mask information for T1w space."""
        info = MaskProcessor.get_mask_info(files_dict_with_masks, "T1w")

        assert info["analysis_space"] == "T1w"
        assert info["mask_key"] == "t1w_brain_mask"
        assert info["total_masks"] == 3

    def test_get_mask_info_no_masks(self):
        """Test getting mask information when no masks exist."""
        files_dict = {"ses-01": {"run-01": {}}}

        info = MaskProcessor.get_mask_info(files_dict, "MNI")

        assert info["total_masks"] == 0
        assert info["all_valid"] is False
        assert len(info["mask_files"]) == 0

    def test_get_mask_info_invalid_masks(self, temp_dir):
        """Test getting mask information with invalid masks."""
        # Create invalid mask file
        invalid_mask = temp_dir / "invalid.nii.gz"
        invalid_mask.write_text("Invalid")

        files_dict = {"ses-01": {"run-01": {"mni_brain_mask": invalid_mask}}}

        info = MaskProcessor.get_mask_info(files_dict, "MNI")

        assert info["total_masks"] == 1
        assert info["all_valid"] is False


class TestMaskProcessorEdgeCases:
    """Test edge cases and error conditions."""

    def test_combine_masks_empty_list(self):
        """Test combining empty list of masks."""
        with pytest.raises(ValueError, match="No mask provided for intersection"):
            MaskProcessor.combine_masks_with_threshold([])

    def test_combine_masks_invalid_threshold(self, multiple_brain_masks):
        """Test combining masks with invalid threshold values."""
        # Test threshold > 1.0 - should raise ValueError
        with pytest.raises(ValueError, match="The threshold should be smaller than 1"):
            MaskProcessor.combine_masks_with_threshold(multiple_brain_masks, threshold=1.5)

        # Test threshold = 0.0 (should work)
        combined = MaskProcessor.combine_masks_with_threshold(multiple_brain_masks, threshold=0.0)
        assert hasattr(combined, "get_fdata")

    def test_create_combined_mask_different_spaces(self, files_dict_with_masks):
        """Test creating combined masks for different analysis spaces."""
        # Test both MNI and T1w
        mni_mask = MaskProcessor.create_combined_mask(files_dict_with_masks, "MNI", 1.0)
        t1w_mask = MaskProcessor.create_combined_mask(files_dict_with_masks, "T1w", 1.0)

        # Both should be valid masks
        assert hasattr(mni_mask, "get_fdata")
        assert hasattr(t1w_mask, "get_fdata")

        # Should have same shape (same underlying mask data in fixtures)
        assert mni_mask.shape == t1w_mask.shape
