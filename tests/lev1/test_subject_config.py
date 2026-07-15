from pathlib import Path

import pytest

from network_glm.lev1.subject_config import Config


class TestConfig:
    """Test class for Config dataclass."""

    def test_config_creation(self):
        """Test basic config creation with required parameters."""
        config = Config(bids_dir="/path/to/bids", fmriprep_dir="/path/to/fmriprep")

        assert config.bids_dir == Path("/path/to/bids")
        assert config.fmriprep_dir == Path("/path/to/fmriprep")
        assert config.output_dir == Path("./results/")
        assert config.subject_id is None
        assert config.task_name is None

    def test_config_with_custom_output_dir(self):
        """Test config creation with custom output directory."""
        config = Config(
            bids_dir="/path/to/bids",
            fmriprep_dir="/path/to/fmriprep",
            output_dir="/custom/output",
        )

        assert config.output_dir == Path("/custom/output")

    def test_config_string_to_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = Config(bids_dir="string/path/bids", fmriprep_dir="string/path/fmriprep")

        assert isinstance(config.bids_dir, Path)
        assert isinstance(config.fmriprep_dir, Path)
        assert isinstance(config.output_dir, Path)

    def test_config_with_subject_and_task(self):
        """Test config creation with subject and task parameters."""
        config = Config(
            bids_dir="/path/to/bids",
            fmriprep_dir="/path/to/fmriprep",
            subject_id="sub-s001",
            task_name="rest",
        )

        assert config.subject_id == "sub-s001"
        assert config.task_name == "rest"


class TestConfigDirectoryMethods:
    """Test Config directory creation methods."""

    def test_get_subject_dirs_without_subject_info(self):
        """Test that get_subject_dirs raises error without subject info."""
        config = Config(bids_dir="/path/to/bids", fmriprep_dir="/path/to/fmriprep")

        with pytest.raises(ValueError, match="subject_id and task_name must be set"):
            config.get_subject_dirs()

    def test_get_subject_dirs_with_subject_info(self):
        """Test get_subject_dirs returns correct paths."""
        config = Config(
            bids_dir="/path/to/bids",
            fmriprep_dir="/path/to/fmriprep",
            output_dir="/results",
            subject_id="sub-s001",
            task_name="rest",
        )

        dirs = config.get_subject_dirs()

        expected_base = Path("/results/sub-s001/task-rest")
        expected_dirs = {
            "quality_control": expected_base / "quality_control",
            "indiv_contrasts": expected_base / "indiv_contrasts",
            "fixed_effects": expected_base / "fixed_effects",
            "simplified_events": expected_base / "simplified_events",
            "task_residuals": expected_base / "task_residuals",
            "masks": expected_base / "masks",
            "base": expected_base,
        }

        assert dirs == expected_dirs

    def test_create_subject_dirs(self, temp_dir):
        """Test create_subject_dirs creates directories."""
        config = Config(
            bids_dir="/path/to/bids",
            fmriprep_dir="/path/to/fmriprep",
            output_dir=temp_dir,
            subject_id="sub-s001",
            task_name="rest",
        )

        dirs = config.create_subject_dirs(clean_existing=False)

        # Check all directories were created
        for dir_path in dirs.values():
            assert dir_path.exists()
            assert dir_path.is_dir()

        # Check directory structure
        expected_base = temp_dir / "sub-s001" / "task-rest"
        assert dirs["base"] == expected_base
        assert dirs["quality_control"] == expected_base / "quality_control"
        assert dirs["masks"] == expected_base / "masks"

    def test_create_subject_dirs_cleans_existing_files(self, temp_dir):
        """Test create_subject_dirs cleans existing files when requested."""
        config = Config(
            bids_dir="/path/to/bids",
            fmriprep_dir="/path/to/fmriprep",
            output_dir=temp_dir,
            subject_id="sub-s001",
            task_name="rest",
        )

        # Create directory structure first
        dirs = config.create_subject_dirs(clean_existing=False)

        # Add some test files
        test_file = dirs["quality_control"] / "test_file.txt"
        test_file.write_text("test content")
        assert test_file.exists()

        # Create dirs again with cleaning
        config.create_subject_dirs(clean_existing=True)

        # File should be removed
        assert not test_file.exists()
        # But directory should still exist
        assert dirs["quality_control"].exists()
