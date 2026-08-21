"""Test file discovery functionality."""

from pathlib import Path

from network_glm.io.file_discovery import FileFinder


class TestFileFinder:
    """Test FileFinder class functionality."""

    def test_file_finder_initialization(self, temp_dir):
        """Test FileFinder initialization."""
        bids_dir = temp_dir / "bids"
        fmriprep_dir = temp_dir / "fmriprep"

        finder = FileFinder(bids_dir, fmriprep_dir)

        assert finder.bids_dir == bids_dir
        assert finder.fmriprep_dir == fmriprep_dir
        assert finder.run_pattern.pattern == r"run-\d+"

    def test_file_finder_with_string_paths(self, temp_dir):
        """Test initialization with string paths."""
        bids_str = str(temp_dir / "bids")
        fmriprep_str = str(temp_dir / "fmriprep")

        finder = FileFinder(bids_str, fmriprep_str)

        assert isinstance(finder.bids_dir, Path)
        assert isinstance(finder.fmriprep_dir, Path)


class TestFileFinderIntegration:
    """Test FileFinder with mock file structures."""

    def create_mock_bids_structure(self, temp_dir):
        """Create mock BIDS directory structure."""
        bids_dir = temp_dir / "bids"
        subj_dir = bids_dir / "sub-s001" / "ses-01" / "func"
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Create events files
        events_files = [
            "sub-s001_ses-01_task-rest_run-01_events.tsv",
            "sub-s001_ses-01_task-rest_run-02_events.tsv",
        ]

        for filename in events_files:
            (subj_dir / filename).write_text("onset\tduration\ttrial_type\n0\t1\trest\n")

        return bids_dir

    def create_mock_fmriprep_structure(self, temp_dir):
        """Create mock fMRIPrep directory structure."""
        fmriprep_dir = temp_dir / "fmriprep"
        subj_dir = fmriprep_dir / "sub-s001" / "ses-01" / "func"
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Create fMRIPrep files for each run
        file_patterns = [
            "sub-s001_ses-01_task-rest_run-01_desc-confounds_timeseries.tsv",
            "sub-s001_ses-01_task-rest_run-01_space-T1w_desc-preproc_bold.nii.gz",
            "sub-s001_ses-01_task-rest_run-01_space-T1w_desc-brain_mask.nii.gz",
            "sub-s001_ses-01_task-rest_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
            "sub-s001_ses-01_task-rest_run-01_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
            "sub-s001_ses-01_task-rest_run-02_desc-confounds_timeseries.tsv",
            "sub-s001_ses-01_task-rest_run-02_space-T1w_desc-preproc_bold.nii.gz",
            "sub-s001_ses-01_task-rest_run-02_space-T1w_desc-brain_mask.nii.gz",
            "sub-s001_ses-01_task-rest_run-02_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
            "sub-s001_ses-01_task-rest_run-02_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
        ]

        for filename in file_patterns:
            (subj_dir / filename).write_text("mock data")

        return fmriprep_dir

    def test_get_files_complete_structure(self, temp_dir):
        """Test get_files with complete file structure."""
        bids_dir = self.create_mock_bids_structure(temp_dir)
        fmriprep_dir = self.create_mock_fmriprep_structure(temp_dir)

        finder = FileFinder(bids_dir, fmriprep_dir)
        files = finder.get_files("s001", "rest")  # Test subject ID normalization

        # Should find complete runs
        assert "ses-01" in files
        assert "run-01" in files["ses-01"]
        assert "run-02" in files["ses-01"]

        # Check all required files are present for run-01
        run_01_files = files["ses-01"]["run-01"]
        expected_files = [
            "events",
            "confounds",
            "t1w_data",
            "t1w_brain_mask",
            "mni_data",
            "mni_brain_mask",
        ]

        for file_type in expected_files:
            assert file_type in run_01_files
            assert isinstance(run_01_files[file_type], Path)

    def test_get_files_incomplete_structure(self, temp_dir):
        """Test get_files with incomplete file structure."""
        bids_dir = self.create_mock_bids_structure(temp_dir)
        fmriprep_dir = temp_dir / "fmriprep"
        fmriprep_dir.mkdir(parents=True, exist_ok=True)

        # Only create partial fMRIPrep structure
        subj_dir = fmriprep_dir / "sub-s001" / "ses-01" / "func"
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Only create confounds file (missing other required files)
        (subj_dir / "sub-s001_ses-01_task-rest_run-01_desc-confounds_timeseries.tsv").write_text(
            "confounds"
        )

        finder = FileFinder(bids_dir, fmriprep_dir)
        files = finder.get_files("sub-s001", "rest")

        # Should return empty dict since runs are incomplete
        assert files == {}

    def test_get_files_with_custom_required_files(self, temp_dir):
        """Test get_files with custom required files list."""
        bids_dir = self.create_mock_bids_structure(temp_dir)
        fmriprep_dir = temp_dir / "fmriprep"
        fmriprep_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal structure with only events and confounds
        subj_dir = fmriprep_dir / "sub-s001" / "ses-01" / "func"
        subj_dir.mkdir(parents=True, exist_ok=True)
        (subj_dir / "sub-s001_ses-01_task-rest_run-01_desc-confounds_timeseries.tsv").write_text(
            "confounds"
        )

        finder = FileFinder(bids_dir, fmriprep_dir)
        files = finder.get_files("sub-s001", "rest", required_files=["events", "confounds"])

        # Should find the run since only events and confounds are required
        assert "ses-01" in files
        assert "run-01" in files["ses-01"]
        assert "events" in files["ses-01"]["run-01"]
        assert "confounds" in files["ses-01"]["run-01"]

    def test_validate_file_completeness(self, temp_dir):
        """Test file completeness validation."""
        bids_dir = self.create_mock_bids_structure(temp_dir)
        fmriprep_dir = self.create_mock_fmriprep_structure(temp_dir)

        finder = FileFinder(bids_dir, fmriprep_dir)
        files = finder.get_files("sub-s001", "rest")

        validation = finder.validate_file_completeness(files, "rest")

        assert validation["total_runs"] == 2
        assert validation["complete_runs"] == 2
        assert validation["total_sessions"] == 1
        assert validation["sessions_with_data"] == 1

    def test_validate_file_completeness_empty(self, temp_dir):
        """Test file completeness validation with no files."""
        bids_dir = temp_dir / "bids"
        fmriprep_dir = temp_dir / "fmriprep"
        bids_dir.mkdir()
        fmriprep_dir.mkdir()

        finder = FileFinder(bids_dir, fmriprep_dir)
        files = finder.get_files("sub-s001", "rest")

        validation = finder.validate_file_completeness(files, "rest")

        assert validation["total_runs"] == 0
        assert validation["complete_runs"] == 0
        assert validation["total_sessions"] == 0
        assert validation["sessions_with_data"] == 0


class TestGetRequiredFilesForSpace:
    """Test get_required_files_for_space with all surface spaces."""

    def test_surface_returns_gifti_keys(self):
        required = FileFinder.get_required_files_for_space("surface")
        assert required == ["events", "confounds", "left_surface", "right_surface"]

    def test_fsnative_returns_gifti_keys(self):
        required = FileFinder.get_required_files_for_space("fsnative")
        assert required == ["events", "confounds", "left_surface", "right_surface"]

    def test_fsaverage6_returns_gifti_keys(self):
        required = FileFinder.get_required_files_for_space("fsaverage6")
        assert required == ["events", "confounds", "left_surface", "right_surface"]

    def test_fslr_returns_cifti_keys(self):
        required = FileFinder.get_required_files_for_space("fsLR")
        assert required == ["events", "confounds", "cifti_bold"]


class TestSurfacePatterns:
    """Test that surface_space parameter selects correct file patterns."""

    def test_get_files_fsaverage6_patterns(self, temp_dir):
        """Files discovered with surface_space='fsaverage6' match fsaverage6 patterns."""
        bids_dir = temp_dir / "bids"
        fmriprep_dir = temp_dir / "fmriprep"
        subj_dir_bids = bids_dir / "sub-s001" / "ses-01" / "func"
        subj_dir_bids.mkdir(parents=True)
        subj_dir_fmriprep = fmriprep_dir / "sub-s001" / "ses-01" / "func"
        subj_dir_fmriprep.mkdir(parents=True)

        # Events
        (subj_dir_bids / "sub-s001_ses-01_task-rest_run-01_events.tsv").write_text(
            "onset\tduration\ttrial_type\n0\t1\trest\n"
        )
        # Confounds
        (
            subj_dir_fmriprep / "sub-s001_ses-01_task-rest_run-01_desc-confounds_timeseries.tsv"
        ).write_text("mock")
        # fsaverage6 surface files
        (
            subj_dir_fmriprep
            / "sub-s001_ses-01_task-rest_run-01_hemi-L_space-fsaverage6_bold.func.gii"
        ).write_text("mock")
        (
            subj_dir_fmriprep
            / "sub-s001_ses-01_task-rest_run-01_hemi-R_space-fsaverage6_bold.func.gii"
        ).write_text("mock")

        finder = FileFinder(bids_dir, fmriprep_dir)
        required = FileFinder.get_required_files_for_space("fsaverage6")
        files = finder.get_files(
            "s001", "rest", required_files=required, surface_space="fsaverage6"
        )

        assert "ses-01" in files
        assert "run-01" in files["ses-01"]
        assert "left_surface" in files["ses-01"]["run-01"]
        assert "fsaverage6" in files["ses-01"]["run-01"]["left_surface"].name

    def test_get_files_fsnative_is_default(self, temp_dir):
        """Without surface_space param, fsnative patterns are used."""
        bids_dir = temp_dir / "bids"
        fmriprep_dir = temp_dir / "fmriprep"
        subj_dir_bids = bids_dir / "sub-s001" / "ses-01" / "func"
        subj_dir_bids.mkdir(parents=True)
        subj_dir_fmriprep = fmriprep_dir / "sub-s001" / "ses-01" / "func"
        subj_dir_fmriprep.mkdir(parents=True)

        (subj_dir_bids / "sub-s001_ses-01_task-rest_run-01_events.tsv").write_text(
            "onset\tduration\ttrial_type\n0\t1\trest\n"
        )
        (
            subj_dir_fmriprep / "sub-s001_ses-01_task-rest_run-01_desc-confounds_timeseries.tsv"
        ).write_text("mock")
        (
            subj_dir_fmriprep
            / "sub-s001_ses-01_task-rest_run-01_hemi-L_space-fsnative_bold.func.gii"
        ).write_text("mock")
        (
            subj_dir_fmriprep
            / "sub-s001_ses-01_task-rest_run-01_hemi-R_space-fsnative_bold.func.gii"
        ).write_text("mock")

        finder = FileFinder(bids_dir, fmriprep_dir)
        required = FileFinder.get_required_files_for_space("surface")
        files = finder.get_files("s001", "rest", required_files=required)

        assert "ses-01" in files
        assert "fsnative" in files["ses-01"]["run-01"]["left_surface"].name


class TestFileFinderEdgeCases:
    """Test edge cases for FileFinder."""

    def test_run_pattern_matching(self, temp_dir):
        """Test that run pattern correctly matches various formats."""
        finder = FileFinder(temp_dir, temp_dir)

        # Test various run patterns
        test_cases = [
            ("run-1", True),
            ("run-01", True),
            ("run-123", True),
            ("run-", False),
            ("run1", False),
            ("task-run-1", True),  # Should match the run-1 part
        ]

        for test_string, should_match in test_cases:
            match = finder.run_pattern.search(test_string)
            if should_match:
                assert match is not None, f"Should match: {test_string}"
            else:
                assert match is None, f"Should not match: {test_string}"

    def test_nonexistent_directories(self, temp_dir):
        """Test behavior with nonexistent directories."""
        nonexistent_bids = temp_dir / "nonexistent_bids"
        nonexistent_fmriprep = temp_dir / "nonexistent_fmriprep"

        finder = FileFinder(nonexistent_bids, nonexistent_fmriprep)
        files = finder.get_files("sub-s001", "rest")

        # Should return empty dict without crashing
        assert files == {}


def test_filefinder_respects_mni_template_and_res(tmp_path):
    """FileFinder builds mni_data/mni_brain_mask patterns from mni_template + mni_res kwargs."""
    bids = tmp_path / "bids"
    fmriprep = tmp_path / "fmriprep"
    func = fmriprep / "sub-s01" / "ses-01" / "func"
    func.mkdir(parents=True)
    base = "sub-s01_ses-01_task-rest_run-1"
    # 2009cAsym res-2 is what the study produces, and what the default looks for.
    (func / f"{base}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz").touch()
    (func / f"{base}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz").touch()
    (func / f"{base}_desc-confounds_timeseries.tsv").touch()
    # events.tsv in BIDS
    bids_func = bids / "sub-s01" / "ses-01" / "func"
    bids_func.mkdir(parents=True)
    (bids_func / f"{base}_events.tsv").touch()

    from network_glm.io.file_discovery import FileFinder

    # Default (2009cAsym res-2) should find the file
    finder_default = FileFinder(bids_dir=bids, fmriprep_dir=fmriprep)
    files = finder_default.get_files(
        subject_id="sub-s01",
        task_name="rest",
        required_files=["mni_data", "mni_brain_mask"],
    )
    flat = str(files)
    assert "MNI152NLin2009cAsym_res-2" in flat

    # Override to a space/res that is not present should find nothing
    finder_override = FileFinder(
        bids_dir=bids,
        fmriprep_dir=fmriprep,
        mni_template="MNI152NLin6Asym",
        mni_res="1",
    )
    files2 = finder_override.get_files(
        subject_id="sub-s01",
        task_name="rest",
        required_files=["mni_data", "mni_brain_mask"],
    )
    # No matching files, so the filter drops everything
    assert files2 == {}
