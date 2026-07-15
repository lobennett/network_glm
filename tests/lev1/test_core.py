"""Test core utility functions."""

import json

import pytest

from network_glm.exclusions import (
    create_exclusion_key,
    load_exclusions,
    load_exclusions_by_type,
    normalize_subject_id,
)


class TestCreateExclusionKey:
    """Test exclusion key creation functionality."""

    def test_create_valid_key(self):
        """Test creating key with all required fields."""
        exclusion = {
            "subject": "sub-s001",
            "session": "ses-01",
            "task": "rest",
            "run": "run-1",
        }
        expected = "sub-s001_ses-01_rest_run-1"
        assert create_exclusion_key(exclusion) == expected

    def test_create_key_missing_required_field(self):
        """Test that missing required field raises KeyError."""
        exclusion = {
            "subject": "sub-s001",
            "session": "ses-01",
            "task": "rest",
            # Missing 'run'
        }
        with pytest.raises(KeyError, match="Missing required keys in exclusion"):
            create_exclusion_key(exclusion)

    def test_create_key_multiple_missing_fields(self):
        """Test handling multiple missing fields."""
        exclusion = {
            "subject": "sub-s001"
            # Missing 'session', 'task', 'run'
        }
        with pytest.raises(KeyError, match="Missing required keys in exclusion"):
            create_exclusion_key(exclusion)

    def test_create_key_empty_values(self):
        """Test creating key with empty string values."""
        exclusion = {
            "subject": "sub-s001",
            "session": "",
            "task": "rest",
            "run": "run-1",
        }
        expected = "sub-s001__rest_run-1"
        assert create_exclusion_key(exclusion) == expected


class TestLoadExclusionsByType:
    """Test loading exclusions by specific type."""

    def test_load_all_types(self, temp_dir):
        """Test loading all exclusion types from file."""
        # Create test exclusions data
        exclusions_data = {
            "fmriprep_exclusions": [
                {
                    "subject": "sub-s001",
                    "session": "ses-01",
                    "task": "rest",
                    "run": "run-1",
                },
                {
                    "subject": "sub-002",
                    "session": "ses-01",
                    "task": "rest",
                    "run": "run-1",
                },
            ],
            "behavioral_exclusions": [
                {
                    "subject": "sub-003",
                    "session": "ses-01",
                    "task": "task",
                    "run": "run-1",
                }
            ],
            "quality_exclusions": [
                {
                    "subject": "sub-004",
                    "session": "ses-01",
                    "task": "rest",
                    "run": "run-2",
                }
            ],
        }

        # Write to temporary file
        exclusions_file = temp_dir / "test_exclusions.json"
        with open(exclusions_file, "w") as f:
            json.dump(exclusions_data, f)

        # Load all types
        result = load_exclusions_by_type(exclusions_file)

        assert len(result) == 3
        assert "fmriprep_exclusions" in result
        assert "behavioral_exclusions" in result
        assert "quality_exclusions" in result

        # Check specific exclusions
        assert len(result["fmriprep_exclusions"]) == 2
        assert len(result["behavioral_exclusions"]) == 1
        assert len(result["quality_exclusions"]) == 1

        assert "sub-s001_ses-01_rest_run-1" in result["fmriprep_exclusions"]
        assert "sub-003_ses-01_task_run-1" in result["behavioral_exclusions"]

    def test_load_specific_types(self, temp_dir):
        """Test loading only specified exclusion types."""
        exclusions_data = {
            "fmriprep_exclusions": [
                {
                    "subject": "sub-s001",
                    "session": "ses-01",
                    "task": "rest",
                    "run": "run-1",
                }
            ],
            "behavioral_exclusions": [
                {
                    "subject": "sub-002",
                    "session": "ses-01",
                    "task": "task",
                    "run": "run-1",
                }
            ],
        }

        exclusions_file = temp_dir / "test_exclusions.json"
        with open(exclusions_file, "w") as f:
            json.dump(exclusions_data, f)

        # Load only fmriprep exclusions
        result = load_exclusions_by_type(exclusions_file, ["fmriprep_exclusions"])

        assert len(result) == 1
        assert "fmriprep_exclusions" in result
        assert "behavioral_exclusions" not in result
        assert len(result["fmriprep_exclusions"]) == 1

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading from nonexistent file returns empty dict."""
        nonexistent_file = temp_dir / "nonexistent.json"
        result = load_exclusions_by_type(nonexistent_file)
        assert result == {}

    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON file returns empty dict."""
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("invalid json content")

        result = load_exclusions_by_type(invalid_file)
        assert result == {}

    def test_load_missing_exclusion_type(self, temp_dir):
        """Test requesting nonexistent exclusion type."""
        exclusions_data = {
            "fmriprep_exclusions": [
                {
                    "subject": "sub-s001",
                    "session": "ses-01",
                    "task": "rest",
                    "run": "run-1",
                }
            ]
        }

        exclusions_file = temp_dir / "test_exclusions.json"
        with open(exclusions_file, "w") as f:
            json.dump(exclusions_data, f)

        # Request nonexistent type
        result = load_exclusions_by_type(exclusions_file, ["missing_type"])
        assert result == {}

    def test_load_invalid_exclusion_structure(self, temp_dir):
        """Test handling exclusions with invalid structure."""
        exclusions_data = {
            "fmriprep_exclusions": [
                {
                    "subject": "sub-s001",
                    "session": "ses-01",
                    "task": "rest",
                    "run": "run-1",
                },
                {"subject": "sub-002"},  # Missing required fields
                {
                    "subject": "sub-003",
                    "session": "ses-01",
                    "task": "rest",
                    "run": "run-1",
                },
            ]
        }

        exclusions_file = temp_dir / "test_exclusions.json"
        with open(exclusions_file, "w") as f:
            json.dump(exclusions_data, f)

        result = load_exclusions_by_type(exclusions_file, ["fmriprep_exclusions"])

        # Should skip invalid exclusion but process valid ones
        assert len(result["fmriprep_exclusions"]) == 2
        assert "sub-s001_ses-01_rest_run-1" in result["fmriprep_exclusions"]
        assert "sub-003_ses-01_rest_run-1" in result["fmriprep_exclusions"]

    def test_load_non_list_exclusion_type(self, temp_dir):
        """Test handling exclusion type that is not a list."""
        exclusions_data = {
            "fmriprep_exclusions": "not a list",
            "behavioral_exclusions": [
                {
                    "subject": "sub-s001",
                    "session": "ses-01",
                    "task": "task",
                    "run": "run-1",
                }
            ],
        }

        exclusions_file = temp_dir / "test_exclusions.json"
        with open(exclusions_file, "w") as f:
            json.dump(exclusions_data, f)

        result = load_exclusions_by_type(exclusions_file)

        # Should skip non-list type but process valid ones
        assert "fmriprep_exclusions" not in result
        assert "behavioral_exclusions" in result
        assert len(result["behavioral_exclusions"]) == 1


class TestLoadExclusions:
    """Test combined exclusions loading functionality."""

    def test_load_combined_exclusions(self, temp_dir):
        """Test loading and combining all exclusions."""
        exclusions_data = {
            "fmriprep_exclusions": [
                {
                    "subject": "sub-s001",
                    "session": "ses-01",
                    "task": "rest",
                    "run": "run-1",
                },
                {
                    "subject": "sub-002",
                    "session": "ses-01",
                    "task": "rest",
                    "run": "run-1",
                },
            ],
            "behavioral_exclusions": [
                {
                    "subject": "sub-003",
                    "session": "ses-01",
                    "task": "task",
                    "run": "run-1",
                },
                {
                    "subject": "sub-s001",
                    "session": "ses-01",
                    "task": "rest",
                    "run": "run-1",
                },  # Duplicate
            ],
        }

        exclusions_file = temp_dir / "test_exclusions.json"
        with open(exclusions_file, "w") as f:
            json.dump(exclusions_data, f)

        result = load_exclusions(exclusions_file)

        # Should combine and deduplicate
        assert len(result) == 3  # Unique exclusions
        assert "sub-s001_ses-01_rest_run-1" in result
        assert "sub-002_ses-01_rest_run-1" in result
        assert "sub-003_ses-01_task_run-1" in result

    def test_load_specific_exclusion_types_only(self, temp_dir):
        """Test loading only specific exclusion types."""
        exclusions_data = {
            "fmriprep_exclusions": [
                {
                    "subject": "sub-s001",
                    "session": "ses-01",
                    "task": "rest",
                    "run": "run-1",
                }
            ],
            "behavioral_exclusions": [
                {
                    "subject": "sub-002",
                    "session": "ses-01",
                    "task": "task",
                    "run": "run-1",
                }
            ],
        }

        exclusions_file = temp_dir / "test_exclusions.json"
        with open(exclusions_file, "w") as f:
            json.dump(exclusions_data, f)

        # Load only behavioral exclusions
        result = load_exclusions(exclusions_file, ["behavioral_exclusions"])

        assert len(result) == 1
        assert "sub-002_ses-01_task_run-1" in result
        assert "sub-s001_ses-01_rest_run-1" not in result

    def test_load_empty_exclusions(self, temp_dir):
        """Test loading file with no exclusions."""
        exclusions_data = {"fmriprep_exclusions": [], "behavioral_exclusions": []}

        exclusions_file = temp_dir / "test_exclusions.json"
        with open(exclusions_file, "w") as f:
            json.dump(exclusions_data, f)

        result = load_exclusions(exclusions_file)
        assert len(result) == 0


class TestNormalizeSubjectId:
    """Test subject ID normalization functionality."""

    def test_adds_prefix_to_plain_id(self):
        """Test adding sub- prefix to plain subject ID."""
        result = normalize_subject_id("001")
        assert result == "sub-001"

    def test_adds_prefix_to_numeric_string(self):
        """Test adding sub- prefix to numeric string."""
        result = normalize_subject_id("123")
        assert result == "sub-123"

    def test_preserves_existing_prefix(self):
        """Test that existing sub- prefix is preserved."""
        result = normalize_subject_id("sub-001")
        assert result == "sub-001"

    def test_preserves_existing_prefix_with_alphanumeric(self):
        """Test that existing sub- prefix is preserved with alphanumeric ID."""
        result = normalize_subject_id("sub-s001")
        assert result == "sub-s001"

    def test_handles_empty_string(self):
        """Test handling empty string input."""
        result = normalize_subject_id("")
        assert result == "sub-"

    def test_handles_only_sub_prefix(self):
        """Test handling input that is only 'sub-'."""
        result = normalize_subject_id("sub-")
        assert result == "sub-"


class TestExclusionKeyMatchesRunnerForAllGenerators:
    """Regression: every generator's compiled entries must produce a key that
    matches the key the lev1 runner constructs for that scan.

    The lev1 runner builds its per-run key via ``_run_base_filename`` as
    ``f'{subj_id}_{session}_task-{task_name}_{run}'`` (task is BARE in the runner;
    the literal ``task-`` prefix is in the format string). For a compiled exclusion
    entry to actually skip that run, ``create_exclusion_key(entry)`` must equal
    that string. This pins the schema invariant that ALL generators store the
    ``task`` field PREFIXED (``task-goNogo``) and ``run`` as ``run-N`` so behavioral
    and motion exclusions are honored identically by lev1.

    NOTE (network_glm lift-and-shift): the monolith also had
    ``test_behavioral_generator_emits_prefixed_task_and_run`` here, which
    exercised the monolith's events/behavioral-QC ``run_qc`` entry point.
    That module is out of scope for network_glm (engine-only extraction), so
    that test was dropped; the schema invariant it protected is still
    covered by the sibling test below using hand-built compiled entries.
    """

    def test_motion_and_behavioral_entries_both_match_runner_key(self, temp_dir):
        from network_glm.lev1.runner import _run_base_filename

        # The scan the runner is iterating over.
        subj_id, session, task_name, run = "sub-s10", "ses-01", "goNogo", "run-1"
        runner_key = _run_base_filename(subj_id, session, task_name, run)

        # Motion-style compiled entry (prefixed task, as motion.py emits).
        motion_entry = {
            "subject": "sub-s10",
            "session": "ses-01",
            "task": "task-goNogo",
            "run": "run-1",
            "source": "motion",
            "action": "exclude",
            "reason": "fd",
        }
        # Behavioral-style compiled entry — must ALSO be prefixed to match.
        behavioral_entry = {
            "subject": "sub-s10",
            "session": "ses-01",
            "task": "task-goNogo",
            "run": "run-1",
            "source": "behavioral-qc",
            "action": "exclude",
            "reason": "omission",
        }

        compiled = temp_dir / "compiled_exclusions.json"
        with open(compiled, "w") as f:
            json.dump([motion_entry, behavioral_entry], f)

        # This is exactly the code path lev1 uses (prepare.py -> load_exclusions).
        exclusions = load_exclusions(compiled)

        # Both generators' entries must land on the runner's key.
        assert create_exclusion_key(motion_entry) == runner_key
        assert create_exclusion_key(behavioral_entry) == runner_key
        assert (
            runner_key in exclusions
        ), f"{runner_key!r} not in {sorted(exclusions)!r}: lev1 would NOT skip this scan"

