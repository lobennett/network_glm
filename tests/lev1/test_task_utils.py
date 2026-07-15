"""Test task utility functions."""

from pathlib import Path

import pytest

from network_glm.task_utils import (
    detect_sample_type,
    get_expected_sessions,
)


class TestDetectSampleType:
    """Test sample type detection functionality."""

    def test_detect_discovery_sample(self):
        """Test detection of discovery sample."""
        bids_path = Path("/data/discovery/bids")
        result = detect_sample_type(bids_path)
        assert result == "discovery"

    def test_detect_validation_sample(self):
        """Test detection of validation sample."""
        bids_path = Path("/data/validation/bids")
        result = detect_sample_type(bids_path)
        assert result == "validation"

    def test_detect_validation_sample_no_explicit_type(self):
        """Test that paths without 'discovery' default to validation."""
        bids_path = Path("/data/study/bids")
        result = detect_sample_type(bids_path)
        assert result == "validation"

    def test_detect_discovery_in_nested_path(self):
        """Test detection with 'discovery' in nested path."""
        bids_path = Path("/projects/study/discovery/subset1/bids")
        result = detect_sample_type(bids_path)
        assert result == "discovery"


class TestGetExpectedSessions:
    """Test expected sessions from YAML config."""

    def test_base_task_sessions(self):
        """Test that base tasks return 5 sessions."""
        assert get_expected_sessions("flanker") == 5
        assert get_expected_sessions("stopSignal") == 5
        assert get_expected_sessions("nBack") == 5

    def test_dual_task_sessions(self):
        """Dual tasks have no fixed expected-session count (expected_sessions: null)."""
        assert get_expected_sessions("stopSignalWDirectedForgetting") is None

    def test_unknown_task_raises(self):
        """Test that unknown task names raise an error."""
        with pytest.raises(FileNotFoundError):
            get_expected_sessions("nonexistentTask")
