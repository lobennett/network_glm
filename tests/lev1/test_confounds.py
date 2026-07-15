"""Tests for confounds processing."""

import pandas as pd

from network_glm.lev1.processing.confounds import get_fc_confounds


class TestGetFcConfounds:
    """Test FC confound extraction for tissue-based regression."""

    def test_returns_all_fc_columns(self):
        """All six expected FC confound columns should be returned."""
        df = pd.DataFrame(
            {
                "csf": [1.0, 2.0],
                "csf_derivative1": [0.1, 0.2],
                "white_matter": [1.0, 2.0],
                "white_matter_derivative1": [0.1, 0.2],
                "global_signal": [1.0, 2.0],
                "global_signal_derivative1": [0.1, 0.2],
                "trans_x": [0.5, 0.6],
            }
        )
        result = get_fc_confounds(df)
        assert "global_signal" in result.columns
        assert "csf" in result.columns
        assert "white_matter" in result.columns
        assert "global_signal_derivative1" in result.columns
        assert "csf_derivative1" in result.columns
        assert "white_matter_derivative1" in result.columns
        assert "trans_x" not in result.columns
        assert len(result.columns) == 6

    def test_partial_columns(self):
        """Should return whatever FC columns are available."""
        df = pd.DataFrame(
            {
                "csf": [1.0],
                "white_matter": [1.0],
                "trans_x": [0.5],
            }
        )
        result = get_fc_confounds(df)
        assert len(result.columns) == 2
        assert "csf" in result.columns
        assert "white_matter" in result.columns

    def test_no_fc_columns_returns_empty(self):
        """Should return empty DataFrame when no FC columns found."""
        df = pd.DataFrame(
            {
                "trans_x": [0.5],
                "rot_x": [0.1],
            }
        )
        result = get_fc_confounds(df)
        assert result.empty
