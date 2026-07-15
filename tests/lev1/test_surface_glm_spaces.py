"""Integration tests for multi-space surface GLM support."""

import numpy as np
import pandas as pd

from network_glm.io.file_discovery import FileFinder
from network_glm.lev1.processing.confounds import get_fc_confounds
from network_glm.lev1.processing.fixed_effects import FixedEffectsAnalyzer
from network_glm.lev1.processing.surface_data import SurfaceGLM


class TestSurfaceGLMSpaces:
    """Test that SurfaceGLM works on different vertex counts."""

    def test_fsaverage6_sized_data(self):
        """SurfaceGLM should handle fsaverage6-sized data (40962 vertices)."""
        np.random.seed(42)
        n_tp, n_verts = 100, 40962
        X = np.random.randn(n_tp, 3)
        Y = X @ np.random.randn(3, n_verts) + np.random.randn(n_tp, n_verts) * 0.5

        dm = pd.DataFrame(X, columns=["r0", "r1", "r2"])
        glm = SurfaceGLM(t_r=1.5, noise_model="ols")
        glm.fit(Y, dm)

        residuals = glm.get_residuals()
        assert residuals.shape == (n_tp, n_verts)
        assert np.std(residuals) < 1.0

    def test_residuals_with_fc_confounds_shape(self):
        """Residuals from SurfaceGLM have correct shape for nilearn.signal.clean."""
        from nilearn.signal import clean as clean_signal

        np.random.seed(42)
        n_tp, n_verts = 100, 500
        X = np.random.randn(n_tp, 3)
        Y = X @ np.random.randn(3, n_verts) + np.random.randn(n_tp, n_verts) * 0.5

        dm = pd.DataFrame(X, columns=["r0", "r1", "r2"])
        glm = SurfaceGLM(t_r=1.5, noise_model="ols")
        glm.fit(Y, dm)
        residuals = glm.get_residuals()

        # Mock FC confounds (same number of timepoints)
        fc_confounds = np.random.randn(n_tp, 6)

        # clean_signal should accept (n_tp, n_verts) with confounds
        cleaned = clean_signal(
            residuals,
            t_r=1.5,
            low_pass=0.1,
            high_pass=0.01,
            confounds=fc_confounds,
            standardize=False,
            detrend=False,
        )
        assert cleaned.shape == (n_tp, n_verts)


class TestFixedEffectsSpaceTag:
    """Test that fixed effects output uses the correct space tag."""

    def test_default_space_is_fsnative(self):
        analyzer = FixedEffectsAnalyzer("sub-s01", "flanker", hemisphere="L")
        assert analyzer.surface_space == "fsnative"

    def test_fsaverage6_space_tag(self):
        analyzer = FixedEffectsAnalyzer(
            "sub-s01", "flanker", hemisphere="L", surface_space="fsaverage6"
        )
        assert analyzer.surface_space == "fsaverage6"

    def test_fslr_space_tag(self):
        analyzer = FixedEffectsAnalyzer("sub-s01", "flanker", hemisphere="L", surface_space="fsLR")
        assert analyzer.surface_space == "fsLR"


class TestFileDiscoverySpaces:
    """Test file discovery parameterization for multiple spaces."""

    def test_surface_patterns_dict_has_all_spaces(self):
        assert "fsnative" in FileFinder.SURFACE_PATTERNS
        assert "fsaverage6" in FileFinder.SURFACE_PATTERNS
        assert "fsLR" in FileFinder.SURFACE_PATTERNS

    def test_fsnative_patterns_are_gifti(self):
        patterns = FileFinder.SURFACE_PATTERNS["fsnative"]
        assert "left_surface" in patterns
        assert "right_surface" in patterns
        assert patterns["left_surface"].endswith(".func.gii")

    def test_fsaverage6_patterns_are_gifti(self):
        patterns = FileFinder.SURFACE_PATTERNS["fsaverage6"]
        assert "left_surface" in patterns
        assert "fsaverage6" in patterns["left_surface"]

    def test_fslr_patterns_are_cifti(self):
        patterns = FileFinder.SURFACE_PATTERNS["fsLR"]
        assert "cifti_bold" in patterns
        assert patterns["cifti_bold"].endswith(".dtseries.nii")


class TestFcConfoundsIntegration:
    """Test FC confound extraction from realistic confounds data."""

    def test_fc_confounds_from_full_confounds_tsv(self):
        """FC confounds should extract correct columns from full confounds."""
        # Simulate a realistic fMRIPrep confounds TSV
        n_tp = 200
        columns = [
            "csf",
            "csf_derivative1",
            "white_matter",
            "white_matter_derivative1",
            "global_signal",
            "global_signal_derivative1",
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "cosine00",
            "cosine01",
            "cosine02",
        ]
        df = pd.DataFrame(
            np.random.randn(n_tp, len(columns)),
            columns=columns,
        )
        fc = get_fc_confounds(df)
        assert len(fc.columns) == 6
        assert fc.shape[0] == n_tp
        # Verify no motion or cosine columns leaked in
        for col in fc.columns:
            assert col in [
                "global_signal",
                "global_signal_derivative1",
                "csf",
                "csf_derivative1",
                "white_matter",
                "white_matter_derivative1",
            ]
