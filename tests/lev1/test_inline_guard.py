"""Inline design-matrix guard: rank deficiency only.

Per-contrast VIF is checked separately by `run_quality_control`
(see test_quality_control_contrast_vif.py); column-VIF on the design
matrix is intentionally not checked here because nuisance regressors
(motion + motion**2, drift bases) routinely have high inter-column VIFs
that don't impair contrast estimation.
"""

from __future__ import annotations

import pytest

try:
    import nilearn  # noqa: F401
except ImportError:
    pytest.skip(
        "neuroimaging dependencies not installed (install with: uv pip install -e '.[lev1]')",
        allow_module_level=True,
    )

import numpy as np
import pandas as pd

from network_glm.lev1.processing.glm import (
    RankDeficientDesignError,
    check_design_matrix_health,
)


def test_rank_deficient_design_matrix_raises():
    # Two perfectly correlated columns -> rank deficient
    n = 100
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    dm = pd.DataFrame(
        {
            "go": a,
            "go_dup": a,  # perfect duplicate of "go"
            "constant": np.ones(n),
        }
    )
    with pytest.raises(RankDeficientDesignError, match="rank"):
        check_design_matrix_health(dm)


def test_clean_design_matrix_passes():
    n = 100
    rng = np.random.default_rng(1)
    dm = pd.DataFrame(
        {
            "go": rng.normal(size=n),
            "stop": rng.normal(size=n),
            "constant": np.ones(n),
        }
    )
    # Should not raise
    check_design_matrix_health(dm)


def test_high_per_column_vif_does_not_raise():
    """Routine motion + motion**2 collinearity must NOT trip the inline guard.

    Per-column VIF can hit hundreds for legitimate nuisance regressors;
    that's contrast-VIF territory and lives in quality_control instead.
    """
    n = 1000
    rng = np.random.default_rng(2)
    x1 = rng.normal(size=n)
    # Build a near-duplicate column that produces a very high per-column VIF
    # but doesn't violate rank (just very high R^2). Should not raise.
    x2 = x1 + rng.normal(scale=0.001, size=n)
    dm = pd.DataFrame({"x1": x1, "x2": x2, "constant": np.ones(n)})
    check_design_matrix_health(dm)
