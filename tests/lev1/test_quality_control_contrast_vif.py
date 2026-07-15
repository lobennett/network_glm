"""Contrast-VIF computation inside `run_quality_control`.

VIFs are computed and saved to the contrastVIFs.csv but do NOT fail QA —
high-VIF runs are surfaced for manual review by the cohort QC step
(`network_glm.cohort.outliers`), not auto-skipped at run time.
"""

from __future__ import annotations

from pathlib import Path

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

from network_glm.lev1.processing.quality_control import run_quality_control


def _design(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "go": rng.normal(size=n),
            "stop": rng.normal(size=n),
            "constant": np.ones(n),
        }
    )


def test_low_contrast_vif_passes_qa(tmp_path: Path):
    """Independent regressors -> contrast VIFs ~1; QA passes."""
    dm = _design()
    contrasts = {"go": "go", "stop": "stop", "go-stop": "go - stop"}
    vifs, any_fail = run_quality_control(
        design_matrix=dm,
        contrasts=contrasts,
        percent_junk=0.05,
        output_dir=tmp_path,
        subject_id="sub-s01",
        session="ses-01",
        run="run-1",
        task_name="goNogo",
    )
    assert all(v < 5.0 for v in vifs.values()), vifs
    assert any_fail is False


def test_high_contrast_vif_does_not_fail_qa(tmp_path: Path):
    """High contrast VIFs are computed and saved but do NOT fail QA.

    The cohort QC step flags them for manual review; lev1 must not
    auto-skip scans on contrast VIF.
    """
    n = 200
    rng = np.random.default_rng(7)
    base = rng.normal(size=n)
    dm = pd.DataFrame(
        {
            "a": base,
            "b": base + rng.normal(scale=0.05, size=n),
            "constant": np.ones(n),
        }
    )
    contrasts = {"a-b": "a - b"}
    vifs, any_fail = run_quality_control(
        design_matrix=dm,
        contrasts=contrasts,
        percent_junk=0.05,
        output_dir=tmp_path,
        subject_id="sub-s01",
        session="ses-01",
        run="run-1",
        task_name="goNogo",
    )
    # VIF is high (>>5) but QA still passes — flagging is downstream
    assert vifs["a-b"] > 5.0, vifs
    assert any_fail is False
    # CSV still emitted with the VIF for cohort QC to consume
    csv_path = tmp_path / "sub-s01_ses-01_task-goNogo_run-1_desc-contrastVIFs.csv"
    assert csv_path.is_file()
    df = pd.read_csv(csv_path)
    assert df.loc[df["contrast"] == "a-b", "VIF"].iloc[0] > 5.0


def test_high_junk_fails_qa(tmp_path: Path):
    """Behavioral junk > 30% remains a true QA failure (not a VIF call)."""
    dm = _design()
    contrasts = {"go": "go"}
    vifs, any_fail = run_quality_control(
        design_matrix=dm,
        contrasts=contrasts,
        percent_junk=0.35,
        output_dir=tmp_path,
        subject_id="sub-s01",
        session="ses-01",
        run="run-1",
        task_name="goNogo",
    )
    assert any_fail is True
