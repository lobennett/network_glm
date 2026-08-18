"""Tests for network_glm.qc.design_plots — design/contrast/correlation figures.

These reproduce the two figures collaborators circulate when reviewing a GLM:

1. the design matrix (scans x regressors) with a companion contrast matrix, and
2. the regressor x regressor correlation matrix.

Nothing here recomputes a GLM. `run_quality_control` already persists
``*_desc-designMatrix.csv`` per run, so plotting is a post-hoc pass over existing
outputs — the same principle `--skip-qc-plots` was built on (persist the data,
render offline).

One deliberate divergence from nilearn: ``plot_design_matrix_correlation`` drops
drift and constant regressors ("The drift and constant regressors are omitted from
the plot"), but those rows are exactly what you want when checking whether a task
regressor is soaking up drift. We compute the correlation ourselves so cosine and
constant columns stay visible.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from network_glm.qc import design_plots  # noqa: E402


def _design(n_scans=120, drift=True, constant=True) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = {
        "go": rng.normal(size=n_scans),
        "nogo_success": rng.normal(size=n_scans),
        "junk": rng.normal(size=n_scans),
        "trans_x": rng.normal(size=n_scans),
    }
    if drift:
        for i in range(3):
            cols[f"cosine0{i}"] = np.cos(np.linspace(0, (i + 1) * np.pi, n_scans))
    if constant:
        cols["constant"] = np.ones(n_scans)
    return pd.DataFrame(cols)


def _write_design(tmp_path, name="sub-s03_ses-01_task-goNogo_run-1_desc-designMatrix.csv"):
    path = tmp_path / name
    _design().to_csv(path, index=False)
    return path


# --- numeric cores ------------------------------------------------------------

def test_design_correlation_keeps_drift_and_constant(tmp_path):
    """The whole point of rolling our own: nilearn would drop these columns."""
    corr = design_plots.design_correlation(_design())

    assert "cosine00" in corr.columns and "constant" in corr.columns
    assert list(corr.columns) == list(corr.index), "must be square + same order"


def test_design_correlation_can_omit_drift_like_nilearn(tmp_path):
    corr = design_plots.design_correlation(_design(), include_drift=False)

    assert not [c for c in corr.columns if c.startswith("cosine")]
    assert "constant" not in corr.columns
    assert "go" in corr.columns


def test_design_correlation_survives_the_zero_variance_constant(tmp_path):
    """A constant column has zero variance, so its correlations are undefined.

    Leave them NaN (they render blank) but force the diagonal to 1.0 so the plot
    reads like every other correlation matrix instead of punching a hole in it.
    """
    corr = design_plots.design_correlation(_design())

    assert corr.loc["constant", "constant"] == pytest.approx(1.0)
    assert np.isnan(corr.loc["constant", "go"])
    assert np.all(np.isclose(np.diag(corr.to_numpy()), 1.0))


def test_contrast_matrix_rows_are_contrasts_columns_are_regressors():
    design = _design()
    contrasts = {"go": "go", "nogo-go": "nogo_success - go"}

    cmat = design_plots.contrast_matrix(design.columns.tolist(), contrasts)

    assert list(cmat.index) == ["go", "nogo-go"]
    assert list(cmat.columns) == design.columns.tolist()
    assert cmat.loc["go", "go"] == pytest.approx(1.0)
    assert cmat.loc["nogo-go", "nogo_success"] == pytest.approx(1.0)
    assert cmat.loc["nogo-go", "go"] == pytest.approx(-1.0)
    assert cmat.loc["go", "trans_x"] == pytest.approx(0.0)


def test_contrast_matrix_skips_unparseable_contrasts():
    """A contrast naming a regressor this run doesn't have is reported, not fatal —
    dropped-column runs are normal (see contrasts.filter_contrasts_for_dropped_columns)."""
    design = _design()

    cmat = design_plots.contrast_matrix(
        design.columns.tolist(), {"go": "go", "bogus": "not_a_regressor"}
    )

    assert list(cmat.index) == ["go"]


def test_parse_entities_from_the_design_matrix_filename():
    ents = design_plots.parse_entities(
        "sub-s03_ses-11_task-stopSignalWFlanker_run-2_desc-designMatrix.csv"
    )
    assert ents == {
        "subject": "sub-s03",
        "session": "ses-11",
        "task": "stopSignalWFlanker",
        "run": "run-2",
    }


# --- figures ------------------------------------------------------------------

def test_plot_run_writes_three_figures(tmp_path):
    csv = _write_design(tmp_path)

    written = design_plots.plot_run(csv, out_dir=tmp_path, contrasts={"go": "go"})

    assert set(written) == {"design", "contrasts", "correlation"}
    for kind, path in written.items():
        assert path.is_file(), kind
        assert path.stat().st_size > 5000, f"{kind} looks empty"
    assert written["design"].name.endswith("_desc-designMatrix.png")
    assert written["correlation"].name.endswith("_desc-designCorrelation.png")
    assert written["contrasts"].name.endswith("_desc-contrastMatrix.png")


def test_plot_run_resolves_contrasts_from_the_task_name(tmp_path):
    """With no contrasts passed, look them up from the task in the filename so the
    command works on lev1 output that was written months ago."""
    csv = _write_design(tmp_path)

    written = design_plots.plot_run(csv, out_dir=tmp_path)

    assert written["contrasts"].is_file()


def test_plot_run_omits_the_contrast_panel_when_none_resolve(tmp_path):
    csv = _write_design(
        tmp_path, name="sub-s03_ses-01_task-notARealTask_run-1_desc-designMatrix.csv"
    )

    written = design_plots.plot_run(csv, out_dir=tmp_path)

    assert "contrasts" not in written
    assert written["design"].is_file() and written["correlation"].is_file()


def test_plot_run_defaults_output_beside_the_input(tmp_path):
    csv = _write_design(tmp_path)

    written = design_plots.plot_run(csv)

    assert written["design"].parent == csv.parent


def test_titles_name_the_subject_and_task(tmp_path):
    """Figures get shared out of context, so they must be self-identifying."""
    csv = _write_design(tmp_path)

    title = design_plots.figure_title(design_plots.parse_entities(csv.name), "design")

    assert "sub-s03" in title and "goNogo" in title and "ses-01" in title


# --- batch + CLI --------------------------------------------------------------

def test_main_plots_every_design_matrix_under_a_directory(tmp_path):
    _write_design(tmp_path, "sub-s03_ses-01_task-goNogo_run-1_desc-designMatrix.csv")
    _write_design(tmp_path, "sub-s10_ses-02_task-goNogo_run-1_desc-designMatrix.csv")
    out = tmp_path / "figs"

    rc = design_plots.main([str(tmp_path), "--out-dir", str(out)])

    assert rc == 0
    assert len(sorted(out.glob("*_desc-designMatrix.png"))) == 2


def test_main_accepts_a_single_csv(tmp_path):
    csv = _write_design(tmp_path)
    out = tmp_path / "figs"

    rc = design_plots.main([str(csv), "--out-dir", str(out)])

    assert rc == 0
    assert (out / csv.name.replace(".csv", ".png")).is_file()


def test_main_errors_when_nothing_matches(tmp_path):
    with pytest.raises(SystemExit):
        design_plots.main([str(tmp_path / "nope.csv")])


def test_cli_routes_design_plots(monkeypatch):
    """`network-glm design-plots ...` must reach this module."""
    from network_glm import cli

    seen = {}
    monkeypatch.setattr(cli, "_design_plots_main", lambda argv: seen.setdefault("argv", argv) or 0)

    cli.main(["design-plots", "somewhere"])

    assert seen["argv"] == ["somewhere"]
