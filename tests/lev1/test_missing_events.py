"""Missing events.tsv handling: skip-with-warning, never crash.

Spec (`docs/archive/superpowers/specs/2026-05-06-lev1-audit-design.md`):
"scan with events.tsv missing -> caller (run.py) skips with a logger warning;
downstream gets nothing for that scan; no crash."

`FileFinder.get_files()` already drops runs that lack any required file via
`_filter_complete_runs`, so lev1 will not crash. The gap surfaced by Task 14
is that the drop is silent: a non-rest scan with BOLD + confounds + no
events.tsv disappears from the output without any log entry, which makes
"why isn't this scan being analyzed?" debugging painful.

This file codifies the desired behavior at two layers:

1. `FileFinder.get_files` / `_filter_complete_runs` - emit a logger warning
   that names the dropped run and the missing file types.
2. `discover_and_validate_files` (run.py) - keeps raising `ValueError` only
   when the entire subject yields zero runs, and the warning from layer (1)
   is visible to the caller.
"""

from __future__ import annotations

import logging
from pathlib import Path

from network_glm.io.file_discovery import FileFinder


def _make_run_files(
    bids_dir: Path,
    fmriprep_dir: Path,
    *,
    subject: str,
    session: str,
    task: str,
    run: str,
    include_events: bool,
) -> None:
    """Build a minimal BIDS + fMRIPrep tree for one run.

    fMRIPrep side always has confounds + T1w/MNI BOLD + masks. BIDS side
    optionally has events.tsv based on include_events.
    """
    bids_func = bids_dir / f"sub-{subject}" / session / "func"
    fmriprep_func = fmriprep_dir / f"sub-{subject}" / session / "func"
    bids_func.mkdir(parents=True, exist_ok=True)
    fmriprep_func.mkdir(parents=True, exist_ok=True)

    base = f"sub-{subject}_{session}_task-{task}_{run}"

    if include_events:
        (bids_func / f"{base}_events.tsv").write_text("onset\tduration\ttrial_type\n0\t1\tgo\n")

    fmriprep_files = [
        f"{base}_desc-confounds_timeseries.tsv",
        f"{base}_space-T1w_desc-preproc_bold.nii.gz",
        f"{base}_space-T1w_desc-brain_mask.nii.gz",
        f"{base}_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
        f"{base}_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz",
    ]
    for fname in fmriprep_files:
        (fmriprep_func / fname).write_text("mock")


def test_missing_events_does_not_crash(tmp_path: Path) -> None:
    """A non-rest scan missing events.tsv is filtered out, not raised."""
    bids_dir = tmp_path / "bids"
    fmriprep_dir = tmp_path / "fmriprep"
    _make_run_files(
        bids_dir,
        fmriprep_dir,
        subject="s01",
        session="ses-01",
        task="stopSignal",
        run="run-01",
        include_events=False,
    )

    finder = FileFinder(bids_dir, fmriprep_dir)
    required = FileFinder.get_required_files_for_space("MNI")
    # Must not raise.
    files = finder.get_files("s01", "stopSignal", required_files=required)

    # Run is dropped because events is required but missing.
    assert files == {}


def test_missing_events_logs_warning(tmp_path: Path, caplog) -> None:
    """When events is the only missing required file, FileFinder warns.

    AUDIT GAP (Task 14): currently `_filter_complete_runs` silently drops
    runs lacking required files. The desired behavior per the spec is
    skip-with-warning so operators see why a scan dropped out. When the
    warning is added, this test passes; until then it documents the gap.
    """
    bids_dir = tmp_path / "bids"
    fmriprep_dir = tmp_path / "fmriprep"
    _make_run_files(
        bids_dir,
        fmriprep_dir,
        subject="s01",
        session="ses-01",
        task="stopSignal",
        run="run-01",
        include_events=False,
    )

    finder = FileFinder(bids_dir, fmriprep_dir)
    required = FileFinder.get_required_files_for_space("MNI")

    with caplog.at_level(logging.WARNING, logger="network_glm.io.file_discovery"):
        files = finder.get_files("s01", "stopSignal", required_files=required)

    assert files == {}
    warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert warnings, "expected a warning when a run is dropped for missing required files"
    msgs = "\n".join(rec.getMessage() for rec in warnings)
    assert (
        "events" in msgs.lower()
    ), f'expected a warning naming the missing file type "events"; got:\n{msgs}'
    # The warning should locate the dropped run (subject/session/run identifiers).
    assert "run-01" in msgs


def test_partial_missing_events_keeps_other_runs(tmp_path: Path, caplog) -> None:
    """Run-01 missing events is dropped+warned; run-02 with events is kept."""
    bids_dir = tmp_path / "bids"
    fmriprep_dir = tmp_path / "fmriprep"
    _make_run_files(
        bids_dir,
        fmriprep_dir,
        subject="s01",
        session="ses-01",
        task="stopSignal",
        run="run-01",
        include_events=False,
    )
    _make_run_files(
        bids_dir,
        fmriprep_dir,
        subject="s01",
        session="ses-01",
        task="stopSignal",
        run="run-02",
        include_events=True,
    )

    finder = FileFinder(bids_dir, fmriprep_dir)
    required = FileFinder.get_required_files_for_space("MNI")

    with caplog.at_level(logging.WARNING, logger="network_glm.io.file_discovery"):
        files = finder.get_files("s01", "stopSignal", required_files=required)

    # run-02 retained; run-01 silently dropped or warned.
    assert "ses-01" in files
    assert "run-02" in files["ses-01"]
    assert "run-01" not in files["ses-01"]

    # Warning should reference run-01, not run-02.
    msgs = "\n".join(rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING)
    assert "run-01" in msgs
    assert "events" in msgs.lower()


def test_missing_events_does_not_warn_for_unrelated_subject(tmp_path: Path, caplog) -> None:
    """Empty subject directory yields {} but should not produce spurious warnings."""
    bids_dir = tmp_path / "bids"
    fmriprep_dir = tmp_path / "fmriprep"
    bids_dir.mkdir()
    fmriprep_dir.mkdir()

    finder = FileFinder(bids_dir, fmriprep_dir)
    required = FileFinder.get_required_files_for_space("MNI")

    with caplog.at_level(logging.WARNING, logger="network_glm.io.file_discovery"):
        files = finder.get_files("sNotPresent", "stopSignal", required_files=required)

    assert files == {}
    # No partial run was found, so there's nothing to warn about.
    msgs = [rec.getMessage() for rec in caplog.records if rec.levelno >= logging.WARNING]
    assert not msgs, f"unexpected warnings for empty subject: {msgs}"
