"""Regression test: every active BIDS scan has a complete fmriprep output AND an events.tsv (if non-rest)."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

import pytest

BIDS_DERIV_PAIRS = [
    (
        Path("/scratch/users/logben/discovery_bids"),
        Path("/scratch/users/logben/discovery_bids/derivatives/fmriprep_25.2.4"),
    ),
    (
        Path("/scratch/users/logben/validation_bids"),
        Path("/scratch/users/logben/validation_bids/derivatives/fmriprep_25.2.4"),
    ),
]

BOLD_RE = re.compile(r"^(sub-\w+)_(ses-\w+)_task-(\w+)_run-(\w+)(?:_echo-\d+)?_bold\.nii\.gz$")
CONFOUNDS_RE = re.compile(
    r"^(sub-\w+)_(ses-\w+)_task-(\w+)_run-(\w+)_desc-confounds_timeseries\.tsv$"
)


def _load_bidsignore(bids_dir: Path) -> list[str]:
    f = bids_dir / ".bidsignore"
    if not f.is_file():
        return []
    return [
        ln.strip()
        for ln in f.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def _is_ignored(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in patterns)


def _active_scans(bids_dir: Path) -> set[tuple[str, str, str, str]]:
    patterns = _load_bidsignore(bids_dir)
    out: set[tuple[str, str, str, str]] = set()
    for fp in bids_dir.rglob("*_bold.nii.gz"):
        parts = fp.relative_to(bids_dir).parts
        if "derivatives" in parts or "sourcedata" in parts:
            continue
        rel = fp.relative_to(bids_dir).as_posix()
        if _is_ignored(rel, patterns):
            continue
        m = BOLD_RE.match(fp.name)
        if not m:
            continue
        out.add((m.group(1), m.group(2), m.group(3), m.group(4)))
    return out


def _processed_scans(deriv_dir: Path) -> set[tuple[str, str, str, str]]:
    out: set[tuple[str, str, str, str]] = set()
    for fp in deriv_dir.rglob("*_desc-confounds_timeseries.tsv"):
        m = CONFOUNDS_RE.match(fp.name)
        if m:
            out.add((m.group(1), m.group(2), m.group(3), m.group(4)))
    return out


@pytest.mark.parametrize("bids_dir,deriv_dir", BIDS_DERIV_PAIRS, ids=lambda p: p.name)
def test_every_active_bids_scan_was_processed(bids_dir: Path, deriv_dir: Path) -> None:
    if not bids_dir.is_dir() or not deriv_dir.is_dir():
        pytest.skip(f"BIDS or fmriprep dir not present: {bids_dir} / {deriv_dir}")
    active = _active_scans(bids_dir)
    processed = _processed_scans(deriv_dir)
    missing = active - processed
    extra = processed - active
    assert not missing, f"BIDS scans not processed by fmriprep: {sorted(missing)[:10]}"
    assert not extra, f"fmriprep processed scans not in BIDS: {sorted(extra)[:10]}"


@pytest.mark.parametrize("bids_dir,_deriv_dir", BIDS_DERIV_PAIRS, ids=lambda p: p.name)
def test_every_task_scan_has_events(bids_dir: Path, _deriv_dir: Path) -> None:
    if not bids_dir.is_dir():
        pytest.skip(f"BIDS dir not present: {bids_dir}")
    active = _active_scans(bids_dir)
    task_scans = [s for s in active if s[2].lower() != "rest"]
    missing = []
    for sub, ses, task, run in task_scans:
        events = bids_dir / sub / ses / "func" / f"{sub}_{ses}_task-{task}_run-{run}_events.tsv"
        if not events.is_file():
            missing.append((sub, ses, task, run))
    assert not missing, f"task scans without events.tsv: {sorted(missing)[:10]}"
