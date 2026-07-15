"""FileFinder uses the BIDS-side session, not the behavioral session.

Five validation subjects (s321, s1445, s1326, s1391, s1258) have +1 offset
between behavioral and BIDS sessions. The offset is resolved at BIDS-creation
time via the reconciliation manifest's ``dest_session`` field, so by the time
lev1 runs, fmriprep paths are written under the BIDS session label and
FileFinder must respect them.

These tests build a fake fmriprep tree under the BIDS-side session for an
offset subject and confirm:
  * FileFinder discovers the run keyed by the BIDS session.
  * No data is invented for the (would-be-wrong) behavioral session.
  * If a stray behavioral-session folder exists alongside, it is reported
    independently and never substituted for the BIDS session.
"""

from __future__ import annotations

from pathlib import Path

from network_glm.io.file_discovery import FileFinder

# (BIDS-side session, behavioral session that should NOT be substituted)
OFFSET_CASES = [
    ("s321", "ses-02", "ses-01"),
    ("s1445", "ses-02", "ses-01"),
    ("s1326", "ses-03", "ses-02"),
    ("s1391", "ses-06", "ses-05"),
    ("s1258", "ses-07", "ses-06"),
]


def _make_full_run(
    bids_dir: Path,
    fmriprep_dir: Path,
    subject: str,
    session: str,
    task: str,
    run: str,
) -> None:
    """Create a complete BIDS+fmriprep file tree for a single run."""
    sub = f"sub-{subject}"
    base = f"{sub}_{session}_task-{task}_{run}"

    bids_func = bids_dir / sub / session / "func"
    bids_func.mkdir(parents=True, exist_ok=True)
    (bids_func / f"{base}_events.tsv").write_text("onset\tduration\ttrial_type\n0\t1\tcue\n")

    fp_func = fmriprep_dir / sub / session / "func"
    fp_func.mkdir(parents=True, exist_ok=True)
    for suffix in (
        "desc-confounds_timeseries.tsv",
        "space-T1w_desc-preproc_bold.nii.gz",
        "space-T1w_desc-brain_mask.nii.gz",
        "space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz",
        "space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz",
    ):
        (fp_func / f"{base}_{suffix}").write_text("mock")


def test_offset_subject_uses_bids_session_s1258(tmp_path: Path) -> None:
    """s1258's first behavioral session maps to BIDS ses-07.

    Build a fake fmriprep tree under ses-07 only and confirm FileFinder
    surfaces the run under the BIDS session key, with all required files.
    """
    bids = tmp_path / "bids"
    fmriprep = tmp_path / "fmriprep_25.2.4"
    _make_full_run(bids, fmriprep, "s1258", "ses-07", "rest", "run-1")

    finder = FileFinder(bids_dir=bids, fmriprep_dir=fmriprep)
    files = finder.get_files("s1258", "rest")

    assert "ses-07" in files, f"expected BIDS session ses-07 in result, got {sorted(files)}"
    assert "ses-06" not in files, (
        "behavioral session ses-06 must not be invented — offset is a "
        "BIDS-creation concern, not a lev1 concern"
    )

    run_files = files["ses-07"]["run-1"]
    for ftype in (
        "events",
        "confounds",
        "t1w_data",
        "t1w_brain_mask",
        "mni_data",
        "mni_brain_mask",
    ):
        assert ftype in run_files, f"missing {ftype}"
        path = run_files[ftype]
        # Path must contain the BIDS session, never the behavioral one.
        assert "ses-07" in str(path)
        assert "ses-06" not in str(path)


def test_all_offset_subjects_resolve_via_bids_session(tmp_path: Path) -> None:
    """Parametric coverage of every documented offset subject.

    For each (subject, BIDS session, behavioral session) tuple we build only
    the BIDS-side tree and confirm FileFinder reports that session — never
    the behavioral one.
    """
    for subject, bids_session, beh_session in OFFSET_CASES:
        sub_root = tmp_path / subject
        bids = sub_root / "bids"
        fmriprep = sub_root / "fmriprep"
        _make_full_run(bids, fmriprep, subject, bids_session, "rest", "run-1")

        finder = FileFinder(bids_dir=bids, fmriprep_dir=fmriprep)
        files = finder.get_files(subject, "rest")

        assert (
            bids_session in files
        ), f"{subject}: expected {bids_session} in result, got {sorted(files)}"
        assert (
            beh_session not in files
        ), f"{subject}: behavioral session {beh_session} must not appear"

        bold = files[bids_session]["run-1"]["mni_data"]
        assert bids_session in bold.parts
        assert beh_session not in bold.parts


def test_stray_behavioral_session_dir_not_promoted(tmp_path: Path) -> None:
    """A stray behavioral-session directory must not contaminate the result.

    If, hypothetically, a behavioral-session folder existed alongside the
    BIDS-side one (e.g. an old artifact), FileFinder must still key by the
    actual on-disk session and never silently rewrite paths. We confirm by
    populating the BIDS session fully and the behavioral session only with
    a stray events file: the behavioral session is reported as its own
    incomplete run (and therefore filtered out), while the BIDS session
    survives intact.
    """
    bids = tmp_path / "bids"
    fmriprep = tmp_path / "fmriprep"
    _make_full_run(bids, fmriprep, "s1258", "ses-07", "rest", "run-1")

    # Stray behavioral-session events file (no fmriprep counterparts).
    stray_func = bids / "sub-s1258" / "ses-06" / "func"
    stray_func.mkdir(parents=True, exist_ok=True)
    (stray_func / "sub-s1258_ses-06_task-rest_run-1_events.tsv").write_text(
        "onset\tduration\ttrial_type\n0\t1\tcue\n"
    )

    finder = FileFinder(bids_dir=bids, fmriprep_dir=fmriprep)
    files = finder.get_files("s1258", "rest")

    # ses-07 (BIDS session) survives with all files.
    assert "ses-07" in files
    assert "run-1" in files["ses-07"]
    # ses-06 (behavioral session) is dropped because its required files are
    # missing — it is *not* substituted for or merged into ses-07.
    assert "ses-06" not in files
    # And the BIDS-session paths are unmodified.
    assert "ses-06" not in str(files["ses-07"]["run-1"]["mni_data"])
