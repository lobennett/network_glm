"""Tests for ``resolve_freesurfer_subject``.

fMRIPrep's longitudinal anat workflow creates per-session FreeSurfer
subjects named ``sub-X_ses-Y`` rather than the plain canonical
``sub-X``. Any code shelling out to a FreeSurfer binary
(``mri_surf2surf``, ``mris_euler_number``, etc.) must pass the on-disk
name or it fails with an opaque ``failed to open GIFTI XML file`` error
deep inside the binary. This helper resolves canonical → on-disk name.

Coverage:

1. **Direct hit** — ``<SUBJECTS_DIR>/sub-X`` exists → return as-is.
2. **Session glob** — only ``<SUBJECTS_DIR>/sub-X_ses-Y`` exists →
   return the on-disk name with the session suffix.
3. **Multiple sessions** — returns a deterministic choice (sorted)
   so the same subject resolves the same way across re-runs.
4. **Missing** — raises ``FileNotFoundError`` with a clear message
   listing both lookup patterns. Silently substituting the canonical
   name would yield the opaque downstream FreeSurfer error.
5. **No false match** — ``sub-s10alt_ses-09`` must NOT resolve as a
   match for ``sub-s10``. The glob anchors to ``{subject}_ses-*``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from network_glm.lev1.processing.surface_data import (
    resolve_freesurfer_subject,
)


def test_returns_canonical_when_direct_dir_exists(tmp_path):
    (tmp_path / "sub-s03").mkdir()
    assert resolve_freesurfer_subject("sub-s03", tmp_path) == "sub-s03"


def test_resolves_session_suffixed_dir_when_canonical_absent(tmp_path):
    (tmp_path / "sub-s10_ses-09").mkdir()
    assert resolve_freesurfer_subject("sub-s10", tmp_path) == "sub-s10_ses-09"


def test_returns_deterministic_choice_for_multiple_sessions(tmp_path):
    (tmp_path / "sub-s10_ses-09").mkdir()
    (tmp_path / "sub-s10_ses-05").mkdir()
    (tmp_path / "sub-s10_ses-13").mkdir()
    # Sorted order makes the choice deterministic across runs
    result = resolve_freesurfer_subject("sub-s10", tmp_path)
    assert result == "sub-s10_ses-05"


def test_canonical_dir_wins_over_session_dir(tmp_path):
    """If both ``sub-X`` and ``sub-X_ses-Y`` exist, prefer canonical.

    Order matters: the canonical dir means we're in single-anat mode and
    FS binaries can be passed the plain name; the session dir is a
    multi-anat artifact that would also work but is the longer name.
    """
    (tmp_path / "sub-s10").mkdir()
    (tmp_path / "sub-s10_ses-09").mkdir()
    assert resolve_freesurfer_subject("sub-s10", tmp_path) == "sub-s10"


def test_raises_when_no_matching_dir_exists(tmp_path):
    """No matching FS subject → FileNotFoundError with a helpful message
    (not a silent canonical-name substitution that would surface later
    as an opaque FreeSurfer binary failure)."""
    with pytest.raises(FileNotFoundError) as exc:
        resolve_freesurfer_subject("sub-s10", tmp_path)
    msg = str(exc.value)
    assert "sub-s10" in msg
    assert "sub-s10_ses-*" in msg


def test_does_not_match_lookalike_subjects(tmp_path):
    """``sub-s10alt_ses-09`` must NOT resolve for ``sub-s10``.

    The glob pattern is ``{canonical}_ses-*``; the underscore boundary
    means any subject id sharing a prefix with another shouldn't false-match.
    """
    (tmp_path / "sub-s10alt_ses-09").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_freesurfer_subject("sub-s10", tmp_path)


def test_accepts_str_and_path_subjects_dir(tmp_path):
    """``subjects_dir`` may be passed as str or Path."""
    (tmp_path / "sub-s03").mkdir()
    assert resolve_freesurfer_subject("sub-s03", str(tmp_path)) == "sub-s03"
    assert resolve_freesurfer_subject("sub-s03", Path(tmp_path)) == "sub-s03"
