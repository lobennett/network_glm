"""Tests for the lev1 provenance-writing helpers (PR4b, additive provenance).

These exercise the small testable helpers that wire the PR4a provenance
primitive into the lev1 run:

- ``_collect_run_inputs`` — flattens the discovered ``files`` dict into the
  list of ACTUAL input paths consumed (events / confounds / BOLD per space).
- ``_write_lev1_provenance`` — writes both ``dataset_description.json`` (at the
  shared results dir) and ``run-manifest.json`` (at the per-subject×task subdir)
  with the right stage / inputs / exclusions source.

Written red-first.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _touch(p: Path, content: bytes = b"x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


# --------------------------------------------------------------------------- #
# _collect_run_inputs
# --------------------------------------------------------------------------- #
def test_collect_run_inputs_volumetric(tmp_path):
    """Volumetric runs contribute events + confounds + mni/t1w BOLD data."""
    from network_glm.lev1.run import _collect_run_inputs

    ev = _touch(tmp_path / "events.tsv")
    cf = _touch(tmp_path / "confounds.tsv")
    bold = _touch(tmp_path / "mni_bold.nii.gz")

    files = {
        "ses-01": {
            "run-1": {
                "events": ev,
                "confounds": cf,
                "mni_data": bold,
                "mni_brain_mask": _touch(tmp_path / "mask.nii.gz"),
            }
        }
    }
    inputs = _collect_run_inputs(files)
    assert ev in inputs
    assert cf in inputs
    assert bold in inputs
    # masks are derived/intermediate, not a study input we hash here
    assert (tmp_path / "mask.nii.gz") not in inputs


def test_collect_run_inputs_surface_both_hemis(tmp_path):
    """Surface runs contribute events + confounds + both hemisphere BOLDs."""
    from network_glm.lev1.run import _collect_run_inputs

    ev = _touch(tmp_path / "events.tsv")
    cf = _touch(tmp_path / "confounds.tsv")
    lh = _touch(tmp_path / "hemi-L.func.gii")
    rh = _touch(tmp_path / "hemi-R.func.gii")

    files = {
        "ses-01": {
            "run-1": {
                "events": ev,
                "confounds": cf,
                "left_surface": lh,
                "right_surface": rh,
            }
        }
    }
    inputs = _collect_run_inputs(files)
    assert set(inputs) >= {ev, cf, lh, rh}


def test_collect_run_inputs_dedupes_and_orders(tmp_path):
    """Multiple runs across sessions are all collected, de-duplicated, sorted."""
    from network_glm.lev1.run import _collect_run_inputs

    ev1 = _touch(tmp_path / "ses-01" / "events.tsv")
    cf1 = _touch(tmp_path / "ses-01" / "confounds.tsv")
    b1 = _touch(tmp_path / "ses-01" / "bold.nii.gz")
    ev2 = _touch(tmp_path / "ses-02" / "events.tsv")
    cf2 = _touch(tmp_path / "ses-02" / "confounds.tsv")
    b2 = _touch(tmp_path / "ses-02" / "bold.nii.gz")

    files = {
        "ses-01": {"run-1": {"events": ev1, "confounds": cf1, "mni_data": b1}},
        "ses-02": {"run-1": {"events": ev2, "confounds": cf2, "mni_data": b2}},
    }
    inputs = _collect_run_inputs(files)
    assert set(inputs) == {ev1, cf1, b1, ev2, cf2, b2}
    assert len(inputs) == len(set(inputs))  # no duplicates
    # deterministic ordering
    assert inputs == sorted(inputs, key=str)


# --------------------------------------------------------------------------- #
# _write_lev1_provenance
# --------------------------------------------------------------------------- #
def test_write_lev1_provenance_writes_both_files(tmp_path, monkeypatch):
    """dataset_description.json at results dir; run-manifest.json at the
    per-subject×task subdir, stage='lev1', recording the passed inputs +
    exclusions source."""
    from network_glm.lev1 import run as lev1_run
    from network_glm import provenance as prov

    # Keep the helper from refusing on the real (possibly dirty) worktree.
    monkeypatch.setattr(prov, "git_is_dirty", lambda: False)

    results_dir = tmp_path / "results"
    base_dir = results_dir / "sub-s10" / "task-flanker"
    base_dir.mkdir(parents=True)

    ev = _touch(tmp_path / "events.tsv")
    cf = _touch(tmp_path / "confounds.tsv")
    excl = _touch(tmp_path / "compiled_exclusions.json", b"[]")

    args = SimpleNamespace(
        subj_id="sub-s10",
        task_name="flanker",
        bids_dir="/tmp/bids",
        fmriprep_dir="/tmp/fmriprep",
        results_dir=str(results_dir),
        exclusions_file=str(excl),
        allow_dirty=False,
    )
    dirs = {"base": base_dir}

    lev1_run._write_lev1_provenance(results_dir, args, dirs, [ev, cf])

    dd = results_dir / "dataset_description.json"
    rm = base_dir / "run-manifest.json"
    assert dd.exists(), "dataset_description.json must be written at the results dir"
    assert rm.exists(), "run-manifest.json must be written at the per-subjectxtask subdir"

    d = json.loads(dd.read_text())
    assert d["Name"] == "lev1"
    assert d["DatasetType"] == "derivative"

    m = json.loads(rm.read_text())
    assert m["stage"] == "lev1"
    recorded = {Path(i["path"]) for i in m["inputs"]}
    assert recorded == {ev, cf}
    assert m["exclusions_source"]["path"] == str(excl)


def test_write_lev1_provenance_passes_allow_dirty_through(tmp_path, monkeypatch):
    """allow_dirty=False on args with a dirty tree must surface (fail loud)."""
    from network_glm.lev1 import run as lev1_run
    from network_glm import provenance as prov

    monkeypatch.setattr(prov, "git_is_dirty", lambda: True)

    results_dir = tmp_path / "results"
    base_dir = results_dir / "sub-s10" / "task-flanker"
    base_dir.mkdir(parents=True)
    excl = _touch(tmp_path / "excl.json", b"[]")

    args = SimpleNamespace(
        subj_id="sub-s10",
        task_name="flanker",
        bids_dir="/tmp/bids",
        fmriprep_dir="/tmp/fmriprep",
        results_dir=str(results_dir),
        exclusions_file=str(excl),
        allow_dirty=False,
    )
    dirs = {"base": base_dir}

    with pytest.raises(RuntimeError):
        lev1_run._write_lev1_provenance(results_dir, args, dirs, [])


def test_write_lev1_provenance_allow_dirty_true_proceeds(tmp_path, monkeypatch):
    """allow_dirty=True writes the manifest even on a dirty tree."""
    from network_glm.lev1 import run as lev1_run
    from network_glm import provenance as prov

    monkeypatch.setattr(prov, "git_is_dirty", lambda: True)

    results_dir = tmp_path / "results"
    base_dir = results_dir / "sub-s10" / "task-flanker"
    base_dir.mkdir(parents=True)
    excl = _touch(tmp_path / "excl.json", b"[]")

    args = SimpleNamespace(
        subj_id="sub-s10",
        task_name="flanker",
        bids_dir="/tmp/bids",
        fmriprep_dir="/tmp/fmriprep",
        results_dir=str(results_dir),
        exclusions_file=str(excl),
        allow_dirty=True,
    )
    dirs = {"base": base_dir}

    lev1_run._write_lev1_provenance(results_dir, args, dirs, [])
    m = json.loads((base_dir / "run-manifest.json").read_text())
    assert m["code_dirty"] is True


def test_lev1_parser_has_allow_dirty_default_false():
    """The lev1 CLI exposes --allow-dirty defaulting to False."""
    from network_glm.lev1.run import get_parser

    parser = get_parser()
    ns = parser.parse_args(
        [
            "--subj-id",
            "s10",
            "--task-name",
            "flanker",
            "--bids-dir",
            "/b",
            "--fmriprep-dir",
            "/f",
            "--exclusions-file",
            "/e.json",
        ]
    )
    assert hasattr(ns, "allow_dirty")
    assert ns.allow_dirty is False

    ns2 = parser.parse_args(
        [
            "--subj-id",
            "s10",
            "--task-name",
            "flanker",
            "--bids-dir",
            "/b",
            "--fmriprep-dir",
            "/f",
            "--exclusions-file",
            "/e.json",
            "--allow-dirty",
        ]
    )
    assert ns2.allow_dirty is True
