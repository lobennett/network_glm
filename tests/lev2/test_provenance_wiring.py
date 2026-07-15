"""Tests for the lev2 provenance-writing helper (PR4b, additive provenance).

Exercises ``_write_lev2_provenance``, which writes a BIDS
``dataset_description.json`` (naming the lev1 source dirs in SourceDatasets)
plus a ``run-manifest.json`` (stage='lev2', recording the discovered lev1
input files) at the run's output dir. Written red-first.
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


def test_write_lev2_provenance_writes_both_files(tmp_path, monkeypatch):
    """dataset_description names the lev1 source dirs; run-manifest records the
    discovered lev1 input files at stage='lev2'."""
    from network_glm.lev2 import run as lev2_run
    from network_glm import provenance as prov

    monkeypatch.setattr(prov, "git_is_dirty", lambda: False)

    output_dir = tmp_path / "lev2_out"
    output_dir.mkdir(parents=True)
    lev1_a = tmp_path / "lev1_discovery"
    lev1_b = tmp_path / "lev1_validation"
    lev1_a.mkdir()
    lev1_b.mkdir()

    f1 = _touch(tmp_path / "fe1.nii.gz")
    f2 = _touch(tmp_path / "fe2.nii.gz")

    args = SimpleNamespace(
        contrast="task-flanker_contrast-incongruent-congruent",
        level1_dirs=[str(lev1_a), str(lev1_b)],
        output_dir=str(output_dir),
        allow_dirty=False,
    )

    lev2_run._write_lev2_provenance(output_dir, args, [lev1_a, lev1_b], [str(f1), str(f2)])

    dd = output_dir / "dataset_description.json"
    rm = output_dir / "run-manifest.json"
    assert dd.exists()
    assert rm.exists()

    d = json.loads(dd.read_text())
    assert d["Name"] == "lev2"
    assert d["DatasetType"] == "derivative"
    # SourceDatasets names the lev1 dirs
    flat = json.dumps(d["SourceDatasets"])
    assert str(lev1_a) in flat
    assert str(lev1_b) in flat

    m = json.loads(rm.read_text())
    assert m["stage"] == "lev2"
    recorded = {Path(i["path"]) for i in m["inputs"]}
    assert recorded == {f1, f2}


def test_write_lev2_provenance_respects_allow_dirty_false(tmp_path, monkeypatch):
    """A dirty tree with allow_dirty=False surfaces (fail loud)."""
    from network_glm.lev2 import run as lev2_run
    from network_glm import provenance as prov

    monkeypatch.setattr(prov, "git_is_dirty", lambda: True)

    output_dir = tmp_path / "lev2_out"
    output_dir.mkdir(parents=True)
    lev1_a = tmp_path / "lev1"
    lev1_a.mkdir()

    args = SimpleNamespace(
        contrast="c",
        level1_dirs=[str(lev1_a)],
        output_dir=str(output_dir),
        allow_dirty=False,
    )

    with pytest.raises(RuntimeError):
        lev2_run._write_lev2_provenance(output_dir, args, [lev1_a], [])


def test_lev2_parser_has_allow_dirty_default_false():
    """The lev2 CLI exposes --allow-dirty defaulting to False."""
    from network_glm.lev2.run import get_parser

    parser = get_parser()
    ns = parser.parse_args(
        [
            "--contrast",
            "c",
            "--level1-dirs",
            "/a",
            "/b",
        ]
    )
    assert hasattr(ns, "allow_dirty")
    assert ns.allow_dirty is False

    ns2 = parser.parse_args(
        [
            "--contrast",
            "c",
            "--level1-dirs",
            "/a",
            "--allow-dirty",
        ]
    )
    assert ns2.allow_dirty is True
