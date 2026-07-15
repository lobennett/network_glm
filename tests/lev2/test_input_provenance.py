"""Tests for lev2 input-provenance-chain closure (PR4c, behavior-preserving).

lev2 inputs are lev1 fixed-effects files. PR4b writes a per-subject×task
``run-manifest.json`` one level above the ``fixed_effects/`` dir
(``<results>/<subj>/task-<task>/run-manifest.json``). PR4c reads those
manifests and summarizes the distinct ``exclusions_source`` SHA / ``code_sha`` /
``config_version`` across the selected inputs, WARNING loudly when they are
provenance-inconsistent (mixed exclusion sets / code versions / configs).

Selection is UNCHANGED by PR4c: ``discover_input_files`` still drops
``_desc-belowMinRuns_`` files via substring; that behavior is pinned in
``test_run.py`` and re-pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path


def _write_manifest(
    base_dir: Path,
    *,
    code_sha: str,
    config_version: str,
    excl_sha: str | None,
) -> Path:
    """Write a minimal lev1 run-manifest.json at ``base_dir`` (the dir that is
    the PARENT of ``fixed_effects/``)."""
    base_dir.mkdir(parents=True, exist_ok=True)
    excl_block = None if excl_sha is None else {"path": "compiled.json", "sha256": excl_sha}
    manifest = {
        "stage": "lev1",
        "code_sha": code_sha,
        "config_version": config_version,
        "exclusions_source": excl_block,
    }
    p = base_dir / "run-manifest.json"
    p.write_text(json.dumps(manifest, indent=2))
    return p


def _fe_file(base_dir: Path, name: str) -> Path:
    """Create a fixed-effects file under ``base_dir/fixed_effects/`` and return it."""
    fe_dir = base_dir / "fixed_effects"
    fe_dir.mkdir(parents=True, exist_ok=True)
    f = fe_dir / name
    f.write_bytes(b"")
    return f


def test_read_input_provenance_summarizes_and_dedupes(tmp_path):
    """Across inputs sharing the same SHAs, the summary records ONE distinct
    value per field (deduped) and reports consistency."""
    from network_glm.lev2.run import _read_input_provenance

    base1 = tmp_path / "sub-s03" / "task-flanker"
    base2 = tmp_path / "sub-s10" / "task-flanker"
    _write_manifest(base1, code_sha="abc123", config_version="v1", excl_sha="dead00")
    _write_manifest(base2, code_sha="abc123", config_version="v1", excl_sha="dead00")
    f1 = _fe_file(base1, "sub-s03_fe.nii.gz")
    f2 = _fe_file(base2, "sub-s10_fe.nii.gz")

    summary = _read_input_provenance([str(f1), str(f2)])

    assert set(summary["code_sha"]) == {"abc123"}
    assert set(summary["config_version"]) == {"v1"}
    assert set(summary["exclusions_source"]) == {"dead00"}
    assert summary["consistent"] is True
    assert summary["n_inputs"] == 2
    assert summary["n_manifests_found"] == 2


def test_read_input_provenance_missing_manifest_is_unknown(tmp_path):
    """A fixed-effects file with no sibling run-manifest contributes 'unknown'
    and does NOT crash."""
    from network_glm.lev2.run import _read_input_provenance

    base = tmp_path / "sub-s03" / "task-flanker"
    # No manifest written.
    f = _fe_file(base, "sub-s03_fe.nii.gz")

    summary = _read_input_provenance([str(f)])

    assert "unknown" in summary["code_sha"]
    assert "unknown" in summary["config_version"]
    assert "unknown" in summary["exclusions_source"]
    assert summary["n_inputs"] == 1
    assert summary["n_manifests_found"] == 0


def test_read_input_provenance_detects_inconsistency(tmp_path):
    """Mixed code SHAs / exclusion sets across inputs => consistent=False."""
    from network_glm.lev2.run import _read_input_provenance

    base1 = tmp_path / "sub-s03" / "task-flanker"
    base2 = tmp_path / "sub-s10" / "task-flanker"
    _write_manifest(base1, code_sha="abc123", config_version="v1", excl_sha="dead00")
    _write_manifest(base2, code_sha="zzz999", config_version="v1", excl_sha="beef11")
    f1 = _fe_file(base1, "sub-s03_fe.nii.gz")
    f2 = _fe_file(base2, "sub-s10_fe.nii.gz")

    summary = _read_input_provenance([str(f1), str(f2)])

    assert set(summary["code_sha"]) == {"abc123", "zzz999"}
    assert set(summary["exclusions_source"]) == {"dead00", "beef11"}
    assert summary["consistent"] is False


def test_read_input_provenance_empty_inputs(tmp_path):
    """No inputs => trivially consistent, empty distinct sets, no crash."""
    from network_glm.lev2.run import _read_input_provenance

    summary = _read_input_provenance([])

    assert summary["n_inputs"] == 0
    assert summary["n_manifests_found"] == 0
    assert summary["consistent"] is True
    assert summary["code_sha"] == []


def test_read_input_provenance_handles_missing_exclusions_block(tmp_path):
    """A manifest with exclusions_source=None records 'none' for that field
    (distinct from 'unknown', which means the manifest itself was absent)."""
    from network_glm.lev2.run import _read_input_provenance

    base = tmp_path / "sub-s03" / "task-flanker"
    _write_manifest(base, code_sha="abc123", config_version="v1", excl_sha=None)
    f = _fe_file(base, "sub-s03_fe.nii.gz")

    summary = _read_input_provenance([str(f)])

    assert "none" in summary["exclusions_source"]
    assert "unknown" not in summary["exclusions_source"]


def test_write_lev2_provenance_warns_on_inconsistent_inputs(tmp_path, monkeypatch, capsys):
    """When selected inputs come from mixed exclusion sets / code versions,
    _write_lev2_provenance prints a loud stderr WARNING and records the
    summary in the lev2 manifest under 'input_provenance'."""
    from network_glm.lev2 import run as lev2_run
    from network_glm import provenance as prov

    monkeypatch.setattr(prov, "git_is_dirty", lambda: False)

    output_dir = tmp_path / "lev2_out"
    output_dir.mkdir(parents=True)
    lev1 = tmp_path / "lev1"

    base1 = lev1 / "sub-s03" / "task-flanker"
    base2 = lev1 / "sub-s10" / "task-flanker"
    _write_manifest(base1, code_sha="abc123", config_version="v1", excl_sha="dead00")
    _write_manifest(base2, code_sha="zzz999", config_version="v1", excl_sha="beef11")
    f1 = _fe_file(base1, "sub-s03_fe.nii.gz")
    f2 = _fe_file(base2, "sub-s10_fe.nii.gz")

    from types import SimpleNamespace

    args = SimpleNamespace(
        contrast="task-flanker_contrast-incongruent-congruent",
        level1_dirs=[str(lev1)],
        output_dir=str(output_dir),
        allow_dirty=False,
    )

    lev2_run._write_lev2_provenance(output_dir, args, [lev1], [str(f1), str(f2)])

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "provenance" in err.lower()

    m = json.loads((output_dir / "run-manifest.json").read_text())
    assert "input_provenance" in m
    ip = m["input_provenance"]
    assert ip["consistent"] is False
    assert set(ip["code_sha"]) == {"abc123", "zzz999"}


def test_write_lev2_provenance_no_warning_when_consistent(tmp_path, monkeypatch, capsys):
    """Consistent inputs => no inconsistency WARNING, summary still recorded."""
    from network_glm.lev2 import run as lev2_run
    from network_glm import provenance as prov

    monkeypatch.setattr(prov, "git_is_dirty", lambda: False)

    output_dir = tmp_path / "lev2_out"
    output_dir.mkdir(parents=True)
    lev1 = tmp_path / "lev1"

    base1 = lev1 / "sub-s03" / "task-flanker"
    base2 = lev1 / "sub-s10" / "task-flanker"
    _write_manifest(base1, code_sha="abc123", config_version="v1", excl_sha="dead00")
    _write_manifest(base2, code_sha="abc123", config_version="v1", excl_sha="dead00")
    f1 = _fe_file(base1, "sub-s03_fe.nii.gz")
    f2 = _fe_file(base2, "sub-s10_fe.nii.gz")

    from types import SimpleNamespace

    args = SimpleNamespace(
        contrast="task-flanker_contrast-incongruent-congruent",
        level1_dirs=[str(lev1)],
        output_dir=str(output_dir),
        allow_dirty=False,
    )

    lev2_run._write_lev2_provenance(output_dir, args, [lev1], [str(f1), str(f2)])

    err = capsys.readouterr().err
    # No inconsistency warning (the only warnings here would be dirty-tree,
    # which is monkeypatched off).
    assert "inconsistent" not in err.lower()
    assert "mixed" not in err.lower()

    m = json.loads((output_dir / "run-manifest.json").read_text())
    assert m["input_provenance"]["consistent"] is True
