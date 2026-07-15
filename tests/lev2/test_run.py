"""Tests for src/network_glm/lev2/run.py."""

from __future__ import annotations

from pathlib import Path


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_discover_input_files_filters_below_min_runs(tmp_path):
    """Files containing _desc-belowMinRuns_ are dropped from the result."""
    from network_glm.lev2.run import discover_input_files

    lev1_dir = tmp_path / "lev1"
    contrast_name = "task-flanker_contrast-incongruent-congruent"
    common = f"{contrast_name}_rtmodel-RTDur"

    # Two untagged (kept), two tagged (dropped).
    _touch(
        lev1_dir
        / "sub-s03"
        / "ses-01"
        / "fixed_effects"
        / f"sub-s03_{common}_stat-fixed-effects.nii.gz"
    )
    _touch(
        lev1_dir
        / "sub-s10"
        / "ses-01"
        / "fixed_effects"
        / f"sub-s10_{common}_stat-fixed-effects.nii.gz"
    )
    _touch(
        lev1_dir
        / "sub-s19"
        / "ses-01"
        / "fixed_effects"
        / f"sub-s19_{common}_desc-belowMinRuns_stat-fixed-effects.nii.gz"
    )
    _touch(
        lev1_dir
        / "sub-s29"
        / "ses-01"
        / "fixed_effects"
        / f"sub-s29_{common}_desc-belowMinRuns_stat-fixed-effects.nii.gz"
    )

    files = discover_input_files([lev1_dir], contrast_name)

    assert len(files) == 2
    subjects = {Path(f).name.split("_")[0] for f in files}
    assert subjects == {"sub-s03", "sub-s10"}
    for f in files:
        assert "_desc-belowMinRuns_" not in f
