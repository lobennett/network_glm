"""Synthetic consumer checks; the fake executable does not implement FSL statistics."""

import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from network_glm.lev2 import run

from tests.lev2.helpers import CONTRAST, CORRP, effect, save


@pytest.mark.parametrize("alias", [False, True])
def test_repeated_roots_rejected_with_paths(tmp_path, alias):
    root = tmp_path / "lev1"
    filename = effect(root, "s03")
    second = root
    if alias:
        second = tmp_path / "alias"
        second.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="[Rr]epeated.*root") as error:
        run.discover_input_files([root, second], CONTRAST)
    assert str(root) in str(error.value)
    assert str(second) in str(error.value)
    assert filename in str(error.value)


@pytest.mark.parametrize("same_root", [False, True])
def test_duplicate_subjects_rejected_without_choosing(tmp_path, same_root):
    a = tmp_path / "a"
    b = a if same_root else tmp_path / "b"
    first = effect(a, "s03")
    second = effect(b, "s03", arm="noRT" if same_root else "RTDur")
    with pytest.raises(ValueError) as error:
        run.discover_input_files([a] if same_root else [a, b], CONTRAST)
    assert first in str(error.value) and second in str(error.value)


def test_mixed_arms_across_distinct_subjects_rejected(tmp_path):
    a = effect(tmp_path, "s03")
    b = effect(tmp_path, "s20", arm="noRT")
    with pytest.raises(ValueError, match="RT") as error:
        run.discover_input_files([tmp_path], CONTRAST)
    assert a in str(error.value) and b in str(error.value)


def test_file_alias_cannot_create_another_subject(tmp_path):
    first = effect(tmp_path / "lev1", "s03")
    second = Path(effect(tmp_path / "lev1", "s20"))
    second.unlink()
    second.symlink_to(first)
    with pytest.raises(ValueError) as error:
        run.run_level2_analysis(CONTRAST, [first, str(second)], tmp_path / "output")
    assert first in str(error.value) and str(second) in str(error.value)


@pytest.mark.parametrize("kind", ["duplicate", "task", "contrast", "arm", "subject-directory"])
def test_direct_call_validates_identity_before_outputs(tmp_path, kind):
    first = effect(tmp_path / "lev1", "s03")
    if kind == "duplicate":
        second = first
    else:
        second = effect(tmp_path / "lev1", "s20",
                        arm="noRT" if kind == "arm" else "RTDur",
                        contrast={"task": "task-other_contrast-task",
                                  "contrast": "task-rest_contrast-other"}.get(kind, CONTRAST))
        if kind == "subject-directory":
            moved = Path(second).with_name(Path(second).name.replace("sub-s20", "sub-s03"))
            Path(second).rename(moved)
            second = str(moved)
    output = tmp_path / "outputs"
    with pytest.raises(ValueError) as error:
        run.run_level2_analysis(CONTRAST, [first, second], output)
    assert second in str(error.value)
    assert not output.exists()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_consumed_map_rejected_before_mask_or_prep(tmp_path, value):
    first = effect(tmp_path / "lev1", "s03")
    second = effect(tmp_path / "lev1", "s20")
    data = np.ones((3, 3, 3)); data[1, 2, 0] = value
    save(Path(second), data)
    output = tmp_path / "outputs"
    with pytest.raises(ValueError, match="nonfinite") as error:
        run.run_level2_analysis(CONTRAST, [first, second], output)
    assert second in str(error.value)
    assert "1, 2, 0" in str(error.value)
    assert not output.exists()
    with pytest.raises(ValueError, match="nonfinite"):
        run.compute_mask([first, second])


@pytest.mark.parametrize("kind", ["4d", "affine"])
def test_existing_geometry_rejections_preserved(tmp_path, kind):
    first = effect(tmp_path / "lev1", "s03")
    second = effect(tmp_path / "lev1", "s20")
    save(Path(second), np.ones((3, 3, 3, 2)) if kind == "4d" else None,
         np.diag([2, 2, 2, 1]) if kind == "affine" else None)
    with pytest.raises(ValueError):
        run.run_level2_analysis(CONTRAST, [first, second], tmp_path / "outputs")
    assert not (tmp_path / "outputs").exists()


def test_coverage_is_strictly_more_than_nine_of_ten_and_keeps_disconnected_voxels(tmp_path):
    files = []
    for i in range(10):
        data = np.zeros((3, 3, 3))
        data[0, 0, 0] = 1
        data[2, 2, 2] = 1
        data[1, 1, 1] = i != 9
        files.append(save(tmp_path / f"{i}.nii.gz", data))
    mask = run.compute_mask(files).get_fdata()
    assert mask[1, 1, 1] == 0
    assert mask[0, 0, 0] == mask[2, 2, 2] == 1
    assert run.compute_mask(files, threshold=1).get_fdata().sum() == 2


def test_disjoint_roots_keep_4d_order_roster_and_two_sided_command(tmp_path, fake_randomise):
    a, b = tmp_path / "a", tmp_path / "b"
    first = effect(a, "s20", 20)
    second = effect(b, "s03", 3)
    files = run.discover_input_files([b, a], CONTRAST)
    assert files == [first, second]
    output = tmp_path / "outputs"
    assert run.main(["--contrast", CONTRAST, "--level1-dirs", str(b), str(a),
                     "--output-dir", str(output), "--seed", "42",
                     "--num-permutations", "10", "--allow-dirty"]) == 0
    result = output / CONTRAST
    np.testing.assert_array_equal(nib.load(result / "input_data4d.nii.gz").get_fdata()[0, 0, 0], [20, 3])
    manifest = json.loads((result / "run-manifest.json").read_text())
    roster = manifest["observations"]
    assert [row["volume_index"] for row in roster] == [0, 1]
    assert [row["subject"] for row in roster] == ["sub-s20", "sub-s03"]
    assert [row["path"] for row in roster] == files
    assert all(row["rt_model"] == "RTDur" for row in roster)
    assert [row["path"] for row in manifest["inputs"]] == files
    assert (result / "design.con").read_text().split("/Matrix\n")[1].strip() == "1"
    assert (result / "design.fts").read_text().split("/Matrix\n")[1].strip() == "1.0"
    assert not (result / "design.mat").exists()
    calls = [json.loads(line) for line in fake_randomise.read_text().splitlines()]
    assert len(calls) == 2
    assert {"--seed=42", "-1", "--fonly", "-T"} <= set(calls[0])
    assert calls[0][calls[0].index("-n") + 1] == "10"
    assert "-R" in calls[1] and "--seed=42" not in calls[1]
    # The relocated script remains usable against the published preparation.
    script = (result / "randomise_call.sh").read_text()
    assert str(result / "group_mask.nii.gz") in script


@pytest.mark.parametrize("mode", ["noop", "fail", "missing-t", "wrong-corrected-name",
                                 "unreadable", "shape", "affine", "nan", "inf",
                                 "missing-preparation"])
def test_failed_attempt_preserves_every_previous_output(tmp_path, monkeypatch, fake_randomise, mode):
    files = [effect(tmp_path / "lev1", "s03"), effect(tmp_path / "lev1", "s20")]
    output = tmp_path / "outputs"
    old = output / CONTRAST
    save(old / CORRP, np.ones((1, 1, 1)))  # stale, deliberately incompatible
    (old / "group_mask.nii.gz").write_bytes(b"previous mask")
    (old / "run-manifest.json").write_text('{"previous": true}')
    before = {p.name: p.read_bytes() for p in old.iterdir()}
    monkeypatch.setenv("TEST_MODE", mode)
    bad = tmp_path / "bad-product.nii.gz"
    if mode == "unreadable":
        bad.write_bytes(b"not an image")
    elif mode in {"shape", "affine", "nan", "inf"}:
        save(bad, np.ones((1, 1, 1)) if mode == "shape" else
             np.full((3, 3, 3), {"nan": np.nan, "inf": np.inf}.get(mode, 1)),
             np.diag([2, 2, 2, 1]) if mode == "affine" else None)
    if bad.exists():
        monkeypatch.setenv("TEST_PRODUCT", str(bad))
    assert run.run_level2_analysis(CONTRAST, files, output, num_permutations=10) is False
    assert {p.name: p.read_bytes() for p in old.iterdir()} == before
    if mode == "fail":
        assert len(fake_randomise.read_text().splitlines()) == 1


def test_successful_replacement_preserves_prior_directory(tmp_path, fake_randomise):
    files = [effect(tmp_path / "lev1", "s03"), effect(tmp_path / "lev1", "s20")]
    output = tmp_path / "outputs"
    old = output / CONTRAST
    old.mkdir(parents=True)
    (old / "historical.txt").write_text("prior result")
    assert run.run_level2_analysis(CONTRAST, files, output, num_permutations=10)
    assert not (old / "historical.txt").exists()
    assert len(list(output.glob(f".{CONTRAST}.previous-*/historical.txt"))) == 1


def cli_args(output, *extra):
    """A real parsed CLI invocation, the only provenance context lev2 accepts."""
    return run.get_parser().parse_args(
        ["--contrast", CONTRAST, "--level1-dirs", str(output), "--output-dir", str(output), *extra]
    )


def test_manifest_failure_cannot_replace_previous_results(tmp_path, monkeypatch, fake_randomise):
    files = [effect(tmp_path / "lev1", "s03")]
    output = tmp_path / "outputs"
    old = output / CONTRAST
    old.mkdir(parents=True)
    (old / "run-manifest.json").write_text("previous manifest")

    def fail(*args, **kwargs):
        raise OSError("synthetic manifest write failure")

    monkeypatch.setattr(run.provenance, "write_run_manifest", fail)
    with pytest.raises(OSError, match="synthetic"):
        run.run_level2_analysis(CONTRAST, files, output, num_permutations=10,
                                provenance_args=cli_args(output, "--allow-dirty"))
    assert (old / "run-manifest.json").read_text() == "previous manifest"
    assert len(list(old.iterdir())) == 1


def test_cli_invalid_roster_returns_failure_without_outputs(tmp_path):
    root = tmp_path / "lev1"
    effect(root, "s03")
    output = tmp_path / "outputs"
    assert run.main(["--contrast", CONTRAST, "--level1-dirs", str(root), str(root),
                     "--output-dir", str(output)]) == 1
    assert not output.exists()


def test_direct_call_without_cli_context_invents_no_provenance(tmp_path, fake_randomise):
    """No invocation is fabricated: absent real args, the optional manifest is omitted."""
    files = [effect(tmp_path / "lev1", "s03"), effect(tmp_path / "lev1", "s20")]
    output = tmp_path / "outputs"
    assert run.run_level2_analysis(CONTRAST, files, output, num_permutations=10)
    published = output / CONTRAST
    assert (published / CORRP).is_file()
    assert not (published / "run-manifest.json").exists()
    assert not (published / "dataset_description.json").exists()


def test_cli_context_keeps_the_dirty_tree_guard(tmp_path, monkeypatch, fake_randomise):
    """A real invocation without --allow-dirty still refuses to stamp a dirty tree."""
    files = [effect(tmp_path / "lev1", "s03")]
    output = tmp_path / "outputs"
    old = output / CONTRAST
    old.mkdir(parents=True)
    (old / "historical.txt").write_text("prior result")
    monkeypatch.setattr(run.provenance, "git_is_dirty", lambda: True)
    with pytest.raises(RuntimeError, match="uncommitted"):
        run.run_level2_analysis(CONTRAST, files, output, num_permutations=10,
                                provenance_args=cli_args(output))
    assert (old / "historical.txt").read_text() == "prior result"


def test_published_directory_is_readable_by_the_umask_not_owner_only(tmp_path, fake_randomise):
    files = [effect(tmp_path / "lev1", "s03")]
    output = tmp_path / "outputs"
    umask = os.umask(0o022)
    os.umask(umask)
    assert run.run_level2_analysis(CONTRAST, files, output, num_permutations=10)
    assert (output / CONTRAST).stat().st_mode & 0o777 == 0o777 & ~umask


def test_replacement_keeps_the_prior_contrast_directory_mode(tmp_path, fake_randomise):
    files = [effect(tmp_path / "lev1", "s03")]
    output = tmp_path / "outputs"
    old = output / CONTRAST
    old.mkdir(parents=True)
    old.chmod(0o750)
    assert run.run_level2_analysis(CONTRAST, files, output, num_permutations=10)
    assert (output / CONTRAST).stat().st_mode & 0o777 == 0o750
