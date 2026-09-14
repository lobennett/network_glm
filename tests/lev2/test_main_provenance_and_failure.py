"""B5 / B5b regression tests for lev2 main().

B5  — provenance is written into the per-contrast output dir
      ({results_dir}/{contrast}/), not the shared {results_dir} root, so the
      SLURM array (one contrast per task, all sharing --output-dir) no longer
      clobbers/races a single root manifest.
B5b — a failed randomise propagates: run_level2_analysis returns False, and
      main() returns non-zero WITHOUT stamping a success provenance manifest.
"""

from __future__ import annotations

import sys
from pathlib import Path

from network_glm.lev2 import run as lev2_run
from tests.lev2.helpers import CONTRAST, effect


def _argv(monkeypatch, results_dir, contrast):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lev2",
            "--contrast",
            contrast,
            "--level1-dirs",
            str(results_dir / "lev1"),
            "--output-dir",
            str(results_dir),
        ],
    )


def test_main_writes_provenance_into_per_contrast_dir(tmp_path, monkeypatch, fake_randomise):
    results_dir = tmp_path / "lev2_out"
    (results_dir / "lev1").mkdir(parents=True)
    contrast = CONTRAST
    _argv(monkeypatch, results_dir, contrast)

    monkeypatch.setattr(lev2_run.provenance, "git_is_dirty", lambda: False)
    effect(results_dir / "lev1", "s03")

    captured = {}

    def _spy_prov(output_dir, args, level1_dirs, input_files, **kwargs):
        captured["dir"] = Path(output_dir)

    monkeypatch.setattr(lev2_run, "_write_lev2_provenance", _spy_prov)

    rc = lev2_run.main()
    assert rc == 0
    # Provenance must be written into the per-contrast subdir, not the root.
    assert captured["dir"].name.startswith(f".{contrast}.attempt-")
    assert (results_dir / contrast).is_dir()


def test_main_returns_nonzero_and_skips_provenance_on_analysis_failure(tmp_path, monkeypatch):
    results_dir = tmp_path / "lev2_out"
    (results_dir / "lev1").mkdir(parents=True)
    contrast = "task-flanker_contrast-incongruent-congruent"
    _argv(monkeypatch, results_dir, contrast)

    monkeypatch.setattr(lev2_run.provenance, "git_is_dirty", lambda: False)
    monkeypatch.setattr(lev2_run, "discover_input_files", lambda dirs, c: ["/fe1.nii.gz"])
    monkeypatch.setattr(lev2_run, "run_level2_analysis", lambda *a, **k: False)

    called = {"prov": False}
    monkeypatch.setattr(
        lev2_run,
        "_write_lev2_provenance",
        lambda *a, **k: called.__setitem__("prov", True),
    )

    rc = lev2_run.main()
    assert rc != 0
    assert called["prov"] is False  # no success manifest stamped for a failed run


def test_run_level2_analysis_returns_false_on_randomise_failure(tmp_path, monkeypatch, fake_randomise):
    monkeypatch.setenv("TEST_MODE", "fail")
    files = [effect(tmp_path / "lev1", "s03")]
    assert lev2_run.run_level2_analysis(CONTRAST, files, tmp_path / "out") is False


def test_run_level2_analysis_returns_true_on_success(tmp_path, fake_randomise):
    files = [effect(tmp_path / "lev1", "s03")]
    assert lev2_run.run_level2_analysis(CONTRAST, files, tmp_path / "out") is True


def test_zero_exit_without_a_corrected_map_is_not_success(tmp_path, monkeypatch, fake_randomise):
    monkeypatch.setenv("TEST_MODE", "noop")
    files = [effect(tmp_path / "lev1", "s03")]
    assert lev2_run.run_level2_analysis(CONTRAST, files, tmp_path / "out") is False


def test_run_level2_analysis_returns_false_on_no_inputs(tmp_path):
    assert lev2_run.run_level2_analysis("c", [], tmp_path) is False


def test_run_level2_analysis_forwards_seed_to_randomise(tmp_path, monkeypatch, fake_randomise):
    """Exercise the helper API variant that accepts a seed, using real preparation."""
    import randomise_prep

    original = randomise_prep.setup_randomise_tfce
    captured = {}

    def setup(*, seed, **kwargs):
        captured["seed"] = seed
        path = original(**kwargs)
        lev2_run._inject_seed(Path(path), seed)
        return path

    monkeypatch.setattr(randomise_prep, "setup_randomise_tfce", setup)
    files = [effect(tmp_path / "lev1", "s03")]
    assert lev2_run.run_level2_analysis(CONTRAST, files, tmp_path / "out", seed=4242)
    assert captured["seed"] == 4242


def test_mask_threshold_defaults_match_cli(monkeypatch):
    """B8: compute_mask / run_level2_analysis default mask_threshold must equal
    the lev2 CLI default (0.9), so a direct caller that omits it does not
    silently get a strict (1.0) all-subjects intersection."""
    import inspect

    cli_default = (
        lev2_run.get_parser().parse_args(["--contrast", "c", "--level1-dirs", "/a"]).mask_threshold
    )
    assert cli_default == 0.9

    sig = inspect.signature
    assert sig(lev2_run.compute_mask).parameters["threshold"].default == cli_default
    assert sig(lev2_run.run_level2_analysis).parameters["mask_threshold"].default == cli_default


def test_compute_mask_connected_defaults_false():
    """J4: the group mask must keep every voxel meeting the coverage threshold
    (connected=False), not just the largest connected component."""
    import inspect

    assert inspect.signature(lev2_run.compute_mask).parameters["connected"].default is False
