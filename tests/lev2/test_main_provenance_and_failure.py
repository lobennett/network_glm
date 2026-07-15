"""B5 / B5b regression tests for lev2 main().

B5  — provenance is written into the per-contrast output dir
      ({results_dir}/{contrast}/), not the shared {results_dir} root, so the
      SLURM array (one contrast per task, all sharing --output-dir) no longer
      clobbers/races a single root manifest.
B5b — a failed randomise propagates: run_level2_analysis returns False, and
      main() returns non-zero WITHOUT stamping a success provenance manifest.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

from network_glm.lev2 import run as lev2_run


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


def test_main_writes_provenance_into_per_contrast_dir(tmp_path, monkeypatch):
    results_dir = tmp_path / "lev2_out"
    (results_dir / "lev1").mkdir(parents=True)
    contrast = "task-flanker_contrast-incongruent-congruent"
    _argv(monkeypatch, results_dir, contrast)

    monkeypatch.setattr(lev2_run.provenance, "git_is_dirty", lambda: False)
    monkeypatch.setattr(lev2_run, "discover_input_files", lambda dirs, c: ["/fe1.nii.gz"])

    def _fake_analysis(contrast_name, input_files, output_dir, *a, **k):
        (Path(output_dir) / contrast_name).mkdir(parents=True, exist_ok=True)
        return True

    monkeypatch.setattr(lev2_run, "run_level2_analysis", _fake_analysis)

    captured = {}

    def _spy_prov(output_dir, args, level1_dirs, input_files):
        captured["dir"] = Path(output_dir)

    monkeypatch.setattr(lev2_run, "_write_lev2_provenance", _spy_prov)

    rc = lev2_run.main()
    assert rc == 0
    # Provenance must be written into the per-contrast subdir, not the root.
    assert captured["dir"] == results_dir / contrast


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


def _stub_randomise(monkeypatch, raises: bool):
    """Stub compute_mask + randomise_prep + subprocess so run_level2_analysis
    runs without FSL. If raises, the randomise subprocess fails."""
    monkeypatch.setattr(
        lev2_run,
        "compute_mask",
        lambda files, threshold=1.0: types.SimpleNamespace(
            to_filename=lambda p: Path(p).write_bytes(b"")
        ),
    )
    fake_mod = types.ModuleType("randomise_prep")
    fake_mod.setup_randomise_tfce = lambda **k: "/tmp/fake_randomise.sh"
    monkeypatch.setitem(sys.modules, "randomise_prep", fake_mod)

    def _run(cmd, **k):
        if raises:
            raise subprocess.CalledProcessError(1, cmd, output="out", stderr="boom")
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(lev2_run.subprocess, "run", _run)


def test_run_level2_analysis_returns_false_on_randomise_failure(tmp_path, monkeypatch):
    _stub_randomise(monkeypatch, raises=True)
    ok = lev2_run.run_level2_analysis("c", ["/fe1.nii.gz"], tmp_path)
    assert ok is False


def test_run_level2_analysis_returns_true_on_success(tmp_path, monkeypatch):
    _stub_randomise(monkeypatch, raises=False)
    ok = lev2_run.run_level2_analysis("c", ["/fe1.nii.gz"], tmp_path)
    assert ok is True


def test_run_level2_analysis_returns_false_on_no_inputs(tmp_path):
    assert lev2_run.run_level2_analysis("c", [], tmp_path) is False


def test_run_level2_analysis_forwards_seed_to_randomise(tmp_path, monkeypatch):
    """The seed is passed to setup_randomise_tfce when its signature accepts it
    (here a **kwargs stub), pinning randomise's permutation RNG."""
    monkeypatch.setattr(
        lev2_run,
        "compute_mask",
        lambda files, threshold=1.0: types.SimpleNamespace(
            to_filename=lambda p: Path(p).write_bytes(b"")
        ),
    )
    captured = {}
    fake_mod = types.ModuleType("randomise_prep")

    def _setup(**kwargs):
        captured.update(kwargs)
        return "/tmp/fake_randomise.sh"

    fake_mod.setup_randomise_tfce = _setup
    monkeypatch.setitem(sys.modules, "randomise_prep", fake_mod)
    monkeypatch.setattr(
        lev2_run.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "ok", "")
    )

    ok = lev2_run.run_level2_analysis("c", ["/fe1.nii.gz"], tmp_path, seed=4242)
    assert ok is True
    assert captured.get("seed") == 4242


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
