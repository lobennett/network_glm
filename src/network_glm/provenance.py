"""Reusable provenance primitives for analysis-stage outputs.

Generalizes the lockfile pattern already used by the source monolith's
exclusions-compiler (UTC ISO timestamp, git SHA, file paths
and counts) into composable helpers any stage can call to record:

- the exact code state (``git_sha`` / ``git_is_dirty``),
- the locked dependency set (``uv_lock_hash``, ``tool_versions``),
- the study-level config version (via :mod:`network_glm.thresholds`),
- the inputs consumed (``file_manifest`` with per-file sha256), and
- a machine-readable ``run-manifest.json`` plus a minimal valid BIDS
  ``dataset_description.json`` for derivative trees.

This module is the canonical home of the git-SHA helper. ``exclusions.base``
re-exports it as ``_git_sha`` so the committed exclusions lockfiles stay
byte-identical and all existing importers keep working (DRY, behavior-preserving).

PR4a adds the primitive + tests only; wiring it into lev1/lev2 is a later PR.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import UTC, datetime
from importlib import metadata as _ilmd
from pathlib import Path
from typing import Any

from network_glm.thresholds import config_version

# Repo root resolved from this file's location:
# src/network_glm/provenance.py -> parents[0]=network_glm, parents[1]=src,
# parents[2]=repo root.
# NOTE (network_glm lift-and-shift): the source monolith had this module one
# directory deeper (core/provenance.py under the old package root, parents[3]);
# the index below was adjusted to match this package's shallower layout.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# UTC ISO timestamp format, byte-identical to the exclusions lockfile.
_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Sensible default tool set recorded when a caller doesn't name packages.
_DEFAULT_TOOLS = ["neuro-workflow", "numpy", "nibabel", "nilearn", "pandas", "scipy"]


def _now_iso() -> str:
    """UTC timestamp in the same format as the exclusions lockfile."""
    return datetime.now(UTC).strftime(_ISO_FMT)


def git_sha() -> str:
    """Return the current git HEAD short SHA, with a ``+dirty`` suffix if the
    working tree has uncommitted changes. Returns ``"unknown"`` if git is
    unavailable or ``_REPO_ROOT`` is not a git repo.

    The subprocess runs with ``cwd=_REPO_ROOT`` so SLURM jobs launched from
    scratch dirs still resolve the correct repo (git fails when CWD is outside
    any repo, which would otherwise drop ``code_sha`` to a sentinel).
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    if not sha:
        return "unknown"
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                cwd=_REPO_ROOT,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return sha
    return f"{sha}+dirty" if dirty else sha


def git_is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes.

    Returns False when ``_REPO_ROOT`` is not a git repo or git is unavailable
    (a non-repo can't be "dirty"; callers wanting strictness check git_sha).
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=_REPO_ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return bool(out)


def uv_lock_hash() -> str:
    """Short sha256 (12 hex chars) of ``uv.lock`` at the repo root.

    Returns ``"unknown"`` if the lock file is absent. Lets a manifest pin the
    exact resolved dependency graph the run was executed against.
    """
    lock = _REPO_ROOT / "uv.lock"
    if not lock.is_file():
        return "unknown"
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:12]


def tool_versions(packages: list[str]) -> dict[str, str]:
    """Installed version per package via ``importlib.metadata.version``.

    Missing packages map to ``"not installed"`` rather than raising, so a
    manifest write never fails on an optional dependency.
    """
    out: dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = _ilmd.version(pkg)
        except _ilmd.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def fsl_version() -> str:
    """Best-effort FSL version string, or ``"unknown"``.

    FSL is not a Python package, so it is absent from :func:`tool_versions`.
    The volumetric lev2 path shells out to FSL ``randomise``; recording the FSL
    version closes the external-tool gap in the run manifest. Reads
    ``$FSLDIR/etc/fslversion`` (the canonical FSL version file); never raises.
    """
    import os

    fsldir = os.environ.get("FSLDIR")
    if not fsldir:
        return "unknown"
    vfile = Path(fsldir) / "etc" / "fslversion"
    try:
        return vfile.read_text().strip().split(":")[0] or "unknown"
    except OSError:
        return "unknown"


def _sha256_file(path: Path) -> str:
    """Stream a file through sha256 (chunked, so large NIfTIs don't blow RAM)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(paths: list[Path]) -> list[dict]:
    """Per-input record of ``{path, size_bytes, sha256}``.

    Fails loud (``FileNotFoundError``) on any missing input rather than
    silently dropping it, so a manifest never claims an input that wasn't
    actually consumed.
    """
    manifest: list[dict] = []
    for p in paths:
        p = Path(p)
        if not p.is_file():
            raise FileNotFoundError(f"file_manifest: input does not exist: {p}")
        manifest.append(
            {
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "sha256": _sha256_file(p),
            }
        )
    return manifest


def require_clean_tree(allow_dirty: bool = False) -> None:
    """Raise ``RuntimeError`` if the working tree is dirty and not allowed.

    Lets stages opt into refusing to write provenance against uncommitted code.
    """
    if not allow_dirty and git_is_dirty():
        raise RuntimeError(
            "refusing to proceed: git working tree has uncommitted changes "
            f"(repo root {_REPO_ROOT}). Commit/stash first, or pass allow_dirty=True."
        )


def write_run_manifest(
    output_dir: str | Path,
    *,
    stage: str,
    args: Any,
    inputs: list[Path] | None = None,
    exclusions_source: str | Path | None = None,
    tools: list[str] | None = None,
    allow_dirty: bool = True,
) -> Path:
    """Write ``<output_dir>/run-manifest.json`` capturing full run provenance.

    Records the code state, locked deps, config version, tool versions, the
    exclusions source (path + sha256), the JSON-safe args, a UTC timestamp,
    host info, the SLURM job id (or null), and a per-file manifest of inputs.

    Args:
        output_dir: directory to write ``run-manifest.json`` into (created).
        stage: pipeline stage label, e.g. ``"lev1"`` / ``"lev2"``.
        args: argparse Namespace or dict of run arguments (made JSON-safe).
        inputs: input files to hash into the manifest. Missing files raise.
        exclusions_source: optional compiled-exclusions file to pin (path+sha).
        tools: package names to record versions for. Defaults to a core set.
        allow_dirty: if False, raise when the working tree has uncommitted
            changes (delegates to :func:`require_clean_tree`).

    Returns:
        Path to the written ``run-manifest.json``.
    """
    require_clean_tree(allow_dirty=allow_dirty)

    # _jsonify keeps one JSON-safe serialization path; vendored into
    # network_glm.exclusions alongside the other lockfile helpers.
    from network_glm.exclusions import _jsonify

    if args is None:
        args_dict: Any = None
    elif hasattr(args, "__dict__") and not isinstance(args, dict):
        args_dict = vars(args)
    else:
        args_dict = dict(args)

    excl_block = None
    if exclusions_source is not None:
        excl_path = Path(exclusions_source)
        excl_block = {
            "path": str(excl_path),
            "sha256": _sha256_file(excl_path) if excl_path.is_file() else None,
        }

    uname = os.uname()
    manifest = {
        "stage": stage,
        "code_sha": git_sha(),
        "code_dirty": git_is_dirty(),
        "uv_lock_hash": uv_lock_hash(),
        "config_version": config_version(),
        "tool_versions": tool_versions(tools if tools is not None else _DEFAULT_TOOLS),
        "exclusions_source": excl_block,
        "args": _jsonify(args_dict) if args_dict is not None else None,
        "created_at": _now_iso(),
        "host": {
            "nodename": uname.nodename,
            "sysname": uname.sysname,
            "release": uname.release,
            "machine": uname.machine,
            "user": os.environ.get("USER"),
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "inputs": file_manifest(inputs) if inputs else [],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "run-manifest.json"
    import json

    out_path.write_text(json.dumps(manifest, indent=2))
    return out_path


def write_dataset_description(
    output_dir: str | Path,
    *,
    name: str,
    pipeline_version: str | None = None,
    source_datasets: list[dict] | None = None,
) -> Path:
    """Write a minimal valid BIDS derivative ``dataset_description.json``.

    ``GeneratedBy`` records the code SHA (in a ``CodeURL``-style field) so the
    derivative tree is traceable to the exact commit that produced it.

    Args:
        output_dir: directory to write into (created).
        name: dataset Name field.
        pipeline_version: Version recorded under GeneratedBy (defaults to the
            installed ``neuro-workflow`` package version, else "unknown").
        source_datasets: BIDS SourceDatasets list (defaults to []).

    Returns:
        Path to the written ``dataset_description.json``.
    """
    if pipeline_version is None:
        try:
            pipeline_version = _ilmd.version("neuro-workflow")
        except _ilmd.PackageNotFoundError:
            pipeline_version = "unknown"

    desc = {
        "Name": name,
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "neuro-workflow",
                "Version": pipeline_version,
                "CodeURL": f"git:{git_sha()}",
            }
        ],
        "SourceDatasets": source_datasets if source_datasets is not None else [],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dataset_description.json"
    import json

    out_path.write_text(json.dumps(desc, indent=2))
    return out_path
