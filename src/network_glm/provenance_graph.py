"""Assemble the full provenance chain into one machine-readable graph.

This is a pure *data exporter*. It walks the pipeline's on-disk artifacts and
stitches them into a single JSON-serializable dict:

    Flywheel snapshot -> reconciliation manifest -> collection .bidsignore
    -> fMRIPrep -> exclusions lockfile -> lev1 -> lev2

Each artifact becomes an ordered ``stages`` entry ``{name, present, path, ...}``.
Stages that expose a git ``code_sha`` (exclusions lockfile, lev1, lev2)
additionally carry it; fMRIPrep carries a tool ``version`` instead. An ``edges``
list records the consecutive hand-offs; when both endpoints expose a
``code_sha`` the edge carries ``sha_mismatch: bool`` (a drift alarm), otherwise
``sha_mismatch: None``.

Missing artifacts are represented as ``present: false`` rather than raising, so
the graph is buildable at any pipeline phase (e.g. lev2 absent pre-Phase-2).

The single-manifest read mirrors ``analysis.lev2.run._read_input_provenance``'s
best-effort JSON pattern (missing/unreadable -> sentinel, never fatal); that
function summarizes MANY lev1 inputs from lev2's vantage, whereas this exporter
reads ONE representative manifest per stage across the whole chain, so a shared
tiny ``_read_json`` helper is the honest DRY boundary rather than reuse of the
lev2-specific input->manifest path mapping.

No UI, no plotting: this module only produces the JSON contract a future
interactive visualization will consume.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from network_glm.provenance import _now_iso, git_sha

# Repo root anchor (monkeypatchable in tests). Same resolution as provenance.py:
# src/network_glm/provenance_graph.py -> parents[2] == repo root.
# NOTE (network_glm lift-and-shift): the source monolith had this module one
# directory deeper (core/provenance_graph.py under the old package root,
# parents[3]); the index below was adjusted to match this package's shallower
# layout.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# fMRIPrep derivative directory name (matches the committed pipeline output).
_FMRIPREP_DIRNAME = "fmriprep_25.2.4"

# Ordered stage names — the provenance chain, source -> sink.
STAGE_ORDER = [
    "flywheel_snapshot",
    "reconciliation",
    "bidsignore",
    "fmriprep",
    "exclusions_lockfile",
    "lev1",
    "lev2",
]


def _read_json(path: Path) -> dict | None:
    """Best-effort JSON read: return the parsed dict or None.

    Mirrors the non-fatal pattern in
    :func:`network_glm.lev2.run._read_input_provenance` — a missing,
    unreadable, or malformed file yields ``None`` (recorded as ``present:
    false``) rather than crashing the whole export.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _flywheel_stage(repo_root: Path, cohort: str) -> dict[str, Any]:
    path = repo_root / "data" / "repro" / f"fw_inventory_{cohort}.json"
    stage: dict[str, Any] = {"name": "flywheel_snapshot", "path": str(path)}
    data = _read_json(path) if path.is_file() else None
    if data is None:
        stage["present"] = False
        return stage
    stage["present"] = True
    # Snapshots may (optionally) record a code_sha and a capture timestamp.
    stage["code_sha"] = data.get("code_sha")
    stage["captured_at"] = data.get("captured_at") or data.get("created_at")
    # Record a lightweight size hint if the inventory lists sessions/acquisitions.
    for key in ("n_sessions", "n_acquisitions", "n_subjects"):
        if key in data:
            stage[key] = data[key]
    return stage


def _reconciliation_stage(repo_root: Path, cohort: str) -> dict[str, Any]:
    path = repo_root / "config" / "manifests" / f"reconciliation_{cohort}.tsv"
    stage: dict[str, Any] = {"name": "reconciliation", "path": str(path)}
    if not path.is_file():
        stage["present"] = False
        return stage
    stage["present"] = True
    # Row count (minus header) as a cheap manifest-size signal; TSV has no sha.
    try:
        n_lines = sum(1 for _ in path.open())
        stage["n_rows"] = max(n_lines - 1, 0)
    except OSError:
        pass
    return stage


def _bidsignore_stage(repo_root: Path, cohort: str) -> dict[str, Any]:
    path = repo_root / "data" / "exclusions" / f"{cohort}_collection.bidsignore"
    stage: dict[str, Any] = {"name": "bidsignore", "path": str(path)}
    stage["present"] = path.is_file()
    if stage["present"]:
        try:
            stage["n_patterns"] = sum(
                1 for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")
            )
        except OSError:
            pass
    return stage


def _fmriprep_stage(bids_root: Path | None) -> dict[str, Any]:
    stage: dict[str, Any] = {"name": "fmriprep"}
    if bids_root is None:
        stage["present"] = False
        stage["path"] = None
        return stage
    path = bids_root / "derivatives" / _FMRIPREP_DIRNAME / "dataset_description.json"
    stage["path"] = str(path)
    data = _read_json(path) if path.is_file() else None
    if data is None:
        stage["present"] = False
        return stage
    stage["present"] = True
    # fMRIPrep exposes a tool VERSION (not a neuro-workflow code_sha).
    version = None
    generated_by = data.get("GeneratedBy")
    if isinstance(generated_by, list) and generated_by:
        version = generated_by[0].get("Version")
    stage["version"] = version
    return stage


def _lockfile_stage(repo_root: Path, cohort: str) -> dict[str, Any]:
    path = repo_root / "data" / "exclusions" / f"{cohort}_lock.json"
    stage: dict[str, Any] = {"name": "exclusions_lockfile", "path": str(path)}
    data = _read_json(path) if path.is_file() else None
    if data is None:
        stage["present"] = False
        return stage
    stage["present"] = True
    stage["code_sha"] = data.get("compiled_at_code_sha")
    stage["compiled_at"] = data.get("compiled_at")
    stage["n_total_entries"] = data.get("n_total_entries")
    stage["n_overrides"] = data.get("n_overrides")
    sources = data.get("sources") or []
    stage["sources"] = [
        {
            "generator": s.get("generator"),
            "code_sha": s.get("code_sha"),
            "ran_at": s.get("ran_at"),
            "n_entries": s.get("n_entries"),
        }
        for s in sources
    ]
    return stage


def _find_manifest(bids_root: Path, glob_pattern: str) -> Path | None:
    """Return the first (sorted) run-manifest.json matching a derivatives glob."""
    try:
        matches = sorted(bids_root.glob(glob_pattern))
    except OSError:
        return None
    return matches[0] if matches else None


def _analysis_stage(name: str, bids_root: Path | None, glob_pattern: str) -> dict[str, Any]:
    """Build a lev1/lev2-style stage from a representative run-manifest.json.

    Globs the derivatives tree for the first matching manifest and lifts its
    ``code_sha`` / ``config_version`` / exclusions-source sha (the same fields
    lev2 already reasons over), so the chain can be checked for code drift.
    """
    stage: dict[str, Any] = {"name": name}
    if bids_root is None:
        stage["present"] = False
        stage["path"] = None
        return stage
    manifest_path = _find_manifest(bids_root, glob_pattern)
    if manifest_path is None:
        stage["present"] = False
        stage["path"] = None
        return stage
    stage["path"] = str(manifest_path)
    data = _read_json(manifest_path)
    if data is None:
        stage["present"] = False
        return stage
    stage["present"] = True
    stage["code_sha"] = data.get("code_sha")
    stage["config_version"] = data.get("config_version")
    excl = data.get("exclusions_source")
    stage["exclusions_sha256"] = excl.get("sha256") if isinstance(excl, dict) else None
    return stage


def _build_edges(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One edge per consecutive stage pair.

    When BOTH endpoints expose a non-null ``code_sha`` the edge carries
    ``sha_mismatch = a != b`` (a drift alarm). Otherwise (e.g. either endpoint
    is absent, or exposes only a tool version like fMRIPrep) the edge records
    ``sha_mismatch = None`` — no comparison is meaningful.
    """
    edges: list[dict[str, Any]] = []
    for a, b in zip(stages, stages[1:], strict=False):
        a_sha = a.get("code_sha")
        b_sha = b.get("code_sha")
        if a_sha and b_sha:
            mismatch: bool | None = a_sha != b_sha
        else:
            mismatch = None
        edges.append(
            {
                "from": a["name"],
                "to": b["name"],
                "from_code_sha": a_sha,
                "to_code_sha": b_sha,
                "sha_mismatch": mismatch,
            }
        )
    return edges


def build_provenance_graph(cohort: str, bids_root: str | Path | None = None) -> dict[str, Any]:
    """Assemble the full Flywheel->lev2 provenance chain for ``cohort``.

    Args:
        cohort: dataset/cohort name (e.g. ``"validation"``), used to locate the
            repo-anchored artifacts (fw inventory, reconciliation manifest,
            collection .bidsignore, exclusions lockfile).
        bids_root: BIDS dataset root whose ``derivatives/`` tree holds fMRIPrep,
            lev1, and lev2 outputs. ``None`` -> those three stages are marked
            ``present: false`` (repo-anchored stages still resolve).

    Returns:
        A JSON-safe dict with ``cohort``, ``generated_at``,
        ``generator_code_sha``, ``bids_root``, an ordered ``stages`` list, and
        an ``edges`` list.
    """
    repo_root = _REPO_ROOT
    bids_path = Path(bids_root) if bids_root is not None else None

    stages = [
        _flywheel_stage(repo_root, cohort),
        _reconciliation_stage(repo_root, cohort),
        _bidsignore_stage(repo_root, cohort),
        _fmriprep_stage(bids_path),
        _lockfile_stage(repo_root, cohort),
        _analysis_stage(
            "lev1", bids_path, "derivatives/lev1_surface/sub-*/task-*/run-manifest.json"
        ),
        _analysis_stage("lev2", bids_path, "derivatives/lev2*/**/run-manifest.json"),
    ]

    return {
        "cohort": cohort,
        "generated_at": _now_iso(),
        "generator_code_sha": git_sha(),
        "bids_root": str(bids_path) if bids_path is not None else None,
        "stages": stages,
        "edges": _build_edges(stages),
    }
