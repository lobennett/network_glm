"""Per-run completion records for conservative, content-checked resume."""

import hashlib
import json
from pathlib import Path

from network_glm import provenance
from network_glm.acquisition import sidecar_path_for

# These change execution or across-run aggregation, not the per-run fit.
_EXECUTION_ARGS = {
    "skip_existing",
    "skip_qc_plots",
    "verbose",
    "allow_dirty",
    "results_dir",
    "exclusions_file",
    "min_runs",
    "within_subject_threshold",
}
_BOLD_KEYS = {"mni_data", "t1w_data", "left_surface", "right_surface", "cifti_bold"}


def run_signature(run_files, args, task_params, sample_type) -> str | None:
    """Hash scientific settings, actual package source, dependencies, and inputs.

    Exclusions are evaluated before reuse and again during fixed effects. Updating
    that lock therefore does not invalidate an otherwise identical per-run fit.
    Missing inputs make a run ineligible for reuse; normal input validation then
    reports the missing file to the caller.
    """
    inputs = {Path(path).resolve() for path in run_files.values()}
    for key in _BOLD_KEYS & run_files.keys():
        sidecar = sidecar_path_for(run_files[key])
        if sidecar.is_file():
            inputs.add(sidecar.resolve())
    if not inputs or any(not path.is_file() for path in inputs):
        return None
    package = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.suffix in {".py", ".yaml"}:
            source.update(
                str(path.relative_to(package)).encode() + b"\0" + path.read_bytes()
            )
    request = {
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in _EXECUTION_ARGS
        },
        "task_params": task_params,
        "sample_type": sample_type,
        "source_sha256": source.hexdigest(),
        "versions": provenance.tool_versions(
            [
                "numpy",
                "scipy",
                "pandas",
                "nibabel",
                "nilearn",
                "statsmodels",
                "pyyaml",
            ]
        ),
        "inputs": provenance.file_manifest(sorted(inputs)),
    }
    return hashlib.sha256(
        json.dumps(request, sort_keys=True, default=str).encode()
    ).hexdigest()


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def mark_running(path: Path) -> None:
    """Invalidate an earlier completion before starting a replacement fit."""
    _write_record(path, {"schema_version": 1, "status": "running"})


def save_completion(path: Path, signature: str, outputs: list[Path]) -> None:
    """Publish a completion only after all declared files have been written."""
    _write_record(
        path,
        {
            "schema_version": 1,
            "status": "completed",
            "signature": signature,
            "outputs": provenance.file_manifest(
                sorted({item.resolve() for item in outputs})
            ),
        },
    )


def can_reuse(path: Path, signature: str | None) -> bool:
    """Old, incomplete, mismatched, or damaged outputs must be recomputed."""
    if signature is None:
        return False
    try:
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            return False
        if record.get("status") != "completed" or record.get("signature") != signature:
            return False
        outputs = record.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return False
        return (
            provenance.file_manifest([Path(item["path"]) for item in outputs])
            == outputs
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def saved_paths(results: dict) -> list[Path]:
    """Collect the scientific files returned by the volume/surface/CIFTI runners."""
    paths = []
    for value in results.values():
        if isinstance(value, Path):
            paths.append(value)
        elif isinstance(value, dict):
            paths.extend(saved_paths(value))
    return paths
