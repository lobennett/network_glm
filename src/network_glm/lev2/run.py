#!/usr/bin/env python3
"""Level 2 GLM Analysis script for group-level statistical analysis."""

# Note: `randomise_prep` is lazy-imported inside run_level2_analysis (the only
# call site). Module-level import would force test environments without the
# `lev1` extras installed to fail at import time, even when only testing
# helpers like discover_input_files. Lazy import surfaces a clear
# ModuleNotFoundError when randomise actually gets called in production.

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

from nilearn.image import math_img
from nilearn.masking import intersect_masks

from network_glm import provenance


def compute_mask(input_files, threshold=0.9, connected=False):
    """
    Computes a group mask by intersecting individual subject masks.

    Each subject's effect size map is converted to a binary mask (1 where
    data is not zero, 0 otherwise). These individual masks are then
    intersected to create a final group mask.

    Parameters
    ----------
    input_files : list of str or Path
        List of paths to first-level effect size files.
    threshold : float, optional
        The proportion of masks in which a voxel must be active to be
        included in the final mask. Default 0.9 (voxel must be present in
        >=90% of subject masks), matching the lev2 CLI default. 1.0 would be
        a strict intersection (voxel must be in all masks).
    connected : bool, optional
        If True, only the largest connected component of the final mask is
        kept. Default False: keep every voxel that meets the coverage
        threshold. For a group statistical mask there is no reason to require
        a single contiguous component, and ``connected=True`` would silently
        drop legitimate gray-matter voxels in smaller disconnected clusters
        (it is appropriate for single-subject brain extraction, not group
        coverage masks).

    Returns
    -------
    nibabel.nifti1.Nifti1Image
        The combined group mask image.
    """
    print("== Generating a mask for each of the input files ==")
    subject_masks = [math_img("img != 0", img=f) for f in input_files]

    print("== Intersecting subject masks to create the final group mask ==")
    group_mask = intersect_masks(subject_masks, threshold=threshold, connected=connected)

    return group_mask


def discover_input_files(level1_dirs: list[Path], contrast_name: str) -> list[str]:
    """
    Discover input files for a specific contrast from multiple level1 output directories.

    Files tagged `_desc-belowMinRuns_` (subjects whose fixed-effects came
    from fewer than `min_runs` retained sessions, see lev1 design 2026-05-07)
    are filtered out automatically.

    Args:
        level1_dirs: List of paths to level1 output directories
        contrast_name: Task_contrast name (e.g., 'task-flanker_contrast-incongruent-congruent')

    Returns:
        List of paths to fixed effects files for this contrast (excluding
        _desc-belowMinRuns_ files).
    """
    all_files: list[str] = []
    n_dropped = 0

    for level1_dir in level1_dirs:
        pattern = (
            level1_dir
            / "sub-*"
            / "*"
            / "fixed_effects"
            / f"*{contrast_name}_rtmodel-*_stat-fixed-effects.nii.gz"
        )
        files = glob.glob(str(pattern))
        kept = [f for f in files if "_desc-belowMinRuns_" not in f]
        n_dropped += len(files) - len(kept)
        all_files.extend(kept)

    if n_dropped:
        print(
            f"discover_input_files: dropped {n_dropped} "
            f"_desc-belowMinRuns files for contrast {contrast_name}"
        )

    return sorted(all_files)


def _input_manifest_path(input_file: str | Path) -> Path:
    """Map a lev1 fixed-effects input file to its PR4b run-manifest.

    The lev2 glob selects files at
    ``<results>/<subj>/task-<task>/fixed_effects/<file>`` and PR4b
    (:func:`network_glm.lev1.run._write_lev1_provenance`) writes the
    per-subject×task ``run-manifest.json`` at the per-subject×task ``base`` dir
    (``dirs['base'] == <results>/<subj>/task-<task>/``), i.e. ONE level above
    the ``fixed_effects/`` dir. So the manifest is the input file's
    grandparent-dir ``run-manifest.json``.
    """
    return Path(input_file).parent.parent / "run-manifest.json"


def _read_input_provenance(input_files: list[str]) -> dict:
    """Summarize the provenance chain of lev2's selected lev1 inputs.

    For each input fixed-effects file, locate and read its PR4b lev1
    ``run-manifest.json`` (see :func:`_input_manifest_path`) and collect the
    distinct ``code_sha`` / ``config_version`` / ``exclusions_source`` SHA
    values across all inputs. This is the provenance-chain closure: lev2 can
    no longer be blind to which exclusion set / lev1 code version / config
    produced its inputs.

    Best-effort and non-fatal: a missing or unreadable manifest contributes the
    sentinel ``"unknown"`` (it does NOT raise — pre-PR4b/legacy lev1 outputs
    have no manifest). A manifest present but with no ``exclusions_source``
    block contributes ``"none"`` (distinct from ``"unknown"``).

    Selection is UNCHANGED — this function never filters inputs; it only reads
    the manifests of whatever ``discover_input_files`` already selected.

    Args:
        input_files: the lev1 fixed-effects files lev2 selected as inputs.

    Returns:
        A JSON-safe dict with:
        - ``n_inputs``: number of input files.
        - ``n_manifests_found``: how many inputs had a readable manifest.
        - ``code_sha`` / ``config_version`` / ``exclusions_source``: sorted
          lists of the DISTINCT values seen (``"unknown"`` / ``"none"``
          sentinels included).
        - ``consistent``: True iff each of those distinct sets has <= 1 entry
          (i.e. all inputs share one exclusion set, code version, and config).
    """
    code_shas: set[str] = set()
    config_versions: set[str] = set()
    excl_shas: set[str] = set()
    n_manifests_found = 0

    for input_file in input_files:
        manifest_path = _input_manifest_path(input_file)
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            # Missing / unreadable / malformed manifest: legacy or pre-PR4b
            # outputs. Record the gap rather than crashing.
            code_shas.add("unknown")
            config_versions.add("unknown")
            excl_shas.add("unknown")
            continue

        n_manifests_found += 1
        code_shas.add(manifest.get("code_sha") or "unknown")
        config_versions.add(manifest.get("config_version") or "unknown")

        excl_block = manifest.get("exclusions_source")
        if not excl_block:
            excl_shas.add("none")
        else:
            excl_shas.add(excl_block.get("sha256") or "none")

    distinct = {
        "code_sha": sorted(code_shas),
        "config_version": sorted(config_versions),
        "exclusions_source": sorted(excl_shas),
    }
    consistent = all(len(v) <= 1 for v in distinct.values())

    return {
        "n_inputs": len(input_files),
        "n_manifests_found": n_manifests_found,
        "consistent": consistent,
        **distinct,
    }


def run_level2_analysis(
    contrast_name: str,
    input_files: list[str],
    output_dir: Path,
    mask_threshold: float = 0.9,
    num_permutations: int = 5000,
    seed: int = 0,
) -> bool:
    """Run level 2 analysis for a specific contrast.

    Returns True on success, False if there were no inputs or randomise failed
    (so the caller can propagate the failure instead of silently exiting 0 and
    stamping a success provenance manifest).

    ``seed`` pins FSL randomise's permutation RNG for reproducibility. It is
    forwarded only if the installed ``setup_randomise_tfce`` accepts a ``seed``
    parameter (older randomise-prep versions do not), so this never breaks on an
    API that predates seed support.
    """
    print(f"Running Level 2 analysis for: {contrast_name}")
    print(f"Found {len(input_files)} input files")

    if not input_files:
        print(f"Error: No input files found for contrast {contrast_name}")
        return False

    contrast_output_dir = output_dir / contrast_name
    contrast_output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing group analysis mask...")
    group_mask_img = compute_mask(input_files, threshold=mask_threshold)
    group_mask_path = contrast_output_dir / "group_mask.nii.gz"
    group_mask_img.to_filename(group_mask_path)
    print(f"--> Group mask saved to: {group_mask_path}")

    print("Setting up FSL randomise...")
    # Lazy import so test environments without the lev1 extras installed
    # can still import + exercise the helpers in this module. In production
    # randomise_prep is in the lev1 extras group; install it via
    # `uv pip install "randomise-prep @ git+https://github.com/jmumford/randomise-prep.git"`
    # if missing. Failure here surfaces a clear ModuleNotFoundError.
    from randomise_prep import setup_randomise_tfce

    randomise_kwargs = dict(
        input_files=input_files,
        group_mask=str(group_mask_path),
        output_directory=str(contrast_output_dir),
        analysis_type="onesample_2sided",
        num_perm=num_permutations,
    )
    # Forward the seed only if the installed setup_randomise_tfce supports it
    # (explicit `seed` param, or **kwargs). Safe no-op on versions that don't.
    import inspect

    _sig = inspect.signature(setup_randomise_tfce)
    if "seed" in _sig.parameters or any(p.kind == p.VAR_KEYWORD for p in _sig.parameters.values()):
        randomise_kwargs["seed"] = seed
    else:
        print(
            "WARNING: installed randomise-prep has no seed parameter; FSL "
            "randomise permutation RNG is not pinned for this run.",
            file=sys.stderr,
        )
    script_path = setup_randomise_tfce(**randomise_kwargs)

    print("Running FSL randomise...")
    try:
        subprocess.run(["bash", script_path], capture_output=True, text=True, check=True)
        print("✓ FSL randomise completed successfully")
        print(f"Results saved to: {contrast_output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"✗ FSL randomise failed: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

    return True


def get_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description="Level 2 GLM Analysis for Network R01 dataset")
    parser.add_argument(
        "--contrast",
        type=str,
        required=True,
        help='Contrast name (e.g., "nBack_twoBack-oneBack")',
    )
    parser.add_argument(
        "--level1-dirs",
        nargs="+",
        type=str,
        required=True,
        help="Level 1 output directories (can specify multiple)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="./level2_output",
        help="Level 2 output directory",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.9,
        help="Threshold for group mask intersection (0.0-1.0)",
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=5000,
        help="Number of permutations (FSL randomise for volume; sign-flip for surface)",
    )
    parser.add_argument(
        "--space",
        choices=["volume", "surface"],
        default="volume",
        help="volume: FSL randomise on NIfTI fixed-effects (default). "
        "surface: self-contained sign-flip permutation group test on the "
        "GIFTI surface fixed-effects (both hemispheres, whole-cortex FWE).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for the surface sign-flip permutation (reproducible).",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        default=False,
        help="Permit recording provenance against an uncommitted (dirty) git "
        "working tree without warning. Without this flag a dirty tree warns "
        "loudly to stderr but the run still proceeds; the manifest records "
        "code_dirty truthfully either way.",
    )
    return parser


def _warn_if_inconsistent_inputs(input_provenance: dict) -> None:
    """Print a loud stderr WARNING when the input provenance chain is mixed.

    Group-level (lev2) results are only interpretable if every contributing
    lev1 fixed-effects map came from the SAME exclusion set, lev1 code version,
    and study config. When that is not the case, mixing them silently is a
    scientific hazard — so we warn loudly (but do NOT fail or change selection;
    the operator decides). The full distinct-value summary is also persisted in
    the lev2 manifest under ``input_provenance`` for the audit trail.
    """
    if input_provenance.get("consistent", True):
        return

    print(
        'WARNING: lev2 inputs are provenance-INCONSISTENT — group results may '
        'mix incompatible lev1 outputs. Distinct values across inputs:\n'
        f'  code_sha:          {input_provenance["code_sha"]}\n'
        f'  config_version:    {input_provenance["config_version"]}\n'
        f'  exclusions_source: {input_provenance["exclusions_source"]}\n'
        f'  ({input_provenance["n_manifests_found"]}/{input_provenance["n_inputs"]} '
        'inputs had a readable lev1 run-manifest; "unknown" = missing manifest, '
        '"none" = no exclusions recorded). Re-run the offending lev1 subjects '
        'against a single exclusion set / code version before trusting lev2.',
        file=sys.stderr,
    )


def _write_lev2_provenance(output_dir, args, level1_dirs, input_files):
    """Write additive provenance for a lev2 contrast run.

    - ``dataset_description.json`` at ``output_dir``, naming the lev1 source
      dirs in ``SourceDatasets``.
    - ``run-manifest.json`` at ``output_dir``, stage='lev2', recording the
      discovered lev1 fixed-effects input files AND (PR4c) an
      ``input_provenance`` summary closing the provenance chain: the distinct
      lev1 ``code_sha`` / ``config_version`` / ``exclusions_source`` SHAs across
      those inputs. A loud stderr WARNING fires if that summary is inconsistent
      (mixed exclusion sets / code versions / configs). Selection is unchanged.

    Called AFTER the contrast's scientific outputs; errors are allowed to
    surface (fail loud). ``allow_dirty`` is threaded from the CLI flag.
    """
    allow_dirty = getattr(args, "allow_dirty", False)
    provenance.write_dataset_description(
        output_dir,
        name="lev2",
        source_datasets=[{"URL": str(d)} for d in level1_dirs],
    )

    # Provenance-chain closure (PR4c): read each input's lev1 run-manifest and
    # summarize the distinct upstream SHAs. Warn loudly on inconsistency.
    input_provenance = _read_input_provenance(input_files)
    _warn_if_inconsistent_inputs(input_provenance)

    manifest_path = provenance.write_run_manifest(
        output_dir,
        stage="lev2",
        args=args,
        inputs=[Path(f) for f in input_files],
        allow_dirty=allow_dirty,
    )

    # Additively fold the input-provenance summary into the written manifest.
    # write_run_manifest's schema is fixed and shared across stages, so we
    # merge the lev2-specific block in here rather than widen the primitive.
    manifest = json.loads(manifest_path.read_text())
    manifest["input_provenance"] = input_provenance
    # External (non-pip) tool versions. nilearn/numpy/scipy are already in the
    # manifest's tool_versions; FSL (the volume randomise engine) is not a Python
    # package, so record it explicitly. "unknown" when FSL is absent (e.g. the
    # surface path, which uses numpy only).
    manifest["external_tool_versions"] = {"fsl": provenance.fsl_version()}
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    """Run level 2 analysis with command line arguments."""
    parser = get_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("Level 2 GLM Analysis")
    print("=" * 60)
    print(f"Contrast: {args.contrast}")
    print(f"Level 1 directories: {args.level1_dirs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mask threshold: {args.mask_threshold}")
    print(f"Permutations: {args.num_permutations}")
    print("=" * 60)
    print()

    # Provenance is ADDITIVE: warn loudly (but do not fail) when stamping a
    # dirty tree, unless --allow-dirty. The manifest records code_dirty truly.
    if provenance.git_is_dirty() and not args.allow_dirty:
        print(
            "WARNING: git working tree is dirty; lev2 provenance will record "
            "code_dirty=true. Commit/stash for a reproducible stamp, or pass "
            "--allow-dirty to silence this warning.",
            file=sys.stderr,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    level1_dirs = [Path(d) for d in args.level1_dirs]
    for level1_dir in level1_dirs:
        if not level1_dir.exists():
            print(f"ERROR: Level 1 directory not found: {level1_dir}")
            return 1

    # Discover input files for the specific contrast. Surface and volume use
    # different fixed-effects file types (.func.gii vs .nii.gz) and engines.
    print(f"Discovering input files for contrast: {args.contrast} (space={args.space})")
    if args.space == "surface":
        from network_glm.lev2.surface import (
            discover_surface_inputs,
            run_surface_level2_analysis,
        )

        surf = discover_surface_inputs(level1_dirs, args.contrast)
        input_files = surf["L"] + surf["R"]
        if not input_files:
            print(f"ERROR: No surface input files found for contrast {args.contrast}")
            return 1
        ok = run_surface_level2_analysis(
            args.contrast,
            level1_dirs,
            output_dir,
            n_perm=args.num_permutations,
            seed=args.seed,
        )
    else:
        input_files = discover_input_files(level1_dirs, args.contrast)
        if not input_files:
            print(f"ERROR: No input files found for contrast {args.contrast}")
            return 1
        ok = run_level2_analysis(
            args.contrast,
            input_files,
            output_dir,
            args.mask_threshold,
            args.num_permutations,
            seed=args.seed,
        )
    if not ok:
        # The analysis failed (e.g. randomise errored). Do NOT stamp a success
        # provenance manifest; propagate a non-zero exit so the SLURM array
        # surfaces the failure instead of reporting success.
        print(f"ERROR: Level 2 analysis failed for {args.contrast}", file=sys.stderr)
        return 1

    # Provenance (ADDITIVE) — written AFTER the contrast's scientific outputs so
    # a manifest error never loses science. Errors are allowed to surface.
    # Written INTO the per-contrast output dir (matching where
    # run_level2_analysis put its scientific outputs), so that the SLURM array
    # — one contrast per task, all sharing --output-dir — does not race/clobber
    # a single manifest at the results-dir root.
    contrast_output_dir = output_dir / args.contrast
    _write_lev2_provenance(contrast_output_dir, args, level1_dirs, input_files)

    print(f"\nLevel 2 GLM analysis completed for {args.contrast}")
    return 0


if __name__ == "__main__":
    exit(main())
