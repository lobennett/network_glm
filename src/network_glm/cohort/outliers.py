"""Cohort-level outlier QC for lev1 contrast maps.

Mirrors Jeanette Mumford's fmri-outlier-detector/run_network.py against
our lev1 output paths. Single file, plain functions + frozen dataclasses,
≤300 lines (per the spec's code-style guardrails).

Public API:
    detect_lev1_outliers(*, lev1_dirs, output_dir, ...) -> None

Computes per-voxel cohort mean/SD for each (task, contrast) group,
flags voxels >n_std SD from the mean, aggregates per-(subject, contrast)
outlier %, reads the per-contrast VIF CSVs lev1 already emits, writes:
    - lev1_outliers.csv     (one row per scan-contrast)
    - lev1_outliers.pdf     (Jeanette-style: panels + histograms)
    - lev1_flagged.tsv      (subset where any flag fires)
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# BIDS-style entity parser for our lev1 output filenames.
# The contrast field may contain underscores (e.g. "stop_success-go"),
# so we match it non-greedily up to the next BIDS entity (_rtmodel- or _stat-).
_FILENAME_RE = re.compile(
    r"^(?P<subject>sub-[A-Za-z0-9]+)"
    r"_(?P<session>ses-[A-Za-z0-9]+)"
    r"_task-(?P<task>[A-Za-z0-9]+)"
    r"_run-(?P<run>\d+)"
    r"_contrast-(?P<contrast>.+?)"
    r"(?:_rtmodel-[A-Za-z0-9_]+?)?"
    r"_stat-effect-size\.nii\.gz$"
)


@dataclass(frozen=True)
class ScanContrast:
    """One (subject, session, run, task, contrast) tuple."""

    subject: str
    session: str
    task: str
    run: str
    contrast: str
    path: Path


def parse_contrast_path(path: Path) -> ScanContrast:
    """Parse one stat-effect-size NIfTI path into a ScanContrast.

    Raises ValueError if the filename doesn't match the expected pattern.
    """
    m = _FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"unrecognized contrast filename: {path.name}")
    return ScanContrast(
        subject=m.group("subject"),
        session=m.group("session"),
        task=m.group("task"),
        run=m.group("run"),
        contrast=m.group("contrast"),
        path=path,
    )


def discover_contrast_files(
    lev1_dirs: list[Path],
    *,
    glob_pattern: str = "sub-s*/task-*/indiv_contrasts/*stat-effect-size.nii.gz",
) -> list[Path]:
    """Find all contrast effect-size NIfTIs across the given lev1 output dirs."""
    out: list[Path] = []
    for d in lev1_dirs:
        out.extend(d.glob(glob_pattern))
    return sorted(out)


@dataclass(frozen=True)
class OutlierResult:
    scan: ScanContrast
    outlier_pct: float  # percent of voxels >n_std SD from cohort mean
    n_voxels: int  # total voxels in the volume


def _load_volume(path: Path) -> np.ndarray:
    import nibabel as nib  # local import keeps top-level cheap

    return nib.load(str(path)).get_fdata()


def compute_cohort_outliers(
    contrast_paths: list[Path],
    *,
    n_std: float = 3.0,
) -> list[OutlierResult]:
    """For each (task, contrast) group, compute per-voxel cohort mean/SD and
    count voxels >n_std SD per subject.

    Returns one OutlierResult per scan-contrast.
    """
    import numpy as np

    scans = [parse_contrast_path(p) for p in contrast_paths]

    # Group by (task, contrast)
    groups: dict[tuple[str, str], list[ScanContrast]] = {}
    for sc in scans:
        groups.setdefault((sc.task, sc.contrast), []).append(sc)

    results: list[OutlierResult] = []
    for (_task, _contrast), members in groups.items():
        # Stack volumes: shape (n_subjects, X, Y, Z)
        stacked = np.stack([_load_volume(m.path) for m in members], axis=0)
        cohort_mean = stacked.mean(axis=0)
        cohort_std = stacked.std(axis=0, ddof=0)
        # Avoid divide-by-zero: voxels with zero cohort std contribute 0 outliers
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(cohort_std > 0, (stacked - cohort_mean) / cohort_std, 0.0)
        outlier_mask = np.abs(z) >= n_std  # shape (n_subjects, X, Y, Z)
        n_vox = int(stacked.shape[1] * stacked.shape[2] * stacked.shape[3])
        for i, member in enumerate(members):
            n_out = int(outlier_mask[i].sum())
            results.append(
                OutlierResult(
                    scan=member,
                    outlier_pct=100.0 * n_out / n_vox if n_vox else 0.0,
                    n_voxels=n_vox,
                )
            )
    return results


_VIF_FILENAME_RE = re.compile(
    r"^(?P<subject>sub-[A-Za-z0-9]+)"
    r"_(?P<session>ses-[A-Za-z0-9]+)"
    r"_task-(?P<task>[A-Za-z0-9]+)"
    r"_run-(?P<run>\d+)"
    r"_desc-contrastVIFs\.csv$"
)


@dataclass(frozen=True)
class FlaggedRow:
    """One row in the per-scan-contrast outputs."""

    subject: str
    session: str
    run: str
    task: str
    contrast: str
    outlier_pct: float
    vif: float | None
    flagged_outliers: bool
    flagged_vif: bool


def discover_vif_files(
    lev1_dirs: list[Path],
    *,
    glob_pattern: str = "sub-s*/task-*/quality_control/*_desc-contrastVIFs.csv",
) -> list[Path]:
    out: list[Path] = []
    for d in lev1_dirs:
        out.extend(d.glob(glob_pattern))
    return sorted(out)


def load_vif_table(
    vif_paths: list[Path],
) -> dict[tuple[str, str, str, str, str], float]:
    """Read each contrastVIFs CSV; return {(subject, session, run, task, contrast) -> vif}."""
    import pandas as pd

    table: dict[tuple[str, str, str, str, str], float] = {}
    for fp in vif_paths:
        m = _VIF_FILENAME_RE.match(fp.name)
        if not m:
            continue
        df = pd.read_csv(fp)
        if not {"contrast", "VIF"}.issubset(df.columns):
            continue
        for _, row in df.iterrows():
            key = (
                m.group("subject"),
                m.group("session"),
                m.group("run"),
                m.group("task"),
                str(row["contrast"]),
            )
            table[key] = float(row["VIF"])
    return table


def assemble_flagged_rows(
    outlier_results: list[OutlierResult],
    vif_table: dict[tuple[str, str, str, str, str], float],
    *,
    outlier_pct_threshold: float,
    vif_threshold: float,
) -> list[FlaggedRow]:
    rows: list[FlaggedRow] = []
    for r in outlier_results:
        sc = r.scan
        key = (sc.subject, sc.session, sc.run, sc.task, sc.contrast)
        vif = vif_table.get(key)
        rows.append(
            FlaggedRow(
                subject=sc.subject,
                session=sc.session,
                run=sc.run,
                task=sc.task,
                contrast=sc.contrast,
                outlier_pct=r.outlier_pct,
                vif=vif,
                flagged_outliers=r.outlier_pct > outlier_pct_threshold,
                flagged_vif=(vif is not None and vif > vif_threshold),
            )
        )
    return rows


def _write_table(rows: list[FlaggedRow], path: Path, delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "session",
        "run",
        "task",
        "contrast",
        "outlier_pct",
        "vif",
        "flagged_outliers",
        "flagged_vif",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "subject": r.subject,
                    "session": r.session,
                    "run": r.run,
                    "task": r.task,
                    "contrast": r.contrast,
                    "outlier_pct": f"{r.outlier_pct:.4f}",
                    "vif": "" if r.vif is None else f"{r.vif:.4f}",
                    "flagged_outliers": int(r.flagged_outliers),
                    "flagged_vif": int(r.flagged_vif),
                }
            )


def write_outliers_csv(rows: list[FlaggedRow], path: Path) -> None:
    _write_table(rows, path, delimiter=",")


def write_flagged_tsv(rows: list[FlaggedRow], path: Path) -> None:
    flagged = [r for r in rows if r.flagged_outliers or r.flagged_vif]
    _write_table(flagged, path, delimiter="\t")


def render_outlier_pdf(
    outlier_results: list[OutlierResult],
    *,
    vif_table: dict[tuple[str, str, str, str, str], float],
    output_path: Path,
) -> None:
    """Render a Jeanette-style PDF: one page per (task, contrast) with subject panels,
    plus an all-cohort histogram and per-contrast histograms.

    Layout ported from Jeanette Mumford's fmri-outlier-detector/plotting_functions.py.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    from nilearn.plotting import plot_stat_map

    # Group results by (task, contrast)
    groups: dict[tuple[str, str], list[OutlierResult]] = {}
    for r in outlier_results:
        groups.setdefault((r.scan.task, r.scan.contrast), []).append(r)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        # 1) One page per (task, contrast) — subject panels
        for (task, contrast), members in sorted(groups.items()):
            n = len(members)
            ncols = min(4, n)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
            axes = np.atleast_1d(axes).flatten()
            for ax, r in zip(axes, members, strict=False):
                key = (r.scan.subject, r.scan.session, r.scan.run, r.scan.task, r.scan.contrast)
                vif = vif_table.get(key)
                vif_label = f"vif={vif:.2f}" if vif is not None else "vif=?"
                title = (
                    f"{r.scan.subject} {r.scan.session} run-{r.scan.run}\n"
                    f"{vif_label}, outlier={r.outlier_pct:.1f}%"
                )
                try:
                    plot_stat_map(
                        str(r.scan.path),
                        axes=ax,
                        title=title,
                        display_mode="z",
                        cut_coords=1,
                        colorbar=False,
                        annotate=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    ax.text(
                        0.5,
                        0.5,
                        f"render failed: {exc}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
            for ax in axes[n:]:
                ax.axis("off")
            fig.suptitle(f"{task} / {contrast}", fontsize=14)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # 2) All-cohort outlier-% histogram
        all_pcts = [r.outlier_pct for r in outlier_results]
        if all_pcts:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(all_pcts, bins=30)
            ax.set_xlabel("outlier voxel %")
            ax.set_ylabel("count")
            ax.set_title("Outlier % across all cohort scan-contrasts")
            pdf.savefig(fig)
            plt.close(fig)

        # 3) Per-contrast outlier-% histogram
        for (task, contrast), members in sorted(groups.items()):
            pcts = [r.outlier_pct for r in members]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(pcts, bins=20)
            ax.set_xlabel("outlier voxel %")
            ax.set_ylabel("count")
            ax.set_title(f"{task} / {contrast} — outlier % per scan")
            pdf.savefig(fig)
            plt.close(fig)


def detect_lev1_outliers(
    *,
    lev1_dirs: list[Path],
    output_dir: Path,
    n_std: float = 3.0,
    vif_threshold: float = 5.0,
    outlier_pct_threshold: float = 10.0,
    contrast_glob: str = "sub-s*/task-*/indiv_contrasts/*stat-effect-size.nii.gz",
    vif_glob: str = "sub-s*/task-*/quality_control/*_desc-contrastVIFs.csv",
    exclusions: set[str] | None = None,
) -> None:
    """Top-level: run the full cohort outlier detection pass.

    Args:
        lev1_dirs: directories containing lev1 subject outputs.
        output_dir: where qa_out/lev1_outliers.{csv,pdf}, lev1_flagged.tsv go.
        n_std: SD threshold for outlier voxel definition (Jeanette default 3.0).
        vif_threshold: per-contrast VIF flag threshold (default 5.0).
        outlier_pct_threshold: per-scan outlier voxel % flag threshold (default 10.0).
        contrast_glob: pattern (relative to each lev1_dir) for effect-size NIfTIs.
        vif_glob: pattern for per-contrast VIF CSVs.
        exclusions: optional set of exclusion keys (as produced by
            ``analysis.core.utils.load_exclusions`` / ``_make_exclusion_key``,
            e.g. ``"sub-s10_ses-05_task-goNogo_run-1"``). Matching scan-contrasts
            are dropped before cohort statistics so excluded scans don't skew the
            per-voxel mean/SD. ``None`` keeps all discovered contrasts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    contrast_paths = discover_contrast_files(lev1_dirs, glob_pattern=contrast_glob)
    if not contrast_paths:
        raise RuntimeError(
            f"no contrast NIfTIs found under {lev1_dirs} with glob {contrast_glob!r}"
        )

    if exclusions:
        contrast_paths = [
            p
            for p in contrast_paths
            if _make_exclusion_key(parse_contrast_path(p)) not in exclusions
        ]

    outlier_results = compute_cohort_outliers(contrast_paths, n_std=n_std)

    vif_paths = discover_vif_files(lev1_dirs, glob_pattern=vif_glob)
    vif_table = load_vif_table(vif_paths)

    rows = assemble_flagged_rows(
        outlier_results,
        vif_table,
        outlier_pct_threshold=outlier_pct_threshold,
        vif_threshold=vif_threshold,
    )

    write_outliers_csv(rows, output_dir / "lev1_outliers.csv")
    write_flagged_tsv(rows, output_dir / "lev1_flagged.tsv")
    render_outlier_pdf(
        outlier_results, vif_table=vif_table, output_path=output_dir / "lev1_outliers.pdf"
    )


def _make_exclusion_key(sc: ScanContrast) -> str:
    """Stable key matching the Project B exclusion registry format."""
    return f"{sc.subject}_{sc.session}_task-{sc.task}_run-{sc.run}"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run cohort outlier QC on lev1 outputs.

    Mirrors the monolith's ``scripts/lev1_outliers.py`` driver.

    Usage:
        network-glm cohort-outliers \\
            --lev1-dir /scratch/.../lev1_discovery \\
            --output-dir /scratch/.../qa_lev1 \\
            [--exclusions-file data/exclusions/discovery_compiled.json] \\
            [--n-std 3.0] [--vif-threshold 5.0] [--outlier-pct-threshold 10.0]
    """
    import argparse
    import sys

    from network_glm.exclusions import load_exclusions

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument(
        "--lev1-dir",
        type=Path,
        action="append",
        required=True,
        help="lev1 subject-output dir (repeatable)",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--exclusions-file",
        type=Path,
        default=None,
        help="Compiled exclusions JSON (e.g. data/exclusions/"
        "discovery_compiled.json). Scans with action 'exclude'/"
        "'trim' are dropped before cohort outlier statistics.",
    )
    p.add_argument(
        "--n-std", type=float, default=3.0, help="SD threshold for outlier voxels (default 3.0)"
    )
    p.add_argument("--vif-threshold", type=float, default=5.0)
    p.add_argument("--outlier-pct-threshold", type=float, default=10.0)
    p.add_argument(
        "--contrast-glob",
        type=str,
        default="sub-s*/task-*/indiv_contrasts/*stat-effect-size.nii.gz",
    )
    p.add_argument(
        "--vif-glob", type=str, default="sub-s*/task-*/quality_control/*_desc-contrastVIFs.csv"
    )
    args = p.parse_args(argv)

    for d in args.lev1_dir:
        if not d.is_dir():
            print(f"Error: lev1 dir not found: {d}", file=sys.stderr)
            return 1

    exclusions = None
    if args.exclusions_file is not None:
        if not args.exclusions_file.is_file():
            print(f"Error: exclusions file not found: {args.exclusions_file}", file=sys.stderr)
            return 1
        exclusions = load_exclusions(args.exclusions_file)
        print(f"Loaded {len(exclusions)} exclusion keys from {args.exclusions_file}")

    detect_lev1_outliers(
        lev1_dirs=args.lev1_dir,
        output_dir=args.output_dir,
        n_std=args.n_std,
        vif_threshold=args.vif_threshold,
        outlier_pct_threshold=args.outlier_pct_threshold,
        contrast_glob=args.contrast_glob,
        vif_glob=args.vif_glob,
        exclusions=exclusions,
    )
    print(f"Wrote lev1 outlier report to {args.output_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
