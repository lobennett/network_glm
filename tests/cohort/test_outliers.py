"""Tests for src/network_glm/cohort/outliers.py — discovery + parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from network_glm.cohort.outliers import (
    discover_contrast_files,
    parse_contrast_path,
)


def test_parse_contrast_path():
    p = Path(
        "/lev1/sub-s03/task-stopSignal/indiv_contrasts/"
        "sub-s03_ses-01_task-stopSignal_run-1_contrast-stop_success-go_rtmodel-rt_centered_stat-effect-size.nii.gz"
    )
    sc = parse_contrast_path(p)
    assert sc.subject == "sub-s03"
    assert sc.session == "ses-01"
    assert sc.task == "stopSignal"
    assert sc.run == "1"
    assert sc.contrast == "stop_success-go"


def test_exclusion_key_matches_compiled_registry_format():
    """Contract: the key derived from a contrast filename (used to filter scans
    in detect_lev1_outliers) must byte-match the key load_exclusions emits from a
    compiled-exclusions entry. If these ever diverge, --exclusions-file silently
    drops nothing — the exact failure mode that made the events .bidsignore filter
    inert. This pins both sides to one format.
    """
    from network_glm.exclusions import create_exclusion_key
    from network_glm.cohort.outliers import _make_exclusion_key

    path = Path(
        "/lev1/sub-s10/task-goNogo/indiv_contrasts/"
        "sub-s10_ses-05_task-goNogo_run-1_contrast-go_stat-effect-size.nii.gz"
    )
    # compiled-exclusions entry shape: task/run carry their BIDS prefixes.
    entry = {"subject": "sub-s10", "session": "ses-05", "task": "task-goNogo", "run": "run-1"}
    assert _make_exclusion_key(parse_contrast_path(path)) == create_exclusion_key(entry)


def test_discover_contrast_files(tmp_path: Path):
    # Build a tiny fixture
    f1 = (
        tmp_path
        / "sub-s03/task-goNogo/indiv_contrasts"
        / "sub-s03_ses-01_task-goNogo_run-1_contrast-go_stat-effect-size.nii.gz"
    )
    f1.parent.mkdir(parents=True)
    f1.touch()
    f2 = (
        tmp_path
        / "sub-s10/task-goNogo/indiv_contrasts"
        / "sub-s10_ses-01_task-goNogo_run-1_contrast-go_stat-effect-size.nii.gz"
    )
    f2.parent.mkdir(parents=True)
    f2.touch()
    # red herring that should NOT match
    (tmp_path / "sub-s03/task-goNogo/indiv_contrasts" / "junk.txt").touch()

    found = discover_contrast_files(
        [tmp_path],
        glob_pattern="sub-s*/task-*/indiv_contrasts/*stat-effect-size.nii.gz",
    )
    assert sorted(p.name for p in found) == sorted([f1.name, f2.name])


import nibabel as nib
import numpy as np

from network_glm.cohort.outliers import (
    compute_cohort_outliers,
)


def _mk_nifti(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(path))


def test_compute_cohort_outliers_known_counts(tmp_path: Path):
    """Five subjects, one contrast, 4×4×4 volume. Two subjects are uniform 0;
    one is uniform 100 — that one should have 100% outlier voxels at n_std=2.
    """
    base = tmp_path
    paths: list[Path] = []
    for i, val in enumerate([0.0, 0.0, 0.0, 0.0, 100.0]):
        sub = f"sub-s{i:02d}"
        p = (
            base
            / sub
            / "task-stopSignal/indiv_contrasts"
            / f"{sub}_ses-01_task-stopSignal_run-1_contrast-go_stat-effect-size.nii.gz"
        )
        _mk_nifti(np.full((4, 4, 4), val, dtype=np.float32), p)
        paths.append(p)

    results = compute_cohort_outliers(paths, n_std=2.0)
    by_sub = {r.scan.subject: r for r in results}
    assert by_sub["sub-s04"].outlier_pct == pytest.approx(100.0)
    assert by_sub["sub-s00"].outlier_pct == pytest.approx(0.0)


def test_compute_cohort_outliers_groups_by_task_contrast(tmp_path: Path):
    """Subjects across two contrasts. Outlier counts are computed within each contrast group."""
    paths: list[Path] = []
    for i in range(3):
        sub = f"sub-s{i:02d}"
        # contrast A
        pa = (
            tmp_path
            / sub
            / "task-goNogo/indiv_contrasts"
            / f"{sub}_ses-01_task-goNogo_run-1_contrast-go_stat-effect-size.nii.gz"
        )
        _mk_nifti(np.full((4, 4, 4), float(i), dtype=np.float32), pa)
        paths.append(pa)
        # contrast B
        pb = (
            tmp_path
            / sub
            / "task-goNogo/indiv_contrasts"
            / f"{sub}_ses-01_task-goNogo_run-1_contrast-stop_stat-effect-size.nii.gz"
        )
        _mk_nifti(np.full((4, 4, 4), float(i) * 10, dtype=np.float32), pb)
        paths.append(pb)

    results = compute_cohort_outliers(paths, n_std=2.0)
    contrasts = {(r.scan.task, r.scan.contrast) for r in results}
    assert contrasts == {("goNogo", "go"), ("goNogo", "stop")}


def test_aggregate_vifs_from_csv(tmp_path: Path):
    """A VIF CSV with two contrasts → both VIFs ingested into the right ScanContrast."""

    sub_dir = tmp_path / "sub-s03/task-goNogo/quality_control"
    sub_dir.mkdir(parents=True)
    (sub_dir / "sub-s03_ses-01_task-goNogo_run-1_desc-contrastVIFs.csv").write_text(
        "contrast,VIF\n" "go,1.5\n" "stop,2.7\n"
    )
    from network_glm.cohort.outliers import discover_vif_files, load_vif_table

    vif_paths = discover_vif_files(
        [tmp_path],
        glob_pattern="sub-s*/task-*/quality_control/*_desc-contrastVIFs.csv",
    )
    assert len(vif_paths) == 1

    table = load_vif_table(vif_paths)
    # Key shape: (subject, session, run, task, contrast) → vif
    assert table[("sub-s03", "ses-01", "1", "goNogo", "go")] == pytest.approx(1.5)
    assert table[("sub-s03", "ses-01", "1", "goNogo", "stop")] == pytest.approx(2.7)


def test_write_outputs_csv_and_flagged(tmp_path: Path):
    from network_glm.cohort.outliers import (
        FlaggedRow,
        write_flagged_tsv,
        write_outliers_csv,
    )

    rows = [
        FlaggedRow(
            subject="sub-s03",
            session="ses-01",
            run="1",
            task="goNogo",
            contrast="go",
            outlier_pct=5.0,
            vif=1.2,
            flagged_outliers=False,
            flagged_vif=False,
        ),
        FlaggedRow(
            subject="sub-s10",
            session="ses-01",
            run="1",
            task="goNogo",
            contrast="go",
            outlier_pct=15.0,
            vif=8.0,
            flagged_outliers=True,
            flagged_vif=True,
        ),
    ]
    out_csv = tmp_path / "lev1_outliers.csv"
    out_tsv = tmp_path / "lev1_flagged.tsv"
    write_outliers_csv(rows, out_csv)
    write_flagged_tsv(rows, out_tsv)
    assert out_csv.is_file()
    assert out_tsv.is_file()
    flagged_lines = out_tsv.read_text().splitlines()
    # Header + only the flagged row
    assert len(flagged_lines) == 2
    assert "sub-s10" in flagged_lines[1]


def test_render_pdf_smoke(tmp_path: Path):
    """End-to-end smoke: 3 synthetic subjects × 1 contrast → PDF written, non-empty."""
    import nibabel as nib
    import numpy as np

    paths: list[Path] = []
    for i in range(3):
        sub = f"sub-s{i:02d}"
        p = (
            tmp_path
            / sub
            / "task-goNogo/indiv_contrasts"
            / f"{sub}_ses-01_task-goNogo_run-1_contrast-go_stat-effect-size.nii.gz"
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        # 8x8x8 with random values — large enough for nilearn slicing
        rng = np.random.default_rng(i)
        nib.save(nib.Nifti1Image(rng.normal(size=(8, 8, 8)).astype(np.float32), np.eye(4)), str(p))
        paths.append(p)

    from network_glm.cohort.outliers import (
        compute_cohort_outliers,
        render_outlier_pdf,
    )

    results = compute_cohort_outliers(paths, n_std=2.0)
    pdf_path = tmp_path / "lev1_outliers.pdf"
    render_outlier_pdf(results, vif_table={}, output_path=pdf_path)
    assert pdf_path.is_file()
    assert pdf_path.stat().st_size > 1000  # at least a few KB; not empty


def test_detect_lev1_outliers_end_to_end(tmp_path: Path):
    import nibabel as nib
    import numpy as np

    # Build a tiny lev1 fixture with 3 subjects × 1 contrast × VIF CSVs
    lev1 = tmp_path / "lev1"
    out = tmp_path / "qa_out"
    for i in range(3):
        sub = f"sub-s{i:02d}"
        # contrast NIfTI
        cp = (
            lev1
            / sub
            / "task-goNogo/indiv_contrasts"
            / f"{sub}_ses-01_task-goNogo_run-1_contrast-go_stat-effect-size.nii.gz"
        )
        cp.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(i)
        nib.save(nib.Nifti1Image(rng.normal(size=(6, 6, 6)).astype(np.float32), np.eye(4)), str(cp))
        # VIF CSV
        qc = lev1 / sub / "task-goNogo/quality_control"
        qc.mkdir(parents=True, exist_ok=True)
        (qc / f"{sub}_ses-01_task-goNogo_run-1_desc-contrastVIFs.csv").write_text(
            "contrast,VIF\ngo,2.0\n"
        )

    from network_glm.cohort.outliers import detect_lev1_outliers

    detect_lev1_outliers(
        lev1_dirs=[lev1],
        output_dir=out,
        n_std=2.0,
        vif_threshold=5.0,
        outlier_pct_threshold=10.0,
    )
    assert (out / "lev1_outliers.csv").is_file()
    assert (out / "lev1_outliers.pdf").is_file()
    assert (out / "lev1_flagged.tsv").is_file()
