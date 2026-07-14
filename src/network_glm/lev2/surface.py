"""Surface group-level (lev2) analysis via sign-flip permutation.

FSL randomise — the volumetric lev2 engine in :mod:`.run` — operates on NIfTI
volumes, and its volumetric TFCE is geometrically invalid on packed surface
vertices. This module is a self-contained one-sample group test on surface
(GIFTI) fixed-effects *effect* maps (the same statistic the volume path group-
analyses): a vertex-wise one-sample t with sign-flip permutation and a
max-statistic null over the WHOLE cortex (both hemispheres combined) for
FWE-corrected p-values. No FSL / PALM dependency.

Inputs are the per-subject surface fixed-effects effect maps produced by lev1:
``{subj}_hemi-{H}_space-{space}_task-…_contrast-…_rtmodel-…_stat-fixed-effects.func.gii``
(``_desc-belowMinRuns_`` files are dropped, matching the volume discover).
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import nibabel as nib
import numpy as np

from network_glm.lev1.processing.surface_data import load_surface_stat_map

HEMIS = ("L", "R")
_SUBJECT_RE = re.compile(r"(sub-[A-Za-z0-9]+)_")


def discover_surface_inputs(level1_dirs: list[Path], contrast_name: str) -> dict[str, list[str]]:
    """Return ``{hemi: sorted[effect .func.gii files]}`` for a contrast.

    Drops ``_desc-belowMinRuns_`` files (subjects with too few retained runs),
    exactly like the volumetric :func:`.run.discover_input_files`.
    """
    out: dict[str, list[str]] = {h: [] for h in HEMIS}
    for level1_dir in level1_dirs:
        for h in HEMIS:
            pattern = str(
                Path(level1_dir)
                / "sub-*"
                / "*"
                / "fixed_effects"
                / f"*_hemi-{h}_*{contrast_name}_rtmodel-*_stat-fixed-effects.func.gii"
            )
            files = [f for f in glob.glob(pattern) if "_desc-belowMinRuns_" not in f]
            out[h].extend(files)
    for h in HEMIS:
        out[h] = sorted(out[h])
    return out


def _subject_of(path: str) -> str:
    m = _SUBJECT_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"Cannot parse subject from surface input filename: {path}")
    return m.group(1)


def load_surface_stack(files: list[str]) -> np.ndarray:
    """Stack per-subject 1-D vertex maps into a ``(n_subjects, n_vertices)`` array."""
    return np.stack([np.asarray(load_surface_stat_map(f), dtype=float) for f in files], axis=0)


def _one_sample_t(data: np.ndarray) -> np.ndarray:
    """Vertex-wise one-sample t across the subject axis (axis 0), NaN-safe."""
    n = data.shape[0]
    mean = np.nanmean(data, axis=0)
    sd = np.nanstd(data, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return mean / (sd / np.sqrt(n))


def sign_flip_permutation_test(
    data: np.ndarray,
    n_perm: int = 5000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """One-sample sign-flip permutation test with whole-array max-statistic FWE.

    Parameters
    ----------
    data : (n_subjects, n_vertices) array
        Per-subject effect-size maps.
    n_perm : int
        Number of sign-flip permutations.
    seed : int
        RNG seed (reproducible).

    Returns
    -------
    (t_obs, fwe_p)
        Both shape ``(n_vertices,)``. Vertices with a non-finite value in any
        subject are NaN in both outputs and excluded from the max statistic.
        ``fwe_p`` is the FWE-corrected p-value
        ``(1 + #{max|t*| >= |t_obs|}) / (n_perm + 1)``.
    """
    data = np.asarray(data, dtype=float)
    n_subj, n_vert = data.shape
    valid = np.all(np.isfinite(data), axis=0)

    t_obs = np.full(n_vert, np.nan)
    fwe_p = np.full(n_vert, np.nan)
    if not valid.any():
        return t_obs, fwe_p

    dv = data[:, valid]
    t_obs[valid] = _one_sample_t(dv)
    abs_obs = np.abs(t_obs[valid])

    rng = np.random.RandomState(seed)
    max_null = np.empty(n_perm)
    for p in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_subj)[:, None]
        max_null[p] = np.nanmax(np.abs(_one_sample_t(signs * dv)))

    # FWE p via the sorted null + searchsorted (memory-light: no (V x P) matrix).
    null_sorted = np.sort(max_null)
    n_ge = n_perm - np.searchsorted(null_sorted, abs_obs, side="left")
    fwe_p[valid] = (1 + n_ge) / (n_perm + 1)
    return t_obs, fwe_p


def _save_gifti(vec: np.ndarray, path: Path) -> None:
    darray = nib.gifti.GiftiDataArray(
        data=np.asarray(vec, dtype=np.float32),
        intent="NIFTI_INTENT_NONE",
        datatype="NIFTI_TYPE_FLOAT32",
    )
    nib.save(nib.GiftiImage(darrays=[darray]), str(path))


def run_surface_level2_analysis(
    contrast_name: str,
    level1_dirs: list[Path],
    output_dir: Path,
    n_perm: int = 5000,
    seed: int = 0,
) -> bool:
    """Whole-cortex one-sample sign-flip group analysis for one surface contrast.

    Loads both hemispheres' effect maps, runs the permutation test over the
    combined cortex (so FWE controls the family-wise error across all vertices
    of both hemispheres), and writes per-hemisphere group t-maps and FWE p-maps
    as GIFTI into ``output_dir / contrast_name``.

    Returns True on success, False if inputs are missing/inconsistent (so the
    caller can propagate the failure without stamping a success manifest).
    """
    inputs = discover_surface_inputs(level1_dirs, contrast_name)
    n_l, n_r = len(inputs["L"]), len(inputs["R"])
    print(f"Surface lev2 for {contrast_name}: {n_l} L / {n_r} R input maps")
    if n_l == 0 or n_r == 0:
        print(f"Error: missing surface inputs for {contrast_name} (L={n_l}, R={n_r})")
        return False

    subj_l = [_subject_of(f) for f in inputs["L"]]
    subj_r = [_subject_of(f) for f in inputs["R"]]
    if subj_l != subj_r:
        print(f"Error: L/R subject sets differ for {contrast_name}: " f"L={subj_l} R={subj_r}")
        return False

    stack_l = load_surface_stack(inputs["L"])  # (N, VL)
    stack_r = load_surface_stack(inputs["R"])  # (N, VR)
    n_vert_l = stack_l.shape[1]
    combined = np.concatenate([stack_l, stack_r], axis=1)  # (N, VL+VR)

    t_obs, fwe_p = sign_flip_permutation_test(combined, n_perm=n_perm, seed=seed)

    out_dir = Path(output_dir) / contrast_name
    out_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "L": (t_obs[:n_vert_l], fwe_p[:n_vert_l]),
        "R": (t_obs[n_vert_l:], fwe_p[n_vert_l:]),
    }
    for h, (t_h, p_h) in split.items():
        _save_gifti(t_h, out_dir / f"{contrast_name}_hemi-{h}_stat-group-t.func.gii")
        _save_gifti(p_h, out_dir / f"{contrast_name}_hemi-{h}_stat-fwe-p.func.gii")
    print(f"--> Surface group maps saved to: {out_dir}")
    return True
