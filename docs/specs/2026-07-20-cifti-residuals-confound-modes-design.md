# Design: network_glm confound-modes + fsLR-CIFTI residual output

**Date:** 2026-07-20
**Status:** Approved (design)
**Scope:** network_glm lev1 only.

## Problem / motivation

Task-residual FC from network_glm's "full" first-level model scores much lower on
the network similarity index (NSI, Du et al. 2026 Neuron) than the task-only
"minimal" residuals from `network_lev1_residuals`. Sweeping the *downstream*
XCP-D denoising (ABCD, off, detrend-only, tissue+GSR, despike) did **not** fix it,
which rules out XCP-D and points at the **first-level design matrix** — specifically
the nuisance regressors network_glm adds that the minimal pipeline omits: the
**24-parameter Friston motion**, the **`motion_outlier` spike regressors**, and the
**fMRIPrep `cosine` DCT high-pass** (see the regressor diff in the session notes).

To test which regressors are responsible, we need to run network_glm's lev1 in
graded nuisance configurations and feed the residuals into the *same* NSI pipeline
as the minimal residuals. That pipeline (Gracie's
`.../pfm_compare/code/fmriprep_xcpd/postproc/concatenate_xcpd_*.py`) consumes
**fsLR den-91k `*_bold.dtseries.nii`** → XCP-D → concatenate → MSHBM/NSI. network_glm
does **not** currently emit that: its `fsLR` space is misclassified as a surface
(GIFTI-per-hemi) space and the runner KeyErrors because fsLR file-discovery yields
the `cifti_bold` dtseries, not `left_surface`/`right_surface`. So this feature also
**makes `--space fsLR` functional**.

## Requirements

1. A single flag that selects the nuisance configuration, giving three one-flag arms
   that differ *only* in the confound regressors.
2. fsLR den-91k **dtseries** residual output that is a drop-in for the postproc glob
   `*_space-fsLR_den-91k_*bold.dtseries.nii`.
3. No change to the existing GIFTI (fsnative/fsaverage6) or volumetric paths, or to
   the task regressors.

## §1 — `--confounds-mode {full, no-motion, task-only}`

Default `full` (current behavior; back-compatible). Threaded from `lev1/run.py`
through `runner` into `confounds.load_and_process_confounds` →
`_get_base_confound_pattern(task_name, sample_type, confounds_mode)`:

| mode | regressors kept from the fMRIPrep confounds TSV |
|---|---|
| `full` | `cosine*` + 24p Friston motion (`trans/rot` × {—,deriv1,power2,deriv1_power2}) + `motion_outlier\d+` |
| `no-motion` | `cosine*` only (drop the 24 motion params **and** the spike regressors) |
| `task-only` | none (drop `cosine*` too) |

For `task-only`, `design.create_design_matrix` already appends a `constant`
intercept when no `cosine00` is present — no separate handling needed. Task
regressors (per `task_config/tasks/*.yaml`) are identical across all three modes.
The existing `thresholds.confounds_cosine_caps` cosine cap still applies to whichever
modes keep `cosine*` (`full`, `no-motion`).

## §2 — fsLR-CIFTI residual path

**Port** `network_lev1_residuals/cifti_io.py` → `network_glm/lev1/processing/cifti_io.py`:
- `load_dtseries(path) -> (data[T,91282] float32, template Cifti2Image)`
- `write_dtseries(data, template, out_path)` — reuses `template.header` (SeriesAxis +
  BrainModelAxis) and `nifti_header` verbatim; validates identical shape. This is the
  robust way to emit a valid den-91k grayordinate file.

**New dispatch branch** in `runner.py`. Introduce `is_cifti_space(space)` (True for
`fsLR`) and route `fsLR` to a new `process_cifti_run` **instead of** the surface
branch (`is_surface_space` must no longer claim `fsLR`; surface = `surface`/`fsaverage6`
only):

1. `data, template = load_dtseries(run_files["cifti_bold"])`; `n_scans = data.shape[0]`.
2. Build events + confounds (confounds-mode) + design matrix — identical to the other
   paths (space-agnostic; `create_design_matrix` unchanged).
3. Zero-variance handling + QC (VIF) as today.
4. Fit the existing `SurfaceGLM(t_r=tr)` on the `(T, 91282)` matrix (it is space-agnostic
   — `run_glm` on a 2-D array + `get_residuals()`).
5. Residuals `Y − Xβ`; optional temporal filtering + `--fc-confounds` via
   `nilearn.signal.clean` (same params as the surface path: `low_pass=0.1`,
   `high_pass=0.01`, `standardize=False`, `detrend=False`) — reuse a shared helper.
6. `write_dtseries(residuals, template, dirs["task_residuals"] /
   f"{base_filename}_space-fsLR_den-91k_task-regressed-residuals.dtseries.nii")`.

**Output format is space-driven** (no separate `--output-space` flag): `fsLR`→dtseries,
`fsnative`/`fsaverage6`→GIFTI (unchanged), `MNI`/`T1w`→nii.gz. `--skip-existing` gets a
dtseries filename branch mirroring the surface/volumetric ones.

**Residuals-focused:** the fsLR path wires **residuals only**. It therefore
**requires `--residuals`**; invoked without it, `process_cifti_run` errors with a
clear message (rather than silently producing nothing). Contrast maps as CIFTI
dscalar are out of scope (the GIFTI/volumetric contrast paths are untouched).

## §3 — Testing

Unit:
- `_get_base_confound_pattern` for each mode selects exactly the intended columns from
  a synthetic confounds frame (full → motion+cosine+spike; no-motion → cosine only,
  no `trans_`/`rot_`/`motion_outlier`; task-only → empty).
- `write_dtseries` round-trip: load a small synthetic dtseries, write it back, assert
  identical data + preserved axes; shape-mismatch raises.
- `process_cifti_run` on a synthetic `(T, N)` dtseries + minimal events → a residual
  dtseries of the right shape/axes; residual is orthogonal to the design columns.

Operational (on `sub-s03`, one real fsLR run per mode):
- Produces a `*_space-fsLR_den-91k_*_task-regressed-residuals.dtseries.nii` that
  `nib.load`s and matches the postproc glob; confounds-mode changes the design-matrix
  column count as expected (full ≫ no-motion > task-only).

## Out of scope
- fsLR contrast (dscalar) outputs.
- Any change to XCP-D, the postproc concatenation, or the NSI computation.
- fsnative/fsaverage6 GIFTI residuals for MSHBM are produced by the existing
  `--space` runs (unchanged), not this feature.

## The experiment this enables
```
network_glm lev1 ... --space fsLR --residuals --confounds-mode {full,no-motion,task-only}
  → XCP-D (abcd) → concatenate_xcpd_* → MSHBM/NSI
```
Interpretation: no-motion recovers NSI ⇒ motion double-regression is the cause;
task-only needed ⇒ cosine also contributes; task-only still low ⇒ the task model itself.
