# network_glm

First- and second-level task-fMRI GLMs for the r01network study, plus cohort-level outlier QC.
Fits per-run GLMs from BIDS events + fMRIPrep derivatives (`lev1`), combines runs within a subject
by fixed effects, runs group-level permutation tests (`lev2`), and flags cohort outliers from the
resulting contrast maps. Volumetric, surface and CIFTI spaces are supported.

## Environment

```bash
uv sync
uv run pytest
```

No container: this is a library, installed as a pinned dependency of
[network_fmri](https://github.com/lobennett/network_fmri), which owns Slurm submission for every
stage. Run the models from there. The numerical stack is pinned; Nilearn 0.14.1 avoids
the withdrawn 0.14.0 integer-image precision bug. `pillow<12` avoids libjpeg build
requirements on Sherlock.

FSL (`lev2` volume randomise) and FreeSurfer (surface smoothing via `mri_surf2surf`) are external
licensed tools, never bundled. network_fmri `module load`s them on the host, and only when a run
actually needs them.

## Running the models

Four subcommands, each fitting or plotting one unit in the foreground:

```
network-glm {lev1|lev2|cohort-outliers|design-plots} ...
```

**Submission lives in network_fmri**, which fans each level out over a Slurm array, sets
per-level resources and loads the host modules the level needs:

```bash
network_fmri glm-lev1 --cohort discovery --base-tasks --results-dir <out> -- \
    --bids-dir <bids> --fmriprep-dir <fmriprep> --exclusions-file <lock.json> --residuals
network_fmri glm-lev2 --lev1-dirs <lev1_out> --all --results-dir <out> -- \
    --num-permutations 5000
network_fmri glm-outliers --results-dir <lev1_out> --
```

Everything after `--` is passed through to the runners below untouched, so this package
remains the only place that defines what those flags mean.

### Direct invocation

Useful for debugging one unit interactively:

```bash
network-glm lev1 \
    --subj-id sub-s03 --task-name flanker \
    --bids-dir ... --fmriprep-dir ... --exclusions-file ... \
    --space MNI
```

`--subj-id` accepts either `s03` or `sub-s03` for file discovery, but it is interpolated
**raw** into output filenames — so pass the `sub-` prefixed form or the outputs are not
BIDS-named. network_fmri always passes the prefixed form.

Task selection for lev1 is `--tasks <names...>`, `--all`, `--base-tasks` or `--dual-tasks`;
lev2 mirrors it for contrasts, discovering them from the lev1 tree via
`network_glm.lev2.discover` when not named explicitly.

### QC figures (`design-plots`)

Post-hoc, over persisted lev1 design matrices — no refit:

```bash
network-glm design-plots /scratch/.../lev1_out --out-dir figures/
```

Renders the design matrix, contrast matrix, and regressor correlation matrix.
`--omit-drift` drops drift/constant regressors from the correlation plot.

---

## Analysis spaces

| `--space` | Data | Notes |
|---|---|---|
| `MNI` (default) | volumetric NIfTI | `--mni-template` (default `MNI152NLin6Asym`), `--mni-res` (default `2`) |
| `T1w` | volumetric NIfTI | native subject space |
| `surface` | fsnative GIFTI | smoothing shells out to FreeSurfer `mri_surf2surf` |
| `fsaverage6` | fsaverage6 GIFTI | |
| `fsLR` | fsLR den-91k CIFTI | dense timeseries; residuals-only, requires `--residuals` |

`lev2` takes a different `--space`: `volume` (FSL randomise) or `surface`
(self-contained sign-flip permutation across both hemispheres, whole-cortex FWE,
seeded by `--seed` for reproducibility).

## Confound models

`--confounds-mode` selects the nuisance regressors in the lev1 design. These are the
arms of the Network Similarity Index experiment (task-residual FC vs rest FC):

| Mode | Regressors |
|---|---|
| `full` (default) | cosine drift + 24p Friston motion + spike regressors |
| `no-motion` | cosine drift only |
| `no-cosine` | 24p motion + spikes, no drift model |
| `task-only` | none |

Related residual flags: `--residuals` (emit them at all), `--fc-confounds` (regress
global signal / WM / CSF, per Du et al. 2025), and `--no-residual-filter` (skip the
0.01–0.1 Hz band-pass in any space, deferring temporal filtering to a downstream
tool such as XCP-D). `--no-residual-filter` does not disable `--fc-confounds`.

Residuals use each voxel/vertex's own coefficients and the original design:
`R = Y - X @ beta`. The AR(1) model estimates beta after whitening; the saved
residual is in the original time coordinates. Surface/CIFTI values retain their
input units; the volume fit uses Nilearn's default percent-signal scaling.
`--skip-qc-plots` affects plots only. Failure to write a requested residual or
contrast fails the run, and failed runs cannot contribute stale maps to fixed effects.

The optional tissue/global-signal regression follows the task fit. These sequential
operations can reintroduce task-related signal; they are not equivalent to one joint
regression. The [residual review](docs/RESIDUALS-REVIEW.md) explains the distinction
and includes a reproducible numerical audit.

### Resuming a run

`--skip-existing` requires a completed per-run record under `task_residuals/` with
matching source code, dependency versions, scientific settings, and input contents.
It also verifies the recorded output contents. This works for contrast-only runs as
well as runs with residuals. Hashing reads the files; reuse is not a metadata-only check.

A change to the exclusion lock or `--min-runs` still refreshes fixed effects without
refitting unchanged included runs. Old output files without completion records are
refitted once. A failed or interrupted replacement fit cannot reuse an older receipt.

Smoothed GIFTI runs currently always refit: their external FreeSurfer meshes and
executable are not yet fingerprinted. Unsmoothed surface, CIFTI, and volume fits can
use the completion check.

Replacement fits retire prior run contrast maps, and fixed-effects refresh retires
both normal and below-minimum maps before writing replacements. These files are
preserved with a terminal `.superseded-<id>` suffix, so level 2 cannot discover them.
This also prevents dropped contrasts and newly excluded subjects from contributing
old maps. Archived files remain recoverable; none are automatically deleted.

Use a separate results directory for each scientific configuration. Contrast filenames
include the RT arm, but residual, QC, and subject-manifest filenames are shared within
a subject/task directory. Sharing that directory across arms can overwrite those files.

## Task battery

Defined in `src/network_glm/task_config/battery.yaml`; per-task regressors and
contrasts live in `src/network_glm/task_config/tasks/<task>.yaml`.

- **8 base tasks** (one paradigm per run): `cuedTS`, `directedForgetting`, `flanker`,
  `goNogo`, `nBack`, `shapeMatching`, `spatialTS`, `stopSignal`
- **11 dual tasks** (two concurrent paradigms per run): `directedForgettingWCuedTS`,
  `directedForgettingWFlanker`, `stopSignalWDirectedForgetting`, `stopSignalWFlanker`,
  `spatialTSWCuedTS`, `flankerWShapeMatching`, `cuedTSWFlanker`,
  `spatialTSWShapeMatching`, `nBackWShapeMatching`, `nBackWSpatialTS`,
  `shapeMatchingWCuedTS`

Battery order is canonical — `--all` / `--base-tasks` / `--dual-tasks` resolution
depends on it. Do not reorder without a behavior-preserving audit.

## Outputs

`lev1` writes per subject × task under `--results-dir`:

```
<results-dir>/sub-<id>/task-<name>/
├── indiv_contrasts/     per-run contrast maps
├── fixed_effects/       within-subject across-run combination
├── quality_control/     contrast VIFs, design diagnostics, QC plots
├── simplified_events/   the event model actually fit
├── task_residuals/      residual timeseries (with --residuals) and completion records
└── masks/               per-run and intersected brain masks
```

Fixed-effects maps computed from fewer than `--min-runs` (default 2) runs are tagged
`_desc-belowMinRuns` and filtered out by `lev2` rather than silently dropped.

Per-run surface contrast maps include their mesh as `_space-fsnative` or
`_space-fsaverage6`. The fixed-effects directory contains the conventional across-run
effect, variance, and fixed-effects Z maps, plus a `_stat-meanRunZ` map for every
contrast. `meanRunZ` is the voxel/vertex-wise arithmetic mean of the contributing
per-run Z maps (ignoring NaNs, and preserving NaN where every run is NaN), as needed
to reproduce the Figure 8c aggregation in Ladwig et al. It is not the conventional
fixed-effects Z map: the latter combines effects and variances before converting the
result to Z.

Each fixed-effects contrast also has a `_stat-fixed-effects.json` sidecar recording
the contrast formula, analysis space, smoothing FWHM, aggregation methods, included
run count and identities, contributing effect/variance/Z files, and applicable
scan-level or contrast-level exclusions.

---

## Codebase overview

```
src/network_glm/
  cli.py            dispatch: lev1 | lev2 | cohort-outliers | design-plots
  lev1/             per-run fitting: prepare -> runner -> processing/*; cache.py checks reuse
  lev1/processing/  events, confounds, design, glm, contrasts, fixed_effects,
                    residuals, masks, quality_control, and the surface/CIFTI IO
  lev2/             group level: run.py (FSL randomise), surface.py (sign-flip
                    permutation), discover.py (contrast names, for fan-out)
  cohort/           outlier QC over lev1 contrast maps
  task_config/      battery.yaml + tasks/<task>.yaml + loader with validation
  config/           thresholds.yaml, packaged as data (read via thresholds.py)
  io/               BIDS + fMRIPrep file discovery
  qc/               post-hoc design/contrast/correlation figures
  exclusions.py     --exclusions-file handling
  provenance.py     input manifests; provenance_graph.py assembles the chain
```

### Provenance

Every lev1/lev2 run records a manifest hashing the inputs the model actually
consumed — events, confounds, and the BOLD timeseries for whichever space ran.
Derived intermediates (brain masks) are intentionally excluded.

Runs against an uncommitted working tree warn loudly to stderr but still proceed;
the manifest records `code_dirty` truthfully either way. `--allow-dirty` suppresses
the warning, not the record.

## Development

```bash
uv sync --frozen --group dev
uv run --frozen pytest -q -ra          # full suite; report skips and failures
uv run --frozen pytest tests/lev1 -v   # first-level only
uv run --frozen pytest tests/lev2 -v   # group-level only
```

Tests use generated fixtures — no cluster or participant imaging data required.
[GitHub Actions](.github/workflows/tests.yml) runs the full suite on pushes and pull
requests with Python 3.13 from `.python-version` and dependencies from `uv.lock`.
Coverage includes first-level volume/surface/CIFTI processing, group surface
sign-flip permutations, volume group command/provenance/error handling, cohort QC,
plots, and the CLI. Dependency installation needs network access; the tests run offline.

FSL `randomise` execution is mocked in the volume group tests; script assertions
check command construction, not native FSL results. Native FSL, FreeSurfer
`mri_surf2surf`, and Sherlock/Slurm execution are not validated in generic Linux CI.
