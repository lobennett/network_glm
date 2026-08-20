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
stage. Run the models from there. `pillow<12` is the one hard pin — pillow 12 needs libjpeg headers
Sherlock lacks.

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
0.01–0.1 Hz band-pass on the CIFTI path, deferring temporal filtering to XCP-D so
you don't double-band-pass).

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
├── task_residuals/      residual timeseries (with --residuals)
└── masks/               per-run and intersected brain masks
```

Fixed-effects maps computed from fewer than `--min-runs` (default 2) runs are tagged
`_desc-belowMinRuns` and filtered out by `lev2` rather than silently dropped.

---

## Codebase overview

```
src/network_glm/
  cli.py            dispatch: lev1 | lev2 | cohort-outliers | design-plots
  lev1/             per-run fitting: prepare -> runner -> processing/*
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
uv run pytest                    # full suite
uv run pytest tests/lev1 -v      # first-level only
```

Tests are offline — no cluster or real imaging data required.
