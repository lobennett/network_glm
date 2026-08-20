# network_glm

First- and second-level task-fMRI GLMs for the r01network study, plus cohort-level
outlier QC. Extracted from `neuro_workflow@246cdf73` as a faithful lift-and-shift.

The package fits per-run GLMs from BIDS events + fMRIPrep derivatives (`lev1`),
combines runs within a subject via fixed effects, runs group-level permutation
tests (`lev2`), and flags cohort outliers from the resulting contrast maps.
Volumetric, surface, and CIFTI analysis spaces are supported.

---

## Environment

The scientific stack is **pinned**, and installs on Sherlock's CentOS 7 (glibc 2.17)
compute nodes at python 3.11 **and** 3.13: numpy, scipy, statsmodels and matplotlib all
resolve to `manylinux_2_17` wheels, and pandas source-builds cleanly. `pillow<12` is the
one real constraint — pillow 12's source build needs libjpeg headers the host lacks.

```bash
uv sync          # local development
uv run pytest    # 396 tests
```

There is no container. This package is a library, installed as a dependency of
[network_fmri](https://github.com/lobennett/network_fmri), which owns Slurm submission for
every stage of the pipeline. Run the models from there.

FSL (`lev2` volume randomise) and FreeSurfer 8.1 (surface smoothing via `mri_surf2surf`)
are external, licensed tools that were never bundled here either. network_fmri
`module load`s them on the host, and only when a run actually needs them.

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
├── cli.py               top-level dispatch → lev1 | lev2 | cohort-outliers
│                        | design-plots
├── lev1/
│   ├── run.py           CLI entry, argument parsing
│   ├── prepare.py       setup + input discovery
│   ├── subject_config.py  per-subject output layout
│   ├── runner.py        per-run processing + fixed-effects orchestration
│   ├── spaces.py        analysis-space helpers (volume/surface/CIFTI branching)
│   └── processing/
│       ├── events.py          BIDS events → design regressors
│       ├── confounds.py       confound selection, dummy scans, spike regressors
│       ├── design.py          design matrix construction
│       ├── glm.py             model fitting
│       ├── contrasts.py       contrast computation and saving
│       ├── fixed_effects.py   across-run combination
│       ├── residuals.py       residual extraction + band-pass filtering
│       ├── cifti_io.py        fsLR den-91k .dtseries.nii read/write
│       ├── surface_data.py    GIFTI surface loading
│       ├── masks.py           brain-mask intersection
│       ├── imaging.py         image-dtype helpers for map saving
│       └── quality_control.py VIFs, design diagnostics
├── lev2/
│   ├── run.py           group-level CLI; volume path via FSL randomise
│   ├── discover.py      contrast names present in a lev1 tree (used to fan out)
│   └── surface.py       sign-flip permutation group test on GIFTI surfaces
├── cohort/outliers.py   cohort-level outlier QC over lev1 contrast maps
├── config/thresholds.yaml  study-level thresholds (package data, see thresholds.py)
├── task_config/
│   ├── battery.yaml     base/dual task lists (canonical order)
│   ├── tasks/*.yaml     per-task regressors, contrasts, parameters
│   └── loader.py        YAML load + contrast validation
├── io/file_discovery.py BIDS + fMRIPrep file discovery
├── qc/design_plots.py   post-hoc design/contrast/correlation figures
├── exclusions.py        scan-level exclusion handling (--exclusions-file)
├── provenance.py        provenance primitives for analysis-stage outputs
├── provenance_graph.py  assembles the full chain into one machine-readable graph
├── thresholds.py        study-level thresholds as code (config/thresholds.yaml, package data)
├── acquisition.py       scanner-acquisition constants
└── task_utils.py        task helpers
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
