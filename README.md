# network_glm

First- and second-level task-fMRI GLMs for the r01network study, plus cohort-level
outlier QC. Extracted from `neuro_workflow@246cdf73` as a faithful lift-and-shift.

The package fits per-run GLMs from BIDS events + fMRIPrep derivatives (`lev1`),
combines runs within a subject via fixed effects, runs group-level permutation
tests (`lev2`), and flags cohort outliers from the resulting contrast maps.
Volumetric, surface, and CIFTI analysis spaces are supported.

---

## Environment

The scientific stack is **pinned** and the project targets **Python 3.11**. This is
not incidental: Sherlock's compute nodes are CentOS 7 (glibc 2.17), where newer
scipy/pillow releases ship no usable wheel and fall back to source builds that fail.
`scipy==1.14.1` is the newest release with a working cp311 manylinux2014 wheel.

**On Sherlock, run from the container.** A bare `uv sync` on a compute node cannot
build scipy.

```bash
apptainer exec /home/groups/russpold/singularity_images/network_glm.sif \
    network-glm --help
```

FSL (`lev2` volume randomise) and FreeSurfer 8.1 (surface smoothing via
`mri_surf2surf`) are deliberately **not** baked into the image — the sbatch
templates `module load` them on the host and they resolve through `apptainer exec`.

For local development and the offline test suite:

```bash
uv sync          # python 3.11, see .python-version
uv run pytest    # 314 tests across 53 files
```

---

## Launching the models

`network-glm` dispatches to five subcommands:

```
network-glm {lev1|lev2|cohort-outliers|design-plots|submit} ...
```

`submit` is the normal entry point on a cluster — it renders an sbatch template,
writes a job list, and submits a Slurm array (one array task per subject × task).
The bare `lev1`/`lev2` commands run a single unit in the foreground and are mostly
for debugging or interactive work.

> `--partition` is **required** on every `submit` command. Sherlock has no default
> partition, and jobs submitted without one are rejected.

### First level (`submit lev1`)

One array task per subject × task.

```bash
network-glm submit lev1 \
    --subjects s03 s10 s19 \
    --base-tasks \
    --bids-dir   /oak/.../bids/discovery \
    --fmriprep-dir /scratch/.../derivatives/fmriprep_25.2.4 \
    --results-dir  /scratch/.../lev1_out \
    --exclusions-file /path/to/discovery_lock.json \
    --space fsLR \
    --residuals \
    --partition russpold
```

Task selection is one of `--tasks <names...>`, `--all`, `--base-tasks`, or
`--dual-tasks`. Subjects are bare IDs (`s03`, not `sub-s03`).

Default resources: **1 CPU, 64 GB, 2 days** — override with `--nthreads`,
`--mem-gb`, `--time`. Use `--print-only` to render the sbatch script without
submitting it.

### Second level (`submit lev2`)

One array task per contrast.

```bash
network-glm submit lev2 \
    --contrasts nBack_twoBack-oneBack \
    --level1-dirs /scratch/.../lev1_out \
    --results-dir /scratch/.../lev2_out \
    --space volume \
    --num-permutations 5000 \
    --partition russpold
```

Contrast selection mirrors lev1: `--contrasts`, `--all`, `--base-tasks`,
`--dual-tasks`. Default resources: **2 CPUs, 4 GB, 4 hours**.

### Cohort outlier QC (`submit outliers`)

Scans a completed lev1 tree for subjects whose contrast maps or design collinearity
fall outside cohort norms.

```bash
network-glm submit outliers \
    --results-dir /scratch/.../lev1_out \
    --partition russpold
```

Defaults: `--n-std 3.0` (SD threshold for outlier voxels), `--vif-threshold 5.0`,
`--outlier-pct-threshold 10.0`. Resources: **2 CPUs, 16 GB, 1 hour**.

### Single run, no Slurm

```bash
network-glm lev1 \
    --subj-id sub-s03 --task-name flanker \
    --bids-dir ... --fmriprep-dir ... --exclusions-file ... \
    --space MNI
```

Note `--subj-id` takes the **`sub-` prefixed** form here, unlike `submit --subjects`.

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
│                        | design-plots | submit
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
│   └── surface.py       sign-flip permutation group test on GIFTI surfaces
├── cohort/outliers.py   cohort-level outlier QC over lev1 contrast maps
├── submit/              Slurm array submission
│   ├── lev1.py / lev2.py / outliers.py
│   ├── _slurm.py        template rendering, sbatch submission, resource resolution
│   └── templates/       lev1.sbatch, lev2.sbatch, outliers.sbatch
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
