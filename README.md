# network_glm

First- and second-level task-fMRI models for the r01network study. The package
fits run-level models from BIDS events and fMRIPrep derivatives, combines runs
within subjects, runs group permutation tests, and detects cohort outliers.

Pipeline submission belongs in
[network_fmri](https://github.com/lobennett/network_fmri). Use this package
directly to inspect or debug one unit of work.

## Setup

```bash
uv sync --frozen
uv run --frozen network-glm --help
```

FSL is required for volume-level permutation tests. FreeSurfer is required for
surface smoothing. These licensed tools are loaded by `network_fmri` on
Sherlock and are not installed by this package.

## Commands

```text
network-glm {lev1|lev2|cohort-outliers|design-plots} ...
```

Fit one subject and task:

```bash
uv run network-glm lev1 \
    --subj-id sub-s03 \
    --task-name flanker \
    --bids-dir /path/to/bids \
    --fmriprep-dir /path/to/fmriprep \
    --exclusions-file /path/to/exclusions.json \
    --space MNI
```

Pass a `sub-` prefixed subject ID so output filenames remain BIDS-compatible.
Select tasks with `--tasks`, `--all`, `--base-tasks`, or `--dual-tasks`.

Create figures from saved first-level design matrices:

```bash
uv run network-glm design-plots /path/to/lev1 --out-dir figures/
```

Run `network-glm <command> --help` for all options.

## Analysis choices

First-level spaces are `MNI`, `T1w`, `surface`, `fsaverage6`, and `fsLR`.
The `fsLR` path is residual-only and requires `--residuals`. Second-level
models use `volume` or `surface`.

`--confounds-mode` accepts:

| Mode | Regressors |
|---|---|
| `full` | cosine drift, 24-parameter motion, and spikes |
| `no-motion` | cosine drift |
| `no-cosine` | motion and spikes |
| `task-only` | none |

Use `--residuals` to save task residuals, `--fc-confounds` to regress global,
white-matter, and CSF signals, and `--no-residual-filter` to skip the 0.01–0.1
Hz filter. The tissue and global-signal regression follows the task fit, so it
is not equivalent to one joint regression. See
[Residual validation](docs/RESIDUALS-REVIEW.md) for the numerical audit.

Task definitions live in `src/network_glm/task_config/`. The order in
`battery.yaml` defines the task-group options. Go/no-go trial regressors
require `trial_id == "test_trial"`, preventing labeled break rows from being
treated as trials. Optional constituent-condition and dual-task RT maps use the
[diagnostic contrast recipe](docs/DIAGNOSTIC-CONTRASTS.md).

## Outputs

First-level results use this layout:

```text
<results-dir>/sub-<id>/task-<name>/
├── indiv_contrasts/
├── fixed_effects/
├── quality_control/
├── simplified_events/
├── task_residuals/
└── masks/
```

Fixed-effects maps based on fewer than `--min-runs` runs are marked
`_desc-belowMinRuns` and excluded from level 2. Sidecars record the model,
space, smoothing, included runs, inputs, and exclusions. Completion records
allow `--skip-existing` to reuse outputs only when code, settings, inputs, and
outputs still match.

Level 2 rejects duplicate subjects, repeated roots, mixed RT arms, nonfinite
maps, incompatible labels, and invalid output geometry. Successful reruns
retain the prior contrast directory as `.<contrast>.previous-<id>`; remove old
copies manually when they are no longer needed.

Use a separate results directory for each scientific configuration because
some filenames are shared across model variants.

## Development

```bash
uv run --frozen pytest -q
uv build
```

Tests use synthetic data and do not require cluster access or real imaging data.
FSL execution is mocked in CI, so those tests verify interfaces rather than
native permutation statistics.
