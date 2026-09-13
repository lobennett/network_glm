# Opt-in coefficient-map diagnostics

The existing Python contrast API can export individual condition or pooled
response-time coefficients from a fitted GLM. This recipe makes seven historical
diagnostic definitions reusable without editing canonical task YAML, refitting the
model, or adding maps to normal `network-glm lev1` output.

The four condition contrasts extract the congruent/incongruent coefficients for
flanker and two cue-switch condition coefficients for cuedTS. The other three
extract the existing pooled `response_time` coefficient for the discovery dual
tasks. These are coefficients relative to the fitted model's implicit baseline;
they are not absolute activation or correlation/reliability estimates. Subtracting
the two constituent **effect-size** maps gives the corresponding difference
effect; subtracting their z maps does not give a difference z map.

## Inputs and output isolation

Use this only with an already fitted `SurfaceGLM` (one hemisphere) or single-run
Nilearn `FirstLevelModel` from the intended task and RT arm. Keep these variables
in scope before executing the example:

- `fitted_glm`: that fitted object. Saved contrast maps are not a serialized fit;
  missing coefficients cannot be recovered from existing cognitive maps alone.
- `design_columns`: the actual fitted design's columns, after zero-variance column
  removal. For `SurfaceGLM`, use `fitted_glm.design_matrix_.columns`; for a
  single-run `FirstLevelModel`, use `fitted_glm.design_matrices_[0].columns`.
- `task_name` and `rt_model`: the task and arm used in that fit, not a request to
  change its model. RTDur includes the pooled RT regressor. noRT and RTepoch do
  not; the latter is only supported for tasks allowed by the canonical loader.
- `diagnostic_dir`: a **new, dedicated directory outside the production lev1
  results tree**, for example `Path("diagnostics/flanker-RTDur/run-01/hemi-L")`.
  The example refuses an existing directory instead of overwriting its contents.
  Use distinct directories for each run, hemisphere, space and scientific arm.
- `base_filename`: the run identity, such as
  `sub-synthetic_ses-01_task-flanker_run-1`.
- `hemisphere` and `surface_space`: e.g. `"L"` and `"fsaverage6"` for GIFTI;
  set both to `None` for a volume fit. This example does not export CIFTI.

The example does not perform participant discovery, fitting, exclusions or
aggregation. Producing a new participant fit is a separate analysis action.
There is no CLI custom-task-config switch implied by this Python recipe.

## Export the explicitly selected maps

This is the executable example exercised by
`tests/lev1/test_diagnostic_recipe.py`. Each formula below is a single coefficient
name, so missing design columns can be passed directly to the existing filter.

```python
from pathlib import Path

from network_glm.lev1.processing.contrasts import (
    compute_run_contrasts,
    filter_contrasts_for_dropped_columns,
)
from network_glm.task_config.loader import RT_MODELS

extra = {
    "flanker": {
        "congruent-baseline": "congruent",
        "incongruent-baseline": "incongruent",
    },
    "cuedTS": {
        "task_stay_cue_switch-baseline": "task_stay_cue_switch",
        "task_switch_cue_switch-baseline": "task_switch_cue_switch",
    },
    "directedForgettingWFlanker": {"response_time": "response_time"},
    "stopSignalWDirectedForgetting": {"response_time": "response_time"},
    "stopSignalWFlanker": {"response_time": "response_time"},
}

if rt_model not in RT_MODELS:
    raise ValueError(f"Unknown rt_model: {rt_model!r}")
requested = extra[task_name].copy()
missing = sorted(set(requested.values()) - set(design_columns))
selected, skipped = filter_contrasts_for_dropped_columns(requested, missing)

saved = {}
if selected:
    output_dir = Path(diagnostic_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    saved = compute_run_contrasts(
        fitted_glm,
        task_name,
        output_dir,
        base_filename,
        contrasts=selected,
        hemisphere=hemisphere,
        surface_space=surface_space,
        rt_model=rt_model,
    )
print("Saved diagnostic contrasts:", sorted(saved))
print("Skipped absent columns:", skipped)
```

`saved` contains effect-size, variance and z-score paths for each selected map.
If no requested coefficient is available, no directory is created. A dropped
column skips its contrast; the recipe never restores a regressor or alters the
fit. The fitted design's columns are the only selection rule: because noRT and
RTepoch designs carry no pooled `response_time` regressor, the RT-only dual-task
requests land in `skipped` under those arms and produce no maps, while `rt_model`
only labels the arm that the supplied fit already used. An unknown task is
rejected by the explicit mapping rather than falling back to defaults.
If export fails partway through, retain that directory for inspection and choose
a new directory for a retry.

## Interpreting and retaining the diagnostics

Exporting additional contrasts from the same fitted object leaves its coefficients,
existing effects and residuals unchanged. Adding these definitions to canonical
YAML would nevertheless change default output generation, source-based reuse
signatures and potentially downstream contrast discovery. Keep them opt-in.

The normal runner does not register these standalone exports in its completion
records. Record the source fit, design, task/arm and exclusion context with your
diagnostic work; do not insert these maps into an existing production completion
record. If explicitly aggregating them later, the existing
`FixedEffectsAnalyzer.compute_all_task_fixed_effects(..., contrasts=selected)`
API accepts custom definitions. That separate action still needs correct run and
contrast exclusions and run-minimum settings. Keep its output outside production:
level-2 discovery can find additional fixed-effects maps placed in a lev1 tree.

These seven mappings come from two exploratory diagnostic efforts — a
condition-betas exploration and a dual-response-time-maps exploration — that
were never published to this repository; the definitions above are the whole of
what they contribute, so nothing here depends on recovering that history.
Those explorations also
produced a break-selector experiment, a dated participant launcher and narrow
event/reliability summaries. They are not prerequisites for this recipe. In
particular, selecting ordinary breaks instead of feedback-labeled breaks changes
the fitted design and is a separate scientific-model alternative. This recipe
does not adopt that experiment or choose a new RT arm.
