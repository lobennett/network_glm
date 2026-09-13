# Residual computation review — 2026-09-13

Reviewed `network_glm` from `160fba7`, including the corresponding dependency
pin in `network_fmri` at `6b5a179`. Sherlock's deployed `network_fmri` checkout and
installed GLM pin matched those revisions. The numerical checks use synthetic data;
they do not establish the impact on existing participant outputs.

## What a surface residual means

For each run, `runner.process_single_run` aligns events and confounds to the already
trimmed BOLD series, samples task regressors at the derivative sidecar's slice-time
reference, validates the design, and fits each hemisphere. The design includes the
selected task/RT regressors, intercept, and nuisance regressors chosen by
`--confounds-mode`. CIFTI uses the same fitting implementation over grayordinates.

`SurfaceGLM` estimates a separate coefficient vector for every vertex. Nilearn groups
vertices by estimated AR(1) coefficient to fit efficiently; sharing an AR label does
not mean sharing task coefficients. With whitening operator W, the model estimates
beta by least squares on `W Y` and `W X`. The exported residual is `Y - X beta`.
It is neither `W(Y - X beta)` nor `Y - (W X) beta`.

AR(1) residuals need not be orthogonal to X in the ordinary Euclidean metric.
The relevant fitted-model check is `(W X).T @ (W R) ≈ 0`. A nonzero ordinary
task/residual correlation alone therefore does not demonstrate an AR(1) fitting bug.

Surface and CIFTI retain the input signal units. The volume path uses
`FirstLevelModel`'s default signal scaling, so its residual amplitudes are in
percent-signal-change units. The paths use the same noise model but their amplitudes
are not directly interchangeable.

## Confirmed defects and corrections

| Priority | Finding at the reviewed revision | Correction |
|---|---|---|
| High | `process_surface_run` used `continue` for `--skip-qc-plots`, bypassing residual writing and hemisphere return values. | Gate only the plotting loop; test real GIFTI outputs for both hemispheres. |
| High | Surface/CIFTI failure dictionaries were ignored; volume filtering could substitute raw data and label it filtered. | Propagate failures; never substitute an unfiltered series after a filtering error. |
| High | The pinned Nilearn 0.14 volume residual accessor subtracts a prediction based on `whitened_design` from original Y. | Reconstruct volume residuals explicitly from original X and per-voxel beta. This changes volume residuals, not fitted contrasts. |
| High | `--skip-existing` checked residual filenames alone, allowing changed models or damaged outputs to be reused. | Require an atomic completion record matching source, dependency versions, model settings, input hashes, and output hashes. Legacy files are refitted once. |
| High | Nilearn 0.14.0 was yanked because maskers cast cleaned integer-image signals back to integers. | Pin the corrected 0.14.1 release and verify that identical int16/float64 inputs yield the same smoothed GLM estimates. |
| High | Contrast and fixed-effects exceptions could be logged while execution continued as successful. Failed runs could leave old maps available for aggregation. | Fail on requested-output errors and exclude failed run identities from fixed effects. |
| High | Refits could leave old dropped-contrast maps; exclusion refresh could leave eligible fixed-effects maps when the current result was below the run minimum or absent. | Retire exact output families with recoverable `.superseded-<id>` suffixes before replacing them. Test with real GIFTI maps and changed exclusions. |
| Moderate | Surface predictions were cast to the input float32 dtype before subtracting the large baseline. | Subtract in float64, then cast when writing GIFTI/CIFTI. |
| Moderate | `--no-residual-filter` worked only for CIFTI. | Honor it for volume and GIFTI as well. |

The changes preserve the AR(1) estimator, task design, confound arms, and default
filter cutoffs. `fit_run_glm` now has one shared parameter dictionary; retaining
results is the only task-versus-residual configuration difference.

The dependency correction follows the
[Nilearn 0.14.1 release notes](https://nilearn.github.io/stable/changes/whats_new.html#version-0-14-1).
The regression test failed on 0.14.0 with a maximum effect-map difference of `0.00814`
between identical values stored as int16 and float64. With 0.14.1 the difference is
below `1e-6`, consistent with the float32 smoothing used for int16 images. This is
separate from the original-coordinate residual reconstruction fix.

## Numerical verification

Run the offline audit from the repository root:

```bash
uv run --frozen python docs/audit_residuals.py
uv run --frozen pytest -q
```

For the seeded AR(1) example, residuals agree with independently solved GLS to a
maximum absolute error of `1.37e-12`; the maximum whitened design/residual inner
product is `1.84e-10`. This verifies per-vertex coefficients, group reconstruction,
and the use of the original rather than whitened prediction.

Regression tests additionally exercise GIFTI writing, CIFTI failure propagation,
volume residual reconstruction, filter failures, small fluctuations around a large
baseline, cache invalidation, and the fixed-effects failure boundary.

## Scientific decision: downstream FC cleaning

The optional sequence remains:

```text
task/nuisance GLM → original-coordinate residual
                 → optional 0.01–0.1 Hz filtering and tissue/global-signal regression
```

The second regression does not know the first design. If a tissue/global signal
shares task variance, subtracting it from task residuals can reintroduce a task
component. Filtering can also change the relationship to earlier regressors.
This is the projection-order problem demonstrated by
[Lindquist et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6865661/).
[Nilearn's clean](https://nilearn.github.io/stable/modules/generated/nilearn.signal.clean.html)
coordinates filtering with the confounds supplied to that call, not with an earlier
GLM design that it never receives.

The audit isolates this issue with OLS: task/residual correlation is approximately
zero after the task fit, becomes `-0.5968` after the separate tissue regression,
and remains approximately zero under joint regression. This deliberately constructed
example is not an estimate of contamination in the Network data.

For the FC analysis, specify which signals must be removed together and which tool
owns temporal filtering. A joint task-plus-nuisance denoising fit, with consistently
filtered data and regressors, is a reasonable candidate to validate. Keep task-contrast
estimation separate so the denoising choice does not silently change those contrasts.
The current defaults were not changed to a new scientific model during this review.

## Operational implications and remaining review points

Use new result directories when comparing scientific configurations or corrected
residual outputs. The completion records support safe resumption and exclusion-driven
fixed-effects refresh; they do not preserve multiple configurations under shared
residual/QC filenames. Keep the old outputs for a paired audit before a cohort rerun.

Smoothed GIFTI fits deliberately do not reuse completion records until consumed
FreeSurfer geometry and executable identity can be fingerprinted. Prior contrast and
fixed-effects maps are preserved with terminal `.superseded-<id>` suffixes that are
excluded from level-2 discovery, including when no eligible runs remain. Inner
aggregation errors and missing variance pairs now fail visibly rather than returning
empty results that could conceal a failed refresh.

Before using existing FC outputs, inspect their actual flags, hemisphere coverage,
timepoint counts, timing alignment, finite/medial-wall vertices, confound availability,
and filtering history. `get_fc_confounds` currently accepts the available subset of
six tissue/global-signal columns; a study-level completeness policy remains to be set.

Other scientific review points are the hard-coded approximate fixed-effects degrees
of freedom (`100` per run), handling of partial nonfinite surface estimates, and
selection among multiple native FreeSurfer session meshes. These are distinct from
the residual-writing defects and need targeted real-data checks before changing
the study's inference or mesh-selection policy.
