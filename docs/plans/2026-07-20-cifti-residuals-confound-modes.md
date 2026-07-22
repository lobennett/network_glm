# network_glm confound-modes + fsLR-CIFTI residuals — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add `--confounds-mode {full,no-motion,task-only}` and a functional `--space fsLR` path that writes task residuals as an fsLR den-91k `.dtseries.nii`, so the NSI experiment (motion vs no-motion vs task-only) can run through XCP-D→MSHBM/NSI.

**Architecture:** (1) a confounds-mode switch in `confounds.py` selects which nuisance columns enter the design matrix; (2) a new CIFTI dispatch branch loads the den-91k dtseries, fits the existing space-agnostic `SurfaceGLM` on the `(T,91282)` matrix, and writes residuals back as a dtseries reusing the input CIFTI header.

**Tech Stack:** Python 3.11, nibabel (cifti2), nilearn (`run_glm`, `signal.clean`), pandas, pytest.

**Spec:** `docs/specs/2026-07-20-cifti-residuals-confound-modes-design.md`
**Branch:** `feat/cifti-residuals-confound-modes` (spec committed at `41b39dd`).

---

## Environment (Task 0 — do first; blocks test execution)

network_glm's neuroimaging deps live in the `.[lev1]` optional extra; tests
`pytest.skip` without them. On Sherlock compute nodes the env must be **Python
3.11** (3.13 lacks wheels for the locked scientific stack) with deps installed
**wheels-forced** (scipy/numpy/nibabel/nilearn source-build otherwise).

- [ ] **Step 1: Build the test venv** (adjust if a blessed recipe exists / CI env differs)

```bash
module load uv/0.9.5
export UV_CACHE_DIR=/scratch/users/logben/uv_cache
uv python install 3.11
ENV=/scratch/users/logben/network_glm_test_venv
uv venv --python 3.11 "$ENV"
uv pip install --python "$ENV/bin/python" --only-binary=:all: -e '.[lev1]' pytest
"$ENV/bin/python" -c "import pytest,nibabel,nilearn,scipy,pandas; print('env ok')"
```

- [ ] **Step 2: Confirm the suite runs (baseline green)**

Run: `UV_PROJECT_ENVIRONMENT=$ENV uv run --no-sync pytest tests/lev1 -q -p no:cacheprovider`
Expected: existing tests pass (no `Skipped: neuroimaging dependencies not installed`).

> All later task test commands assume `module load uv/0.9.5`,
> `export UV_PROJECT_ENVIRONMENT=/scratch/users/logben/network_glm_test_venv`,
> and run with `uv run --no-sync pytest … -p no:cacheprovider`.
> Note: tasks 1–2 (confounds regex, spaces) only need pandas+pytest; tasks 3–5
> need nibabel/nilearn/scipy from the `.[lev1]` extra.

---

## File Structure
- Modify `src/network_glm/lev1/processing/confounds.py` — `confounds_mode` param.
- Modify `src/network_glm/lev1/run.py` — `--confounds-mode` CLI arg.
- Modify `src/network_glm/lev1/spaces.py` — `is_cifti_space`; drop fsLR from `is_surface_space`.
- Create `src/network_glm/lev1/processing/cifti_io.py` — dtseries load/write (port).
- Modify `src/network_glm/lev1/processing/residuals.py` — `process_cifti_residuals`.
- Modify `src/network_glm/lev1/runner.py` — `process_cifti_run` + CIFTI dispatch + skip-existing + confounds-mode threading.
- Tests: `tests/lev1/test_confounds_modes.py`, `test_cifti_io.py`, `test_cifti_residuals.py`, `test_spaces_cifti.py`.

---

## Task 1: `--confounds-mode` confound selection

**Files:** Modify `processing/confounds.py`; Modify `lev1/run.py`; Modify `runner.py:332`; Test `tests/lev1/test_confounds_modes.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/lev1/test_confounds_modes.py
import pandas as pd
from network_glm.lev1.processing.confounds import load_and_process_confounds


def _write_confounds(tmp_path):
    cols = {
        "cosine00": [1.0, 1.0, 1.0], "cosine01": [0.1, 0.2, 0.3],
        "trans_x": [0.0, 0.1, 0.2], "trans_x_derivative1": [0, 0.1, 0.1],
        "trans_x_power2": [0, 0.01, 0.04], "rot_z": [0.0, 0.01, 0.02],
        "motion_outlier00": [0, 1, 0],
        "csf": [1.0, 1.1, 1.2], "white_matter": [2.0, 2.1, 2.2],
        "global_signal": [3.0, 3.1, 3.2],
    }
    p = tmp_path / "confounds.tsv"
    pd.DataFrame(cols).to_csv(p, sep="\t", index=False)
    return p


def test_full_keeps_motion_cosine_spikes(tmp_path):
    df = load_and_process_confounds(_write_confounds(tmp_path), "stopSignal",
                                    "validation", confounds_mode="full")
    assert any(c.startswith("cosine") for c in df.columns)
    assert "trans_x" in df.columns and "motion_outlier00" in df.columns
    assert "csf" not in df.columns and "global_signal" not in df.columns


def test_no_motion_drops_motion_and_spikes(tmp_path):
    df = load_and_process_confounds(_write_confounds(tmp_path), "stopSignal",
                                    "validation", confounds_mode="no-motion")
    assert any(c.startswith("cosine") for c in df.columns)
    assert not any(c.startswith(("trans_", "rot_")) for c in df.columns)
    assert not any(c.startswith("motion_outlier") for c in df.columns)


def test_task_only_drops_everything(tmp_path):
    df = load_and_process_confounds(_write_confounds(tmp_path), "stopSignal",
                                    "validation", confounds_mode="task-only")
    assert df.shape[1] == 0
    assert len(df) == 3  # rows preserved for design-matrix concat
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --no-sync pytest tests/lev1/test_confounds_modes.py -q -p no:cacheprovider`
Expected: FAIL — `load_and_process_confounds() got an unexpected keyword argument 'confounds_mode'`.

- [ ] **Step 3: Implement.** In `processing/confounds.py`:

Change `_get_base_confound_pattern` to build the pattern from a cosine part + a
motion part and honor the mode:

```python
def _get_base_confound_pattern(task_name: str, sample_type: str,
                               confounds_mode: str = "full") -> str:
    from network_glm.thresholds import confounds_cosine_caps
    cosine = "cosine"
    max_idx = confounds_cosine_caps().get(sample_type, {}).get(task_name)
    if max_idx is not None:
        cosine = f"cosine0[0-{int(max_idx)}]"
    motion = (
        "trans_[xyz]$|trans_[xyz]_derivative1$|trans_[xyz]_power2$|"
        "trans_[xyz]_derivative1_power2$|rot_[xyz]$|rot_[xyz]_derivative1$|"
        "rot_[xyz]_power2$|rot_[xyz]_derivative1_power2$|motion_outlier\\d+"
    )
    if confounds_mode == "no-motion":
        return cosine
    return f"{cosine}|{motion}"
```

Add `confounds_mode: str = "full"` to `load_and_process_confounds`'s signature
and short-circuit `task-only` to an empty-column frame (rows preserved) before
filtering:

```python
    if dummy_scans > 0:
        confounds_df = confounds_df.iloc[dummy_scans:].reset_index(drop=True)

    if confounds_mode == "task-only":
        # No nuisance regressors; design.create_design_matrix adds an intercept.
        return confounds_df.iloc[:, :0].reset_index(drop=True)

    pattern = _get_base_confound_pattern(task_name, sample_type, confounds_mode)
    if additional_patterns:
        pattern = "|".join([pattern] + additional_patterns)
    return confounds_df.filter(regex=pattern).reset_index(drop=True)
```

In `lev1/run.py`, add the CLI arg (next to `--fc-confounds`):

```python
    parser.add_argument(
        "--confounds-mode",
        choices=["full", "no-motion", "task-only"],
        default="full",
        help="Nuisance regressors in the lev1 design: full (cosine+24p motion+spikes), "
        "no-motion (cosine only), task-only (none). NSI-experiment arms.",
    )
```

In `runner.py`, thread it into the confounds call (currently `runner.py:332`):

```python
    selected_confounds = load_and_process_confounds(
        run_files["confounds"], args.task_name, sample_type, dummy_scans=0,
        confounds_mode=getattr(args, "confounds_mode", "full"),
    )
```

- [ ] **Step 4: Run to verify pass** — `uv run --no-sync pytest tests/lev1/test_confounds_modes.py -q -p no:cacheprovider` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/network_glm/lev1/processing/confounds.py src/network_glm/lev1/run.py \
        src/network_glm/lev1/runner.py tests/lev1/test_confounds_modes.py
git commit -m "feat(lev1): --confounds-mode {full,no-motion,task-only}

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `is_cifti_space` + de-classify fsLR as surface

**Files:** Modify `lev1/spaces.py`; Test `tests/lev1/test_spaces_cifti.py`.

- [ ] **Step 1: Failing test**

```python
# tests/lev1/test_spaces_cifti.py
from network_glm.lev1.spaces import is_surface_space, is_cifti_space, resolve_surface_space


def test_fslr_is_cifti_not_surface():
    assert is_cifti_space("fsLR") is True
    assert is_surface_space("fsLR") is False


def test_surface_spaces_unchanged():
    assert is_surface_space("surface") is True
    assert is_surface_space("fsaverage6") is True
    assert is_cifti_space("surface") is False
    assert is_cifti_space("MNI") is False
    assert resolve_surface_space("surface") == "fsnative"
```

- [ ] **Step 2: Run to verify fail** — `ImportError: cannot import name 'is_cifti_space'`.

- [ ] **Step 3: Implement** in `lev1/spaces.py`:

```python
def is_surface_space(space: str) -> bool:
    """True for per-hemi GIFTI surface spaces (fsLR is CIFTI, not here)."""
    return space in ("surface", "fsaverage6")


def is_cifti_space(space: str) -> bool:
    """True for CIFTI dense-timeseries spaces (fsLR den-91k)."""
    return space == "fsLR"
```

Leave `resolve_surface_space` unchanged (its `fsLR` entry is harmless/unused now).

- [ ] **Step 4: Run to verify pass** — `uv run --no-sync pytest tests/lev1/test_spaces_cifti.py -q -p no:cacheprovider` → pass. Also run `tests/lev1/test_surface_glm_spaces.py` to confirm no regression.

- [ ] **Step 5: Commit**

```bash
git add src/network_glm/lev1/spaces.py tests/lev1/test_spaces_cifti.py
git commit -m "feat(lev1): is_cifti_space; fsLR no longer classified as surface

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: CIFTI dtseries I/O (port)

**Files:** Create `processing/cifti_io.py`; Test `tests/lev1/test_cifti_io.py`.

- [ ] **Step 1: Failing test**

```python
# tests/lev1/test_cifti_io.py
import nibabel as nib
import numpy as np
import pytest
from network_glm.lev1.processing.cifti_io import load_dtseries, write_dtseries


def _make_dtseries(path, n_t=10, n_gray=91282):
    # minimal valid CIFTI-2 dtseries: SeriesAxis x BrainModelAxis
    bm = nib.cifti2.BrainModelAxis.from_mask(np.ones(n_gray, dtype=bool),
                                             name="CIFTI_STRUCTURE_CORTEX_LEFT")
    ax = nib.cifti2.SeriesAxis(start=0, step=1.49, size=n_t)
    hdr = nib.Cifti2Header.from_axes((ax, bm))
    data = np.random.RandomState(0).randn(n_t, n_gray).astype(np.float32)
    nib.Cifti2Image(data, header=hdr).to_filename(str(path))
    return data


def test_load_returns_data_and_template(tmp_path):
    p = tmp_path / "x.dtseries.nii"
    data = _make_dtseries(p, n_t=8, n_gray=1000)
    loaded, tmpl = load_dtseries(p)
    assert loaded.shape == (8, 1000)
    np.testing.assert_allclose(loaded, data, rtol=1e-5)
    assert tmpl.shape == (8, 1000)


def test_write_roundtrip_preserves_axes(tmp_path):
    p = tmp_path / "x.dtseries.nii"
    _make_dtseries(p, n_t=8, n_gray=1000)
    _, tmpl = load_dtseries(p)
    resid = np.zeros((8, 1000), dtype=np.float32)
    out = write_dtseries(resid, tmpl, tmp_path / "resid.dtseries.nii")
    reloaded, tmpl2 = load_dtseries(out)
    assert reloaded.shape == (8, 1000)
    assert tmpl2.header.get_axis(0).size == 8


def test_write_shape_mismatch_raises(tmp_path):
    p = tmp_path / "x.dtseries.nii"
    _make_dtseries(p, n_t=8, n_gray=1000)
    _, tmpl = load_dtseries(p)
    with pytest.raises(ValueError, match="template shape"):
        write_dtseries(np.zeros((7, 1000), dtype=np.float32), tmpl,
                       tmp_path / "bad.dtseries.nii")
```

- [ ] **Step 2: Run to verify fail** — `ModuleNotFoundError: ...cifti_io`.

- [ ] **Step 3: Implement** `processing/cifti_io.py` (verbatim port from
`network_lev1_residuals/src/network_lev1_residuals/cifti_io.py`):

```python
"""CIFTI dense-timeseries (.dtseries.nii) IO for fsLR den-91k residuals.

The task GLM runs on the (T, 91282) data matrix and the residual is the same
shape, so the residual dtseries reuses the input image's CIFTI + NIfTI headers
verbatim — the robust way to preserve the BrainModelAxis/SeriesAxis.
"""
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def load_dtseries(path) -> Tuple[np.ndarray, "nib.cifti2.Cifti2Image"]:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    return data, img


def write_dtseries(data: np.ndarray, template: "nib.cifti2.Cifti2Image", out_path) -> Path:
    out_path = Path(out_path)
    if tuple(data.shape) != tuple(template.shape):
        raise ValueError(
            f"residual shape {tuple(data.shape)} != template shape {tuple(template.shape)}; "
            "header reuse requires identical (timepoints, grayordinates)."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.cifti2.Cifti2Image(
        np.asarray(data, dtype=np.float32),
        header=template.header,
        nifti_header=template.nifti_header,
    ).to_filename(str(out_path))
    return out_path
```

- [ ] **Step 4: Run to verify pass** — `uv run --no-sync pytest tests/lev1/test_cifti_io.py -q -p no:cacheprovider` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/network_glm/lev1/processing/cifti_io.py tests/lev1/test_cifti_io.py
git commit -m "feat(lev1): cifti_io dtseries load/write (ported)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `process_cifti_residuals`

**Files:** Modify `processing/residuals.py`; Test `tests/lev1/test_cifti_residuals.py`.

- [ ] **Step 1: Failing test**

```python
# tests/lev1/test_cifti_residuals.py
import nibabel as nib
import numpy as np
import pandas as pd
from network_glm.lev1.processing.cifti_io import load_dtseries
from network_glm.lev1.processing.residuals import process_cifti_residuals
from network_glm.lev1.processing.surface_data import SurfaceGLM


def _make_dtseries(path, n_t=40, n_gray=500):
    bm = nib.cifti2.BrainModelAxis.from_mask(np.ones(n_gray, dtype=bool),
                                             name="CIFTI_STRUCTURE_CORTEX_LEFT")
    ax = nib.cifti2.SeriesAxis(start=0, step=1.49, size=n_t)
    hdr = nib.Cifti2Header.from_axes((ax, bm))
    rng = np.random.RandomState(1)
    data = rng.randn(n_t, n_gray).astype(np.float32)
    nib.Cifti2Image(data, header=hdr).to_filename(str(path))
    return data


def test_process_cifti_residuals_writes_dtseries(tmp_path):
    p = tmp_path / "in.dtseries.nii"
    data = _make_dtseries(p, n_t=40, n_gray=500)
    _, template = load_dtseries(p)
    # simple design: intercept + one regressor
    dm = pd.DataFrame({"reg": np.linspace(0, 1, 40), "constant": np.ones(40)})
    glm = SurfaceGLM(t_r=1.49).fit(data, dm)
    out = process_cifti_residuals(glm, template, tmp_path, "sub-x_ses-01_task-t_run-1",
                                  tr=1.49, fc_confounds=None, low_pass=None, high_pass=None)
    assert out["success"] is True
    saved = out["saved_path"]
    assert saved.name.endswith("_space-fsLR_den-91k_task-regressed-residuals.dtseries.nii")
    reloaded, _ = load_dtseries(saved)
    assert reloaded.shape == (40, 500)
```

- [ ] **Step 2: Run to verify fail** — `ImportError: cannot import name 'process_cifti_residuals'`.

- [ ] **Step 3: Implement** in `processing/residuals.py` (add near
`process_surface_residuals`; reuse the same filtering semantics):

```python
def cifti_residual_filename(base_filename: str) -> str:
    """Canonical fsLR den-91k residual dtseries name (shared with --skip-existing)."""
    return f"{base_filename}_space-fsLR_den-91k_task-regressed-residuals.dtseries.nii"


def process_cifti_residuals(
    surface_glm,
    template,
    output_dir: Path,
    base_filename: str,
    tr: float = 1.49,
    low_pass: float | None = 0.1,
    high_pass: float | None = 0.01,
    fc_confounds: np.ndarray | None = None,
) -> dict:
    """Compute Y - X*beta from a fitted SurfaceGLM over CIFTI grayordinates,
    optionally band-pass + FC-confound clean, and write an fsLR den-91k dtseries
    reusing the input CIFTI header. Mirrors process_surface_residuals."""
    from network_glm.lev1.processing.cifti_io import write_dtseries

    output_dir = Path(output_dir)
    result = {"success": True, "saved_path": None, "errors": []}
    try:
        residuals = surface_glm.get_residuals()  # (T, n_grayordinates)
        if low_pass is not None or high_pass is not None or fc_confounds is not None:
            residuals = clean_signal(
                residuals, t_r=tr, low_pass=low_pass, high_pass=high_pass,
                confounds=fc_confounds, standardize=False, detrend=False,
            )
        out_path = write_dtseries(
            residuals.astype(np.float32), template,
            output_dir / cifti_residual_filename(base_filename),
        )
        result["saved_path"] = out_path
        logger.info("Saved CIFTI residuals: %s", out_path)
    except Exception as e:
        result["success"] = False
        result["errors"].append(str(e))
        logger.error("Failed to process CIFTI residuals: %s", e)
    return result
```

(`clean_signal` and `np`/`Path`/`logger` are already imported at the top of
`residuals.py`.)

- [ ] **Step 4: Run to verify pass** — `uv run --no-sync pytest tests/lev1/test_cifti_residuals.py -q -p no:cacheprovider` → pass.

- [ ] **Step 5: Commit**

```bash
git add src/network_glm/lev1/processing/residuals.py tests/lev1/test_cifti_residuals.py
git commit -m "feat(lev1): process_cifti_residuals writes fsLR den-91k dtseries

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Wire the CIFTI branch into the runner

**Files:** Modify `lev1/runner.py`; Test `tests/lev1/test_cifti_residuals.py` (add an end-to-end case).

- [ ] **Step 1: Failing test** — append to `tests/lev1/test_cifti_residuals.py`:

```python
def test_process_cifti_run_requires_residuals(tmp_path):
    import argparse
    from network_glm.lev1.runner import process_cifti_run
    args = argparse.Namespace(residuals=False, task_name="stopSignal", subj_id="sub-x")
    try:
        process_cifti_run({}, pd.DataFrame(), args, {}, "sub-x_ses-01_task-t_run-1", 1.49, None)
        assert False, "expected ValueError when --residuals not set"
    except ValueError as e:
        assert "residuals" in str(e).lower()
```

- [ ] **Step 2: Run to verify fail** — `ImportError: cannot import name 'process_cifti_run'`.

- [ ] **Step 3: Implement** in `runner.py`.

(a) Import the CIFTI helpers at the top with the other residual imports:

```python
from network_glm.lev1.processing.cifti_io import load_dtseries
from network_glm.lev1.processing.residuals import (
    cifti_residual_filename,
    process_cifti_residuals,
)
from network_glm.lev1.spaces import is_cifti_space  # alongside is_surface_space
```

(b) Add `process_cifti_run` (near `process_surface_run`):

```python
def process_cifti_run(run_files, design_matrix, args, dirs, base_filename, tr, fc_confounds=None):
    """Fit a GLM over fsLR den-91k grayordinates and write residuals as a dtseries."""
    from network_glm.lev1.processing.surface_data import SurfaceGLM, load_surface_data  # noqa
    if not getattr(args, "residuals", False):
        raise ValueError("--space fsLR is residuals-only; pass --residuals.")
    data, template = load_dtseries(run_files["cifti_bold"])  # (T, 91282)
    glm = SurfaceGLM(t_r=tr).fit(data, design_matrix)
    return process_cifti_residuals(
        glm, template, dirs["task_residuals"], base_filename, tr, fc_confounds=fc_confounds
    )
```

(c) In `process_single_run`, add a CIFTI branch for **n_scans** (before the
surface/volumetric branch that starts `if is_surface_space(args.space):`):

```python
    if is_cifti_space(args.space):
        if "cifti_bold" not in run_files:
            raise ValueError(f"Missing cifti_bold for {session}/{run}")
        _cifti_data, _cifti_template = load_dtseries(run_files["cifti_bold"])
        n_scans = _cifti_data.shape[0]
        del _cifti_data  # reloaded inside process_cifti_run; keep peak memory low
    elif is_surface_space(args.space):
        ...  # unchanged
    else:
        ...  # unchanged volumetric
```

(d) Add the CIFTI dispatch alongside the existing surface/volumetric dispatch
(where `if is_surface_space(args.space): process_surface_run(...)`):

```python
    if is_cifti_space(args.space):
        process_cifti_run(run_files, design_matrix, args, dirs, base_filename, tr, fc_confounds)
    elif is_surface_space(args.space):
        process_surface_run(...)  # unchanged
    else:
        process_volumetric_run(...)  # unchanged
```

(e) Add the CIFTI `--skip-existing` branch (near the surface/vol residual-path
checks around `runner.py:287`):

```python
    elif is_cifti_space(args.space) and args.residuals:
        cifti_res = dirs["task_residuals"] / cifti_residual_filename(base_filename)
        if cifti_res.exists():
            logger.info("Skipping %s (outputs already exist)", run_key)
            return True
```

- [ ] **Step 4: Run to verify pass** — `uv run --no-sync pytest tests/lev1/test_cifti_residuals.py -q -p no:cacheprovider` → pass.

- [ ] **Step 5: Full lev1 suite** — `uv run --no-sync pytest tests/lev1 -q -p no:cacheprovider` → all pass (no regressions to surface/volumetric).

- [ ] **Step 6: Commit**

```bash
git add src/network_glm/lev1/runner.py tests/lev1/test_cifti_residuals.py
git commit -m "feat(lev1): wire fsLR CIFTI dispatch (process_cifti_run + skip-existing)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Operational validation (controller-run, on Sherlock)

- [ ] **Step 1: One real fsLR run per mode on sub-s03** (validation cohort, one session/task with a `space-fsLR_den-91k_bold.dtseries.nii`), to a scratch out dir:

```bash
for MODE in full no-motion task-only; do
  uv run --no-sync <lev1 entrypoint> --space fsLR --residuals \
    --confounds-mode "$MODE" --subj-id sub-s03 --task-name <task> ... \
    --output <scratch>/nsi_arms/$MODE
done
```

- [ ] **Step 2: Verify each output** — a `*_space-fsLR_den-91k_*_task-regressed-residuals.dtseries.nii`
that `nib.load`s, has shape `(T, 91282)`, and matches Gracie's glob
`*_space-fsLR_den-91k_*bold.dtseries.nii` pattern (rename/desc as needed for her
`concatenate_xcpd_*`). Confirm design-matrix column counts differ across modes
(full ≫ no-motion > task-only) in the run logs.

- [ ] **Step 3: Open PR** for `feat/cifti-residuals-confound-modes`; hand the three
arms to the XCP-D→NSI pipeline.

---

## Notes / out of scope
- fsLR **contrasts** (dscalar) are not implemented (residuals-only path).
- No changes to XCP-D, Gracie's postproc, or the NSI computation.
- fsnative/fsaverage6 GIFTI residuals for MSHBM come from the existing
  `--space surface`/`fsaverage6` runs (unchanged).
