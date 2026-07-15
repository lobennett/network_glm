"""J5: surface lev2 group analysis via sign-flip permutation."""

from pathlib import Path

import nibabel as nib
import numpy as np

from network_glm.lev2.surface import (
    discover_surface_inputs,
    run_surface_level2_analysis,
    sign_flip_permutation_test,
)

CONTRAST = "task-flanker_contrast-cong"


def _write_gii(path: Path, vec: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = nib.gifti.GiftiDataArray(
        data=vec.astype(np.float32), intent="NIFTI_INTENT_NONE", datatype="NIFTI_TYPE_FLOAT32"
    )
    nib.save(nib.GiftiImage(darrays=[d]), str(path))


def _fe_name(sub, hemi, below=False):
    desc = "_desc-belowMinRuns" if below else ""
    return (
        f"{sub}_hemi-{hemi}_space-fsaverage6_{CONTRAST}_rtmodel-RTDur"
        f"{desc}_stat-fixed-effects.func.gii"
    )


def _make_lev1(tmp_path, subjects, n_vert=6, signal_vertex=0, signal=3.0, seed=1):
    """Build a lev1 dir with per-subject L/R surface fixed-effects effect maps.
    A strong positive group effect is planted at signal_vertex."""
    rng = np.random.RandomState(seed)
    lev1 = tmp_path / "lev1"
    for _i, sub in enumerate(subjects):
        for hemi in ("L", "R"):
            vec = rng.randn(n_vert) * 0.3
            vec[signal_vertex] += signal
            fe_dir = lev1 / sub / "task-flanker" / "fixed_effects"
            _write_gii(fe_dir / _fe_name(sub, hemi), vec)
    return lev1


def test_sign_flip_detects_signal_and_nulls():
    rng = np.random.RandomState(2)
    n_subj, n_vert = 20, 8
    data = rng.randn(n_subj, n_vert) * 0.5
    data[:, 0] += 3.0  # strong positive group effect at vertex 0
    t_obs, fwe_p = sign_flip_permutation_test(data, n_perm=500, seed=0)

    assert t_obs[0] > 0
    assert fwe_p[0] < 0.05  # signal vertex survives FWE
    # signal vertex is the most significant; null vertices are far less so
    assert fwe_p[0] <= fwe_p[1:].min()
    finite = fwe_p[np.isfinite(fwe_p)]
    assert finite.min() >= 1.0 / 501 and finite.max() <= 1.0


def test_sign_flip_nan_vertex_excluded():
    rng = np.random.RandomState(3)
    data = rng.randn(10, 5)
    data[4, 2] = np.nan  # one subject missing at vertex 2
    t_obs, fwe_p = sign_flip_permutation_test(data, n_perm=100, seed=0)
    assert np.isnan(t_obs[2]) and np.isnan(fwe_p[2])
    assert np.isfinite(fwe_p[[0, 1, 3, 4]]).all()


def test_sign_flip_deterministic():
    rng = np.random.RandomState(4)
    data = rng.randn(12, 6)
    a = sign_flip_permutation_test(data, n_perm=200, seed=7)
    b = sign_flip_permutation_test(data, n_perm=200, seed=7)
    np.testing.assert_array_equal(a[1], b[1])


def test_discover_drops_below_min_runs(tmp_path):
    subs = ["sub-s03", "sub-s10"]
    lev1 = _make_lev1(tmp_path, subs)
    # add a belowMinRuns file for a third subject in L
    _write_gii(
        lev1 / "sub-s19" / "task-flanker" / "fixed_effects" / _fe_name("sub-s19", "L", below=True),
        np.zeros(6, dtype=np.float32),
    )
    found = discover_surface_inputs([lev1], CONTRAST)
    assert len(found["L"]) == 2 and len(found["R"]) == 2
    assert not any("belowMinRuns" in f for f in found["L"])


def test_run_surface_level2_writes_per_hemi_maps(tmp_path):
    subs = [f"sub-s{n:02d}" for n in range(1, 9)]
    lev1 = _make_lev1(tmp_path, subs, n_vert=6, signal_vertex=0)
    out = tmp_path / "lev2_surface"
    ok = run_surface_level2_analysis(CONTRAST, [lev1], out, n_perm=300, seed=0)
    assert ok is True
    cdir = out / CONTRAST
    for hemi in ("L", "R"):
        t_path = cdir / f"{CONTRAST}_hemi-{hemi}_stat-group-t.func.gii"
        p_path = cdir / f"{CONTRAST}_hemi-{hemi}_stat-fwe-p.func.gii"
        assert t_path.exists() and p_path.exists()
        fwe = nib.load(str(p_path)).darrays[0].data
        assert fwe[0] < 0.05  # planted signal survives whole-cortex FWE


def test_run_surface_level2_fails_on_subject_mismatch(tmp_path):
    lev1 = _make_lev1(tmp_path, ["sub-s03", "sub-s10"])
    # remove one R map so L/R subject sets differ
    (lev1 / "sub-s10" / "task-flanker" / "fixed_effects" / _fe_name("sub-s10", "R")).unlink()
    assert run_surface_level2_analysis(CONTRAST, [lev1], tmp_path / "o", n_perm=50) is False


def test_run_surface_level2_fails_on_no_inputs(tmp_path):
    (tmp_path / "lev1").mkdir()
    assert (
        run_surface_level2_analysis(CONTRAST, [tmp_path / "lev1"], tmp_path / "o", n_perm=50)
        is False
    )
