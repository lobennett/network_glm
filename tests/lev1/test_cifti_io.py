import nibabel as nib
import numpy as np
import pytest
from network_glm.lev1.processing.cifti_io import load_dtseries, write_dtseries


def _make_dtseries(path, n_t=10, n_gray=91282):
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
