"""Human-readable provenance identifies consumed inputs and residual conventions."""

import hashlib
import json
from argparse import Namespace
from importlib.metadata import version

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from network_glm import provenance
from network_glm.acquisition import sidecar_path_for
from network_glm.lev1 import cache, run
from network_glm.lev1.processing.glm import fit_run_glm
from network_glm.lev1.processing.residuals import process_run_residuals


def args_for(tmp_path, **overrides):
    values = dict(space="MNI", residuals=True, no_residual_filter=False,
                  fc_confounds=False, bids_dir="bids", fmriprep_dir="fmriprep",
                  exclusions_file=str(tmp_path / "exclusions.json"), allow_dirty=True)
    values.update(overrides)
    return Namespace(**values)


def test_serialized_package_identity_is_installed_network_glm(tmp_path):
    manifest = provenance.write_run_manifest(tmp_path, stage="lev1", args={})
    description = provenance.write_dataset_description(tmp_path, name="lev1")
    tools = json.loads(manifest.read_text())["tool_versions"]
    generator = json.loads(description.read_text())["GeneratedBy"][0]
    assert tools["network_glm"] == version("network_glm")
    assert "neuro-workflow" not in tools
    assert generator["Name"] == "network_glm"
    assert generator["Version"] == version("network_glm")


@pytest.mark.parametrize("key,mask_key,extension", [
    ("mni_data", "mni_brain_mask", ".nii.gz"),
    ("t1w_data", "t1w_brain_mask", ".nii.gz"),
    ("left_surface", None, ".func.gii"),
    ("cifti_bold", None, ".dtseries.nii"),
])
def test_consumed_timing_and_mask_hashes_change_in_existing_records(tmp_path, key, mask_key, extension):
    bold = tmp_path / ("bold" + extension)
    bold.write_bytes(b"synthetic identity fixture")
    sidecar = sidecar_path_for(bold)
    sidecar.write_text('{"StartTime": 0.5}')
    run_files = {key: bold}
    if mask_key:
        mask = tmp_path / "mask.nii.gz"
        mask.write_bytes(b"mask A")
        run_files[mask_key] = mask
    files = {"ses-01": {"run-1": run_files}}
    args = args_for(tmp_path)

    def record():
        run._write_lev1_provenance(tmp_path, args, {"base": tmp_path / "task"},
                                   run._collect_run_inputs(files))
        return {row["path"]: row["sha256"] for row in
                json.loads((tmp_path / "task/run-manifest.json").read_text())["inputs"]}

    before = record()
    assert before[str(sidecar)] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    if mask_key:
        assert before[str(mask)] == hashlib.sha256(mask.read_bytes()).hexdigest()
    signature = cache.run_signature(run_files, args, {}, "discovery")
    sidecar.write_text('{"StartTime": 0.7}')
    if mask_key:
        mask.write_bytes(b"mask B")
    after = record()
    assert before[str(bold)] == after[str(bold)]
    assert before[str(sidecar)] != after[str(sidecar)]
    if mask_key:
        assert before[str(mask)] != after[str(mask)]
    assert signature != cache.run_signature(run_files, args, {}, "discovery")


def test_only_consumed_surface_sidecar_and_combined_mask_are_recorded(tmp_path):
    left = tmp_path / "left.func.gii"
    left.touch()
    right = tmp_path / "right.func.gii"
    right.touch()
    sidecar_path_for(left).write_text("{}")
    sidecar_path_for(right).write_text("{}")
    combined = tmp_path / "combined.nii.gz"
    combined.touch()
    inputs = run._collect_run_inputs(
        {"ses-01": {"run-1": {"left_surface": left, "right_surface": right}}},
        combined_mask_path=combined,
    )
    assert set(inputs) == {left, right, sidecar_path_for(left), combined}


@pytest.mark.parametrize("space,units", [("MNI", "percent signal change"),
                                        ("T1w", "percent signal change"),
                                        ("surface", "input BOLD units"),
                                        ("fsaverage6", "input BOLD units"),
                                        ("fsLR", "input BOLD units")])
@pytest.mark.parametrize("no_filter", [False, True])
def test_residual_units_and_filter_settings_are_explicit(tmp_path, space, units, no_filter):
    args = args_for(tmp_path, space=space, no_residual_filter=no_filter, fc_confounds=True)
    run._write_lev1_provenance(tmp_path, args, {"base": tmp_path}, [])
    residuals = json.loads((tmp_path / "run-manifest.json").read_text())["residuals"]
    assert residuals["requested"] is True
    assert residuals["units"] == units
    assert residuals["definition"] == "Y - X @ beta (unwhitened design)"
    assert residuals["filtering"]["high_pass_hz"] == (None if no_filter else 0.01)
    assert residuals["filtering"]["low_pass_hz"] == (None if no_filter else 0.1)
    assert residuals["filtering"]["detrend"] is False
    assert residuals["filtering"]["standardize"] is False
    assert residuals["fc_confounds_requested"] is True
    assert residuals["sample_mask"] is None


def test_no_residual_request_does_not_claim_residual_outputs(tmp_path):
    run._write_lev1_provenance(tmp_path, args_for(tmp_path, residuals=False), {"base": tmp_path}, [])
    assert json.loads((tmp_path / "run-manifest.json").read_text())["residuals"] == {"requested": False}


def test_metadata_writes_leave_real_residual_and_contrast_arrays_unchanged(tmp_path):
    rng = np.random.default_rng(71)
    design = pd.DataFrame({"task": rng.normal(size=100), "constant": 1.0})
    data = 1000 + rng.normal(size=(2, 2, 2, 100))
    image = nib.Nifti1Image(data, np.eye(4))
    mask = nib.Nifti1Image(np.ones((2, 2, 2), np.uint8), np.eye(4))
    model = fit_run_glm(image, design, "residual", "sub-test", 1.49, mask_img=mask)
    result = process_run_residuals(model, tmp_path, "sub-test", mask_img=mask)
    assert result["success"]
    path = result["saved_paths"]["filtered"][0]
    before_bytes = path.read_bytes()
    before_effect = model.compute_contrast("task", output_type="effect_size").get_fdata()
    run._write_lev1_provenance(tmp_path, args_for(tmp_path), {"base": tmp_path}, [])
    assert "residuals" in json.loads((tmp_path / "run-manifest.json").read_text())
    assert path.read_bytes() == before_bytes
    repeated = process_run_residuals(model, tmp_path / "repeat", "sub-test", mask_img=mask)
    np.testing.assert_array_equal(nib.load(repeated["saved_paths"]["filtered"][0]).get_fdata(),
                                  nib.load(path).get_fdata())
    np.testing.assert_array_equal(model.compute_contrast("task", output_type="effect_size").get_fdata(),
                                  before_effect)
