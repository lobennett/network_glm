"""The DCT-cosine cap is config-driven (thresholds.yaml confounds.cosine_max_index),
preserving the historical discovery/nBack behavior and applying nowhere else."""

import network_glm.thresholds as thr
from network_glm.lev1.processing.confounds import _get_base_confound_pattern


def test_discovery_nback_caps_cosines_from_config():
    # The committed config caps discovery/nBack to cosine00..04.
    pat = _get_base_confound_pattern("nBack", "discovery")
    assert "cosine0[0-4]" in pat
    assert "cosine|" not in pat  # full cosine token replaced


def test_other_sample_task_keeps_full_cosine_set():
    pat = _get_base_confound_pattern("flanker", "validation")
    assert pat.startswith("cosine|")  # uncapped
    assert "cosine0[0-" not in pat


def test_cap_is_config_driven(monkeypatch):
    monkeypatch.setattr(thr, "confounds_cosine_caps", lambda: {"validation": {"flanker": 2}})
    pat = _get_base_confound_pattern("flanker", "validation")
    assert "cosine0[0-2]" in pat
    # a task without a configured cap is unaffected
    assert _get_base_confound_pattern("goNogo", "validation").startswith("cosine|")
