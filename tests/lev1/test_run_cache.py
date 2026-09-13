"""Reuse requires the same scientific request and intact completed outputs."""

from argparse import Namespace

import pytest

from network_glm.lev1 import cache


def test_resume_matches_content_and_settings_but_allows_exclusion_refresh(tmp_path):
    source = tmp_path / "events.tsv"
    source.write_text("first input")
    output = tmp_path / "residuals.func.gii"
    output.write_text("completed output")
    record = tmp_path / "completed.json"
    args = Namespace(rt_model="RTDur", exclusions_file="motion.json", min_runs=2)
    signature = cache.run_signature({"events": source}, args, {"tr": 1.49}, "discovery")
    cache.save_completion(record, signature, [output])
    assert cache.can_reuse(record, signature)

    args.exclusions_file = "final.json"
    args.min_runs = 3
    args.skip_existing = True
    assert (
        cache.run_signature({"events": source}, args, {"tr": 1.49}, "discovery")
        == signature
    )

    args.rt_model = "noRT"
    changed = cache.run_signature({"events": source}, args, {"tr": 1.49}, "discovery")
    assert not cache.can_reuse(record, changed)
    args.rt_model = "RTDur"
    source.write_text("other input")  # same length, different bytes
    changed = cache.run_signature({"events": source}, args, {"tr": 1.49}, "discovery")
    assert not cache.can_reuse(record, changed)
    output.write_text("corrupted output")  # same length as completed output
    assert not cache.can_reuse(record, signature)


@pytest.mark.parametrize("contents", ["{}", "[]", "{broken", '{"status": "running"}'])
def test_invalid_completion_record_cannot_skip(tmp_path, contents):
    record = tmp_path / "complete.json"
    record.write_text(contents)
    assert not cache.can_reuse(record, "signature")


def test_starting_a_rerun_invalidates_previous_completion(tmp_path):
    output = tmp_path / "out"
    output.write_text("valid")
    record = tmp_path / "complete.json"
    cache.save_completion(record, "signature", [output])
    cache.mark_running(record)
    assert not cache.can_reuse(record, "signature")


def test_bold_timing_sidecar_is_part_of_signature(tmp_path):
    bold = tmp_path / "sub-test_bold.func.gii"
    bold.write_text("BOLD")
    sidecar = tmp_path / "sub-test_bold.json"
    sidecar.write_text('{"StartTime": 0.5}')
    args = Namespace(space="surface")
    first = cache.run_signature({"left_surface": bold}, args, {"tr": 1.49}, "discovery")
    sidecar.write_text('{"StartTime": 0.7}')
    assert (
        cache.run_signature({"left_surface": bold}, args, {"tr": 1.49}, "discovery")
        != first
    )
