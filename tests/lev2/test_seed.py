"""randomise's RNG must be pinned, or identical runs give different p-values.

The installed randomise-prep has no `seed` parameter, so run.py injects `--seed` into the
generated script instead. If that ever stops working the caller raises rather than quietly
producing unreproducible permutations.
"""
from pathlib import Path

from network_glm.lev2.run import _inject_seed

# Exactly what randomise_prep.prep writes.
SCRIPT = ('#!/bin/bash\n'
          'randomise \\\n'
          '  -i "/x/in.nii.gz" \\\n'
          '  -o "/x/out" \\\n'
          '  -m "/x/mask.nii.gz" \\\n'
          '  -d "/x/design.mat" \\\n'
          '  -t "/x/design.con" \\\n'
          '  -T -n 5000\n')


def test_seed_is_injected(tmp_path):
    p = tmp_path / "randomise_call.sh"
    p.write_text(SCRIPT)
    assert _inject_seed(p, 42) is True
    text = p.read_text()
    assert "--seed 42" in text
    # Still a valid continued command, and the original flags survive.
    assert text.startswith("#!/bin/bash\nrandomise \\\n  --seed 42 \\\n")
    assert '-i "/x/in.nii.gz"' in text and "-T -n 5000" in text


def test_injection_is_idempotent(tmp_path):
    p = tmp_path / "randomise_call.sh"
    p.write_text(SCRIPT)
    _inject_seed(p, 42)
    assert _inject_seed(p, 42) is True
    assert p.read_text().count("--seed") == 1


def test_no_randomise_call_reports_failure(tmp_path):
    p = tmp_path / "other.sh"
    p.write_text("#!/bin/bash\necho hi\n")
    assert _inject_seed(p, 42) is False


def test_missing_script_reports_failure(tmp_path):
    assert _inject_seed(tmp_path / "nope.sh", 42) is False
