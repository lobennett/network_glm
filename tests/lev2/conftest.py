"""Synthetic external executable for volume group-boundary tests."""

import os
import sys

import pytest

from tests.lev2.helpers import save


@pytest.fixture
def fake_randomise(tmp_path, monkeypatch):
    """Keep the real prep/concatenation/shell; replace only the external executable."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    executable = bindir / "randomise"
    executable.write_text(f"#!{sys.executable}\n" + '''import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['TEST_ARGV'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
assert Path(args[args.index('-m') + 1]).is_file()
mode = os.environ.get('TEST_MODE', 'success')
if mode == 'fail' and '--fonly' in args:
    sys.exit(7)
if mode == 'noop':
    sys.exit(0)
prefix = args[args.index('-o') + 1]
suffixes = (['_fstat1', '_tfce_fstat1', '_tfce_corrp_fstat1']
            if '--fonly' in args else ['_tstat1'])
for suffix in suffixes:
    if mode == 'missing-t' and suffix == '_tstat1':
        continue
    if mode == 'wrong-corrected-name' and suffix == '_tfce_corrp_fstat1':
        suffix = '_tfce_corrp_fstat99'
    shutil.copyfile(os.environ['TEST_PRODUCT'], prefix + suffix + '.nii.gz')
if mode == 'missing-preparation' and '-R' in args:
    Path(args[args.index('-i') + 1]).unlink()
''')
    executable.chmod(0o755)
    log = tmp_path / "argv.jsonl"
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("TEST_ARGV", str(log))
    monkeypatch.setenv("TEST_PRODUCT", save(tmp_path / "product.nii.gz"))
    return log
