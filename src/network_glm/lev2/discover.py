"""Find the contrasts a set of level-1 output directories contains.

How lev1 names its fixed-effects files is this package's knowledge, so a caller wanting to
fan out one job per contrast asks here rather than globbing the layout itself. Previously
this lived in the submit layer; submission moved to network_fmri, the layout knowledge did
not.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path


def discover_contrasts_from_lev1_dirs(
    lev1_dirs: list[str],
    task_filter: list[str] | None = None,
    space: str = "volume",
) -> list[str]:
    """Glob fixed-effects files and extract contrast names.

    Contrast names can contain underscores, so the capture spans everything between
    ``task-`` and the next BIDS-entity boundary (``_rtmodel-`` if present, ``_stat-`` as a
    fallback), not just up to the first underscore.
    """
    if space == "surface":
        leaf = "*_hemi-L_*_stat-fixed-effects.func.gii"
    else:
        leaf = "*_stat-fixed-effects.nii.gz"
    contrasts = set()
    for lev1_dir in lev1_dirs:
        pattern = str(Path(lev1_dir) / "sub-*" / "*" / "fixed_effects" / leaf)
        for fpath in glob.glob(pattern):
            fname = Path(fpath).name
            m = re.search(r"(task-[^_]+_contrast-.+?)_(?:rtmodel-|stat-)", fname)
            if not m:
                continue
            contrast_id = m.group(1)
            if task_filter is None:
                contrasts.add(contrast_id)
                continue
            task_m = re.search(r"task-([^_]+)", contrast_id)
            if task_m and task_m.group(1) in task_filter:
                contrasts.add(contrast_id)
    return sorted(contrasts)
