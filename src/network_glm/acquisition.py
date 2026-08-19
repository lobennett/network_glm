"""Scanner-acquisition constants shared across the pipeline.

Single source of truth (RSE rubric RF-6) for acquisition parameters that
multiple modules must agree on. A prior divergence between ``events/create.py``
and ``analysis/task_config/loader.py`` would silently corrupt event-onset
offsets, so these values are defined here exactly once and imported elsewhere.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TR_SECONDS = 1.49  # repetition time of the BOLD acquisition (seconds)
N_DUMMY = 7  # dummy volumes discarded upstream by scripts/trim_bold.py


def sidecar_path_for(bold_path: str | Path) -> Path:
    """Return the JSON sidecar path that accompanies a BOLD derivative.

    BIDS filenames contain no dots other than the extension, so everything
    from the first dot onward is the extension. This handles ``.nii.gz``,
    ``.dtseries.nii`` and ``.func.gii`` with one rule.
    """
    bold_path = Path(bold_path)
    stem = bold_path.name.split(".")[0]
    return bold_path.with_name(f"{stem}.json")


def resolve_slice_time_ref(bold_path: str | Path) -> float:
    """Seconds to add to ``frame_times`` so the model aligns with the data.

    fMRIPrep slice-time corrects each BOLD series to a reference time and
    records that time as ``StartTime`` in the derivative sidecar (alongside
    ``SliceTimingCorrected``). Sampling the regressors at that same reference
    is what keeps the design matrix aligned with the resampled data.

    The pipeline previously assumed ``TR/2``. That is only correct when the
    slices span the full TR: AFNI's ``3dTshift -tzero`` defaults to the mean
    of the slice times, which equals ``TR/2`` only for a gapless acquisition.
    Our sequence acquires 51 slices over 0-1.4023 s of a 1.49 s TR, leaving
    ~88 ms of dead time, so the true reference is 0.701 s -- 44 ms earlier
    than ``TR/2`` (exactly half the dead time).

    Returns 0.0 when slice-timing correction was not applied, since in that
    case the volumes sit at their nominal onsets and need no shift.

    Raises:
        FileNotFoundError: no sidecar, so the reference is unknowable.
        ValueError: sidecar claims STC but omits ``StartTime``.
    """
    sidecar = sidecar_path_for(bold_path)
    if not sidecar.exists():
        raise FileNotFoundError(
            f"No JSON sidecar for {bold_path} (expected {sidecar}); cannot "
            "determine the slice-timing reference."
        )

    with open(sidecar, encoding="utf-8") as fh:
        metadata = json.load(fh)

    if not metadata.get("SliceTimingCorrected", False):
        logger.info(
            "%s reports no slice-timing correction; using a zero frame_times offset.",
            sidecar.name,
        )
        return 0.0

    if "StartTime" not in metadata:
        raise ValueError(
            f"{sidecar} reports SliceTimingCorrected=true but has no StartTime. "
            "Refusing to assume TR/2, which would silently misalign the model."
        )

    return float(metadata["StartTime"])
