"""Scanner-acquisition constants shared across the pipeline.

Single source of truth (RSE rubric RF-6) for acquisition parameters that
multiple modules must agree on. A prior divergence between ``events/create.py``
and ``analysis/task_config/loader.py`` would silently corrupt event-onset
offsets, so these values are defined here exactly once and imported elsewhere.
"""

TR_SECONDS = 1.49  # repetition time of the BOLD acquisition (seconds)
N_DUMMY = 7  # dummy volumes discarded upstream by scripts/trim_bold.py
