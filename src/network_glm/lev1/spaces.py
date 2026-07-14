"""Analysis-space helpers shared across the lev1 pipeline.

Tiny leaf module (no dependencies) so both the setup phase
(:mod:`.prepare`), the per-run runner (:mod:`.runner`) and the CLI
(:mod:`.run`) can classify the ``--space`` argument without importing each
other.
"""


def is_surface_space(space: str) -> bool:
    """Check if the analysis space is a surface space."""
    return space in ("surface", "fsaverage6", "fsLR")


def resolve_surface_space(space: str) -> str | None:
    """Return the surface template name, or None for volumetric."""
    mapping = {
        "surface": "fsnative",
        "fsaverage6": "fsaverage6",
        "fsLR": "fsLR",
    }
    return mapping.get(space)
