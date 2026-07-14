"""Subject-level analysis configuration for the lev1 GLM pipeline.

Defines :class:`Config`, the resolved bids/fmriprep/output paths plus the
subject/task identity that :mod:`.prepare` builds and the per-run
:mod:`.runner` consumes. Lives under ``lev1/`` rather than a top-level
``config`` module to avoid a name collision with the unrelated dataset-registry
config module in the source monolith this package was lifted from.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for network level 1 analysis.

    Args:
        bids_dir: Path to BIDS dataset directory
        fmriprep_dir: Path to fMRIPrep derivatives directory
        output_dir: Path to output directory (default: ./results/)
        subject_id: Subject ID for creating subject-specific directories
        task_name: Task name for creating task-specific directories
    """

    bids_dir: Path
    fmriprep_dir: Path
    output_dir: Path = Path("./results/")
    subject_id: str | None = None
    task_name: str | None = None

    def __post_init__(self):
        """Convert string paths to Path objects if needed."""
        self.bids_dir = Path(self.bids_dir)
        self.fmriprep_dir = Path(self.fmriprep_dir)
        self.output_dir = Path(self.output_dir)

    def get_subject_dirs(self) -> dict[str, Path]:
        """Get subject-specific output directories.

        Returns:
            Dictionary mapping directory names to Path objects

        Raises:
            ValueError: If subject_id or task_name not set
        """
        if not self.subject_id or not self.task_name:
            raise ValueError("subject_id and task_name must be set to create subject directories")

        base_dir = self.output_dir / self.subject_id / f"task-{self.task_name}"

        return {
            "quality_control": base_dir / "quality_control",
            "indiv_contrasts": base_dir / "indiv_contrasts",
            "fixed_effects": base_dir / "fixed_effects",
            "simplified_events": base_dir / "simplified_events",
            "task_residuals": base_dir / "task_residuals",
            "masks": base_dir / "masks",
            "base": base_dir,
        }

    def create_subject_dirs(self, clean_existing: bool = True) -> dict[str, Path]:
        """Create subject-specific output directories.

        Args:
            clean_existing: If True, remove existing files in directories

        Returns:
            Dictionary mapping directory names to created Path objects
        """
        dirs = self.get_subject_dirs()

        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

            if clean_existing:
                existing_files = [f for f in dir_path.rglob("*") if f.is_file()]
                if existing_files:
                    logger.warning(
                        "Removing %d existing file(s) from %s",
                        len(existing_files),
                        dir_path,
                    )
                    for file_path in existing_files:
                        file_path.unlink()

        return dirs
