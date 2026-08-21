"""File discovery utilities for BIDS and fMRIPrep data."""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class FileFinder:
    """Find and organize BIDS and fMRIPrep files."""

    def __init__(
        self,
        bids_dir: Path,
        fmriprep_dir: Path,
        mni_template: str = "MNI152NLin2009cAsym",
        mni_res: str = "2",
    ):
        """Initialize file finder.

        Args:
            bids_dir: Path to BIDS dataset directory
            fmriprep_dir: Path to fMRIPrep derivatives directory
            mni_template: fMRIPrep MNI template name to look for in derivative
                filenames. Defaults to 'MNI152NLin2009cAsym', the study's requested
                MNI space. MNI152NLin6Asym is also produced, but only because
                --cifti-output pulls it in for the fsLR path.
            mni_res: Resolution suffix for the MNI BOLD/mask files
                (e.g., '1', '2'). Defaults to '2'.
        """
        self.bids_dir = Path(bids_dir)
        self.fmriprep_dir = Path(fmriprep_dir)
        self.mni_template = mni_template
        self.mni_res = mni_res
        self.run_pattern = re.compile(r"run-\d+")

    # Surface file patterns keyed by space name
    SURFACE_PATTERNS = {
        "fsnative": {
            "left_surface": "hemi-L_space-fsnative_bold.func.gii",
            "right_surface": "hemi-R_space-fsnative_bold.func.gii",
        },
        "fsaverage6": {
            "left_surface": "hemi-L_space-fsaverage6_bold.func.gii",
            "right_surface": "hemi-R_space-fsaverage6_bold.func.gii",
        },
        "fsLR": {
            "cifti_bold": "space-fsLR_den-91k_bold.dtseries.nii",
        },
    }

    @staticmethod
    def get_required_files_for_space(space: str) -> list[str]:
        """Get list of required file types for a given analysis space.

        Args:
            space: Analysis space ('T1w', 'MNI', 'surface', 'fsaverage6', or 'fsLR')

        Returns:
            List of required file type keys

        Examples:
            >>> FileFinder.get_required_files_for_space('T1w')
            ['events', 'confounds', 't1w_data', 't1w_brain_mask']
        """
        if space == "T1w":
            return ["events", "confounds", "t1w_data", "t1w_brain_mask"]
        elif space == "MNI":
            return ["events", "confounds", "mni_data", "mni_brain_mask"]
        elif space in ("surface", "fsnative", "fsaverage6"):
            return ["events", "confounds", "left_surface", "right_surface"]
        elif space == "fsLR":
            return ["events", "confounds", "cifti_bold"]
        else:
            # Default: all files
            return [
                "events",
                "confounds",
                "t1w_data",
                "t1w_brain_mask",
                "mni_data",
                "mni_brain_mask",
            ]

    def get_files(
        self,
        subject_id: str,
        task_name: str,
        required_files: list[str] | None = None,
        surface_space: str | None = None,
    ) -> dict[str, dict[str, dict[str, Path]]]:
        """Parse BIDS and fMRIPrep directories for required files.

        Args:
            subject_id: Subject ID (will add 'sub-' prefix if missing)
            task_name: Task name to search for
            required_files: List of required file types (default: all standard files)
            surface_space: Surface space name ('fsnative', 'fsaverage6', 'fsLR')
                to select correct file patterns. Default None uses fsnative.

        Returns:
            Nested dictionary: {session: {run: {file_type: path}}}

        Examples:
            >>> finder = FileFinder('/data/bids', '/data/fmriprep')
            >>> files = finder.get_files('001', 'rest')
            >>> files['ses-01']['run-1']['events']
            PosixPath('/data/bids/sub-s001/ses-01/func/sub-s001_ses-01_task-rest_run-1_events.tsv')
        """
        if not subject_id.startswith("sub-"):
            subject_id = f"sub-{subject_id}"

        if required_files is None:
            required_files = [
                "events",
                "confounds",
                "t1w_data",
                "t1w_brain_mask",
                "mni_data",
                "mni_brain_mask",
            ]

        files = {}

        # Select surface patterns based on surface_space
        surface_patterns = self.SURFACE_PATTERNS.get(
            surface_space or "fsnative", self.SURFACE_PATTERNS["fsnative"]
        )

        # File patterns for fMRIPrep derivatives
        fmriprep_patterns = {
            "confounds": "desc-confounds_timeseries.tsv",
            "t1w_data": "space-T1w_desc-preproc_bold.nii.gz",
            "t1w_brain_mask": "space-T1w_desc-brain_mask.nii.gz",
            "mni_data": f"space-{self.mni_template}_res-{self.mni_res}_desc-preproc_bold.nii.gz",
            "mni_brain_mask": f"space-{self.mni_template}_res-{self.mni_res}_desc-brain_mask.nii.gz",
            **surface_patterns,
        }

        # Get events from BIDS directory
        self._discover_events_files(files, subject_id, task_name)

        # Get fMRIPrep derivatives
        self._discover_fmriprep_files(files, subject_id, task_name, fmriprep_patterns)

        # Filter to complete runs only
        return self._filter_complete_runs(files, required_files)

    def _discover_events_files(self, files: dict, subject_id: str, task_name: str) -> None:
        """Discover BIDS events files."""
        bids_subj_dir = self.bids_dir / subject_id
        pattern = f"ses-*/func/*task-{task_name}_*events.tsv"

        for events_file in bids_subj_dir.glob(pattern):
            session = events_file.parts[-3]
            run_match = self.run_pattern.search(events_file.name)

            if run_match:
                run = run_match.group(0)
                files.setdefault(session, {}).setdefault(run, {})["events"] = events_file

    def _discover_fmriprep_files(
        self, files: dict, subject_id: str, task_name: str, patterns: dict[str, str]
    ) -> None:
        """Discover fMRIPrep derivative files."""
        fmriprep_subj_dir = self.fmriprep_dir / subject_id
        pattern = f"ses-*/func/*task-{task_name}_*"

        for file_path in fmriprep_subj_dir.glob(pattern):
            session = file_path.parts[-3]
            run_match = self.run_pattern.search(file_path.name)

            if not run_match:
                continue

            run = run_match.group(0)
            files.setdefault(session, {}).setdefault(run, {})

            # Match file types by pattern
            for file_type, pattern_str in patterns.items():
                if pattern_str in file_path.name:
                    files[session][run][file_type] = file_path

    def _filter_complete_runs(
        self, files: dict, required_files: list[str]
    ) -> dict[str, dict[str, dict[str, Path]]]:
        """Filter to only include runs with all required files.

        A run is dropped when it's missing any required file type. To make
        skip-with-warning visible to operators (rather than a silent drop),
        each dropped run that has at least one discovered file emits a
        WARNING naming the missing file types. Runs with no discovered files
        at all are not warned about (those represent absent scans, not
        partially-incomplete ones).
        """
        filtered_files = {}

        for session, runs in files.items():
            for run, file_dict in runs.items():
                missing = [ft for ft in required_files if ft not in file_dict]
                if not missing:
                    filtered_files.setdefault(session, {})[run] = file_dict
                elif file_dict:
                    logger.warning(
                        "Skipping %s/%s: missing required file(s): %s",
                        session,
                        run,
                        ", ".join(sorted(missing)),
                    )

        return filtered_files

    def validate_file_completeness(
        self, files: dict[str, dict[str, dict[str, Path]]], task_name: str
    ) -> dict[str, int]:
        """Validate file completeness and return summary.

        Args:
            files: Files dictionary from get_files()
            task_name: Task name for validation

        Returns:
            Dictionary with validation statistics
        """
        total_runs = sum(len(runs) for runs in files.values())
        complete_runs = total_runs  # Already filtered by _filter_complete_runs

        return {
            "total_runs": total_runs,
            "complete_runs": complete_runs,
            "total_sessions": len(files),
            "sessions_with_data": len([s for s in files.values() if s]),
        }
