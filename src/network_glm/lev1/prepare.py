"""Setup / discovery phase for the lev1 GLM pipeline.

Establishes the analysis context the per-run :mod:`.runner` consumes: the
:class:`~network_glm.lev1.subject_config.Config`, exclusion sets,
output directories (``setup_analysis``); the discovered BIDS/fMRIPrep file map
(``discover_and_validate_files``); and the combined brain mask for volumetric
space (``setup_masks``).
"""

import logging
from pathlib import Path

from network_glm.task_utils import detect_sample_type, get_expected_sessions
from network_glm.exclusions import (
    load_exclusions,
    load_exclusions_by_type,
    normalize_subject_id,
)
from network_glm.io.file_discovery import FileFinder
from network_glm.lev1.processing.masks import MaskProcessor
from network_glm.lev1.spaces import is_cifti_space, is_surface_space, resolve_surface_space
from network_glm.lev1.subject_config import Config

logger = logging.getLogger(__name__)


def setup_analysis(args):
    """Set up configuration, exclusions, and output directories.

    Returns:
        Tuple of (config, sample_type, expected_sessions, exclusions,
                  exclusions_by_type, dirs).
    """
    args.subj_id = normalize_subject_id(args.subj_id)

    config = Config(
        bids_dir=Path(args.bids_dir),
        fmriprep_dir=Path(args.fmriprep_dir),
        output_dir=Path(args.results_dir),
        subject_id=args.subj_id,
        task_name=args.task_name,
    )

    sample_type = detect_sample_type(config.bids_dir)
    expected_sessions = get_expected_sessions(args.task_name)

    logger.info(
        "Level 1 GLM: %s / %s / %s / %s", args.subj_id, args.task_name, args.space, sample_type
    )

    # Load exclusions (compiled exclusions are all action='exclude'; the legacy
    # behavioral-salvage "trim" path was dead and has been removed).
    exclusions_by_type = load_exclusions_by_type(args.exclusions_file)
    exclusions = load_exclusions(args.exclusions_file)
    logger.info("Loaded %d exclusions", len(exclusions))

    # Create subject-specific directories
    dirs = config.create_subject_dirs(clean_existing=True)

    return config, sample_type, expected_sessions, exclusions, exclusions_by_type, dirs


def discover_and_validate_files(config, args):
    """Discover BIDS and fMRIPrep files and validate completeness.

    Returns:
        files dict from FileFinder.
    """
    finder = FileFinder(
        config.bids_dir,
        config.fmriprep_dir,
        mni_template=getattr(args, "mni_template", "MNI152NLin2009cAsym"),
        mni_res=getattr(args, "mni_res", "2"),
    )
    required_files = FileFinder.get_required_files_for_space(args.space)
    surface_space = resolve_surface_space(args.space)
    files = finder.get_files(
        args.subj_id,
        args.task_name,
        required_files=required_files,
        surface_space=surface_space,
    )

    if not files:
        raise ValueError("No complete runs found! Check that all required files are present.")

    validation = finder.validate_file_completeness(files, args.task_name)
    logger.info(
        "Found %d complete runs across %d sessions",
        validation["complete_runs"],
        validation["total_sessions"],
    )

    return files


def setup_masks(files, args, dirs):
    """Process brain masks for volumetric analysis.

    Returns:
        Path to combined mask or None for surface/CIFTI space.
    """
    if is_surface_space(args.space) or is_cifti_space(args.space):
        logger.info("Skipping mask processing for surface/CIFTI space")
        return None

    mask_info = MaskProcessor.get_mask_info(files, args.space)
    logger.info("Found %d %s brain masks", mask_info["total_masks"], args.space)

    if not mask_info["all_valid"]:
        raise ValueError("Some masks are invalid - cannot proceed")

    mask_filename = f"{args.subj_id}_space-{args.space}_desc-combinedMask.nii.gz"
    combined_mask_path = dirs["masks"] / mask_filename
    MaskProcessor.create_combined_mask(
        files, args.space, args.within_subject_threshold, combined_mask_path
    )
    logger.info("Combined mask saved: %s", combined_mask_path)
    return combined_mask_path
