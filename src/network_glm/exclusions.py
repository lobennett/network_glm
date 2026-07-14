import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_exclusion_key(exclusion: dict[str, str]) -> str:
    """Create a standardized exclusion key from exclusion data.

    Args:
        exclusion: Dictionary with subject, session, task, run keys

    Returns:
        Formatted exclusion key string

    Raises:
        KeyError: If required keys are missing from exclusion
    """
    required_keys = ["subject", "session", "task", "run"]
    missing_keys = [key for key in required_keys if key not in exclusion]

    if missing_keys:
        raise KeyError(f"Missing required keys in exclusion: {missing_keys}")

    subj = exclusion["subject"]
    ses = exclusion["session"]
    task = exclusion["task"]
    run = exclusion["run"]
    return f"{subj}_{ses}_{task}_{run}"


def _is_flat_list_format(data) -> bool:
    """Detect whether exclusion data is the compiled-lockfile flat-list format.

    Compiled-lockfile format: a JSON list of dicts, each with a 'source' field.
    Legacy format: a JSON dict mapping type names to lists of dicts.
    """
    return isinstance(data, list)


def _load_exclusions_flat(
    data: list, exclusion_types: list[str] | None = None
) -> dict[str, set[str]]:
    """Parse compiled exclusions (flat list) into the keyed-dict format.

    Entries with action 'exclude' or 'trim' are included. Entries with
    action 'force-include' are skipped. The 'source' field becomes the key.
    """
    result: dict[str, set[str]] = {}
    for entry in data:
        action = entry.get("action", "")
        if action not in ("exclude", "trim"):
            continue
        source = entry.get("source", "unknown")
        if exclusion_types is not None and source not in exclusion_types:
            continue
        try:
            key = create_exclusion_key(entry)
            result.setdefault(source, set()).add(key)
        except KeyError as e:
            logger.warning("Skipping invalid exclusion entry: %s", e)
    for source, keys in result.items():
        logger.info("Loaded %d %s exclusions", len(keys), source)
    return result


def load_exclusions_by_type(
    exclusions_file: str | Path, exclusion_types: list[str] | None = None
) -> dict[str, set[str]]:
    """Load exclusions by type from JSON file.

    Accepts both the legacy keyed-dict format (``{"motion_exclusions": [...], ...}``)
    and the compiled-lockfile flat-list format (``[{"source": "motion", ...}, ...]``).

    Args:
        exclusions_file: Path to exclusions JSON file
        exclusion_types: List of exclusion types to load. If None, loads all types.

    Returns:
        Dictionary mapping exclusion type to set of exclusion keys

    Examples:
        >>> exclusions = load_exclusions_by_type('exclusions.json', ['fmriprep_exclusions'])
        >>> fmriprep_keys = exclusions['fmriprep_exclusions']
    """
    exclusions_file = Path(exclusions_file)

    if not exclusions_file.exists():
        logger.warning("Exclusions file not found: %s", exclusions_file)
        return {}

    try:
        with open(exclusions_file) as f:
            exclusions_data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load exclusions from %s: %s", exclusions_file, e)
        return {}

    if _is_flat_list_format(exclusions_data):
        logger.debug("Detected compiled-lockfile exclusions format")
        return _load_exclusions_flat(exclusions_data, exclusion_types)

    result = {}

    # If no specific types requested, load all available types
    if exclusion_types is None:
        exclusion_types = [
            key for key in exclusions_data.keys() if isinstance(exclusions_data[key], list)
        ]

    for exclusion_type in exclusion_types:
        if exclusion_type not in exclusions_data:
            logger.warning('Exclusion type "%s" not found in %s', exclusion_type, exclusions_file)
            continue

        excluded_keys = set()
        exclusions_list = exclusions_data[exclusion_type]

        if not isinstance(exclusions_list, list):
            logger.warning('Exclusion type "%s" is not a list, skipping', exclusion_type)
            continue

        for exclusion in exclusions_list:
            try:
                key = create_exclusion_key(exclusion)
                excluded_keys.add(key)
            except KeyError as e:
                logger.warning('Skipping invalid exclusion in "%s": %s', exclusion_type, e)
                continue

        result[exclusion_type] = excluded_keys
        logger.info("Loaded %d %s exclusions", len(excluded_keys), exclusion_type)

    return result


def load_exclusions(
    exclusions_file: str | Path, exclusion_types: list[str] | None = None
) -> set[str]:
    """Load all exclusions from JSON file and return a combined set of exclusion keys.

    Args:
        exclusions_file: Path to exclusions JSON file
        exclusion_types: List of exclusion types to include. If None, includes all types.

    Returns:
        Set of all exclusion keys across specified types

    Examples:
        >>> # Load all exclusion types
        >>> all_exclusions = load_exclusions('exclusions.json')
        >>> # Load only fMRIPrep exclusions
        >>> fmriprep_only = load_exclusions('exclusions.json', ['fmriprep_exclusions'])
    """
    exclusions_by_type = load_exclusions_by_type(exclusions_file, exclusion_types)

    # Combine all exclusion sets
    all_exclusions = set()
    for exclusion_set in exclusions_by_type.values():
        all_exclusions.update(exclusion_set)

    total_count = len(all_exclusions)
    if total_count > 0:
        logger.info("Total exclusions loaded: %d", total_count)
        if len(all_exclusions) <= 10:  # Only show keys if reasonable number
            logger.info("Excluded keys: %s", sorted(all_exclusions))

    return all_exclusions


def load_contrast_exclusions(
    exclusions_file: str | Path,
) -> set[tuple[str, str]]:
    """Load per-contrast exclusions (action 'exclude-contrast') from the compiled file.

    Returns a set of ``(scan_key, contrast)`` pairs, where ``scan_key`` matches
    :func:`create_exclusion_key` (``{sub}_{ses}_task-{task}_{run}``). These drop a
    single contrast's per-run fixed-effects input, not the whole scan (see
    fixed_effects.FixedEffectsAnalyzer.find_contrast_files). Scan-level actions
    (exclude/trim) are ignored here — they are handled by load_exclusions.
    """
    exclusions_file = Path(exclusions_file)
    if not exclusions_file.exists():
        return set()
    try:
        with open(exclusions_file) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load contrast exclusions from %s: %s", exclusions_file, e)
        return set()
    if not isinstance(data, list):
        return set()  # only the compiled-lockfile flat-list format carries exclude-contrast
    out: set[tuple[str, str]] = set()
    for entry in data:
        if entry.get("action") != "exclude-contrast":
            continue
        contrast = entry.get("contrast")
        if not contrast:
            logger.warning("Skipping exclude-contrast entry with no contrast: %s", entry)
            continue
        try:
            out.add((create_exclusion_key(entry), contrast))
        except KeyError as e:
            logger.warning("Skipping invalid exclude-contrast entry: %s", e)
    if out:
        logger.info("Loaded %d per-contrast exclusions", len(out))
    return out


def normalize_subject_id(subject: str) -> str:
    """Normalize subject ID by adding 'sub-' prefix if not already present.

    Args:
        subject: Subject ID string

    Returns:
        Subject ID with 'sub-' prefix
    """
    if subject.startswith("sub-"):
        return subject
    return f"sub-{subject}"
