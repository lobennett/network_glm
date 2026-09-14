"""Subject identity and RT-arm integrity for one-sample group inputs."""

import re
from pathlib import Path


def observation_roster(input_files, contrast_name, *, surface=False):
    """Validate identities in input order, allowing one map per surface hemisphere."""
    spatial_entities = r"_hemi-(?P<hemisphere>[LR])_space-[^_]+" if surface else ""
    extension = r"\.func\.gii" if surface else r"\.nii\.gz"
    pattern = re.compile(
        r"(?P<subject>sub-[A-Za-z0-9]+)" + spatial_entities
        + r"_task-(?P<task>[^_]+)_contrast-(?P<contrast>.+)"
        r"_rtmodel-(?P<rt_model>[^_]+)_stat-fixed-effects" + extension
    )
    roster = []
    subjects = {}
    sources = {}
    for filename in input_files:
        path = Path(filename)
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Cannot identify subject/task/contrast/RT arm in {filename}")
        row = match.groupdict()
        resolved = path.resolve()
        if resolved in sources:
            raise ValueError(f"Repeated input file: {sources[resolved]} and {filename}")
        sources[resolved] = str(filename)
        if f"task-{row['task']}_contrast-{row['contrast']}" != contrast_name:
            raise ValueError(f"Incompatible task/contrast in {filename}; expected {contrast_name}")
        parent_subject = next((p.name for p in path.parents if p.name.startswith("sub-")), None)
        if parent_subject is not None and parent_subject != row["subject"]:
            raise ValueError(f"Subject directory disagrees with filename: {filename}")
        identity = (row["subject"], row.get("hemisphere"))
        if identity in subjects:
            raise ValueError(
                f"Duplicate subject {row['subject']}: {subjects[identity]} and {filename}"
            )
        if roster and row["rt_model"] != roster[0]["rt_model"]:
            raise ValueError(f"Mixed RT model arms: {roster[0]['path']} and {filename}")
        subjects[identity] = str(filename)
        roster.append({**row, "path": str(filename)})
    return roster
