import subprocess
import sys
from pathlib import Path


def get_image_path(image_dir, pipeline_name, version):
    return Path(image_dir) / f"{pipeline_name}_{version}.sif"


def ensure_image(image_dir, pipeline_name, version, docker_uri):
    path = get_image_path(image_dir, pipeline_name, version)
    if path.exists():
        print(f"Image found: {path}")
        return path

    print(f"Image not found at {path}, pulling...")
    cmd = ["apptainer", "pull", str(path), f"{docker_uri}:{version}"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Error: failed to pull {docker_uri}:{version}", file=sys.stderr)
        sys.exit(1)

    return path
