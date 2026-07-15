"""Shared SLURM rendering/submission helpers for the submit layer.

Ported from the monolith's ``neuro_workflow.core.slurm`` (str.format-based
template rendering + ``sbatch`` submission via a temp file). Kept tiny and
dependency-free so ``network_glm.submit`` stays importable without the
scientific stack.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def render_template(template_path: str | Path, context: dict) -> str:
    """Render an sbatch template (``str.format``-style ``{placeholders}``)."""
    template = Path(template_path).read_text()
    return template.format(**context)


def submit_sbatch(script_content: str) -> str:
    """Write ``script_content`` to a temp file and submit it via ``sbatch``."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sbatch", delete=False) as f:
        f.write(script_content)
        f.flush()
        print(f"Sbatch script written to: {f.name}")
        result = subprocess.run(["sbatch", f.name], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error submitting job: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        print(result.stdout.strip())
        return result.stdout.strip()


def make_log_dir(results_dir: str | Path) -> Path:
    """``<results_dir>/logs``, created on disk (``parents=True``)."""
    log_dir = Path(results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def write_list_file(log_dir: str | Path, filename: str, lines: list[str]) -> Path:
    """Write ``<log_dir>/<filename>`` with one entry per line (trailing newline)."""
    list_file = Path(log_dir) / filename
    list_file.write_text("\n".join(lines) + "\n")
    return list_file


def build_mail_line(mail_user: str | None) -> str:
    """Build the SBATCH mail directives, or an empty string if unset."""
    if mail_user:
        return f"#SBATCH --mail-user={mail_user}\n#SBATCH --mail-type=ALL"
    return ""


def resolve_resources(args, defaults: dict) -> dict:
    """Resolve resource values: use the CLI override if not None, else default."""
    return {
        key: getattr(args, key, None) if getattr(args, key, None) is not None else val
        for key, val in defaults.items()
    }
