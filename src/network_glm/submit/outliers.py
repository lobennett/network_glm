"""SLURM submit for ``network-glm cohort-outliers``.

Faithful port of the monolith's bare ``scripts/run_lev1_outliers.sbatch``:
a single (non-array) job with small fixed resources that forwards its
arguments straight through to the cohort-outliers CLI. The monolith script
hardcoded those resources; here they're CLI-overridable (still with the same
defaults) since network_glm has no per-dataset config to source them from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from network_glm.submit._slurm import (
    build_mail_line,
    make_log_dir,
    render_template,
    resolve_resources,
    submit_sbatch,
)

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Matches the monolith's scripts/run_lev1_outliers.sbatch hardcoded values.
DEFAULT_RESOURCES = {"nthreads": 2, "mem_gb": 16, "time": "01:00:00"}

DEFAULT_CONTAINER_IMAGE = "/home/groups/russpold/singularity_images/network_glm.sif"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit a network-glm cohort-outliers SLURM job",
    )
    parser.add_argument("--results-dir", required=True, help="Where to write job logs")
    parser.add_argument("--partition", required=True, help="SLURM partition (Sherlock: no default)")
    parser.add_argument("--nthreads", type=int, default=None)
    parser.add_argument("--mem-gb", type=int, default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--mail-user", default=None)
    parser.add_argument("--job-name", default="lev1_outliers")
    parser.add_argument("--container-image", default=DEFAULT_CONTAINER_IMAGE)
    parser.add_argument(
        "--print-only", action="store_true", default=False,
        help="Render the sbatch script to stdout without submitting",
    )
    parser.add_argument(
        "cohort_args",
        nargs=argparse.REMAINDER,
        help="Args forwarded verbatim to `network-glm cohort-outliers` "
        "(e.g. --lev1-dir ... --output-dir ... --exclusions-file ...); "
        "pass after `--`.",
    )
    return parser


def build_context(args: argparse.Namespace) -> dict:
    log_dir = make_log_dir(args.results_dir)
    resources = resolve_resources(args, DEFAULT_RESOURCES)

    cohort_args = list(args.cohort_args)
    if cohort_args and cohort_args[0] == "--":
        cohort_args = cohort_args[1:]

    return {
        "job_name": args.job_name,
        "nthreads": resources["nthreads"],
        "mem_gb": resources["mem_gb"],
        "time": resources["time"],
        "partition": args.partition,
        "log_dir": str(log_dir),
        "mail_line": build_mail_line(args.mail_user),
        "container_image": args.container_image,
        "cohort_args": " ".join(cohort_args),
    }


def main(argv: list[str] | None = None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)

    if not args.cohort_args:
        print(
            "Error: no cohort-outliers args given (pass them after `--`, e.g. "
            "`network-glm submit outliers --results-dir d --partition p -- "
            "--lev1-dir ... --output-dir ...`)",
            file=sys.stderr,
        )
        return 1

    ctx = build_context(args)
    template_path = TEMPLATE_DIR / "outliers.sbatch"
    script = render_template(template_path, ctx)

    if args.print_only:
        print(script)
        return 0

    print("--- Generated sbatch script ---")
    print(script)
    print("--- Submitting ---")
    submit_sbatch(script)
    return 0


if __name__ == "__main__":
    sys.exit(main())
