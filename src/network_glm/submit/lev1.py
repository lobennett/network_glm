"""SLURM array-submit for ``network-glm lev1``.

Faithful (behavior-preserving) port of the monolith's ``Lev1Pipeline``
(``neuro_workflow.pipelines.lev1``), minus the multi-dataset registry
(``pipeline_config.json`` / ``~/.neuro_workflow/datasets.json``) that
network_glm does not carry — callers pass ``--bids-dir``/``--fmriprep-dir``/
``--results-dir``/``--partition`` explicitly instead of a ``--dataset`` name,
matching how ``network_glm.lev1.run`` itself takes these directly.

One SLURM array task == one (subject, task) pair, exactly like the monolith.
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
    write_list_file,
)
from network_glm.task_config.loader import get_all_tasks, get_base_tasks, get_dual_tasks

TEMPLATE_DIR = Path(__file__).parent / "templates"

DEFAULT_RESOURCES = {"nthreads": 1, "mem_gb": 64, "time": "2-00:00:00"}

DEFAULT_CONTAINER_IMAGE = "/home/groups/russpold/singularity_images/network_glm.sif"


def resolve_tasks(
    *,
    all_tasks: bool,
    base: bool,
    dual: bool,
    tasks: list[str] | None,
) -> list[str]:
    """Resolve the effective task list from the mutually-exclusive CLI flags.

    Exactly one of ``all_tasks``/``base``/``dual``/``tasks`` should be truthy
    (enforced by the argparse mutually-exclusive group in :func:`get_parser`);
    this helper is pure so it's directly unit-testable without argparse.
    """
    if all_tasks:
        return get_all_tasks()
    if base:
        return get_base_tasks()
    if dual:
        return get_dual_tasks()
    return list(tasks or [])


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a network-glm lev1 SLURM array job")

    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--tasks", nargs="+", help="Task name(s) to run")
    task_group.add_argument(
        "--all", dest="tasks_flag", action="store_const", const="all", help="Run all tasks"
    )
    task_group.add_argument(
        "--base-tasks", dest="tasks_flag", action="store_const", const="base",
        help="Run base (single-task) paradigms",
    )
    task_group.add_argument(
        "--dual-tasks", dest="tasks_flag", action="store_const", const="dual",
        help="Run dual-task paradigms",
    )

    parser.add_argument(
        "--subjects", nargs="+", required=True, help="Bare subject IDs (e.g. s03 s10)"
    )
    parser.add_argument("--bids-dir", required=True, help="BIDS directory path")
    parser.add_argument("--fmriprep-dir", required=True, help="fMRIPrep derivatives directory")
    parser.add_argument("--results-dir", required=True, help="Level-1 output directory")
    parser.add_argument("--exclusions-file", required=True, help="Path to exclusions JSON file")
    parser.add_argument(
        "--space",
        default="MNI",
        choices=["MNI", "T1w", "surface", "fsaverage6", "fsLR"],
        help="Analysis space (default: MNI)",
    )
    parser.add_argument(
        "--threshold", type=float, default=1.0, help="Within-subject mask threshold (default: 1.0)"
    )
    parser.add_argument("--smoothing-fwhm", type=float, default=None, help="Spatial smoothing FWHM in mm")
    parser.add_argument("--residuals", action="store_true", default=False)
    parser.add_argument("--fc-confounds", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--skip-qc-plots", action="store_true", default=False)
    parser.add_argument("--min-runs", type=int, default=2)

    parser.add_argument("--partition", required=True, help="SLURM partition (Sherlock: no default)")
    parser.add_argument("--nthreads", type=int, default=None, help="CPUs per task")
    parser.add_argument("--mem-gb", type=int, default=None, help="Memory in GB")
    parser.add_argument("--time", default=None, help="SLURM time limit (D-HH:MM:SS)")
    parser.add_argument("--mail-user", default=None)
    parser.add_argument("--job-name", default="lev1")
    parser.add_argument("--container-image", default=DEFAULT_CONTAINER_IMAGE)
    parser.add_argument(
        "--print-only", action="store_true", default=False,
        help="Render the sbatch script to stdout without submitting",
    )
    return parser


def build_context(args: argparse.Namespace) -> dict:
    tasks_flag = getattr(args, "tasks_flag", None)
    tasks = resolve_tasks(
        all_tasks=tasks_flag == "all",
        base=tasks_flag == "base",
        dual=tasks_flag == "dual",
        tasks=args.tasks,
    )

    results_dir = Path(args.results_dir)
    log_dir = make_log_dir(results_dir)
    pairs = [(subj, task) for subj in args.subjects for task in tasks]
    job_list_file = write_list_file(
        log_dir, "job_list.txt", [f"{subj} {task}" for subj, task in pairs]
    )

    resources = resolve_resources(args, DEFAULT_RESOURCES)

    extra_flags = []
    if args.smoothing_fwhm is not None:
        extra_flags.append(f"--smoothing-fwhm {args.smoothing_fwhm}")
    if args.residuals:
        extra_flags.append("--residuals")
    if args.fc_confounds:
        extra_flags.append("--fc-confounds")
    if args.skip_existing:
        extra_flags.append("--skip-existing")
    if args.skip_qc_plots:
        extra_flags.append("--skip-qc-plots")
    if args.min_runs is not None:
        extra_flags.append(f"--min-runs {args.min_runs}")

    return {
        "job_name": args.job_name,
        "n_jobs": len(pairs),
        "nthreads": resources["nthreads"],
        "mem_gb": resources["mem_gb"],
        "time": resources["time"],
        "partition": args.partition,
        "log_dir": str(log_dir),
        "mail_line": build_mail_line(args.mail_user),
        "job_list_file": str(job_list_file),
        "bids_dir": args.bids_dir,
        "fmriprep_dir": args.fmriprep_dir,
        "results_dir": str(results_dir),
        "exclusions_file": args.exclusions_file,
        "space": args.space,
        "threshold": args.threshold,
        "extra_flags": " ".join(extra_flags),
        "container_image": args.container_image,
    }


def main(argv: list[str] | None = None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)

    if not args.subjects:
        print("Error: --subjects must be non-empty", file=sys.stderr)
        return 1

    ctx = build_context(args)
    if ctx["n_jobs"] == 0:
        print("Error: no (subject, task) pairs to submit", file=sys.stderr)
        return 1

    template_path = TEMPLATE_DIR / "lev1.sbatch"
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
