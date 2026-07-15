"""SLURM array-submit for ``network-glm lev2``.

Faithful port of the monolith's ``Lev2Pipeline`` (``neuro_workflow.pipelines.
lev2``): discovers contrast names from lev1 fixed-effects outputs (or takes
them explicitly), writes a contrast_list.txt, and submits one array task per
contrast. Same dataset-registry simplification as :mod:`network_glm.submit.
lev1` — explicit ``--lev1-dirs``/``--results-dir``/``--partition`` instead of
a ``--dataset`` name.
"""

from __future__ import annotations

import argparse
import glob
import re
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
from network_glm.task_config.loader import get_base_tasks, get_dual_tasks

TEMPLATE_DIR = Path(__file__).parent / "templates"

DEFAULT_RESOURCES = {"nthreads": 2, "mem_gb": 4, "time": "04:00:00"}

DEFAULT_CONTAINER_IMAGE = "/home/groups/russpold/singularity_images/network_glm.sif"


def discover_contrasts_from_lev1_dirs(
    lev1_dirs: list[str],
    task_filter: list[str] | None = None,
    space: str = "volume",
) -> list[str]:
    """Glob fixed-effects files and extract contrast names.

    Mirrors the monolith's ``_discover_contrasts_from_lev1_dirs``: contrast
    names can contain underscores, so the capture spans everything between
    ``task-`` and the next BIDS-entity boundary (``_rtmodel-`` if present,
    ``_stat-`` as a fallback), not just the first underscore.
    """
    if space == "surface":
        leaf = "*_hemi-L_*_stat-fixed-effects.func.gii"
    else:
        leaf = "*_stat-fixed-effects.nii.gz"
    contrasts = set()
    for lev1_dir in lev1_dirs:
        pattern = str(Path(lev1_dir) / "sub-*" / "*" / "fixed_effects" / leaf)
        for fpath in glob.glob(pattern):
            fname = Path(fpath).name
            m = re.search(r"(task-[^_]+_contrast-.+?)_(?:rtmodel-|stat-)", fname)
            if m:
                contrast_id = m.group(1)
                if task_filter is None:
                    contrasts.add(contrast_id)
                else:
                    task_m = re.search(r"task-([^_]+)", contrast_id)
                    if task_m and task_m.group(1) in task_filter:
                        contrasts.add(contrast_id)
    return sorted(contrasts)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a network-glm lev2 SLURM array job")

    parser.add_argument("--lev1-dirs", nargs="+", required=True, help="Level-1 results directories")
    parser.add_argument("--results-dir", required=True, help="Level-2 output directory")

    contrast_group = parser.add_mutually_exclusive_group(required=True)
    contrast_group.add_argument("--contrasts", nargs="+", help="Specific contrast names")
    contrast_group.add_argument(
        "--all", dest="contrasts_flag", action="store_const", const="all",
        help="All contrasts discovered from --lev1-dirs",
    )
    contrast_group.add_argument(
        "--base-tasks", dest="contrasts_flag", action="store_const", const="base",
        help="Contrasts belonging to base tasks",
    )
    contrast_group.add_argument(
        "--dual-tasks", dest="contrasts_flag", action="store_const", const="dual",
        help="Contrasts belonging to dual tasks",
    )

    parser.add_argument(
        "--space",
        choices=["volume", "surface"],
        default="volume",
        help="volume: FSL randomise on NIfTI fixed-effects (default). "
        "surface: sign-flip permutation on GIFTI surface fixed-effects.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.9)
    parser.add_argument("--num-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--partition", required=True, help="SLURM partition (Sherlock: no default)")
    parser.add_argument("--nthreads", type=int, default=None)
    parser.add_argument("--mem-gb", type=int, default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--mail-user", default=None)
    parser.add_argument("--job-name", default="lev2")
    parser.add_argument("--container-image", default=DEFAULT_CONTAINER_IMAGE)
    parser.add_argument(
        "--print-only", action="store_true", default=False,
        help="Render the sbatch script to stdout without submitting",
    )
    return parser


def build_context(args: argparse.Namespace) -> dict:
    space = args.space
    contrasts_flag = getattr(args, "contrasts_flag", None)
    if contrasts_flag == "all":
        contrasts = discover_contrasts_from_lev1_dirs(args.lev1_dirs, space=space)
    elif contrasts_flag == "base":
        contrasts = discover_contrasts_from_lev1_dirs(
            args.lev1_dirs, task_filter=get_base_tasks(), space=space
        )
    elif contrasts_flag == "dual":
        contrasts = discover_contrasts_from_lev1_dirs(
            args.lev1_dirs, task_filter=get_dual_tasks(), space=space
        )
    else:
        contrasts = args.contrasts

    results_dir = Path(args.results_dir)
    log_dir = make_log_dir(results_dir)
    contrast_list_file = write_list_file(log_dir, "contrast_list.txt", contrasts)

    resources = resolve_resources(args, DEFAULT_RESOURCES)

    # Surface uses the self-contained numpy sign-flip test (no FSL); volume
    # uses FSL randomise. FSL is not in the container — loaded via `ml` at
    # runtime, exactly as the monolith does.
    module_loads = "" if space == "surface" else "module load biology fsl"

    return {
        "job_name": args.job_name,
        "n_contrasts": len(contrasts),
        "nthreads": resources["nthreads"],
        "mem_gb": resources["mem_gb"],
        "time": resources["time"],
        "partition": args.partition,
        "log_dir": str(log_dir),
        "mail_line": build_mail_line(args.mail_user),
        "contrast_list_file": str(contrast_list_file),
        "lev1_dirs": " ".join(args.lev1_dirs),
        "results_dir": str(results_dir),
        "space": space,
        "module_loads": module_loads,
        "mask_threshold": args.mask_threshold,
        "num_permutations": args.num_permutations,
        "seed": args.seed,
        "container_image": args.container_image,
    }


def main(argv: list[str] | None = None) -> int:
    parser = get_parser()
    args = parser.parse_args(argv)

    ctx = build_context(args)
    if ctx["n_contrasts"] == 0:
        print("Error: no contrasts found", file=sys.stderr)
        return 1

    template_path = TEMPLATE_DIR / "lev2.sbatch"
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
