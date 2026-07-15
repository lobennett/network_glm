#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --mem={mem_gb}G
#SBATCH --cpus-per-task={nthreads}
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err
{mail_line}

set -euo pipefail

apptainer exec "{container_image}" network-glm cohort-outliers {cohort_args}
