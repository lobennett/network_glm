#!/bin/bash
#SBATCH -J {job_name}
#SBATCH --array=1-{n_contrasts}
#SBATCH --cpus-per-task={nthreads}
#SBATCH --mem={mem_gb}G
#SBATCH --time={time}
#SBATCH -p {partition}
#SBATCH -o {log_dir}/%x-%A-%a.out
#SBATCH -e {log_dir}/%x-%A-%a.err
{mail_line}

CONTRAST="$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{contrast_list_file}")"

# FSL (volume randomise) is a licensed, non-pip tool and is NOT baked into
# the network_glm container -- loaded on the host and available to the
# apptainer exec below. No-op for --space surface (self-contained numpy
# sign-flip permutation, no FSL needed).
{module_loads}

apptainer exec "{container_image}" network-glm lev2 \
    --contrast "$CONTRAST" \
    --level1-dirs {lev1_dirs} \
    --output-dir "{results_dir}" \
    --space {space} \
    --mask-threshold {mask_threshold} \
    --num-permutations {num_permutations} \
    --seed {seed}
