#!/bin/bash
#SBATCH -J {job_name}
#SBATCH --array=1-{n_jobs}
#SBATCH --cpus-per-task={nthreads}
#SBATCH --mem={mem_gb}G
#SBATCH --time={time}
#SBATCH -p {partition}
#SBATCH -o {log_dir}/%x-%A-%a.out
#SBATCH -e {log_dir}/%x-%A-%a.err
{mail_line}

LINE="$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{job_list_file}")"
read -r SUBJ_ID TASK_NAME <<< "$LINE"

# Surface smoothing (--smoothing-fwhm with a surface --space) shells out to
# FreeSurfer's mri_surf2surf; load FreeSurfer so it's on PATH.  No-op for
# volumetric/MNI runs that don't smooth on the surface. FreeSurfer is a
# licensed, non-pip tool and is NOT baked into the network_glm container --
# it's loaded on the host and available to the apptainer exec below.
module load biology freesurfer/8.1.0

apptainer exec "{container_image}" network-glm lev1 \
    --subj-id "$SUBJ_ID" \
    --task-name "$TASK_NAME" \
    --bids-dir "{bids_dir}" \
    --fmriprep-dir "{fmriprep_dir}" \
    --results-dir "{results_dir}" \
    --exclusions-file "{exclusions_file}" \
    --space {space} \
    --within-subject-threshold {threshold} {extra_flags}
