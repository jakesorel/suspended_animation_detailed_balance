#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=00:05:00   # walltime
#SBATCH -J "SA_for_loop_jobs"   # job name
#SBATCH --output=../bash_out/for_loop_2nd_jobs_output.out
#SBATCH --error=../bash_out/for_loop_2nd_jobs_error.out
#SBATCH -n 1
#SBATCH --partition=ncpu
#SBATCH --mem=100M



sbatch --array [0-2499] --output=../bash_out/output_%A_%a.out --error=../bash_out/error_%A_%a.out run_simulation.sh ${SLURM_ARRAY_TASK_ID}

