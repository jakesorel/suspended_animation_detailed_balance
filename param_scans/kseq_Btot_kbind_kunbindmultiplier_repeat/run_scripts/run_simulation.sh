#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=23:59:00   # walltime
#SBATCH -J "SA_simulations"   # job name
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G



eval "$(conda shell.bash hook)"
source activate regression_modelling

echo ${SLURM_TMPDIR}
export TMPDIR=/camp/home/cornwaj/working/suspended_animation_detailed_balance/param_scans/kseq_Btot_kbind_kunbindmultiplier_repeat/scan_results/tmp
echo ${SLURM_TMPDIR}
python run_simulations.py ${SLURM_ARRAY_TASK_ID}
