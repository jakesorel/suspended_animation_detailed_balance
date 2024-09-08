#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=00:59:00   # walltime
#SBATCH -J "SA_simulations"   # job name
#SBATCH -n 1
#SBATCH --partition=ncpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G

eval "$(conda shell.bash hook)"
source activate regression_modelling


python run_k_unbind_B_tot_scan.py ${SLURM_ARRAY_TASK_ID}
