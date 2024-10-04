#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=2-23:59:00   # walltime
#SBATCH -J "SA_simulations"   # job name
#SBATCH -n 1
#SBATCH --partition=ncpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

eval "$(conda shell.bash hook)"
source activate regression_modelling


python run_fit_DE.py ${SLURM_ARRAY_TASK_ID}
