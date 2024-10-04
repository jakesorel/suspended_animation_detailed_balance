#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=1-23:59:00   # walltime
#SBATCH -J "SA_simulations"   # job name
#SBATCH -n 1
#SBATCH --partition=ncpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G

eval "$(conda shell.bash hook)"
source activate regression_modelling


python run_fit_DE.py ${SLURM_ARRAY_TASK_ID}
