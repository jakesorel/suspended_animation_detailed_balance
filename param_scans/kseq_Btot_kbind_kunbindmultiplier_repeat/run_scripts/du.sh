#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=23:59:00   # walltime
#SBATCH -J "disk_utility"   # job name
#SBATCH --output=disk_utility.out
#SBATCH --error=disk_utility..out
#SBATCH -n 1
#SBATCH --partition=cpu
#SBATCH --mem=50M

du -h --max-depth=3
