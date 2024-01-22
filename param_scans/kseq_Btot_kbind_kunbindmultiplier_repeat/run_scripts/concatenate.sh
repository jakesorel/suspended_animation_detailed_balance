#!/bin/bash
#SBATCH --job-name=concatenate_csv
#SBATCH --output=../bash_out/concatenate_csv.out
#SBATCH --error=../bash_out/concatenate_csv.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --array=1-241
ml parallel

# Set the path to the source directory
source_dir="../scan_results/summary_tchosen/by_time"

# Set the path to the destination directory
destination_dir="../scan_results/summary_tchosen/by_time_concatenated"

# Get the sorted list of subdirectories
subdirectories=("$source_dir"/*)

# Get the current subdirectory
current_dir=${subdirectories[SLURM_ARRAY_TASK_ID-1]}

# Check if the directory exists
if [ -z "$current_dir" ]; then
  echo "No more subdirectories to process."
  exit 0
fi

# Set the path to the current source directory
current_source_dir="$source_dir/$current_dir"

# Set the path to the destination file
destination_file="$destination_dir/$current_dir.csv"

# Concatenate all *.csv files in the current subdirectory
find . -name "$current_source_dir"/"*.csv" -print0 | parallel -0 cat > "$destination_file"

echo "Concatenation completed for $current_dir. Result saved to $destination_file."
