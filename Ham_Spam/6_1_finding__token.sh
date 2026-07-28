#!/bin/bash
#SBATCH --account=project_2005092
#SBATCH --partition=small
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-12
#SBATCH --output=logs/token_%A_%a.out
#SBATCH --error=logs/token_%A_%a.err

# Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

source /projappl/project_2005092/nima/persian_registers/cleanlab-venv/bin/activate

INPUT_DIR=/scratch/project_2005092/nima/annotated_data/kept
OUTPUT_DIR=/scratch/project_2005092/nima/token_counts

FILES=($(ls ${INPUT_DIR}/*.jsonl.zst | sort))

python 6_1_finding__token.py \
    --input "${FILES[$SLURM_ARRAY_TASK_ID]}" \
    --output_dir "${OUTPUT_DIR}"