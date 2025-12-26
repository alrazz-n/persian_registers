#!/bin/bash
#SBATCH --job-name=XLMR_finetune
#SBATCH --account=project_2002026
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=XLMR_finetune_%j.out
#SBATCH --error=XLMR_finetune_%j.err

# Activate virtual environment
source venv/bin/activate

# Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# Run the Python script
python3 2_XLMR_finetune.py