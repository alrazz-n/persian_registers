#!/bin/bash
#SBATCH --job-name=XLMR_zeroshot
#SBATCH --account=project_2002026
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=XLMR_zeroshot_%j.out
#SBATCH --error=XLMR_zeroshot_%j.err

# 1. Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# 2. Activate venv
source cleanlab-venv/bin/activate

# 4. Run script
python3 5_XLMR_zeroshot_finetune.py
