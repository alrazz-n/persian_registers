#!/bin/bash
#SBATCH --job-name=XLMR_finetune
#SBATCH --account=project_2002026
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=XLMR_finetune_%j.out
#SBATCH --error=XLMR_finetune_%j.err

# 1. Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# 2. Activate venv
source /projappl/project_2005092/nima/persian_registers/cleanlab-venv/bin/activate

# 4. Run script
python3 2_XLMR_finetune.py