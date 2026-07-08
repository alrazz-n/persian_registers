#!/bin/bash
#SBATCH --job-name=XLMR_optim
#SBATCH --account=project_2002026
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=XLMR_optim_%j.out
#SBATCH --error=XLMR_optim_%j.err

# 1. Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# 2. Activate venv
source cleanlab-venv/bin/activate

# 4. Run script
python3 2_XLMR_optim.py