#!/bin/bash
#SBATCH --job-name=bge-m3_optim_multilingual
#SBATCH --account=project_2002026
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=bge-m3_optim_multi_ling_%j.out
#SBATCH --error=bge-m3_optim_multi_ling_%j.err

# 1. Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# 2. Activate venv
source cleanlab-venv/bin/activate

# 4. Run script
python3 4_bge-m3_optim_multiling.py
