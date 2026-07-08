#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --account=project_2002026
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=inference_%j.out
#SBATCH --error=inference_%j.err

# 1. Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# 2. Activate venv
source cleanlab-venv/bin/activate

# 4. Run script
python3 7_inference.py