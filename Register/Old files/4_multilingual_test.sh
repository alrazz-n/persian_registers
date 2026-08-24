#!/bin/bash
#SBATCH --job-name=finetune_multilingual
#SBATCH --account=project_2002026
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=multi_ling_finetune_%j.out
#SBATCH --error=multi_ling_finetune_%j.err


#1. Purge everything
module purge

#2. Load the system dependencies
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# 3. Activate your specific venv last
source cleanlab-venv/bin/activate

# Run the Python script
python3 4_XLMR_multiling_finetune.py
