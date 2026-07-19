#!/bin/bash
#SBATCH --job-name=Predict_spam_ham
#SBATCH --account=project_2002026
#SBATCH --partition=gpusmall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16 
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=Predict_spam_ham_%j.out
#SBATCH --error=Predict_spam_ham_%j.err

# 1. Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# 2. Activate venv
source /projappl/project_2005092/nima/persian_registers/cleanlab-venv/bin/activate

# 4. Run script
python3 4_Prediction_Spam_ham.py
