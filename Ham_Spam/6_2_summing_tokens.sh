#!/bin/bash
#SBATCH --job-name=Sum_token
#SBATCH --account=project_2005092
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=sum_token%j.out
#SBATCH --error=sum_token%j.err

# 1. Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# 2. Activate venv
source /projappl/project_2005092/nima/persian_registers/cleanlab-venv/bin/activate

# 4. Run script
python3 6_2_summing_tokens.py
