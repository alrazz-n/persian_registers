#!/bin/bash
#SBATCH --job-name=Correlation
#SBATCH --account=project_2005092
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=Correlation_%j.out
#SBATCH --error=Correlation_%j.err

# 1. Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# 2. Activate venv
source /projappl/project_2005092/nima/persian_registers/cleanlab-venv/bin/activate

# 4. Run script
python3 Correlations.py