#!/bin/bash
#SBATCH --job-name=Dist
#SBATCH --account=project_2002026
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=distribution_%j.out
#SBATCH --error=distribution_%j.err

# Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# Activate venv
source /projappl/project_2005092/nima/persian_registers/cleanlab-venv/bin/activate

# Run script
python3 /projappl/project_2005092/nima/binary/plot_notebook/1_dist.py

