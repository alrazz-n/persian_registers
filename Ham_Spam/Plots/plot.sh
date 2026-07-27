#!/bin/bash
#SBATCH --job-name=junk_score_analysis_test
#SBATCH --account=project_2002026
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=junk_score_analysis_%j.out
#SBATCH --error=junk_score_analysis_%j.err

# Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# Activate venv
source /projappl/project_2005092/nima/persian_registers/cleanlab-venv/bin/activate

# Run script
python3 /projappl/project_2005092/nima/binary/plot.py