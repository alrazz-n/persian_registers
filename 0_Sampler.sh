#!/bin/bash
#SBATCH --job-name=Sampler
#SBATCH --account=project_2002026
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --mem=128
#SBATCH --time=01:00:00
#SBATCH --output=Sampler_%j.out
#SBATCH --error=bSampler_%j.err

# 1. Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# 2. Activate venv
source /projappl/project_2005092/nima/persian_registers/cleanlab-venv/bin/activate

# 4. Run script
python3 /projappl/project_2005092/nima/testing/0_Sampler.py