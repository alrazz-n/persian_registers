#!/bin/bash
#SBATCH --job-name=n_token
#SBATCH --account=project_2002026
#SBATCH --partition=small
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=n_token%j.out
#SBATCH --error=n_token%j.err

# Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.6

# Activate venv
source /projappl/project_2005092/nima/persian_registers/cleanlab-venv/bin/activate

echo "Python:"
which python
which python3

python --version
python3 --version

python -c "import sys; print(sys.executable)"
python -c "import transformers; print(transformers.__file__)"
python -c "import huggingface_hub; print(huggingface_hub.__version__)"

# Run script
python3 /projappl/project_2005092/nima/binary/6_finding_n_token.py