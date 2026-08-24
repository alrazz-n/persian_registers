#!/bin/bash
#SBATCH --job-name=MultiCore_test
#SBATCH --account=project_462001491
#SBATCH --partition=standard-g

#SBATCH --array=0-9%4

#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=MultiCore_test_%A_%a.out
#SBATCH --error=MultiCore_test_%A_%a.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch


echo "Start: $(date)"
python3 1_multi_Core_test.py

echo "================================================"
echo "Done: $(date)"
echo "================================================"