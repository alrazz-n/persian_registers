#!/bin/bash
#SBATCH --job-name=MultiCore_test_raw_BGE
#SBATCH --account=project_462001491
#SBATCH --partition=standard-g

#SBATCH --array=0-3%4

#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=MultiCore_test_BGE_%A_%a.out
#SBATCH --error=MultiCore_test_BGE_%A_%a.err

module purge
module use /appl/local/csc/modulefiles
module load pytorch

# Activate your Python environment
source /projappl/project_462001491/nima/Fine_tuning-env/bin/activate

echo "================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Start:     $(date)"
echo "================================================"

echo ""
echo "Python:"
which python3

echo ""
echo "Python version:"
python3 --version

echo "================================================"
echo "Running evaluation"
echo "================================================"

python3 1_multi_Core_BGE_m3_test.py

EXIT_CODE=$?


# ============================================================
# Finished
# ============================================================

echo ""
echo "================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully."
else
    echo "Evaluation FAILED."
    echo "Exit code: $EXIT_CODE"
fi

echo "Done: $(date)"
echo "================================================"

exit $EXIT_CODE