#!/bin/bash
#SBATCH --job-name=register_test
#SBATCH --account=project_462001491
#SBATCH --partition=dev-g

#SBATCH --array=0

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

#SBATCH --output=register_test_%j.out
#SBATCH --error=register_test_%j.err


# ============================================================
# Environment
# ============================================================

module purge
module use /appl/local/csc/modulefiles
module load pytorch


# ============================================================
# Information
# ============================================================

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

python3 1_multi_Core_test.py

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