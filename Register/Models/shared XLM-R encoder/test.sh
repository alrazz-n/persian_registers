#!/bin/bash
#SBATCH --job-name=test_SharedRep
#SBATCH --account=project_462001491
#SBATCH --partition=dev-g

#SBATCH --array=0

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

#SBATCH --output=test_SharedRep_%A_%a.out
#SBATCH --error=test_SharedRep_%A_%a.err


module purge
module use /appl/local/csc/modulefiles
module load pytorch

# Activate your Python environment
source /projappl/project_462001491/nima/Fine_tuning-env/bin/activate

echo "================================================"
echo "Fine-tuning job"
echo "================================================"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Array ID:  $SLURM_ARRAY_TASK_ID"
echo "Start:     $(date)"

echo ""
echo "Python:"
which python3

echo ""
echo "Python version:"
python3 --version


echo "================================================"
echo "Running test"
echo "================================================"


python3 1_shared_multiCore_XLM-R.py

EXIT_CODE=$?


echo ""
echo "================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "test job completed successfully."
else
    echo "test job FAILED."
    echo "Exit code: $EXIT_CODE"
fi

echo "Done: $(date)"

echo "================================================"

exit $EXIT_CODE