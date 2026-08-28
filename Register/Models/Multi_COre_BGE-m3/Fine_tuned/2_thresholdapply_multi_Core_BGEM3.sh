#!/bin/bash

#SBATCH --job-name=MultiCore_Threshold_XLMR
#SBATCH --account=project_462001491
#SBATCH --partition=standard-g

#SBATCH --array=0-3%4

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

#SBATCH --output=MultiCore_Threshold_XLMR_%A_%a.out
#SBATCH --error=MultiCore_Threshold_XLMR_%A_%a.err


module purge
module use /appl/local/csc/modulefiles
module load pytorch


source /projappl/project_462001491/nima/Fine_tuning-env/bin/activate


echo "================================================"
echo "Test Evaluation"
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
echo "Running test evaluation"
echo "================================================"


python3 2_thresholdapply_multi_Core_BGEM3.py

EXIT_CODE=$?


echo ""
echo "================================================"


if [ $EXIT_CODE -eq 0 ]; then

    echo "Test evaluation completed successfully."

else

    echo "Test evaluation FAILED."
    echo "Exit code: $EXIT_CODE"

fi


echo "Done: $(date)"

echo "================================================"


exit $EXIT_CODE
