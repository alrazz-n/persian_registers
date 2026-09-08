#!/bin/bash
#SBATCH --job-name=TookaBERT-L
#SBATCH --account=project_462001491
#SBATCH --partition=standard-g

#SBATCH --array=0-3%4

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

#SBATCH --output=TookaBERT-L_%A_%a.out
#SBATCH --error=TookaBERT-L_%A_%a.err


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

echo ""
echo "GPU:"
nvidia-smi

echo "================================================"
echo "Running fine-tuning"
echo "================================================"


python3 1_TookaBERT-L_fine_tune.py

EXIT_CODE=$?


echo ""
echo "================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Fine-tuning completed successfully."
else
    echo "Fine-tuning FAILED."
    echo "Exit code: $EXIT_CODE"
fi

echo "Done: $(date)"

echo "================================================"

exit $EXIT_CODE
