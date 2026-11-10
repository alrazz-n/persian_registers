#!/bin/bash
#SBATCH --job-name=Qualitative_analysis
#SBATCH --account=project_462001491
#SBATCH --partition=dev-g

#SBATCH --output=Qualitative_analysis_%j.out
#SBATCH --error=Qualitative_analysis_%j.err

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

#SBATCH --array=0

# 1. Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch

# 2. Activate venv
source /projappl/project_462001491/nima/Fine_tuning-env/bin/activate

echo "Qualitative analysis job"

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
# 4. Run script
python3 3_qualitaitve_analysis.py

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
