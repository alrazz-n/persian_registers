#!/bin/bash
#SBATCH --job-name=Dist_HPLTPerRef
#SBATCH --account=project_462001491
#SBATCH --partition=small

#SBATCH --output=Dist_HPLTPerRef_%j.out
#SBATCH --error=Dist_HPLTPerRef_%j.err

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# 1. Load modules
module purge
module use /appl/local/csc/modulefiles
module load pytorch

# 2. Activate venv
source /projappl/project_462001491/nima/Fine_tuning-env/bin/activate

echo "HPLT3 PerRef Distibution JOB"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"

echo ""
echo "Python:"
which python3

echo ""
echo "Python version:"
python3 --version

echo ""
echo "CPU allocation:"
echo "$SLURM_CPUS_PER_TASK"

echo ""

echo "================================================"
echo "Running test"
echo "================================================"
# 4. Run script
python3 2_whole_HLPT3PerRef_distibution.py

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
