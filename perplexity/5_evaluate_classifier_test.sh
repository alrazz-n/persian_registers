#!/bin/bash
#SBATCH --job-name=reviewer_eval
#SBATCH --account=project_462001491
#SBATCH --partition=small-g

#SBATCH --array=0-2

#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

#SBATCH --output=reviewer_eval%A_%a.out
#SBATCH --error=reviewer_eval%A_%a.err

#============================================================
#ENVIRONMENT
#============================================================

module purge
module use /appl/local/csc/modulefiles
module load pytorch

source /projappl/project_462001491/nima/Fine_tuning-env/bin/activate

#============================================================
#JOB INFORMATION
#============================================================

echo "================================================"
echo "Reviewer LM evaluation"
echo "================================================"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date)"

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

#============================================================
#SELECT CORPUS
#============================================================

CORPORA=("hplt3" "random20" "perref")

CORPUS=${CORPORA[$SLURM_ARRAY_TASK_ID]}

echo ""
echo "Corpus: $CORPUS"

echo "================================================"

#============================================================
#EVALUATE
#============================================================

python3 5_evaluate_classifier_test.py \
    --corpus "$CORPUS"


EXIT_CODE=$?

#============================================================
#FINISH
#============================================================

echo ""
echo "================================================"

if [ $EXIT_CODE -eq 0 ]; then

echo "Evaluation completed successfully."


else

echo "Evaluation FAILED."
echo "Exit code: $EXIT_CODE"


fi

echo "Finished: $(date)"

echo "================================================"

exit $EXIT_CODE