#!/bin/bash
#SBATCH --job-name=sc003_corr_lvl1
#SBATCH --output=/home/igw180000/scratch/sc003_corr_lvl1_%j.out
#SBATCH --error=/home/igw180000/scratch/sc003_corr_lvl1_%j.err
#SBATCH --partition=rodrigues,cpu-preempt
#SBATCH --exclude=c-05-08
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Use all memory available on the assigned node.
#SBATCH --mem=0
#SBATCH --time=48:00:00

set -euo pipefail

unset PYTHONHOME
unset PYTHONPATH

source /opt/ohpc/pub/unpackaged/apps/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate scintkit

# Each Python worker reads one large file. Keep numerical libraries from
# creating additional threads inside those four worker processes.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Slurm executes a spool copy, so resolve the program from the directory in
# which sbatch was invoked rather than from BASH_SOURCE.
if [[ -f "$SLURM_SUBMIT_DIR/src/scintkit/space_receiver_processing/reduce_cross_correlations_lvl1.py" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
    SCRIPT="$PROJECT_ROOT/src/scintkit/space_receiver_processing/reduce_cross_correlations_lvl1.py"
elif [[ -f "$SLURM_SUBMIT_DIR/reduce_cross_correlations_lvl1.py" ]]; then
    SCRIPT="$SLURM_SUBMIT_DIR/reduce_cross_correlations_lvl1.py"
    PROJECT_ROOT="$(cd -- "$SLURM_SUBMIT_DIR/../../.." && pwd)"
else
    echo "Could not find reduce_cross_correlations_lvl1.py from: $SLURM_SUBMIT_DIR" >&2
    echo "Submit from the scintkit_summer root or src/scintkit/space_receiver_processing." >&2
    exit 2
fi

export PYTHONPATH="$PROJECT_ROOT/src"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Workers/CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated memory: ${SLURM_MEM_PER_NODE:-all node memory}"
echo "Project: $PROJECT_ROOT"

cd "$PROJECT_ROOT"
python -u "$SCRIPT" \
    --input-dir /titan/frodrigues/corrs_sc003 \
    --output-dir /titan/frodrigues/corrs_sc003/lvl1 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --seconds-per-sample 0.05 \
    --combined-name sc003_corrs_all_lvl1.pq
