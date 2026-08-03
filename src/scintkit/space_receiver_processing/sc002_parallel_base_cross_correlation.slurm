#!/bin/bash
#SBATCH --job-name=sc002_corr
#SBATCH --output=/home/igw180000/scratch/sc002_corr_%j.out
#SBATCH --error=/home/igw180000/scratch/sc002_corr_%j.err
#SBATCH --partition=rodrigues
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail

unset PYTHONHOME
unset PYTHONPATH

source /opt/ohpc/pub/unpackaged/apps/miniconda/4.12.0/etc/profile.d/conda.sh
conda activate scintkit

# Keep each worker single-threaded so four Python workers use the four CPUs
# allocated by Slurm instead of oversubscribing the node through BLAS threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Slurm runs a spool copy of this file, so BASH_SOURCE points under
# /var/spool/slurm rather than at the checkout. Resolve the program from the
# directory in which sbatch was invoked.
if [[ -f "$SLURM_SUBMIT_DIR/src/scintkit/space_receiver_processing/parallel_base_cross_correlation.py" ]]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
    SCRIPT_DIR="$PROJECT_ROOT/src/scintkit/space_receiver_processing"
elif [[ -f "$SLURM_SUBMIT_DIR/parallel_base_cross_correlation.py" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
    PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
else
    echo "Could not find parallel_base_cross_correlation.py from: $SLURM_SUBMIT_DIR" >&2
    echo "Submit from the scintkit_summer root or src/scintkit/space_receiver_processing." >&2
    exit 2
fi
export PYTHONPATH="$PROJECT_ROOT/src"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Project: $PROJECT_ROOT"

cd "$PROJECT_ROOT"
python -u "$SCRIPT_DIR/parallel_base_cross_correlation.py"
