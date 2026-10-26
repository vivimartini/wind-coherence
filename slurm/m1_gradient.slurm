#!/bin/bash
#SBATCH --job-name=wn-mini-m1
#SBATCH --account=mlmi-vm481-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/rds/user/vm481/hpc-work/wind-coherence/logs/m1_%j.out
#SBATCH --error=/rds/user/vm481/hpc-work/wind-coherence/logs/m1_%j.err

# Wilkes3 Ampere. rhel8/default-amp is required for the GPU driver stack.
# Its CUDA toolkit is 11.4. The JAX wheel brings its own CUDA 12 runtime, so
# the toolkit module is unloaded and the pip NVIDIA libraries are used.

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module unload cuda 2>/dev/null || true

set -euo pipefail
unset JAX_PLATFORMS || true

source /rds/user/vm481/hpc-work/wind-coherence/venv-wn/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export JAX_COMPILATION_CACHE_DIR=/rds/user/vm481/hpc-work/wind-coherence/jax-cache

echo "=== node $(hostname) job ${SLURM_JOB_ID} ==="
nvidia-smi
python -c "import jax, haiku; print('jax', jax.__version__, 'backend', jax.default_backend(), 'devices', jax.devices())"

exec python /home/vm481/hpc-work/wind-coherence/pilot/m1_suite.py
