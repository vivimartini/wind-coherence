#!/bin/bash
#SBATCH --job-name=wn-mem-spike
#SBATCH --account=mlmi-vm481-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/vm481/hpc-work/wind-coherence/logs/mem_spike_%j.out
#SBATCH --error=/rds/user/vm481/hpc-work/wind-coherence/logs/mem_spike_%j.err

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module unload cuda 2>/dev/null || true
set -euo pipefail
unset JAX_PLATFORMS || true
source /rds/user/vm481/hpc-work/wind-coherence/venv-wn/bin/activate
cd /home/vm481/hpc-work/wind-coherence
echo "job ${SLURM_JOB_ID} gres ${SLURM_GPUS_ON_NODE:-unset}"
nvidia-smi -L
exec python pilot/mem_spike.py
