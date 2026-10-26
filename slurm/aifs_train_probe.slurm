#!/bin/bash
#SBATCH --job-name=aifs-train-probe
#SBATCH --account=mlmi-vm481-sl2-gpu
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/rds/user/vm481/hpc-work/wind-coherence/logs/aifs_probe_%j.out
#SBATCH --error=/rds/user/vm481/hpc-work/wind-coherence/logs/aifs_probe_%j.err

# Forward and backward with the CRPS loss through the power chain, 2 members,
# 1, 2, 4 and 7 six-hour steps, on one 80 GB A100. Decides AIFS ENS or Mini.
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module unload cuda 2>/dev/null || true
set -euo pipefail
unset JAX_PLATFORMS || true
export WN_ROOT=/rds/user/vm481/hpc-work/wind-coherence
export C_INCLUDE_PATH="$WN_ROOT/pydev/usr/include/python3.12${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export TRITON_CACHE_DIR="$WN_ROOT/triton-cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRITON_CACHE_DIR"
probe=$(mktemp -d)
printf '#include <Python.h>\nint f(void){return 0;}\n' > "$probe/h.c"
if ! gcc "$probe/h.c" -shared -fPIC -o "$probe/h.so" 2> "$probe/h.err"; then
    echo "refusing to run: Python.h does not compile on this node"
    cat "$probe/h.err"
    exit 1
fi
echo "python_headers ok"
source "$WN_ROOT/venv-aifs/bin/activate"
cd /home/vm481/hpc-work/wind-coherence
nvidia-smi -L
total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if (( total < 70000 )); then
    echo "refusing to run: need a full 80 GB A100, this device has ${total} MiB"
    exit 1
fi
python pilot/aifs_train_probe.py &
pid=$!
peak=0
while kill -0 "$pid" 2>/dev/null; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 || true)
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used > peak )); then
        peak=$used
    fi
    sleep 5
done
set +e
wait "$pid"
status=$?
set -e
echo "nvidia_peak_mib ${peak} exit ${status}"
exit "$status"
