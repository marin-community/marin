#!/bin/bash
#SBATCH --job-name=bridge_all
#SBATCH --account=ehpc524
#SBATCH --qos=gp_ehpc
#SBATCH --partition=gpp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --time=0-24:00:00
#SBATCH --output=/gpfs/scratch/ehpc524/ot/logs/bridge_all_%j.out
#SBATCH --error=/gpfs/scratch/ehpc524/ot/logs/bridge_all_%j.err

# Apptainer Bridge on MareNostrum 5 (MN5) — server AND workers co-located on
# a COMPUTE node.
#
# KEY MN5 DIFFERENCE vs JUWELS/Jusuf/JURECA:
#   On JUWELS the bridge server runs on the Jupiter/login node and only the
#   workers are on compute nodes (see start_bridge.sh + juwels_workers.sbatch).
#   On MN5 that does NOT work: BSC's login-node process killer repeatedly
#   reaps the long-lived bridge HTTP server, so the listener keeps going down.
#   The fix is to run the server AND the workers together on a `gpp` COMPUTE
#   node under account `ehpc524` (no login-node touchpoint at all), so the
#   killer never sees the server.
#
#   Server rendezvous for the co-located case: the server binds 0.0.0.0:PORT
#   on the SLURM BatchHost (rank-0 gpp node); its URL is published to
#   $URL_FILE so the RL launcher and any external client can pick it up at
#   run time. Workers on every node (including the BatchHost) connect to that
#   BatchHost URL directly over the cluster fabric.

set -uo pipefail

module --force purge
module load singularity/4.1.5


export BRIDGE_PROXY_HOST=127.0.0.1  # local ssh -L tunnel on this gpp node,
export BRIDGE_PROXY_PORT=9052       # forwards to glogin2:9055 -> VM SOCKS -> internet

OT_BASE="/gpfs/scratch/ehpc524/ot"
CONDA_ENV="/gpfs/scratch/ehpc524/envs/vllm018"
SIF_CACHE="${OT_BASE}/sif_cache"
STAGING_BASE="${OT_BASE}/staging"
WORKERS_PER_NODE="${WORKERS_PER_NODE:-128}"
BRIDGE_PORT="${BRIDGE_PORT:-9920}"

PYTHON="${CONDA_ENV}/bin/python3"
# Deployed copies of the bridge server/worker on MN5 scratch. These are
# `python -m harbor.environments.apptainer.{server,worker}` extracted to a
# plain path so the gpp node doesn't need the editable install on its PYTHONPATH.
SERVER_PY="${OT_BASE}/bridge/server.py"
WORKER_PY="${OT_BASE}/bridge/worker.py"
SERVER_LOG="${OT_BASE}/logs/bridge_server_${SLURM_JOB_ID}.log"
URL_FILE="${OT_BASE}/bridge/bridge_url.txt"

export PATH="${CONDA_ENV}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export BRIDGE_AGENT_TOOLS="${OT_BASE}/agent_tools"

# BatchHost = rank-0 node; that's where the server lives. MN5 keeps the
# server OFF the login node (see header) — the BatchHost is a gpp compute node.
SERVER_HOST="${SLURM_BATCH_HOST:-$(hostname -s)}"
BRIDGE_URL="http://${SERVER_HOST}:${BRIDGE_PORT}"

echo "============================================="
echo "MN5 Bridge (server + workers on gpp compute node)"
echo "  job:       ${SLURM_JOB_ID}"
echo "  account:   ehpc524   partition: gpp"
echo "  nodes:     ${SLURM_JOB_NUM_NODES}  (${SLURM_JOB_NODELIST})"
echo "  server:    ${SERVER_HOST}:${BRIDGE_PORT}  (compute node, not login)"
echo "  workers/n: ${WORKERS_PER_NODE}"
echo "  url file:  ${URL_FILE}"
echo "============================================="

mkdir -p "${OT_BASE}/logs" "${OT_BASE}/bridge" "${SIF_CACHE}" "${STAGING_BASE}"

# 1. Start server (with self-respawn) on the BatchHost gpp node only.
#    Self-respawn guards against any in-process crash; the login-node killer
#    is not a factor here because we are on a compute node.
start_server_loop() {
    while true; do
        echo "[server] $(date) launching server.py on $(hostname -s):${BRIDGE_PORT}"
        "${PYTHON}" "${SERVER_PY}" --port "${BRIDGE_PORT}" --host 0.0.0.0 \
            >> "${SERVER_LOG}" 2>&1
        rc=$?
        echo "[server] $(date) server exited rc=${rc}; respawning in 5s"
        sleep 5
    done
}
start_server_loop &
SERVER_PID=$!

# Wait for the server to actually bind, then publish its URL.
for i in $(seq 1 60); do
    if curl -sf --max-time 2 "http://localhost:${BRIDGE_PORT}/status" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
if ! curl -sf --max-time 2 "http://localhost:${BRIDGE_PORT}/status" >/dev/null 2>&1; then
    echo "FATAL: server did not come up on ${BRIDGE_PORT} within 60s"
    tail -50 "${SERVER_LOG}" || true
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
fi

printf '%s\n' "${BRIDGE_URL}" > "${URL_FILE}.tmp"
mv "${URL_FILE}.tmp" "${URL_FILE}"
echo "[server] published URL to ${URL_FILE}: ${BRIDGE_URL}"

cleanup() {
    echo "[bridge_all] cleanup: killing server pid=${SERVER_PID}"
    kill ${SERVER_PID} 2>/dev/null || true
    rm -f "${URL_FILE}" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

# 2. Launch one worker process per node (including the BatchHost, so the
#    server's own gpp node also runs workers).
srun --ntasks-per-node=1 bash -c "
    set -u
    module --force purge
    module load singularity/4.1.5
    export PATH=${CONDA_ENV}/bin:\$PATH
    export LD_LIBRARY_PATH=${CONDA_ENV}/lib:\${LD_LIBRARY_PATH:-}
    export BRIDGE_AGENT_TOOLS=${BRIDGE_AGENT_TOOLS}
    hostname
    mkdir -p ${STAGING_BASE}
    bash ${OT_BASE}/mn5_start_proxy.sh >/dev/null 2>&1 || true
    ${PYTHON} -u ${WORKER_PY} \
        --bridge-url ${BRIDGE_URL} \
        --sif-cache ${SIF_CACHE} \
        --staging-base ${STAGING_BASE} \
        --num-workers ${WORKERS_PER_NODE}
"
SRUN_RC=$?

echo "[bridge_all] srun exited rc=${SRUN_RC}; tearing down server"
exit ${SRUN_RC}
