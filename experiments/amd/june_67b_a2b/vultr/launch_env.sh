# Launch environment for every Snowball phase on the Vultr 2x8 MI355X cluster. Source it from a Slurm batch script
# at the checkout root, before the first srun:
#   source experiments/amd/june_67b_a2b/vultr/launch_env.sh
# It sets PY (the pinned JAX stack), PYTHONPATH, the RCCL, XLA and JAX settings, W&B paths and the open-file limit,
# and defines srun_with_hang_recovery. rehearsal.py refuses to start multi-node without the settings marked
# "checked" below and logs them all at startup.

WORK=/work/openathena-ihodes/asartran

# Pinned stack: JAX 0.12.0 on ROCm 10.1 (the `rocm` extra). The settings below were tested on it.
# jax-rocm10-plugin 0.12.1 (and 0.12.1.post1) has an exact, deterministic embedding scatter-add; it is not adopted.
PY=$WORK/rocm-extra-test/.venv/bin/python
pp=$PWD
for d in "$PWD"/lib/*/src; do pp=$pp:$d; done
export PYTHONPATH=$pp
# The venv ships its own ROCm libraries; the system ROCm 7.2 ones on the module path would shadow them.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/opt/rocm' | paste -sd: -)

# Checkpoints keep many files open at once. The compute nodes refuse raising the soft limit past the hard limit.
ulimit -n "$(ulimit -Hn)"

# Checked. With Tree or LL, a cross-node all-reduce can read a peer's input before it is written, which doubles
# the loss on some steps. Simple costs about 21% step time on the 67B step; NCCL_ALGO=Ring hangs at startup.
export NCCL_PROTO=Simple
# Checked. Command buffers (HIP graphs) corrupt device memory during training on this stack.
export XLA_FLAGS="--xla_gpu_enable_command_buffer= ${XLA_FLAGS:-}"
# RCCL warns that it runs slower without this.
export HSA_NO_SCRATCH_RECLAIM=1
# A preallocated pool is one fixed range that fragments into OOMs after a few steps; growing by region does not.
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export RAGGED_DOT_IMPL=triton JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/tmp/$USER/triton-$SLURM_JOB_ID XDG_CACHE_HOME=/tmp/$USER/xdg-$SLURM_JOB_ID

# Pensando ANP RCCL net plugin (about 1.45x the all-reduce bandwidth). Only the plugin is put on the library path,
# not the rest of the system ROCm 7.2 libraries.
export NCCL_NET_PLUGIN=librccl-anp.so NCCL_IB_TC=104 NCCL_MIN_NCHANNELS=32 HSA_FORCE_FINE_GRAIN_PCIE=1 \
  NCCL_GDR_FLUSH_DISABLE=1 NCCL_GDRCOPY_ENABLE=0 NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_IGNORE_CPU_AFFINITY=1 \
  RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
ANP_DIR=/tmp/$USER/anp-$SLURM_JOB_ID
srun --ntasks-per-node=1 bash -c "mkdir -p $ANP_DIR && ln -sf /opt/rocm-7.2.0/lib/librccl-anp.so $ANP_DIR/"
export LD_LIBRARY_PATH=$ANP_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN} NCCL_DEBUG_SUBSYS=INIT,NET

# /home is over quota; keep W&B's files on /work.
export WANDB_API_KEY=$(cat $WORK/secrets/wandb_api_key) WANDB_ENTITY=marin-community WANDB_RESUME=allow
export WANDB_DIR=$WORK/wandb/files WANDB_CACHE_DIR=$WORK/wandb/cache WANDB_CONFIG_DIR=$WORK/wandb/config \
  WANDB_DATA_DIR=$WORK/wandb/data
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"

# With the ANP plugin, RCCL communicator creation sometimes never completes: two ranks never join XLA's
# "Initialize clique" rendezvous and the others wait forever, logging that they "may be stuck".
STUCK_TIMEOUT=${STUCK_TIMEOUT:-300}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-3600}
STARTUP_RETRIES=${STARTUP_RETRIES:-2}
# A collective can also hang mid-run, with no abort and no non-finite value. The longest gap between two logged
# steps of a healthy 67B run is a checkpoint save plus an eval; HANG_TIMEOUT must exceed it.
HANG_TIMEOUT=${HANG_TIMEOUT:-2700}
# entity/project/run-id of a W&B run to post an alert to when a hang is detected; unset, the alert is only logged.
HANG_ALERT_WANDB=${HANG_ALERT_WANDB:-}

# last_logged_step LOG...: the highest step any of the logs has logged metrics for, empty if none.
last_logged_step() {
  tail -qc 1000000 "$@" 2>/dev/null | grep -o '"event": "log", "step": [0-9]*' | grep -o '[0-9]*$' | sort -n | tail -1
}

# hang_alert TEXT: log TEXT and, with HANG_ALERT_WANDB set, post it as a W&B alert on that run.
hang_alert() {
  echo "=== ALERT $1 at $(date -u +%FT%T)"
  [ -n "$HANG_ALERT_WANDB" ] || return 0
  timeout 180 "$PY" - "$HANG_ALERT_WANDB" "$1" <<'PYEOF'
import sys

import wandb

entity, project, run_id = sys.argv[1].split("/")
run = wandb.init(entity=entity, project=project, id=run_id, resume="allow")
run.alert(title="MI355X run hung", text=sys.argv[2], level=wandb.AlertLevel.WARN)
run.finish()
PYEOF
}

# stop_srun PID: stop an srun step, killing its tasks if they ignore SIGTERM for 30 s.
stop_srun() {
  kill -TERM "$1"
  sleep 30
  kill -KILL "$1" 2>/dev/null
  wait "$1"
}

# srun_with_hang_recovery LOG_PREFIX ARGS...: srun one process per node with all 8 GPUs, writing each task's output
# to LOG_PREFIX-<task>.log (tasks 0-9).
# Startup: if no task logs a step within STUCK_TIMEOUT seconds of the first "Initialize clique ... may be stuck"
# warning, or within STARTUP_TIMEOUT seconds of launch, it kills the step and launches again, up to STARTUP_RETRIES
# times; a hung attempt's logs are kept as LOG_PREFIX-<task>.hung<N>.log.
# Mid-run: once a step is logged, if no later step is logged within HANG_TIMEOUT seconds, it raises hang_alert,
# kills the step and returns 125, so the caller restarts from the latest checkpoint. Otherwise returns srun's status.
srun_with_hang_recovery() {
  local prefix=$1; shift
  local retry=0 pid start stuck_since step last_step progress_at
  while :; do
    srun --ntasks-per-node=1 --gpus-per-node=8 --kill-on-bad-exit=1 --output="$prefix-%t.log" "$@" &
    pid=$!
    start=$SECONDS
    stuck_since=
    last_step=
    while kill -0 "$pid" 2>/dev/null; do
      sleep 10
      step=$(last_logged_step "$prefix"-[0-9].log)
      if [ -n "$step" ]; then
        if [ "$step" != "$last_step" ]; then
          last_step=$step
          progress_at=$SECONDS
        elif [ $((SECONDS - progress_at)) -ge "$HANG_TIMEOUT" ]; then
          hang_alert "job $SLURM_JOB_ID ($prefix) logged no step after step $last_step for $((SECONDS - progress_at)) s; killing it to resume from the latest checkpoint"
          stop_srun "$pid"
          return 125
        fi
        continue
      fi
      if [ -z "$stuck_since" ] && grep -qs 'Initialize clique.*may be stuck' "$prefix"-[0-9].log; then
        stuck_since=$SECONDS
      fi
      if { [ -n "$stuck_since" ] && [ $((SECONDS - stuck_since)) -ge "$STUCK_TIMEOUT" ]; } ||
        [ $((SECONDS - start)) -ge "$STARTUP_TIMEOUT" ]; then
        hang_alert "job $SLURM_JOB_ID ($prefix) startup hang after $((SECONDS - start)) s (retry $retry of $STARTUP_RETRIES)"
        stop_srun "$pid"
        for f in "$prefix"-[0-9].log; do mv -f "$f" "${f%.log}.hung$retry.log"; done
        retry=$((retry + 1))
        [ "$retry" -gt "$STARTUP_RETRIES" ] && return 124
        continue 2
      fi
    done
    wait "$pid"
    return $?
  done
}
