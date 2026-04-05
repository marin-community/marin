# Cluster Troubleshooting

## Ray Worker Port Collision (manual TPU workers)

### Symptom

Job logs show the autoscaler repeatedly printing:

```
No available node types can fulfill resource requests {'ray-worker-manual-<id>': 1.0}*N
```

The dashboard shows fewer workers than expected (e.g. `59.0/59.0` when 64 are needed).
The TPU pod is `READY` and `HEALTHY` in GCP, but most hosts never join the Ray cluster.

### Root Cause

When `ray start` launches on a worker, several internal components (`runtime_env_agent`,
`dashboard_agent_grpc`, `dashboard_agent_http`, `metrics_export`) pick random ports. The
default worker port range is **10002-19999** (9998 ports). If any random port lands inside
that range, Ray raises:

```
ValueError: Ray component worker_ports is trying to use a port number <PORT> that is used
by other components.
```

The container has `--restart=on-failure`, so it retries, but each retry picks new random
ports — some succeed, some hit collisions again. After many failures Docker applies
exponential backoff, making retries extremely slow.

### Diagnosis

1. Check how many TPU hosts are actually in Ray vs. the pod size:

```bash
gcloud compute tpus tpu-vm describe <TPU_NAME> --zone=<ZONE> --format="yaml(acceleratorType,networkEndpoints.len(),state,health)"
```

2. SSH into a failing worker and check container logs:

```bash
gcloud alpha compute tpus tpu-vm ssh <TPU_NAME> --quiet --worker=<N> --zone=<ZONE> \
  --command='docker logs ray_docker --tail=30 2>&1'
```

Look for `ValueError: Ray component worker_ports is trying to use a port number ...`

3. Check restart count:

```bash
gcloud alpha compute tpus tpu-vm ssh <TPU_NAME> --quiet --worker=<N> --zone=<ZONE> \
  --command='docker inspect ray_docker --format="{{.State.Status}} restart={{.RestartCount}}"'
```

High restart counts (30+) confirm crash-looping with exponential backoff.

### Fix

Pin the problematic ports outside the worker range by modifying `/tmp/entry.sh` on
each worker, then recreate the containers:

```bash
gcloud alpha compute tpus tpu-vm ssh <TPU_NAME> --quiet --worker=all --zone=<ZONE> --command='
docker rm -f ray_docker 2>/dev/null || true

sed -i "s|ray start --address=\${HEAD_IP}:6379 --block|ray start --address=\${HEAD_IP}:6379 --dashboard-agent-grpc-port=9991 --dashboard-agent-listen-port=9992 --runtime-env-agent-port=9990 --metrics-export-port=9993 --block|" /tmp/entry.sh

docker run -d --net=host --name=ray_docker --init --privileged --restart=on-failure --rm=0 --privileged \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --shm-size=100gb \
  -v /tmp:/tmp \
  -e MARIN_PREFIX=gs://marin-us-central2 \
  -e BUCKET=marin-us-central2 \
  -e MARIN_LOCAL_CACHE_DIR=/tmp/marin-cache \
  -e AUTOSCALER_HEARTBEAT_TIMEOUT_S=600 \
  -e TPU_MIN_LOG_LEVEL=3 -e TPU_STDERR_LOG_LEVEL=3 -e TPU_LOG_DIR=disabled \
  -v "/var/run/docker.sock:/var/run/docker.sock" \
  -e RAY_AUTH_MODE=token -e RAY_AUTH_TOKEN_PATH=/home/ray/.ray/auth_token \
  <DOCKER_IMAGE> \
  /bin/bash /tmp/entry.sh
'
```

If some workers still fail after this (high restart count from previous crashes causes
Docker backoff), target them individually:

```bash
for w in <WORKER_IDS>; do
  gcloud alpha compute tpus tpu-vm ssh <TPU_NAME> --quiet --worker=$w --zone=<ZONE> --command='
  docker rm -f ray_docker 2>/dev/null || true
  docker run -d --net=host --name=ray_docker ...  # same as above
  '
done
```

Workers take 1-2 minutes to boot and join Ray. Verify with:

```bash
gcloud alpha compute tpus tpu-vm ssh <TPU_NAME> --quiet --worker=0 --zone=<ZONE> --command='
docker exec ray_docker python3 -c "
import ray
ray.init(address=\"auto\")
nodes = ray.nodes()
alive = [n for n in nodes if n[\"Alive\"]]
with_resource = [n for n in alive if \"<RESOURCE_NAME>\" in n.get(\"Resources\", {})]
print(f\"Workers with resource: {len(with_resource)}\")
"'
```

### Permanent Fix

The `ray start` command in `scripts/ray/cluster.py` (`_initialize_manual_worker`) should
pin these ports by default. Add to the entry script template:

```bash
ray start --address=${HEAD_IP}:6379 \
  --dashboard-agent-grpc-port=9991 \
  --dashboard-agent-listen-port=9992 \
  --runtime-env-agent-port=9990 \
  --metrics-export-port=9993 \
  --block
```

---

## GCS Runtime Env Package Download Failure

### Symptom

```
OSError: Failed to download runtime_env file package gcs://_ray_pkg_<hash>.zip from the
GCS to the Ray worker node.
```

### Root Cause

The working directory package in Ray's GCS was uploaded before worker containers were
restarted. The new containers can't find or download the cached package.

### Fix

Force a new package hash by touching any file in the repo, then resubmit:

```bash
touch .ray_pkg_bust
# resubmit the job (new hash triggers fresh upload)
uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster <CLUSTER> \
  -e WANDB_API_KEY "$WANDB_KEY" \
  -- python <EXPERIMENT_SCRIPT>
# clean up
rm .ray_pkg_bust
```

Alternatively, make any trivial whitespace change to a non-versioned file.

---

## TPU Hardware Fault (Core Halted / scheckne)

### Symptom

Job crashes with:

```
jax.errors.JaxRuntimeError: INTERNAL: Core halted unexpectedly: INTERNAL:
Accelerator device halted prematurely, perhaps due to an on-device check-failure.
Node 0 halted unexpectedly at tag:pc TensorCoreSequencer:1:0x175
(from TensorCoreSequencer:1:0x21f): scheckne:
```

Often surfaces during checkpoint save (in `_single_device_array_to_np_array`),
but the fault is hardware — not caused by the save itself.

### Why It Appears at Checkpoint Time

During training, JAX runs asynchronously — compute is dispatched to the TPU without
blocking. A chip fault during a training step may go unnoticed. Checkpoint save triggers
a synchronous device-to-host transfer (reading all parameters back to CPU), which is the
first point JAX blocks and discovers the TPU is dead.

Additionally, checkpoint changes the memory access pattern (burst reads of all parameters
vs. the streaming compute of training), which can stress a marginal chip differently.

### Diagnosis

If the crash is **reproducible** (happens on every checkpoint), it's almost certainly a
bad TPU chip, not a transient fault. A one-time crash may just be a cosmic ray event —
resubmit and see if it recurs.

### Fix: Delete and Recreate the TPU Pod

```bash
# 1. Stop the job
uv run scripts/ray/cluster.py --cluster <CLUSTER> stop-job <SUBMISSION_ID>

# 2. Delete the queued resource (also removes the VM)
gcloud compute tpus queued-resources delete <TPU_NAME> --zone=<ZONE> --force --quiet

# 3. Verify the VM is gone (should return NOT_FOUND)
gcloud compute tpus tpu-vm describe <TPU_NAME> --zone=<ZONE> 2>&1

# 4. Re-add the worker (this creates a new queued resource + TPU VM)
uv run scripts/ray/cluster.py --config infra/<CLUSTER>.yaml add-worker <TPU_TYPE> \
  --capacity reserved --name <TPU_NAME>

# 5. Wait for ACTIVE state (can take 5-10 minutes for large pods)
gcloud compute tpus queued-resources describe <TPU_NAME> --zone=<ZONE> --format="yaml(state)"

# 6. Initialize Ray on all workers
uv run scripts/ray/cluster.py --cluster <CLUSTER> init-worker <TPU_NAME>

# 7. Apply port collision fix (see above section), then verify all workers joined
```

After re-init, expect the port collision issue — follow the fix in the section above.
Verify all workers are registered before resubmitting.

### Notes

- The checkpoint will likely be corrupt. The job should resume from the previous good
  checkpoint automatically.
- `--force` on `queued-resources delete` also deletes the underlying TPU VM.
- Reserved capacity (`--capacity reserved`) should re-provision quickly. On-demand may
  take longer or fail if capacity is unavailable.

---

## Full Manual Worker Recovery Playbook

When a manual TPU worker is in a bad state (hardware fault, stale containers, missing
from Ray), this is the nuclear option:

```bash
TPU_NAME=ray-worker-manual-<NAME>
ZONE=us-central2-b
CLUSTER=marin-big-run

# 1. Delete and recreate
gcloud compute tpus queued-resources delete $TPU_NAME --zone=$ZONE --force --quiet
uv run scripts/ray/cluster.py --config infra/${CLUSTER}.yaml add-worker <TPU_TYPE> \
  --capacity reserved --name $TPU_NAME

# 2. Wait for ACTIVE
watch -n 10 gcloud compute tpus queued-resources describe $TPU_NAME --zone=$ZONE --format="yaml(state)"

# 3. Init workers into Ray
uv run scripts/ray/cluster.py --cluster $CLUSTER init-worker $TPU_NAME

# 4. Clean stale state and apply port fix on all workers
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --quiet --worker=all --zone=$ZONE --command='
docker rm -f ray_docker 2>/dev/null || true
sudo rm -rf /tmp/ray/session_*

sed -i "s|ray start --address=\${HEAD_IP}:6379 --block|ray start --address=\${HEAD_IP}:6379 --dashboard-agent-grpc-port=9991 --dashboard-agent-listen-port=9992 --runtime-env-agent-port=9990 --metrics-export-port=9993 --block|" /tmp/entry.sh

docker run -d --net=host --name=ray_docker --init --privileged --restart=on-failure --rm=0 --privileged \
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 --shm-size=100gb \
  -v /tmp:/tmp \
  -e MARIN_PREFIX=gs://marin-us-central2 \
  -e BUCKET=marin-us-central2 \
  -e MARIN_LOCAL_CACHE_DIR=/tmp/marin-cache \
  -e AUTOSCALER_HEARTBEAT_TIMEOUT_S=600 \
  -e TPU_MIN_LOG_LEVEL=3 -e TPU_STDERR_LOG_LEVEL=3 -e TPU_LOG_DIR=disabled \
  -v "/var/run/docker.sock:/var/run/docker.sock" \
  -e RAY_AUTH_MODE=token -e RAY_AUTH_TOKEN_PATH=/home/ray/.ray/auth_token \
  us-central2-docker.pkg.dev/hai-gcp-models/marin/marin_cluster:20260129 \
  /bin/bash /tmp/entry.sh
'

# 5. Wait ~2 minutes, then verify all workers joined
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --quiet --worker=0 --zone=$ZONE --command='
docker exec ray_docker python3 -c "
import ray
ray.init(address=\"auto\")
nodes = ray.nodes()
alive = [n for n in nodes if n[\"Alive\"]]
with_resource = [n for n in alive if \"'$TPU_NAME'\" in n.get(\"Resources\", {})]
print(f\"Workers: {len(with_resource)}\")
"'

# 6. Fix any stragglers individually (check for high restart counts)
# Then resubmit the job
```
