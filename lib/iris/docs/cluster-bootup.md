# Cluster Bootup Plan

## Goal

Establish a working workflow for booting an Iris cluster on GCP (eu-west4) using the CLI-driven controller VM approach. This plan delivers:

- **What**: A tested end-to-end cluster bootup sequence with probing/monitoring tools
- **Why**: Validate that the autoscaler-based architecture works in production
- **Scope**: eu-west4 cluster, GCP TPU provider, controller VM lifecycle, worker autoscaling

---

## Non-Goals

- **Multi-region support**: Focus on eu-west4 only for initial validation
- **Production hardening**: This is a testing/validation exercise, not production deployment
- **Performance optimization**: Focus on correctness first

---

## Design Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Developer Machine                                                │
│  ├─ iris cluster --config=eu-west4.yaml start                   │
│  ├─ scripts/probe-controller.py (SSH tunnel + health probes)   │
│  └─ iris controller-rpc (via SSH tunnel)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ GCP eu-west4                                                     │
│  ├─ Controller VM (n2-standard-4)                               │
│  │   └─ Docker: iris-controller container                       │
│  │       ├─ Controller service (port 10000)                     │
│  │       ├─ Integrated Autoscaler                               │
│  │       └─ Scheduler                                           │
│  │                                                               │
│  └─ TPU Slices (created by autoscaler on demand)               │
│      ├─ v5litepod-16 slice → 4 worker VMs                      │
│      └─ Each VM runs iris-worker container                      │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. `iris cluster start` creates controller GCE VM with bootstrap script
2. Bootstrap installs Docker, pulls image, starts controller container
3. Controller runs autoscaler which watches for pending task demand
4. When jobs submitted, autoscaler creates TPU slices
5. Workers discover controller via GCP metadata, register via heartbeat
6. Controller assigns tasks to workers

### Key Design Decision: Config Propagation

**Current Gap**: The controller container starts without cluster config - it doesn't know about scale groups!

The bootstrap script runs:
```bash
sudo docker run ... {docker_image}
# Container CMD: python -m iris.cluster.controller.main serve --host 0.0.0.0 --port 10000
```

No `--config` is passed, so the autoscaler has no scale groups configured.

**Solution**: Copy config file to VM and mount into container:
```bash
# Bootstrap script receives config as heredoc, writes to /etc/iris/config.yaml
cat > /etc/iris/config.yaml << 'IRIS_CONFIG_EOF'
{config_yaml}
IRIS_CONFIG_EOF

# Mount config into container
sudo docker run -d --name {container_name} \
    --network=host \
    -v /etc/iris/config.yaml:/etc/iris/config.yaml:ro \
    -v /var/cache/iris:/var/cache/iris \
    {docker_image} \
    python -m iris.cluster.controller.main serve \
        --host 0.0.0.0 --port {port} --config /etc/iris/config.yaml
```

This approach:
- Keeps config as readable YAML file on VM (easier debugging)
- Avoids shell escaping issues with JSON
- File is mounted read-only into container

### Key Design Decision: SSH Tunnel Context Helper

Probe scripts need to access the controller API, which requires SSH port forwarding. A context manager helper handles this:

```python
@contextmanager
def controller_tunnel(zone: str, project: str, local_port: int = 10000) -> Iterator[str]:
    """Establish SSH tunnel to controller and yield the local URL.

    Usage:
        with controller_tunnel("europe-west4-b", "hai-gcp-models") as url:
            response = httpx.get(f"{url}/health")
    """
    vm_name = discover_controller_vm(zone, project)
    # Start gcloud compute ssh with -L port forwarding in background
    proc = subprocess.Popen([
        "gcloud", "compute", "ssh", vm_name,
        f"--project={project}", f"--zone={zone}",
        "--", "-L", f"{local_port}:localhost:10000", "-N"
    ])
    try:
        # Wait for tunnel to be ready
        wait_for_port(local_port)
        yield f"http://localhost:{local_port}"
    finally:
        proc.terminate()
        proc.wait()
```

This allows probe commands to transparently handle SSH tunneling.

---

## Files Modified or Created

### Directory Tree

```
lib/iris/
  scripts/
    probe-controller.py <new>       # Probe/monitor controller (with SSH tunnel helper)
    cleanup-cluster.py <new>        # Clean existing iris state
    validate-cluster.py <new>       # Run validation jobs
  examples/
    eu-west4.yaml <new>             # Production config for eu-west4
  src/iris/cluster/
    vm/controller.py <modified>     # Mount config file into container
    vm/config.py <modified>         # Add to_dict() for serialization
  docs/
    cluster-bootup.md <new>         # This plan (already created)
```

---

## Implementation Status

### Summary

| Component | Status | Notes |
|-----------|--------|-------|
| `scripts/probe-controller.py` | ✅ Done | Multiple probe commands implemented |
| `scripts/cleanup-cluster.py` | ✅ Done | Dry-run + delete modes |
| `scripts/validate-cluster.py` | ✅ Done | 3 TPU validation tests with --tpu-type option |
| `examples/eu-west4.yaml` | ✅ Done | Production config ready |
| `vm/controller.py` config mount | ✅ Done | Heredoc + volume mount |
| `vm/config.py` to_dict() | ✅ Done | Serialization for YAML |
| `Dockerfile.controller` gcloud | ✅ Done | Added Google Cloud SDK |
| Docker image rebuilt/pushed | ✅ Done | Image deployed to cluster |
| End-to-end validation | ⏳ In Progress | TPU provisioning required |

### Completed Work

**Scripts (all functional):**
- `probe-controller.py`: discover, ssh-status, tunnel, health, autoscaler-status, list-workers, logs, bootstrap-logs, list-jobs
- `cleanup-cluster.py`: lists VMs/TPUs, dry-run by default, `--no-dry-run` to delete
- `validate-cluster.py`: 3 TPU test cases with --tpu-type option (simple TPU job, compute with args, 2-job scheduler test)

**Core changes:**
- `vm/controller.py`: Config written via heredoc to `/etc/iris/config.yaml`, mounted read-only into container, passed via `--config` flag
- `vm/config.py`: Added `to_dict()` method for YAML serialization

### Bug Found & Fixed

**Issue**: Controller container missing `gcloud` CLI (see `docs/autoscale-debug-report.md`)

The autoscaler calls `gcloud compute tpus tpu-vm list` but the Docker image didn't include Google Cloud SDK. Container crashed in restart loop.

**Fix applied**: Updated `Dockerfile.controller` to install `google-cloud-cli`:
```dockerfile
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*
```

---

## Next Steps

1. **Rebuild and push controller image**:
   ```bash
   docker build -f Dockerfile.controller -t europe-west4-docker.pkg.dev/hai-gcp-models/marin/iris-controller:latest .
   docker push europe-west4-docker.pkg.dev/hai-gcp-models/marin/iris-controller:latest
   ```

2. **Restart controller** (pulls new image):
   ```bash
   uv run scripts/cleanup-cluster.py --zone europe-west4-b --project hai-gcp-models --no-dry-run
   uv run python -m iris.cli cluster --config examples/eu-west4.yaml start
   ```

3. **Validate end-to-end**:
   ```bash
   uv run scripts/probe-controller.py --zone europe-west4-b --project hai-gcp-models health
   uv run scripts/validate-cluster.py --zone europe-west4-b --project hai-gcp-models
   ```

---

## Debugging Process

You MUST write individual operation scripts to scripts/<script-name>.py , no matter how simple the command.

e.g. if you want to query gcloud you would write scripts/query-gcloud.py with a click interface and use uv run to run it.

## Known Issues

1. **Code duplication**: `controller_tunnel()` helper is duplicated in probe-controller.py and validate-cluster.py (~50 lines each). Consider extracting to shared module.

2. **Heredoc edge case**: If config YAML contains literal string `IRIS_CONFIG_EOF`, bootstrap would break. Unlikely in practice.

---

## Original Plan Review

**Potential issues (from planning phase):**

1. ~~**Config YAML heredoc**~~: Implemented with quoted heredoc (`<< 'IRIS_CONFIG_EOF'`)
2. **TPU quota**: May hit quota limits in eu-west4 - need to verify quota before testing
3. ~~**Image availability**~~: Fixed by adding gcloud to Dockerfile
4. **SSH tunnel reliability**: Tunnel may drop during long-running probe commands - add reconnection logic

**Suggestions for improvements:**

1. Add retry logic to probe commands for transient failures
2. Consider adding `--wait` flag to cleanup to wait for deletion
3. ✅ Add structured output (JSON) option to probe commands for scripting (done in autoscaler-status)
4. Add `--timeout` option to tunnel context for long-running operations
