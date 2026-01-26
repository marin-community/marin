# Demo Cluster Boot Validation

This document provides step-by-step instructions to validate that the Iris cluster
demo works correctly in `--cluster` mode with real GCP infrastructure.

## Prerequisites

- GCP project: `hai-gcp-models`
- Zone: `europe-west4-b`
- Config file: `examples/eu-west4.yaml`
- Docker images already pushed to Artifact Registry

## Step 1: Clean Up Existing Resources

Before starting a fresh test, clean up any existing controller VMs and TPU slices.

```bash
cd lib/iris

# First, check what resources exist (dry-run)
uv run python scripts/cluster-tools.py \
    --zone europe-west4-b \
    --project hai-gcp-models \
    cleanup

# If resources exist, delete them
uv run python scripts/cluster-tools.py \
    --zone europe-west4-b \
    --project hai-gcp-models \
    cleanup --no-dry-run
```

Verify cleanup completed:

```bash
uv run python scripts/cluster-tools.py \
    --zone europe-west4-b \
    --project hai-gcp-models \
    discover
```

Expected output: `No controller VM found.`

## Step 2: Launch Cluster via demo_cluster.py --cluster

Start the demo cluster in cluster mode. This will:
1. Create a controller VM in GCP
2. Wait for the controller to become healthy
3. Establish an SSH tunnel for local access
4. Submit seed jobs to validate the cluster

```bash
cd lib/iris

uv run python examples/demo_cluster.py \
    --cluster \
    --config examples/eu-west4.yaml \
    --verbose
```

### Expected Behavior

1. **Controller VM Creation** (1-2 minutes):
   - Script should print "Starting new controller VM..."
   - Or "Found existing controller VM: iris-controller-XXXXX" if one exists

2. **SSH Tunnel Establishment** (~30 seconds):
   - Script should print "Establishing SSH tunnel to ..."
   - Then "SSH tunnel established on port XXXXX"

3. **Seed Jobs** (5-10 minutes for TPU provisioning):
   - "Seeding cluster with demo jobs..."
   - Jobs will trigger TPU slice creation via autoscaler
   - Each job should complete with `JOB_STATE_SUCCEEDED`

4. **Final Output**:
   - "All seed jobs succeeded!"
   - "Cluster is running. Press Ctrl+C to disconnect."

### Troubleshooting

If the controller fails to start, check bootstrap logs:

```bash
uv run python scripts/cluster-tools.py \
    --zone europe-west4-b \
    --project hai-gcp-models \
    bootstrap-logs --tail 500
```

If jobs fail, check controller logs:

```bash
uv run python scripts/cluster-tools.py \
    --zone europe-west4-b \
    --project hai-gcp-models \
    logs --tail 200
```

Check autoscaler status:

```bash
uv run python scripts/cluster-tools.py \
    --zone europe-west4-b \
    --project hai-gcp-models \
    autoscaler-status
```

## Step 3: Validate Cluster is Healthy

With the demo_cluster.py still running (or in a separate terminal), run the
validation suite:

```bash
# In a separate terminal with the tunnel still active
uv run python scripts/cluster-tools.py \
    --zone europe-west4-b \
    --project hai-gcp-models \
    validate
```

All 3 validation tests should pass:
- Simple TPU job (v5litepod-16)
- Compute job with args (v5litepod-16)
- Scheduler test (2 concurrent TPU jobs)

## Step 4: Test Jupyter Notebook Connection

The `--cluster` mode doesn't launch Jupyter automatically. To test the notebook
connection, start Jupyter manually with the tunnel active.

### 4a: Establish SSH Tunnel (if not already running)

```bash
# Terminal 1: Keep the tunnel open
uv run python scripts/cluster-tools.py \
    --zone europe-west4-b \
    --project hai-gcp-models \
    tunnel
```

### 4b: Start Jupyter Locally

```bash
# Terminal 2: Start Jupyter with env vars pointing to the tunnel
cd lib/iris
export IRIS_CONTROLLER_ADDRESS="http://localhost:10000"
export IRIS_WORKSPACE="$(pwd)"

uv run jupyter notebook --notebook-dir=examples
```

### 4c: Run the TPU Demo Notebook

1. Open `examples/tpu-demo.ipynb` in the Jupyter interface
2. Run all cells sequentially
3. Verify each cell completes without errors

Expected validations:
- Cell 2: Connects to controller successfully
- Cell 4: Simple JAX computation runs on TPU
- Cell 6: Multi-device pmap computation works
- Cell 8: Coscheduled multi-host job completes

## Step 5: Validate Notebook Programmatically

For CI/automated testing, run the notebook programmatically:

```bash
cd lib/iris
export IRIS_CONTROLLER_ADDRESS="http://localhost:10000"
export IRIS_WORKSPACE="$(pwd)"

uv run python -c "
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

notebook_path = Path('examples/tpu-demo.ipynb')
with open(notebook_path) as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
print('All cells executed successfully!')
"
```

## Step 6: Cleanup After Testing

When finished testing, clean up all GCP resources:

```bash
# First Ctrl+C any running demo_cluster.py or tunnel processes

# Then clean up all iris resources
uv run python scripts/cluster-tools.py \
    --zone europe-west4-b \
    --project hai-gcp-models \
    cleanup --no-dry-run
```

## Success Criteria

The demo cluster test is successful if:

1. [ ] Cleanup completes without errors
2. [ ] `demo_cluster.py --cluster` starts controller VM
3. [ ] SSH tunnel establishes successfully
4. [ ] All seed jobs complete with `JOB_STATE_SUCCEEDED`
5. [ ] `cluster-tools.py validate` passes all 3 tests
6. [ ] Jupyter notebook can connect to the cluster
7. [ ] `tpu-demo.ipynb` runs all cells without errors
8. [ ] Final cleanup removes all resources

## Notes

- TPU provisioning can take 5-10 minutes per job. Be patient.
- If TPUs are unavailable in the zone, jobs will remain pending indefinitely.
- The `--verbose` flag provides detailed logging for debugging.
- Controller VM remains running after demo_cluster.py exits. Always cleanup.
