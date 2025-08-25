# Manual Ray Worker Script Comparison

This document compares the two manual Ray worker scripts and documents all helper functions from the levanter TPU infrastructure.

## Script Comparison

### Files Analyzed
- **Current Version**: `/Users/ahmed/code/stash-view/infra/manual_ray_worker_launch.py`
- **Original Version**: `/Users/ahmed/code/stash-view/original_manual_ray_worker.py`

### Key Differences

#### 1. Docker Container Management Strategy

**Current Version (manual_ray_worker_launch.py)**:
- Uses a **sleep container** approach with direct command execution
- Container runs `sleep 3600` to keep it alive
- Commands executed via `docker exec` after container is running
- Separates gcsfuse mounting from container initialization
- More granular control over setup process

**Original Version (original_manual_ray_worker.py)**:
- Uses **custom entrypoint script** approach
- Creates temporary shell script (`/tmp/entry.sh`) with all setup commands
- Container runs the entrypoint script directly on startup
- All setup happens inside container during initialization
- Ray worker runs in a retry loop (`while true; do ... done`)

#### 2. Setup Command Handling

**Current Version**:
```python
# Run setup commands inside container (excluding gcsfuse)
for command in setup_commands:
    if command.startswith("gcsfuse "):
        continue  # Skip gcsfuse - handled separately
    setup_cmd = f"docker exec {container_name} bash -c '{command}'"
    tpu_ssh(tpu_name, zone, 1, setup_cmd)
```

**Original Version**:
```python
# Write all setup commands to entrypoint script
with tempfile.NamedTemporaryFile("w", prefix="entry", suffix=".sh") as f:
    f.write("#!/bin/bash\n")
    for command in setup_commands:
        f.write(command + "\n")
```

#### 3. gcsfuse Mounting

**Current Version**:
- Handles gcsfuse mounting separately with explicit error handling
- Uses dedicated `docker exec` command for mounting
- Provides verification commands for troubleshooting

**Original Version**:
- gcsfuse mounting handled as part of general setup commands
- No special treatment or error handling for gcsfuse

#### 4. Ray Worker Startup

**Current Version**:
```python
# Start Ray worker directly via docker exec
ray_start_command = f"docker exec -d {container_name} {entry_command}"
tpu_ssh(tpu_name, zone, 1, ray_start_command)
```

**Original Version**:
```python
# Ray worker runs in retry loop within entrypoint script
f.write("while true; do\n")
f.write(entry_command + "\n")
f.write("sleep 10\n")
f.write("done\n")
```

#### 5. Imports

**Current Version**:
- Does NOT import `tempfile`
- Does NOT import `run_command` from levanter.infra.tpus

**Original Version**:
- Imports `tempfile` for creating entrypoint script
- Imports `run_command` from levanter.infra.tpus

#### 6. Error Handling and Logging

**Current Version**:
- More detailed logging with verification commands
- Explicit handling of gcsfuse setup
- Better separation of concerns

**Original Version**:
- Simpler approach with fewer logs
- Built-in retry mechanism for Ray worker
- All-or-nothing setup approach

## Helper Functions from levanter/src/levanter/infra/tpus.py

### Function: `setup_vm_docker(tpu_name, zone, node_count)`
**Purpose**: Initialize Docker permissions and volumes on TPU VM

**Detailed Steps**:
1. **Add user to docker group**: Runs `sudo usermod -aG docker {current_user}` to give current user Docker permissions
2. **Create levanter volume**: Creates a Docker volume named "levanter" using local driver for persistent storage
3. **Remove existing container**: Forcibly removes any existing "levanter" container with `sudo docker rm -f levanter`
4. **Execute via SSH**: All commands executed on TPU VM via `tpu_ssh()` function

### Function: `list_tpus(zone)`
**Purpose**: Get list of all TPU queued resources in a zone

**Detailed Steps**:
1. **Execute gcloud command**: Runs `gcloud alpha compute tpus queued-resources list`
2. **Filter by zone**: Uses `--zone={zone}` parameter
3. **Format output**: Returns JSON with name and state using `--format=json(name.basename(), state)`
4. **Parse and return**: JSON output parsed and returned as Python object

### Function: `describe_tpu_queued_resource(tpu_name, zone)`
**Purpose**: Get detailed information about a specific TPU queued resource

**Detailed Steps**:
1. **Try gcloud describe**: Attempts `gcloud alpha compute tpus queued-resources describe {tpu_name}`
2. **Suppress errors**: Uses `stderr=subprocess.DEVNULL` to hide error output
3. **Parse JSON response**: Returns parsed JSON with name and state information
4. **Handle failures**: Returns `None` if TPU doesn't exist or command fails
5. **Error resilience**: Catches `subprocess.CalledProcessError` and returns gracefully

### Function: `describe_tpu_vm(tpu_name, zone)`
**Purpose**: Get detailed information about a specific TPU VM (different from queued resource)

**Detailed Steps**:
1. **Execute gcloud describe**: Runs `gcloud alpha compute tpus tpu-vm describe {tpu_name}`
2. **Zone specification**: Uses `--zone={zone}` parameter for location
3. **JSON formatting**: Returns structured data with `--format=json(name.basename(), state)`
4. **Error handling**: Returns `None` if VM doesn't exist, catches `subprocess.CalledProcessError`
5. **Silent operation**: Uses `--quiet` flag and suppresses stderr

### Function: `start_tpu_vm_queued_resources(tpu_name, *, tpu_type, capacity_type, version, zone, node_count)`
**Purpose**: Create and wait for TPU queued resource to become active

**Detailed Steps**:
1. **Enable alpha components**: Runs `gcloud components install alpha --quiet` to ensure alpha API access
2. **Set default version**: Uses "tpu-ubuntu2204-base" if no version specified
3. **Check existing TPU**: Calls `describe_tpu_queued_resource()` to see if TPU already exists
4. **Handle existing states**:
   - If FAILED/SUSPENDED: Deletes existing TPU with `gcloud alpha compute tpus queued-resources delete`
   - If other states: Logs current state and returns early
5. **Build creation command**: Constructs gcloud command with accelerator type and zone
6. **Handle capacity types**:
   - best-effort/preemptible: Adds `--best-effort` and `--provisioning-model spot`
   - reserved: Adds `--reserved` flag
   - spot: Adds `--spot` and provisioning model
   - on-demand: No additional flags
7. **Configure node setup**:
   - Single node: Uses `--node-id={tpu_name}`
   - Multiple nodes: Uses `--node-count={node_count}`
8. **Execute creation**: Calls `run_command()` with constructed command
9. **Wait for completion**: Polls every 60 seconds checking TPU status
10. **Handle completion states**:
    - ACTIVE: Success, breaks loop
    - FAILED: Raises RuntimeError with error message
    - Other: Continues waiting, logs status and time waited

### Function: `launch_job(command, tpu_name, tpu_type, capacity_type, zone, node_count, full_image_id, env, foreground, version=None)`
**Purpose**: Complete workflow to launch a job on TPU with Docker

**Detailed Steps**:
1. **Create TPU**: Calls `start_tpu_vm_queued_resources()` with all TPU parameters
2. **Setup Docker environment**: Calls `setup_vm_docker()` to configure Docker permissions and volumes
3. **Build Docker command**: Uses `make_docker_run_command()` to construct proper Docker run command with:
   - Image ID specification
   - Command to execute
   - Environment variables
   - Foreground/background mode
4. **Execute on TPU**: Uses `tpu_ssh()` to run Docker command on the TPU VM
5. **Handle multi-node**: Automatically handles single vs multi-node TPU configurations

### Function: `run_command(*args, **kwargs)`
**Purpose**: Execute shell commands with logging

**Detailed Steps**:
1. **Log command**: Prints "Running: {command}" for visibility
2. **Execute command**: Uses `subprocess.check_call()` to run command
3. **Handle arguments**: Accepts arbitrary args and kwargs to pass through
4. **Error propagation**: Raises exceptions from subprocess if command fails

### Function: `add_ssh_key(ssh_key_filename)`
**Purpose**: Add SSH key to ssh-agent if not already present

**Detailed Steps**:
1. **Get key fingerprint**: Runs `ssh-keygen -lf {filename}` to get key hash
2. **Parse fingerprint**: Extracts SHA256 hash from output (format: "3072 SHA256:... key-name (RSA)")
3. **List existing keys**: Runs `ssh-add -l` to get currently loaded keys
4. **Check for duplicates**: Compares key hash against existing keys
5. **Add if needed**: If key not found, runs `ssh-add {filename}` to load it
6. **Handle errors**: Catches and re-raises `subprocess.CalledProcessError`

### Function: `tpu_ssh(tpu_name, zone, node_count, *args, ignore_failure=False)`
**Purpose**: Execute commands on TPU VM via SSH with automatic key management

**Detailed Steps**:
1. **Add SSH key**: Calls `add_ssh_key()` with Google Compute Engine key (`~/.ssh/google_compute_engine`)
2. **Handle key errors**: Catches SSH key addition failures and warns but continues
3. **Choose execution mode**:
   - Single node: Uses direct `gcloud alpha compute tpus tpu-vm ssh`
   - Multi-node: Delegates to `_tpu_ssh_multislice()`
4. **Build SSH command**: Constructs gcloud command with:
   - TPU name and zone
   - `--worker=all` to execute on all workers
   - `--quiet` for minimal output
   - `--command` with joined arguments
5. **Execute command**: Uses `run_command()` to execute the SSH command
6. **Handle failures**: If `ignore_failure=True`, logs errors but continues; otherwise re-raises

### Function: `_tpu_ssh_multislice(tpu_name, zone, node_count, *args, ignore_failure=False)`
**Purpose**: Execute commands on multi-node TPU slices in parallel

**Detailed Steps**:
1. **Create thread pool**: Uses `concurrent.futures.ProcessPoolExecutor()` for parallel execution
2. **Submit jobs**: Creates futures for each node (0 to node_count-1):
   - Each node accessed as `{tpu_name}-{i}`
   - Same SSH command structure as single node
3. **Execute in parallel**: All SSH commands run simultaneously across nodes
4. **Collect results**: Uses `concurrent.futures.as_completed()` to wait for all
5. **Handle individual failures**: 
   - If `ignore_failure=True`: Logs failures and continues
   - If `ignore_failure=False`: Re-raises first encountered exception
6. **Wait for completion**: Blocks until all nodes complete their commands

### Function: `get_current_tpu_metadata(key: str) -> Optional[str]`
**Purpose**: Retrieve TPU metadata from Google Cloud metadata service (only works ON a TPU VM)

**Detailed Steps**:
1. **Build metadata URL**: Constructs URL using `GCE_TPU_ACCELERATOR_ENDPOINT + key`
2. **Set headers**: Uses `GCE_TPU_HEADERS = {"Metadata-Flavor": "Google"}` required by metadata service
3. **Make HTTP request**: Uses `requests.get()` to query metadata endpoint
4. **Check response**:
   - Status 200 + non-empty text: Returns the metadata value
   - Other responses: Logs debug message about failure
5. **Handle network errors**: Catches `requests.RequestException` and logs debug message
6. **Return value**: Returns metadata string or `None` if unavailable

### Function: `get_current_tpu_is_preempted() -> bool`
**Purpose**: Check if current TPU VM has been preempted (only works ON a TPU VM)

**Detailed Steps**:
1. **Query preemption endpoint**: Makes HTTP GET to `http://metadata.google.internal/computeMetadata/v1/instance/preempted`
2. **Set required headers**: Uses `{"Metadata-Flavor": "Google"}` header
3. **Parse response**:
   - Status 200: Returns `True` if response text is "TRUE", `False` otherwise
   - Other status: Logs warning and returns `False`
4. **Handle network errors**: Catches `requests.RequestException`, logs debug message, and re-raises
5. **Boolean conversion**: Converts string response to boolean value

## Head Node Temporary Unavailability Scenarios

### Scenario: Head Node Becomes Temporarily Unavailable

Let's trace what happens when the Ray head node goes down for 5 minutes and then comes back online:

#### Current Version Behavior (manual_ray_worker_launch.py)

**Initial State:**
1. Script launches TPU and Docker container with `sleep 3600`
2. Ray worker started with: `docker exec -d ray ray start --address=HEAD_IP:6379 --block`
3. Script exits, Ray worker process running inside container

**When Head Node Goes Down:**
1. **Ray worker detects connection loss**: The `--block` flag means the ray process will exit when it loses connection to head node
2. **Ray process exits**: Inside the container, the Ray worker process terminates
3. **Container keeps running**: The container continues running `sleep 3600`, but Ray worker is dead
4. **No automatic recovery**: Nothing restarts the Ray worker process

**When Head Node Comes Back:**
1. **Worker stays dead**: Ray worker process already exited and won't restart itself
2. **Container still alive**: Container running `sleep 3600` but no Ray worker
3. **Manual intervention required**: You must manually SSH into TPU and restart Ray worker
4. **Result**: **WORKER REMAINS OFFLINE UNTIL MANUAL RESTART**

#### Original Version Behavior (original_manual_ray_worker.py)

**Initial State:**
1. Script creates entrypoint script `/tmp/entry.sh` with retry loop:
   ```bash
   #!/bin/bash
   # setup commands...
   while true; do
       ray start --address=HEAD_IP:6379 --block
       sleep 10
   done
   ```
2. Container runs this entrypoint script directly
3. Script exits, container running the retry loop

**When Head Node Goes Down:**
1. **Ray worker detects connection loss**: The `--block` flag causes ray process to exit
2. **Ray process exits**: Returns to the shell script (retry loop)
3. **Retry loop activates**: Script immediately goes to `sleep 10` then retries
4. **Continuous retry attempts**: Every 10 seconds, attempts `ray start --address=HEAD_IP:6379 --block`

**When Head Node Comes Back:**
1. **Next retry attempt**: Within 10 seconds of head node recovery, retry loop attempts connection
2. **Successful reconnection**: `ray start` succeeds and worker rejoins cluster
3. **Automatic recovery**: No manual intervention required
4. **Result**: **WORKER AUTOMATICALLY RECONNECTS WITHIN 10 SECONDS**

### Detailed Timeline Comparison

| Time | Event | Current Version | Original Version |
|------|-------|----------------|------------------|
| T+0 | Head node goes down | Ray worker exits | Ray worker exits |
| T+1 | | Container runs `sleep 3600` | Script waits 10s, retries connection |
| T+2 | | Container runs `sleep 3600` | Script retries connection (fails) |
| T+3 | | Container runs `sleep 3600` | Script retries connection (fails) |
| T+4 | | Container runs `sleep 3600` | Script retries connection (fails) |
| T+5 | Head node comes back | Container runs `sleep 3600` | Script retries connection (fails) |
| T+6 | | **Worker offline** | **Script retries connection (succeeds!)** |
| T+7 | | **Worker offline** | Worker online and processing tasks |

### Why This Matters for Distributed Training

**Training Job Impact:**
- **Current Version**: Training job **permanently stalled** until manual intervention
- **Original Version**: Training job **resumes automatically** within 10 seconds

**Operational Overhead:**
- **Current Version**: Requires monitoring and manual worker restarts
- **Original Version**: Self-healing, no intervention needed

**Common Causes of Head Node Unavailability:**
1. **Autoscaler operations**: Ray autoscaler may restart head node during scaling
2. **Memory pressure**: Head node OOM and restart
3. **Network partitions**: Temporary connectivity issues in cloud
4. **Maintenance**: Planned or unplanned infrastructure updates
5. **Spot instance preemption**: If head node is on spot/preemptible instance

### Container Architecture Comparison

**Current Version Container:**
```
Container Process Tree:
└── sleep 3600 (PID 1, keeps container alive)
    
Background Process:
└── ray start --address=HEAD:6379 --block (separate process, exits on disconnect)
```

**Original Version Container:**
```
Container Process Tree:
└── /tmp/entry.sh (PID 1, entrypoint script)
    └── while true; do
        ├── ray start --address=HEAD:6379 --block (exits on disconnect)
        ├── sleep 10
        └── (loop continues)
```

The key difference: In the original version, **the retry loop IS the main container process**, so when Ray exits, control returns to the loop. In the current version, Ray runs as a **detached background process** with no supervision.

## The gcsfuse Problem: Why We Changed Approaches

Based on `/Users/ahmed/code/marin/manual_worker_gcs.md`, the **original approach was abandoned because it couldn't reliably mount gcsfuse**, not because the retry logic was bad. Here's what happened:

### Original Approach gcsfuse Failure
```bash
# Original entrypoint script approach
#!/bin/bash
gcsfuse --implicit-dirs --only-dir gcsfuse_mount $BUCKET /opt/gcsfuse_mount || true
# ... other setup commands ...
while true; do
    ray start --address=HEAD_IP:6379 --block
    sleep 10
done
```

**Problem**: gcsfuse commands failed inside the entrypoint script due to:
1. **Docker `/tmp` mount permission restrictions**
2. **FUSE device access timing issues** during container startup
3. **Script execution permissions** - `/tmp/entry.sh` not accessible inside container

### Current Approach Success
```python
# Current: Container runs sleep, then direct docker exec
docker_command = [..., "sleep", "3600"]
tpu_ssh(tpu_name, zone, 1, *docker_command)

# gcsfuse via direct docker exec (THIS WORKS!)
gcsfuse_command = f"docker exec {container_name} gcsfuse --implicit-dirs --only-dir gcsfuse_mount {bucket_name} /opt/gcsfuse_mount"
tpu_ssh(tpu_name, zone, 1, gcsfuse_command)

# Ray start via direct docker exec (NO RETRY!)
ray_start_command = f"docker exec -d {container_name} {entry_command}"
tpu_ssh(tpu_name, zone, 1, ray_start_command)
```

**Success**: Direct `docker exec` commands work perfectly, but **no retry logic**.

## The Solution: Hybrid Approach

We can have **both reliable gcsfuse AND retry logic** by combining the best of both approaches:

### Option 1: External Retry Loop (Recommended)

Keep the current reliable gcsfuse approach, but add external process supervision:

```python
# After successful gcsfuse mount
logger.info("Starting Ray worker with retry logic...")

# Create a retry script that runs OUTSIDE the container
retry_script_content = f"""#!/bin/bash
while true; do
    echo "Starting Ray worker..."
    docker exec {container_name} ray start --address={head}:6379 --block
    echo "Ray worker exited, retrying in 10 seconds..."
    sleep 10
done
"""

# Copy retry script to TPU host
with tempfile.NamedTemporaryFile("w", prefix="retry", suffix=".sh", delete=False) as f:
    f.write(retry_script_content)
    f.flush()
    run_command(*f"gcloud compute tpus tpu-vm scp {f.name} {tpu_name}:/tmp/ray_retry.sh --zone={zone} --worker=all".split())

# Make executable and run in background on HOST
tpu_ssh(tpu_name, zone, 1, "chmod +x /tmp/ray_retry.sh")
tpu_ssh(tpu_name, zone, 1, "nohup /tmp/ray_retry.sh > /tmp/ray_worker.log 2>&1 &")
```

**Advantages**:
- ✅ **Reliable gcsfuse**: Uses proven `docker exec` approach
- ✅ **Retry logic**: Ray worker automatically restarts on head node reconnection
- ✅ **Host-level supervision**: Retry script runs on TPU host, not inside container
- ✅ **No container permission issues**: Script runs outside Docker `/tmp` restrictions

### Option 2: Container-Level Process Supervisor

Use a process supervisor inside the container:

```python
# After gcsfuse mount, start a supervisor inside container
supervisor_command = f"""docker exec -d {container_name} bash -c '
while true; do
    echo "Starting Ray worker..."
    ray start --address={head}:6379 --block
    echo "Ray worker exited, retrying in 10 seconds..."
    sleep 10
done'"""

tpu_ssh(tpu_name, zone, 1, supervisor_command)
```

**Advantages**:
- ✅ **Reliable gcsfuse**: Uses proven `docker exec` approach  
- ✅ **Retry logic**: Built into container execution
- ✅ **Simpler implementation**: No external script files

### Option 3: Fix Original Approach gcsfuse

Go back to entrypoint script but fix gcsfuse mounting by using the working approach:

```python
# Create entrypoint that does gcsfuse CORRECTLY
entrypoint_content = f"""#!/bin/bash
# Use the approach we know works
echo "Mounting gcsfuse..."
gcsfuse --implicit-dirs --only-dir gcsfuse_mount {bucket_name} /opt/gcsfuse_mount

# Then retry loop for Ray
while true; do
    echo "Starting Ray worker..."
    ray start --address={head}:6379 --block
    echo "Ray worker exited, retrying in 10 seconds..."
    sleep 10
done
"""

# But run this entrypoint in a way that avoids the permission issues
# (This requires more research into why gcsfuse works via docker exec but not entrypoint)
```

## Timeline Comparison with Hybrid Solution

| Time | Event | Current Version | Hybrid Option 1 |
|------|-------|----------------|------------------|
| T+0 | Head node goes down | Ray worker exits | Ray worker exits |
| T+1 | | Container runs `sleep 3600` | Host retry script detects exit, waits 10s |
| T+2 | | Container runs `sleep 3600` | Host script: `docker exec ray start` (fails) |
| T+3 | | Container runs `sleep 3600` | Host script: `docker exec ray start` (fails) |
| T+4 | | Container runs `sleep 3600` | Host script: `docker exec ray start` (fails) |
| T+5 | Head node comes back | Container runs `sleep 3600` | Host script: `docker exec ray start` (fails) |
| T+6 | | **Worker offline** | **Host script: `docker exec ray start` (succeeds!)** |
| T+7 | | **Worker offline** | Worker online and processing tasks |

## Recommended Implementation

**Use Option 1 (External Retry Loop)** because:

1. **Leverages proven gcsfuse approach**: Keeps the reliable `docker exec gcsfuse` that we know works
2. **Minimal risk**: Doesn't change the working gcsfuse setup
3. **Host-level supervision**: Avoids all Docker container permission issues
4. **Easy to debug**: Retry logic runs on host where you can easily access logs
5. **Preserves container simplicity**: Container just needs to stay alive with `sleep`

The key insight: **We don't need retry logic INSIDE the container** - we just need something that can restart the Ray worker when it dies. Running the retry loop on the TPU host with `docker exec` commands gives us the best of both worlds.

## Summary

The **original version's retry mechanism provides crucial production reliability**, and the **current version's gcsfuse approach actually works**. By combining them with external process supervision, we get both reliable gcsfuse mounting AND automatic Ray worker recovery without the permission issues that plagued the entrypoint script approach.