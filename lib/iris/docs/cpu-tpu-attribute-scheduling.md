# CPU-Agnostic Scheduling Design

## Problem Statement

The Iris scheduler currently requires exact device type matching between jobs and workers:

```python
# In WorkerCapacity.can_fit_job():
job_device_type = get_device_type(res.device)
if job_device_type != self.device_type:
    return False
```

This creates a scheduling problem for CPU-only workloads:

| Job Device | Worker Device | Current Behavior | Desired Behavior |
|------------|---------------|------------------|------------------|
| (none)     | CPU           | Scheduled        | Scheduled        |
| (none)     | TPU           | **Rejected**     | Scheduled        |
| (none)     | GPU           | **Rejected**     | Scheduled        |
| GPU        | TPU           | Rejected         | Rejected         |
| TPU        | GPU           | Rejected         | Rejected         |

A job with no device specified gets `device_type = "cpu"` (the default), but this prevents it from running on TPU or GPU hosts. In practice, TPU VMs and GPU VMs can run CPU workloads just fine - they have CPUs. The current behavior forces users to maintain separate CPU-only workers for CPU tasks, which wastes resources on accelerator hosts.

### Concrete Example

Consider a multi-host TPU training job that needs:
1. 4 TPU tasks (one per TPU VM host) - requires TPU device
2. 1 coordinator task (CPU-only) - runs the training loop orchestration

Currently, the coordinator task cannot be scheduled on any of the TPU VMs, even though they have available CPU capacity. Users must provision a separate CPU-only worker pool just for coordinator tasks.

## Design Principles

1. **CPU is universal**: Every host has a CPU. Jobs that only need CPU should be schedulable anywhere.

2. **Accelerators are specific**: GPU jobs need GPU hardware. TPU jobs need TPU hardware. These cannot run on "lesser" devices.

3. **Explicit device requests are honored**: If a job explicitly requests a device, respect that request precisely.

4. **Backwards compatible**: Existing jobs and worker configurations continue to work.

## Semantics: "No Device" vs "CPU Device"

The proto defines three device types:

```protobuf
message DeviceConfig {
  oneof device {
    CpuDevice cpu = 1;
    GpuDevice gpu = 2;
    TpuDevice tpu = 3;
  }
}
```

We need to distinguish between:

1. **No device specified** (`device` field not set): Job only needs CPU. Can run on any host.
2. **Explicit CPU device** (`device.cpu` set): Job explicitly requests CPU-only workers. Should only run on CPU-only workers.

This distinction matters for workloads that are sensitive to resource contention. For example, a benchmark job might explicitly request `CpuDevice` to avoid running on a GPU host where GPU driver overhead or other workloads could affect results.

However, for the initial implementation, we propose treating both cases identically: jobs without accelerator requirements can run anywhere. The explicit CPU case can be revisited if users demonstrate a need for it.

### Decision: Treat "no device" and "cpu device" as equivalent

For simplicity, we will treat both "no device specified" and "explicit CPU device" the same way: the job can run on any worker. This covers the vast majority of use cases and avoids adding complexity for an edge case with no known users.

If we later need to support "CPU-only workers" as a hard requirement, we can add an explicit constraint like `device-type=cpu` that users can add to their jobs.

## Device Compatibility Matrix

The scheduling rule becomes:

| Job Device Type | Worker Device Type | Compatible? |
|-----------------|-------------------|-------------|
| cpu (or none)   | cpu               | Yes         |
| cpu (or none)   | gpu               | Yes         |
| cpu (or none)   | tpu               | Yes         |
| gpu             | cpu               | No          |
| gpu             | gpu               | Yes (if variant matches) |
| gpu             | tpu               | No          |
| tpu             | cpu               | No          |
| tpu             | gpu               | No          |
| tpu             | tpu               | Yes (if variant matches) |

In pseudocode:

```python
def device_compatible(job_device_type: str, worker_device_type: str) -> bool:
    """Check if a job's device requirement is compatible with a worker's device."""
    if job_device_type == "cpu":
        # CPU jobs can run anywhere
        return True
    # Accelerator jobs require matching device type
    return job_device_type == worker_device_type
```

## Interaction with Coscheduling

Coscheduled TPU jobs specify:
1. A device requirement (TPU with specific topology)
2. A `group_by` attribute (e.g., `tpu-name`) for atomic placement
3. Constraints (e.g., `tpu-name=my-tpu`)

The device compatibility check happens before coscheduling logic:

```
1. Filter workers by constraints (including device attributes)
2. Filter workers by device compatibility  <-- This is where the change applies
3. Filter workers by resource capacity (CPU, memory)
4. Group by coscheduling attribute
5. Find group with enough workers
```

For TPU jobs, the device type is "tpu", so they will only match TPU workers (step 2 filters out CPU/GPU workers). The change only affects CPU jobs, which will now pass through step 2 for all worker types.

### No Impact on TPU Coscheduling

TPU jobs that use coscheduling will:
1. Have `device.tpu` set (device_type = "tpu")
2. Be rejected by CPU-only and GPU workers (device_compatible returns False)
3. Only match TPU workers
4. Continue to use `group_by` for atomic placement

The proposed change does not affect this flow because TPU jobs are not "cpu" device type.

## Implementation

### Option A: Modify `can_fit_job` (Recommended)

Change the device compatibility logic in `WorkerCapacity.can_fit_job()`:

```python
def can_fit_job(self, job: ControllerJob) -> bool:
    """Check if this capacity can fit the job's resource requirements."""
    res = job.request.resources

    if res.cpu > self.available_cpu:
        return False

    if res.memory_bytes > self.available_memory:
        return False

    job_device_type = get_device_type(res.device)

    # CPU jobs can run on any worker (CPU is universal)
    # Accelerator jobs require matching device type
    if job_device_type != "cpu" and job_device_type != self.device_type:
        return False

    job_variant = get_device_variant(res.device)
    if job_variant and job_variant != "auto" and job_variant != self.device_variant:
        return False

    if job_device_type == "gpu" and get_gpu_count(res.device) > self.available_gpus:
        return False

    return True
```

**Pros:**
- Minimal change (one condition modification)
- Centralizes device compatibility logic
- Easy to understand and test

**Cons:**
- Implicitly changes behavior for all CPU jobs

### Option B: Add Device Compatibility Function

Extract device compatibility into a separate function for clarity:

```python
def device_compatible(job_device_type: str, worker_device_type: str) -> bool:
    """Check if a job's device requirement is compatible with a worker's device.

    CPU jobs can run on any worker since every host has a CPU.
    Accelerator jobs (GPU, TPU) require the specific hardware.
    """
    if job_device_type == "cpu":
        return True
    return job_device_type == worker_device_type


def can_fit_job(self, job: ControllerJob) -> bool:
    """Check if this capacity can fit the job's resource requirements."""
    res = job.request.resources

    if res.cpu > self.available_cpu:
        return False

    if res.memory_bytes > self.available_memory:
        return False

    job_device_type = get_device_type(res.device)
    if not device_compatible(job_device_type, self.device_type):
        return False

    # ... rest unchanged
```

**Pros:**
- More explicit about the compatibility semantics
- Easier to extend if we need more complex rules later
- Better documentation via function name and docstring

**Cons:**
- Slightly more code

### Recommendation: Option B

Option B is preferred because it makes the design decision explicit and self-documenting. The `device_compatible` function name clearly communicates the intent.

## Migration and Backwards Compatibility

This change is **backwards compatible** for existing jobs:

1. **CPU jobs on CPU workers**: Continue to work (now pass with `device_compatible("cpu", "cpu") = True`)
2. **GPU jobs on GPU workers**: Continue to work (device types match)
3. **TPU jobs on TPU workers**: Continue to work (device types match)
4. **CPU jobs on accelerator workers**: Now work (previously rejected, now accepted)

The only behavioral change is that CPU jobs can now be scheduled on accelerator workers. This is the desired behavior and should not break any existing workloads.

### Potential Concern: Resource Contention

One concern is that CPU jobs might now compete with accelerator jobs for placement on accelerator workers. However:

1. Accelerator jobs have specific device requirements that will be respected
2. The scheduler uses first-fit, so accelerator jobs waiting in the queue will be scheduled when workers become available
3. If resource isolation is a concern, users can add explicit constraints (e.g., `device-type=cpu` on workers designated for CPU-only workloads)

## Testing

### Unit Tests to Add

```python
def test_cpu_job_schedules_on_tpu_worker(scheduler, state, job_request, worker_metadata):
    """CPU job (no device specified) can be scheduled on a TPU worker."""
    # Register a TPU worker
    meta = worker_metadata(tpu_name="v5litepod-16")
    register_worker(state, "w1", "addr1", meta)

    # Submit a CPU job (no device specified)
    req = job_request()  # No device in request
    tasks = submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    assert len(result.assignments) == 1
    assert result.assignments[0][0] == tasks[0]
    assert result.assignments[0][1].worker_id == WorkerId("w1")


def test_cpu_job_schedules_on_gpu_worker(scheduler, state, job_request, worker_metadata):
    """CPU job can be scheduled on a GPU worker."""
    meta = worker_metadata(gpu_count=8, gpu_name="H100")
    register_worker(state, "w1", "addr1", meta)

    req = job_request()
    tasks = submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    assert len(result.assignments) == 1
    assert result.assignments[0][1].worker_id == WorkerId("w1")


def test_gpu_job_does_not_schedule_on_cpu_worker(scheduler, state, worker_metadata):
    """GPU job cannot be scheduled on a CPU-only worker."""
    meta = worker_metadata()  # CPU-only
    register_worker(state, "w1", "addr1", meta)

    # GPU job
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(
                gpu=cluster_pb2.GpuDevice(variant="H100", count=1)
            ),
        ),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    assert len(result.assignments) == 0


def test_gpu_job_does_not_schedule_on_tpu_worker(scheduler, state, worker_metadata):
    """GPU job cannot be scheduled on a TPU worker."""
    meta = worker_metadata(tpu_name="v5litepod-16")
    register_worker(state, "w1", "addr1", meta)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="gpu-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(
                gpu=cluster_pb2.GpuDevice(variant="H100", count=1)
            ),
        ),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    assert len(result.assignments) == 0


def test_tpu_job_does_not_schedule_on_cpu_worker(scheduler, state, worker_metadata):
    """TPU job cannot be scheduled on a CPU-only worker."""
    meta = worker_metadata()  # CPU-only
    register_worker(state, "w1", "addr1", meta)

    req = cluster_pb2.Controller.LaunchJobRequest(
        name="tpu-job",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(
            cpu=1,
            memory_bytes=1024**3,
            device=cluster_pb2.DeviceConfig(
                tpu=cluster_pb2.TpuDevice(variant="v5litepod-16")
            ),
        ),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    assert len(result.assignments) == 0


def test_cpu_job_prefers_cpu_worker_when_available(scheduler, state, job_request, worker_metadata):
    """When both CPU and TPU workers available, CPU job can go to either.

    Note: We don't implement preference logic - this test just verifies
    the job can be scheduled. First-fit will pick the first available worker.
    """
    meta_cpu = worker_metadata()
    register_worker(state, "w1", "addr1", meta_cpu)

    meta_tpu = worker_metadata(tpu_name="v5litepod-16")
    register_worker(state, "w2", "addr2", meta_tpu)

    req = job_request()
    tasks = submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
    )

    # Job should be scheduled (to either worker)
    assert len(result.assignments) == 1
```

## Future Considerations

### Device Preference (Not Implemented)

In the future, we might want to add preference logic:
- Prefer CPU workers for CPU jobs (to leave accelerator capacity for accelerator jobs)
- Score workers based on "fit" (don't waste accelerator capacity on CPU jobs)

This would require changes to the scheduler's first-fit algorithm to consider worker scoring. This is out of scope for the initial implementation.

### Explicit CPU-Only Constraint (Not Implemented)

If users need to ensure their CPU job runs on a CPU-only worker (not an accelerator worker), they could add a constraint:

```python
constraint = req.constraints.add()
constraint.key = "device-type"
constraint.op = cluster_pb2.CONSTRAINT_OP_EQ
constraint.value.string_value = "cpu"
```

This would require workers to report a `device-type` attribute. This is not implemented but could be added if there is user demand.

## Implementation Plan

1. **Add `device_compatible` function** to `scheduler.py`
2. **Modify `can_fit_job`** to use the new function
3. **Add unit tests** covering all device type combinations
4. **Update documentation** if any user-facing docs mention device scheduling

Estimated effort: 1-2 hours including tests.

## Summary

| Aspect | Decision |
|--------|----------|
| CPU job on any worker | Allowed |
| GPU/TPU job on mismatched worker | Rejected |
| "No device" vs "CPU device" | Treated identically |
| Impact on coscheduling | None |
| Backwards compatibility | Maintained |
| Device preference logic | Deferred |
