---
name: debug-tpu
description: Identify and replace bad TPU nodes causing job failures. Use when logs show "No accelerator found", "FAILED_PRECONDITION", or "Device or resource busy".
---

# Skill: TPU Bad-Node Recovery

Identify and replace bad TPU nodes, then hand back to the active monitoring loop for job recovery.

## Trigger Patterns

Invoke this skill when job logs contain any of:
- `RuntimeError: No accelerator found. Please run on a TPU or GPU.`
- `Failed to cleanup driver after error: INTERNAL: FAILED_PRECONDITION`
- `Device or resource busy`

These indicate a bad TPU node rather than a code or config bug.

## Recovery Steps

### 1. Identify the bad node

Extract the worker IP from the error logs. Look for the IP address associated with the failing task or worker process.

### 2. Find the TPU VM name

Map the IP to a TPU VM name. Methods:
```bash
# List TPU VMs in the zone and match by IP
gcloud compute tpus tpu-vm list --zone <ZONE> --format="table(name,networkEndpoints[0].ipAddress)"
```

### 3. Delete the bad TPU VM

```bash
gcloud compute tpus tpu-vm delete <TPU_VM_NAME> --zone <ZONE> --quiet
```

The cluster will automatically provision a replacement node.

### 4. Return to monitoring

After the bad node is deleted, return to the active babysit loop (**babysit-job** or **babysit-zephyr**) and proceed with the RECOVER step (stop the job, resubmit).

## Guardrails

- This is the **only** cluster mutation allowed without explicit user consent during monitoring.
- Only delete the specific bad node — do not restart or recreate the entire cluster.
- If multiple nodes are bad simultaneously, report to the user before bulk-deleting.
- If the same node (or replacement) fails again after recovery, report to the user rather than retrying.

## See Also

- **babysit-job** — generic job monitoring loop
- **babysit-zephyr** — Zephyr pipeline monitoring
