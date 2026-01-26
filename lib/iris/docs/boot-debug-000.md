# Boot Debug Log 000

## Session Info
- Date: 2026-01-25
- Agent: Senior Engineer
- Config: examples/eu-west4.yaml
- Project: hai-gcp-models
- Zone: europe-west4-b

---

## Step 1: Cleanup Existing Resources

### Step 1.1: Check existing resources (dry-run)

**Time:** ~14:00 UTC

**Command:**
```bash
uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models cleanup
```

**Output:**
```
Scanning zone europe-west4-b in project hai-gcp-models...
(DRY-RUN mode - no changes will be made)

Found 1 controller VM(s):
  - iris-controller-iris

No TPU slices found.

Would delete 1 resource(s). Use --no-dry-run to delete.
```

**Status:** SUCCESS

**Notes:** Found 1 existing controller VM from a previous session. No TPU slices present.

---

### Step 1.2: Delete existing resources

**Time:** ~14:00 UTC

**Command:**
```bash
uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models cleanup --no-dry-run
```

**Output:**
```
Scanning zone europe-west4-b in project hai-gcp-models...

Found 1 controller VM(s):
  - iris-controller-iris

No TPU slices found.

Deleting resources...
Deleting VM: iris-controller-iris

Deleted 1/1 resource(s).
```

**Status:** SUCCESS

**Notes:** Controller VM successfully deleted.

---

### Step 1.3: Verify cleanup with discover

**Time:** ~14:01 UTC

**Command:**
```bash
uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models discover
```

**Output:**
```
Searching for controller VM in europe-west4-b...
No controller VM found.
```

**Status:** SUCCESS

**Notes:** Cleanup verified. Zone is clean.

---

## Step 2: Launch Cluster via demo_cluster.py --cluster

**Time:** ~14:01 UTC

**Command:**
```bash
uv run python examples/demo_cluster.py --cluster --config examples/eu-west4.yaml --verbose
```

**Output:**
```
(running...)
```

**Status:** IN_PROGRESS

---
