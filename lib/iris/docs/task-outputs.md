# Temporary task outputs

Iris preserves files written below `$IRIS_OUTPUT_DIR` as one archive per task attempt. Production clusters store `outputs.tar.zst` in region-local temporary object storage. Local clusters store it in the cluster's temporary directory.

## Write an output

Use the environment variable instead of a relative `./scratch` path:

```python
import os
from pathlib import Path

output_dir = Path(os.environ["IRIS_OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "profile.heap").write_bytes(profile_bytes)
```

Iris captures the directory after the task command exits successfully or with an application error. Each retry has a separate archive keyed by its attempt UID. An empty directory produces no object.

Inspect an attempt to find its archive:

```bash
uv run iris --cluster marin attempt describe /user/job/0:0
uv run fsutil cp gs://.../outputs.tar.zst .
```

`attempt describe` reports `uploaded`, `empty`, `failed`, or `unavailable`. Capture failures do not change the command's exit code or consume a retry. Cancellation, preemption, Pod deletion, and worker loss can make the archive unavailable.

## Limits and retention

The execution cluster owns the policy:

```yaml
task_outputs:
  destination: temporary_object_storage
  ttl_days: 7
  max_bytes: 2147483648
  max_entries: 10000
  finalization_timeout:
    seconds: 300
```

`destination: temporary_object_storage` resolves through `marin_temp_bucket` on the execution cluster. Objects use its lifecycle-managed `tmp/ttl=<N>d/` prefix. Federated tasks therefore write in the region where they run.

`LocalCluster` replaces the destination with `local`. The runtime maps `$IRIS_OUTPUT_DIR` to an attempt-local host directory and stores the archive below the cluster's temporary root. Both disappear when the local cluster closes; the CLI labels this as local-cluster retention.

The byte limit counts regular-file bytes before compression. The entry limit counts files, directories, symlinks, and skipped special files. Iris stores symlinks without following them. Devices, sockets, and FIFOs are skipped and reported. Finalization holds the task allocation until capture completes or reaches its deadline.

## Data access

Iris archives the requested files without inspecting or sanitizing their contents. Profiles, environment dumps, and nested archives can contain object-storage or W&B credentials. Anyone with read access to the regional temporary bucket can read the archive. Write only files that are safe for that bucket's readers.

Large or durable outputs should write directly to object storage while the task runs. `$IRIS_OUTPUT_DIR` is intended for bounded diagnostics produced by local-only tools.
