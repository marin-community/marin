---
name: use-fsutil
description: Use Marin's fsutil CLI for local object-storage operations across GCS, CoreWeave S3, R2, and local paths. Use whenever a task needs to list, inspect, read, size, find, copy, move, synchronize, hash, browse, or remove data at gs:// or s3:// URLs from a checkout. Do not launch an Iris job solely to access object storage.
---

# Use fsutil

Use `uv run fsutil` from the Marin checkout for operations on `gs://` and
`s3://` paths. It routes each declared bucket to the backend and endpoint in
`config/*.yaml`.

Read `docs/references/fsutil.md` for commands, behavior, and safety constraints.
Let the task determine which command and verification are appropriate.

Use this interface before provider-specific CLIs, hand-built S3 endpoints,
custom transfer scripts, or Iris jobs. Do not launch Iris solely to access
object storage. For Python code that needs the same bucket routing, use
`rigging.filesystem.buckets.filesystem_for(url)`.
