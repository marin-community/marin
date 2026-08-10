# Warp-finalized row Fold H100 bundle rejection

The first submission of the fixed warp-finalize benchmark was rejected by the
Iris client before job creation. The generated workspace archive was
`37,568,983` bytes (`35.828574 MiB`), above the controller's 25 MiB limit. No
job ID, Kubernetes pod, GPU allocation, or benchmark process existed.

This is a bootstrap failure, not a performance result. The candidate remains
unmeasured in this artifact. The sealed barrier-tree baseline remains
`0.089681 ms`, compared with `0.061414 ms` for matched XLA.

Static validation at source revision
`571d227bb83935db1e878f309157672551594d06` passed 24 focused tests and
targeted Pyrefly. The ordinary-JAX RMS backward recovery produced the same
three semantic fingerprints for separate and coalesced physical schedules.
The warp-finalized feature kernel contains one `__syncthreads()` and no
barrier-tree stride loop.

The submission requested one H100, one CPU, 16 GB host memory, 50 GB disk,
batch priority, zero retries, and a one-hour timeout. `submission.txt` records
the client rejection. `release-proof.txt` shows that the exact Iris job prefix,
task-label pod query, and matching local holder-session query were empty.
