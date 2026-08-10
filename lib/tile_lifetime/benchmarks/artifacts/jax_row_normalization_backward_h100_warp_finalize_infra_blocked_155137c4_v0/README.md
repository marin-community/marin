# Warp-finalized row Fold H100 infrastructure block

The fixed warp-finalize benchmark did not run. Two corrected minimal-holder
launch attempts were rejected before job creation, so this artifact contains no
H100 latency, correctness, determinism, or handler-count result.

The minimal holder was validated locally before submission:

- Iris bundle: `481,889` bytes (`0.459565 MiB`), below the 25 MiB limit.
- Bundle SHA-256: `b5f27c0152f7130d00327b7654eb21f3f2dc5c448ecd7a9120fbd0d92ba9253c`.
- Exact pushed source: `155137c49565590ce09232e0a67ede303ecc7911`.
- Source archive SHA-256: `03bb2f6a04cdc81533e398090bb2f2bba8d65f31d6278d25fa059eccf8be4643`.
- Linux dependency dry-run: JAX, jaxlib, CUDA plugin, PJRT, and NVCC `0.10.1`
  / `13.3.73` resolved successfully.
- The holder project declares `[dependency-groups] dev = []`.

The first minimal submission used the locally installed Iris client dated
`2026-07-21`. The controller requires at least `2026-07-27`, so it rejected the
request before job insertion.

The final authorized submission used Iris from `origin/main` revision
`64913f5302` with client date `2026-08-08`. Its read-only controller RPC passed,
but `job run` performs an additional CoreWeave S3 setup step. The current-main
`cw-us-east-02a.yaml` lacks the required `stores` entry, so the client raised
before opening a controller tunnel or uploading the validated bundle.

The exact job prefix and task-label pod queries were empty afterward. No local
GPU holder session existed. The sealed barrier-tree baseline therefore remains
`0.089681 ms`, compared with `0.061414 ms` for matched XLA; warp-finalize remains
unmeasured.

