# Partitioned SM90 H100 compile gate

This artifact preserves the single authorized H100 invocation of the generic
`PartitionedGemmSm90` executor. It is a negative environment-boundary result,
not GPU compilation, correctness, or performance evidence.

## Outcome

Iris job `/dlwh/shuttle-quack-partitioned-sm90-h100-20260809` requested one
H100, one CPU, 16 GB host memory, 50 GB disk, batch priority, a one-hour
timeout, and zero retries. It reached terminal failure after 29.42 seconds;
there were no preemptions or retries.

The job cloned and checked out QuACK
`84ef91df9bec87c7e4938517234fafb07ef844dd`, applied the recorded generic
partitioned-mainloop patch, and installed:

- Torch `2.13.0+cu130`;
- CUDA runtime `13.0` through Torch;
- CUTLASS DSL `4.6.1` and its CUDA 12/13 libraries;
- NVIDIA driver `595.71.05` on an H100 80 GB HBM3.

The preflight then imported
`tile_lifetime.quack_partitioned_gemm_adapter`. Python first executed the
package initializer, which imports `jax_collective_transport`; the deliberately
minimal environment did not install JAX. Import stopped with:

```text
ModuleNotFoundError: No module named 'jax'
```

The patched QuACK module was therefore not imported, CuTe did not compile a
device program, the correctness harness did not run, and no timing samples or
throughput claim exist. The failure does not identify a QuACK/CuTe executor
limitation. It identifies a missing preflight dependency caused by Shuttle's
eager package initializer.

Per the one-invocation instruction, no retry or tuning run followed. The exact
Kubernetes task-label query returned no pod after artifact retrieval, so the
H100 allocation is released.

## Revisions and integrity

- Shuttle authoring/harness revision: `233b253f4dde835d317c7d4107ee83209e157168`
- Minimal Iris bundle revision: `e6d7f0c615a437b368752281d628fece6487cf3e`
- QuACK revision: `84ef91df9bec87c7e4938517234fafb07ef844dd`
- QuACK extension patch SHA-256:
  `0bbb2354cff80b2fdf475fce12cef277f961591623b0c078160c27f09e5658db`
- Retrieved archive SHA-256:
  `da5d36d024465b93bc955ecb01e91fdee82fb88b6d72771aa0d67bd3aa3e9b1b`

`raw/iris.log` contains the full task log and base64 archive. The decoded
archive under `shuttle-partitioned-sm90/` contains the dependency-install logs,
device inventory, exact environment record, preflight traceback, and exit
status.

