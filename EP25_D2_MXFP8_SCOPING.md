# R6-4 MXFP8 toolchain scoping assessment

## Superseded checkpoint

This assessment predates the coordinator's comparison against the known-green
`research/mcwitt/7282-uniform-mxfp8` worktree. That comparison recovered the
exact CUTLASS dependency graph: DSL 4.5.2, the NVIDIA package index, and an
impossible-platform override for `nvidia-cutlass-dsl-libs-base==4.5.2`.
The worktree now carries that graph, and
`EP25_D2_RELAY_COMMANDS.md` contains a numerics-only v3 submission.

The direction remains open under the amended round-6 fleet policy regardless
of the v3 result. The historical reconstruction below is retained to explain
the earlier checkpoint.

## What the record establishes

- `research/mcwitt/7282-mxfp8-blackwell` commit `42f7d9fa2` produced the first
  green grouped-kernel run, `/mwittmann/mxfp8-002-g8`. The recorded command was
  `python experiments/grug/moe/standalone/bench_mxfp8_grouped.py` through the
  usual one-GB200 Iris submit.
- The submit recorded for the preceding one-GB200 run used
  `--gpu GB200x1 --enable-extra-resources --cpu 16 --memory 64g --extra gpu`.
  It did not select a task image or set `CUDA_TOOLKIT_PATH`.
- Commit `42f7d9fa2` resolved `jax[cuda13]==0.10.1` and
  `nvidia-cutlass-dsl[cu13]>=4.5.2,<4.6`. The logbook calls the working payload
  the stock aarch64 `nvidia-cutlass-dsl` 4.5.2 wheel and records no manual
  wheel-install step.
- The Iris config at `42f7d9fa2` selected
  `ghcr.io/marin-community/iris-task:latest`. The repository does not record
  the image digest pulled by `/mwittmann/mxfp8-002-g8`.
- Commit `d064dc173` later made the 4.5.2 environment deterministic by
  excluding `nvidia-cutlass-dsl-libs-base`; the logbook reports 32/32 clean
  CUDA 13 payloads after that change. The grouped kernel also needs the
  integer-pointer-to-gmem normalization recorded in `42f7d9fa2`; the vendored
  adapter in this worktree already contains it.
- The v2 bundle resolves CUTLASS DSL 4.6.0 with `libs-cu12` excluded, but
  `ep25d2-mxfp8-numerics-20260725-v2` still failed after 12.29 seconds with
  the same libNVVM `sm_100a` diagnostic. The v2 job predates the environment
  sentinel, so its loaded extension and libNVVM paths are unknown.
- The current Levanter GPU extra also pins `quack-kernels[cu13]==0.6.1`,
  which requires CUTLASS DSL 4.6.0. The green 4.5.2 bundle did not have this
  dependency constraint.

The GitHub issue body was available, but API access to issue comments failed
in this sandbox. The local 7282 logbook has the job names, submit shape,
dependency versions, failures, fixes, and results. It does not contain the
missing image or expanded environment data.

## Missing artifacts

Reproducing the green job requires all of the following:

1. The immutable digest of `ghcr.io/marin-community/iris-task:latest` pulled
   by `/mwittmann/mxfp8-002-g8`, or that job's archived pod specification.
2. The expanded `CUDA_TOOLKIT_PATH`, `LD_LIBRARY_PATH`, and libNVVM path from
   the green container. No explicit values are present in the submit record.
3. The green job's setup log or shipped bundle manifest, including installed
   CUTLASS wheel filenames and hashes. The record says "stock 4.5.2" but does
   not preserve the successful per-pod installation result from before the
   deterministic lock fix.
4. A supported way to isolate CUTLASS DSL 4.5.2 from the current QuACK 0.6.1
   dependency. No side environment or wheel overlay was used or documented by
   the 2.2 PF/s job.

The numerical script now prints one `CUTLASS_ENV_SENTINEL` JSON record before
compilation. It includes the `cutlass` module and loaded `_cutlass_ir`
extension paths, all CUTLASS dist-info names and versions, the payload owner
inferred from wheel `RECORD` hashes, `CUDA_TOOLKIT_PATH`, `LD_LIBRARY_PATH`,
and the libNVVM path selected by `cuda.pathfinder`.

## Estimated effort

- If the archived g8 pod specification, image digest, and setup log are
  available: 1–2 engineer hours to pin the image, reproduce the 4.5.2
  environment, run the sentinel-only probe, and submit one numerical job.
- If those artifacts are unavailable: 4–8 engineer hours plus GB200 queue
  time. This requires either a dedicated 4.5.2 environment that coexists with
  the 4.6.0 training stack, or a kernel/toolchain port validated against
  CUTLASS DSL 4.6.0. Both are new integration work, not a reconstruction of
  the recorded green run.

This is the fourth focused toolchain stop. Further cluster attempts need one
of the missing artifacts or approval for the 4.5.2-isolation/4.6.0-port work.
