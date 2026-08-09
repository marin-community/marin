# Command-buffer replay: dependency-resolution failure

## Status

This artifact records an unmeasured environment-bootstrap failure at Shuttle
revision `2bfe5844388d4b73db0f532bbf81d78372cbace3`. It contains no component or
command-buffer correctness, determinism, handler-count, or performance result.

The command-buffer replay remained gated until the preceding Contract/Map H100
holder reported explicit release. One batch-priority holder was then allocated
with one H100, one requested CPU, 32 GB host memory, and 50 GB disk. A detached
archive of the exact measured revision was transferred and its SHA-256 was
verified in the pod.

The complete natural-Grug environment was resolved before the required
`--dependency-preflight-only` command. Resolution failed while unnecessarily
building the `vllm` serve dependency pulled in by `marin-core`; its build script
required `CUDA_HOME`. The dependency preflight therefore never ran. JAX did not
initialize a backend, no CUDA API or benchmark command executed, and the sole
authorized `shared_map_fused_reverses --command-buffer-candidate
normalized_exp_pair` process never started.

This is a dependency-selection failure, not evidence about the generated
program. The next attempt must use a narrow task environment containing the
natural Grug runtime dependencies plus JAX/JAXLIB/plugin/PJRT 0.11.0 and the
known Torch/Triton AOT build stack, without installing the unrelated Levanter
serve/vLLM extra. It must target canonical revision `239372d31d` or later so it
also includes the corrected JAX FFI layout order. A new allocation requires
explicit authorization after that fixed-layout component replay releases.

## Source and holder

- Shuttle source: `2bfe5844388d4b73db0f532bbf81d78372cbace3`.
- Detached archive: 41 MB, SHA-256
  `c39f7f878d19b8d1af60df8674dcc77effd045a42a2361de6c7eacf3b6f59cd1`.
- Iris holder client: `eafa4d49f7c55fbf2abb26b5d92c1ac7d093f9fb`.
- Holder bundle: 468 bytes with an explicit empty `dev` dependency group.
- Iris job: `/dlwh/dev-gpu-dlwh-shuttle-cmdbuf-2bfe`.

## Release proof

The allocator terminated the holder immediately after the new fixed-layout
gate arrived. Iris reports `JOB_STATE_KILLED`, exit code 0, no task failures,
and no active matching job. The local session file is absent and the exact
task-label Kubernetes query returns no pod.

