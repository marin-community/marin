# Debugging log for H100 Contract/Map v8 NVTX tracking

The single reviewed H100 v8 job used source
`675e3399984129c0b76e3d14f9c123fb46fb0518` and immutable image digest
`sha256:d011ebdfc00d5b3a423d872b86388126636e4755cc5eb722c078f6c6d2ce598a`.
All ordinary-XLA, source-ordered, and fast numerical gates passed. The first
steady timing range then failed in `NvtxRange.__enter__`, before its timed
ordinary-XLA execution. The job was not retried or relaunched.

The sealed artifact is
`lib/tile_lifetime/benchmarks/artifacts/h100_contract_map_evidence_eighth_launch_failure_675e33_v0/`.

## Hypothesis 1: outermost level zero was rejected

This hypothesis is inconsistent with the submitted source. The wrapper uses
signed `ctypes.c_int` results and rejects only values below zero. It already
accepts the documented outermost level `0` and nested positive levels. Reaching
the exception proves that the live call returned a negative value, although the
old diagnostic did not record the exact value.

The exact pinned package is
`cuda-nvtx-13-2_13.2.86-1_amd64.deb`. Its NVTX 3 header defines
`NVTX_NO_PUSH_POP_TRACKING` as `-2`; the implementation returns that value when
no push/pop callback is installed. The same header specifies that successful
push and pop calls return zero-based levels and errors return negative values.
The pinned `libnvtx3interop.so.1.1.0` has SONAME `libnvtx3interop.so.1` and the
image exposes it through the compatibility link `libnvToolsExt.so`.

The failed task did not log the signed return or the presence of
`NVTX_INJECTION64_PATH`/`NVTX_INJECTION32_PATH`. Its generated Nsight Systems
report remained in the failed worker's temporary directory, so the retained
evidence cannot distinguish missing injection from a tool callback that
declined push/pop level tracking.

## Changes to make

The production wrapper now reports a bounded diagnostic containing the signed
return, the `-2` classification, requested and resolved library paths, the
`dladdr` library path when available, and presence-only injection flags. It
never emits injection-path values or the full environment. Push level `0` and
positive nested levels remain successful; negative push/pop results fail, and a
nonmatching nonnegative pop level also fails.

The H100 image build now runs this exact production wrapper under pinned Nsight
Systems 2026.1.3 with `--trace=nvtx`, `--capture-range=nvtx`, and
`--capture-range-end=stop`. It names the exact capture range and permits the
unregistered string used by the ctypes call. The CPU-only smoke requires a
balanced nonnegative push/pop result, a generated report, and exactly one
matching exported NVTX event. The wrapper and harness are read-only BuildKit
mounts and are absent from the final image. The smoke does not load the CUDA
driver or query a device.

## Results

The sealed v8 artifact verifies all pre-timing gates and the exact first
failure. Local fake-library tests cover outermost zero, positive nesting,
negative push and pop, `-2`, mismatched levels, body exceptions, ctypes
signatures, bounded diagnostics, and the runner's toolkit-library boundary.
The image smoke validator tests balanced result and exported-event acceptance,
plus missing, duplicate, lookalike, malformed, negative, unbalanced, and
oversized evidence.

The exact pinned image smoke has not run yet. No library or profiler binding was
changed because the retained v8 evidence does not prove which binding mechanism
failed. A future authorized image build must pass the new smoke before a tag can
be published.

## Future work

- [ ] Run the reviewed image workflow once to observe the pinned CPU-only NVTX
  smoke.
- [ ] If the smoke reports `-2`, use its bounded library and injection identity
  to repair the proven binding failure before another H100 launch.
- [ ] Re-run the 24-record evidence protocol only after the image smoke and
  source review pass.
