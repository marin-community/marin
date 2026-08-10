# Capture-safe Contract/Map H100 replay

## Result

TLTC-XLA-063 is an unaccepted command-buffer accounting result. The sole
measured invocation used Shuttle revision `870ce22524`, one H100, and the fixed
four-warmup, 30-sample, 1,000-iteration counterbalanced protocol.

The generated forward and reverse host-handler counters were:

| Checkpoint | Forward | Reverse |
| --- | ---: | ---: |
| After correctness | 1 | 1 |
| After determinism | 3 | 3 |
| After warmup | 4 | 4 |
| After measurement | 6 | 6 |

Six host callbacks for 30,007 logical generated executions show that XLA
captured and replayed the command-buffer-compatible handlers. Two additional
captures occurred after the warmup checkpoint, so the strict requirement that
counts remain fixed from warmup through measurement failed.

The process completed all timed loops before the assertion fired, but the
harness serializes its result only after handler-count validation. No raw timing
distribution or performance ratio was written, and no timing claim can be
recovered from this run. The experiment was not retried.

## Preflight and guards

Compile, link, load, and symbol resolution passed before device execution. Fresh
handler counts were `(0, 0)`. The generated source has both
`kCmdBufferCompatible` traits, no CUDA launch-status query, no allocation,
handles, autotuning, synchronization, atomics, or opaque semantic dependency.

The run reached the final handler-count gate. Therefore the natural-JAX,
ordered-CPU, layout, and three-trial deterministic-hash guards completed without
raising. Their numeric details were not serialized because the later count gate
failed.

JAX, JAXLIB, `jax-cuda13-plugin`, and `jax-cuda13-pjrt` were all 0.11.0.
The environment was Torch-free. CUDA compilation used NVCC 13.3.73 for
`sm_90a`. The exact source archive SHA-256 was
`2662c0e5b1bc15f112f4d7a791cbf7270464a5479b3cf0ea5e1777e1dcad02bd`.

## Release

The holder `/dlwh/dev-gpu-dlwh-shuttle-contract-map-capture-h100` was
explicitly terminated immediately after evidence copy. The controller reports
`JOB_STATE_KILLED` and `active_jobs=[]`; local holder state is absent; the exact
task-label pod query is empty.
