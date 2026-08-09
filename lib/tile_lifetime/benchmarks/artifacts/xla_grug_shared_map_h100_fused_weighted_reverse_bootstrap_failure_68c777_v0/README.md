# Fused weighted-reverse H100 bootstrap failure

This artifact preserves the only attempted physical-H100 invocation of the
natural Grug train-step boundary with the generated bounded Contract–Map–Fold
composition at source revision `68c7770410`. The benchmark did not reach HLO
recovery, handler compilation, correctness, or timing. It exited while
importing the private `jaxlib._hlo` module used by the benchmark harness.

The allocated image initially contained CPU-only JAX 0.10.1. Installing the
repository-pinned `jax[cuda13]==0.10.1` made the H100 visible but did not provide
`jaxlib._hlo`; `jax-runtime-audit.txt` records `jaxlib._hlo None`. The exact
failure is preserved in `replay.stderr`. No retry was made, so this artifact
contains no performance or acceptance claim about the generated candidate.

The attempted command is preserved in `invocation.txt`. It requested the fused
composition with four warmups and 30 counterbalanced samples, but failed before
any warmup. `benchmark-started`, `benchmark-finished`, and `exit-code.txt`
record the attempt boundary. GPU, driver, clock, power, Python, and package
state are preserved alongside the logs.

The device was one NVIDIA H100 80GB HBM3 with driver 595.71.05 and a 700 W
power limit. Clocks were not pinned. One GPU and four CPUs were requested at
batch priority. The allocation was released immediately after the artifact was
copied, and both local session state and pod lookup verified that it was no
longer active.

This failure is infrastructure/harness compatibility, not a generated-code
failure: the generated fused handler source was never compiled or loaded. A
future replay requires either a JAX build exposing the benchmark's pinned HLO
API or a separately reviewed migration of the harness away from
`jaxlib._hlo`; it must be authorized as a new physical replay.
