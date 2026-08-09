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
failure: the generated fused handler source was never compiled or loaded.

Follow-up commit `274e1b9238` removes the direct `jaxlib._hlo` dependency. HLO
proto import now obtains the module type through the public
`lower(...).compiler_ir(dialect="hlo")` path. The repository-pinned JAX/JAXLIB
0.10.1 runtime passes that proto roundtrip and provides the required text parser
as `jaxlib._jax.hlo_module_from_text`, but it does not provide
`jax.extend.xla.register_hlo_module_transformation`. There is no public HLO
text parser, so the parser remains an isolated, audited compatibility boundary.
`harness-compatibility-audit.json` preserves this result.

A future replay must run the CPU preflight successfully before allocating a
GPU. The exact prerequisites are a matched JAX/JAXLIB build that provides the
public compiler-IR proto roundtrip, one compatible HLO text parser, and both
`register_hlo_module_transformation` and
`clear_hlo_module_transformation` under `jax.extend.xla`. The previously
successful environment used JAX/JAXLIB 0.11.0. The physical replay remains
separately authorized work.
