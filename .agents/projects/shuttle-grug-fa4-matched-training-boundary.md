# Experimental Shuttle and Grug FA4 diagnostic boundary

This harness is not an accepted plugin frontend or performance gate. It does
not authorize an H100 run.

## Claim boundary

The primary expert comparison is Grug's `gpu_fa4_cute` implementation on one
H100 at `B=1, S=2048, Hq=32, Hkv=8, D=128`, BF16 causal self-attention, and
scale `1/sqrt(D)`. This is the Grug BSHD segmented FA4 implementation, not the
THD adapter and not Torch SDPA.

The legal saved-state split is:

```text
forward: Q[B,S,Hq,D], K/V[B,S,Hkv,D]
      -> O[B,S,Hq,D] bf16, LSE[B,Hq,S] fp32 natural-log

reverse: Q, K, V, O, LSE, dO[B,S,Hq,D]
      -> dQ[B,S,Hq,D], dK/dV[B,S,Hkv,D]
```

Grug's custom VJP saves exactly `Q, K, V, O, LSE, lower_bounds, valid` and
passes the output cotangent to `segmented_flash_attention_backward`. Shuttle
reconstructs the ordinary causal index domain from generic
`DomainRestriction` semantics. Metadata generation is included in each public
composed timing and reported separately for component timings.

The generated forward exposes natural-log LSE. Its physical streaming kernel
uses base-2 exponentials internally and converts only at the ABI boundary. The
saved-state reverse converts the natural-log input back to its internal base-2
representation. This matches Grug's public saved-state ABI without changing
the generic normalized-exponential Fold semantics.

## Frontend provenance

The diagnostic starts from ordinary JAX attention algebra and a JAX-produced
VJP graph. It exports through `jax.export` and losslessly imports StableHLO
operations and values. The attention-specific selector then regenerates the
executable reverse with Shuttle's symbolic reference VJP. JAX therefore owns
the source reverse evidence, but not the generated reverse in this harness.
The selector remains in an explicitly experimental tile-lifetime module.

## Oracle audit

The benchmark records hashes for these Grug sources and the exact runtime
configuration selected on the device:

- `levanter.grug.attention._core.attention`, where `gpu_fa4_cute` dispatches;
- `_fa4_cute.gpu_fa4_cute_attention`, including causal metadata and scale;
- `_fa4_cute_backend.segmented_flash_attention_forward`;
- `_fa4_cute_backend.segmented_flash_attention_backward`;
- `_fa4_cute_backend` custom-VJP forward/residual/reverse functions;
- `_fa4_cute_config.flash4_cute_kernel_config`.

The checked-in base Grug model currently leaves attention implementation
unspecified, and the checked-in MoE model defaults it to `None`. Therefore this
comparison is against Grug's available FA4 implementation, not a claim that
all checked-in Grug configurations select FA4 by default.

The checked-in MoE training configuration also permits whole-block
recomputation. This benchmark intentionally measures the saved-O/LSE schedule
candidate. It does not claim that this saved-state policy is already selected
by the full Grug training step.

## Pre-timing gates

Before timing, the harness must:

1. verify all logical shapes, dtypes, layouts, causal semantics, GQA ratio,
   scale, and LSE encoding;
2. compare generated forward state and generated gradients with ordinary JAX;
3. compare Grug forward state and gradients with the same ordinary JAX
   reference;
4. require generated outputs to repeat bitwise;
5. apply the reviewed bounded expert-oracle repeatability policy under
   `allow_rounding_reorder` and serialize every repeat hash and error;
6. fail on nonfinite values or any numerical-policy violation;
7. prove generated handler/library dependencies contain no FA4 or Torch
   semantic kernel.

These checks describe a future diagnostic replay. This checkpoint does not
authorize an H100 allocation. Torch Flash-SDPA remains a secondary comparison.
