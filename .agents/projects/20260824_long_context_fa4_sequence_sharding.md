# Spec: context-sharded Q support in the FA4/CuTe attention path (all-gather-KV CP)

**Branch:** `long-context/fa4-sequence-sharding` (worktree
`~/projects/marin.long-context-fa4-sequence-sharding`)
**Item:** "The production FA4/CuTe wrapper rejects sequence-sharded Q/K/V"
(`_assert_sequence_axis_unsharded`, `lib/levanter/src/levanter/grug/attention/_fa4_cute.py:208-217`)
— from `.agents/projects/20260824_535b_long_context_cooldown.md` and #8374.
**Do not open PRs/issues or post anything to GitHub. Commit locally on this branch only; do not
push.** First step: `git merge long-context/mesh-context-axis` (local branch; it provides the
`context` mesh axis and `_batch_axes` semantics — see its spec under `.agents/projects/`).

## Goal

Make the production GB200 attention path (`gpu_fa4_cute`) accept Q whose sequence axis is
sharded over the `context` mesh axis, with K/V sequence-replicated — the "all-gather-KV" flavor
of context parallelism. This is the flavor the finished 262K TPU run used (Q on `context`, KV
replicated, splash backend) and is the production-critical path for GB200 262K, since the
Transformer Engine route is blocked at runtime (cuDNN `BAD_PARAM`; see
`.agents/logbooks/grug-context-parallel-attention.md` on
`origin/codex/research/grug-context-parallel-attention`). Ring attention and SWA halo
optimization are explicitly out of scope (follow-ups).

## Prior art

- TPU precedent: `b2cf4fe1aa`, `7b7e82f39d`, `ec319e905a` on `origin/june_tpu_67b_a2b` — model
  reshards Q to `P(batch_axes, "context", head, None)`, K/V to seq-`None`, RoPE cos/sin cache
  resharded to seq=`"context"`, segment_ids Q-sharded/KV-replicated. Read those diffs; the
  attention-boundary wiring for the hero model should mirror them.
- Splash's existing sharded wrapper (`_core.py:308-440`) computes q-seq shard factors and
  handles a sharded Q sequence dim; it is the in-repo example of a CP-capable shard_map.

## The central correctness problem

`gpu_fa4_cute_attention` encodes causality as per-q-token int32 `lower_bounds` + bool `valid`
(`fa4_bounds`), consumed by `fa4_cute_attention_forward` (`_fa4_cute_backend.py:659`, a
`jax.custom_vjp` over the CUTLASS FFI). Determine how the kernel derives each Q row's causal
*upper* bound: if it assumes the q row's global position equals its index within the passed Q
block, then a context-sharded Q block starting at global position `s0` computes the wrong
causal frontier and MUST be given an offset (or explicit upper bounds). Read
`_fa4_cute_backend.py` and `_fa4_cute_kernels.py` before designing; do not guess. The design
must state exactly where positions enter (bounds arrays vs. kernel-internal `arange`).

## Scope

1. **Sharded wrapper** (`_fa4_cute.py:220-280`): teach `_fa4_cute_attention_forward_sharded`
   to handle Q/lower_bounds/valid sharded over `context` on the sequence dim while K/V remain
   seq-replicated:
   - Replace the blanket `_assert_sequence_axis_unsharded` on q/bounds/valid with logic that
     accepts (only) `context` sharding there; keep rejecting seq-sharded K/V with a clear error
     naming the all-gather-KV contract.
   - Give the shard_map explicit `in_specs`/`out_specs` (today in_specs are inferred): Q/out
     `P(batch_axes, "context", head_axis, None)`, K/V `P(batch_axes, None, head_axis, None)`,
     bounds/valid `P(batch_axes, "context")`.
   - Inside the body, apply whatever q-position offset the kernel needs (per the analysis
     above). `jax.lax.axis_index("context")` gives the shard index inside shard_map.
2. **Backward**: with K/V replicated over `context` in in_specs, shard_map's transpose must
   produce a `psum` over `context` for dK/dV. Verify this actually happens with
   `check_vma=False` (it may not — if not, wrap the kernel call so K/V grads are explicitly
   psummed over `context`). This is the highest-risk silent-wrongness point; the parity test
   below must cover gradients.
3. **Bounds precompute**: `fa4_cute_segment_bounds` (`_fa4_cute.py:346`) and
   `_packed_segment_causal_lower_bounds` run under jit on `[B, S]` arrays — confirm they work
   with a context-sharded input (associative scan over a sharded axis) or reshard internally
   and document the cost.
4. **Hero model wiring** (`experiments/grug/moe_hero_ep/model.py`): mirror the TPU commits at
   the attention boundary only — reshard Q to context-sharded, K/V (`kv_spec`, :513) stays
   seq-replicated, RoPE cache and segment-id/bounds pinning (:1245, :1264-1291) become
   context-aware. Rotary application must use *global* positions on each shard. Do NOT touch
   MoE/loss/metrics code — that is `long-context/moe-context-sharding`'s scope; coordinate by
   keeping your edits to attention-adjacent functions so the branches merge cleanly.
5. **SWA layers**: sliding_window=2048 layers work unchanged through the same bounds mechanism
   (bounds already encode the window); confirm rather than special-case.

## Out of scope

Ring attention, KV halo exchange for SWA, TE backend, MoE/loss changes, launcher changes,
`gpu_fa4_thd` (leave its behavior as-is; raise a clear NotImplementedError if it receives
context-sharded input).

## Verification

- **Parity test (the core deliverable)**: CPU or single-GPU multi-device
  (`XLA_FLAGS=--xla_force_host_platform_device_count=8`) — build a mesh with
  `context_axis_size in (2, 4)`, run the *reference* attention path on identical inputs
  unsharded vs. context-sharded, compare outputs AND input gradients (fixed cotangent) to tight
  tolerance; include packed segments (multiple docs per row) and a sliding-window case. If the
  FA4 CUTLASS kernel itself cannot run on CPU, structure the test so the sharding wrapper logic
  (specs, offsets, psums) is exercised with the reference kernel body substituted, and add a
  GPU-marked test with the real kernel for later cluster runs. Beware the vacuous-pass trap:
  a test that cannot fail on CPU against unfixed code proves nothing — check the test fails
  when the offset/psum logic is deliberately broken.
- `uv run pyrefly check`; `./infra/pre-commit.py --all-files --fix`; new files under `lib/`
  need `git add -f` (repo-global gitignore trap).
- `uv run --no-project infra/ci/run_tests.py` passes; existing FA4 tests unchanged.

## Risks

- Wrong q-position offset or missing dK/dV psum → silently wrong training. Gradient parity
  tests are mandatory, and must be shown to fail against a deliberately broken variant.
- Memory: all-gather-KV replicates full-sequence K/V per rank. At hero shape (262144 seq,
  global layers KV6, head 128, bf16) that is ~0.8 GiB of K+V per global layer instance —
  acceptable for the first cut; note real numbers in the spec-implementation notes.
