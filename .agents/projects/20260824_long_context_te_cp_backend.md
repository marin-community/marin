# Spec: Transformer Engine context-parallel backend in the Grug attention dispatcher

**Branch:** `long-context/te-cp-backend` (worktree `~/projects/marin.long-context-te-cp-backend`)
**Item:** "Grug has no production Transformer Engine context-parallel backend in its attention
dispatcher" — from `.agents/projects/20260824_535b_long_context_cooldown.md` and #8374/#8141.
**Do not open PRs/issues or post anything to GitHub. Commit locally on this branch only; do not
push.** First step: `git merge long-context/mesh-context-axis` (local branch; provides the
`context` mesh axis — see its spec under `.agents/projects/`).

## Context and honest status

TE 2.17.1 currently fails at runtime on GB200: cuDNN returns `CUDNN_STATUS_BAD_PARAM` from
`fused_attn_f16_arbitrary_seqlen.cu:934` while sizing the backward workspace, reproduced on
NVIDIA's own unmodified CP4 example — the blocker is the TE/cuDNN toolchain, not our shapes.
See `.agents/logbooks/grug-context-parallel-attention.md` on
`origin/codex/research/grug-context-parallel-attention` for the full record (build recipe,
constraints, failure). This branch therefore delivers a complete, opt-in, import-guarded
production backend whose pure-JAX parts are fully unit-tested, and whose TE-touching parts are
covered by GPU-marked integration tests that will pass once a fixed TE/toolchain lands. Do not
water down correctness to "it imports"; do not claim runtime validation that did not happen.

## Prior art (port from it, don't reinvent)

`origin/codex/research/grug-context-parallel-attention` — read these via `git show`:

- `lib/levanter/scripts/bench/bench_grug_context_parallel_attention.py`: the `TransformerEngineApi`
  lazy-import shim (explicit symbol contract, actionable ImportError); the exact `fused_attn`
  call (positional `(q,k,v), None, SequenceDescriptor, None`; `qkv_layout=THD_THD_THD`,
  `attn_mask_type=PADDING_CAUSAL_MASK`, `context_parallel_strategy=RING|ALL_GATHER`,
  `context_parallel_causal_load_balanced=True`, `context_parallel_axis="context"`,
  `stripe_size`, `window_size`, `max_segments_per_seq`); striped causal load balancing via
  `reorder_causal_load_balancing(strategy=ReorderStrategy.Striped, cp_size, seq_dim, stripe_size)`
  applied identically to Q/K/V and to `segment_ids`/`segment_positions` before
  `SequenceDescriptor.from_segment_ids_and_pos`; validity preflights
  (`seq % (2*cp*stripe) == 0`, `is_fused_attn_kernel_available(...)`); `MeshResource(dp_resource=...,
  cp_resource="context")` + `te.autocast`; `NVTE_FUSED_RING_ATTENTION_USE_SCAN=0` before import.
- `experiments/grug/moe_hero_fsdp/context_parallel_attention_benchmark.py`:
  `TRANSFORMER_ENGINE_SETUP_SCRIPT` and `TRANSFORMER_ENGINE_BUILD_ENV` — the only known-good
  TE-on-Iris-GB200 install recipe (cccl/cudnn-frontend/te_cu13 order, libnccl symlink,
  `--no-build-isolation` etc.). Keep verbatim.
- `experiments/grug/dispatch.py` diff: adds `pip_packages`, `extra_env_vars`, `setup_scripts`
  params to `dispatch_grug_training_run`. Port this — it is independently useful and needed to
  launch any TE run.

Recorded TE constraints to honor in code (assert, don't document-only): Ring requires
stripe_size=1; THD CP supports only padding-causal mask, no bias, no dropout, vanilla softmax;
`max_segments_per_seq` is static; supported layouts exclude fully-packed T3HD.

## Design

1. **New implementation name** `"gpu_te_cp"` in `GrugAttentionImplementation`
   (`lib/levanter/src/levanter/grug/attention/_core.py:30-35`) plus a dispatch arm in
   `attention()` (`:443-472`), lazily importing a new module
   `lib/levanter/src/levanter/grug/attention/_te_cp.py`. Never auto-selected; config only.
2. **`_te_cp.py`** contains:
   - the `TransformerEngineApi` shim (ported);
   - pure-JAX striping helpers: `stripe_for_cp(x, *, cp_size, stripe_size, seq_dim)` and its
     exact inverse `unstripe_from_cp` — the benchmark never un-permutes, but production MUST:
     Grug's causal SConv, fused RoPE, labels/loss and packed-segment semantics need natural
     token order, so hidden states stay in **contiguous** sequence shards and
     striping/unstriping happens only at the attention boundary (this is the logbook's recorded
     design decision). These helpers and their inverse property are the main unit-testable
     surface.
   - metadata construction: segment_ids/segment_positions → striped → `SequenceDescriptor`;
     positions must carry original in-segment indices (causality survives the permutation).
   - the `fused_attn` wrapper with the preflight checks, strategy enum
     (`ring` | `all_gather`), and BSHD→THD handling as needed. Map Grug's `AttentionMask`
     (segment_ids, sliding_window) onto TE arguments; raise on unsupported combinations.
3. **Config**: `GrugModelConfig.attention_implementation = "gpu_te_cp"` selects it; add
   whatever minimal knobs are needed (strategy, stripe_size) — prefer a small frozen dataclass
   over loose fields, follow existing config style in the model configs.
4. **Launcher plumbing**: port the dispatch.py extension + TE setup script/env constants into a
   sensible home (e.g. `experiments/grug/te_setup.py`), so a future GPU validation run needs no
   new machinery.

## Out of scope

Fixing the TE/cuDNN toolchain; ring-vs-all-gather perf work; FA4 changes; MoE/loss; making
`gpu_te_cp` the default anywhere.

## Verification

- Unit tests (CPU, no TE installed): stripe/unstripe round-trip on odd shapes; striped
  segment-metadata invariants (equal per-rank token counts; per-token (segment_id, position)
  pairs preserved as a multiset; positions untouched by permutation); preflight rejections
  (bad divisibility, ring with stripe>1, unsupported mask combos) raise with clear messages;
  dispatcher raises an actionable ImportError mentioning the setup script when TE is absent.
- GPU-marked integration test (skipped without TE): 1-process fused_attn forward+backward at a
  small shape, plus a parity check against `reference_attention` on identical inputs. Mark it
  so default test runs skip it (follow module TESTING.md marker conventions; do not weaken the
  default marker expression).
- `uv run pyrefly check`; `./infra/pre-commit.py --all-files --fix`; `git add -f` for new files
  under `lib/` (gitignore trap); `uv run --no-project infra/ci/run_tests.py` passes.

## Risks

- TE API drift between 2.17.1 and the eventual fixed version — keep the shim's symbol list the
  single point of contact.
- Silent divergence between striping applied to tensors vs. metadata — the shared-helper design
  (one function used for both) is the mitigation; tests must cover it.
