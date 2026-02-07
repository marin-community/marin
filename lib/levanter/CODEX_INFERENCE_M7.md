# CODEX Inference M7: TPU Ragged Paged Attention

Status: COMPLETE for M7 exit target (2026-02-06)

## Goal

Re-enable TPU ragged paged attention (kernel path) for long-generation sampling without prefill VMEM OOM.

M7 exit target from `CODEX_REFACTOR_KV.md`:
- TPU kernel path passes 1-prompt long generation without VMEM OOM.

## What Was Broken

With `model.use_tpu_ragged_paged_attention: true`, long-generation runs failed during prefill in
`ragged_paged_attention` / `pallas_call` with `RESOURCE_EXHAUSTED` scoped VMEM OOM.

Representative pre-patch failures:
- E1 (`...m7_kernel_on.yaml`): FAIL at prefill (`max_prefill_size=4096`, `max_pages=80`)
- E8 (`...m7_kernel_on_boundary512.yaml`): FAIL at prefill (`max_prefill_size=512`, `max_pages=16`)

Kernel-off control remained stable:
- E2 (`...m7_kernel_off.yaml`): PASS, full 2048 generation.

This isolated the failure to TPU kernel path behavior, not scheduler/reset mechanics.

## Root Cause (Final)

In `lib/levanter/src/levanter/layers/attention.py`, TPU ragged kernel call path was not forwarding
model-config block-size controls (`ragged_paged_q_block_size`, `ragged_paged_kv_block_pages`).

That meant TPU kernel was invoked with `num_queries_per_block=None` and
`num_kv_pages_per_block=None`, forcing JAX kernel auto block-size selection. On this v5p stack,
that auto path produced prefill-time VMEM OOM in previously failing runs.

## Code Change

Patched `lib/levanter/src/levanter/layers/attention.py`:

1. `ragged_paged_attention(...)` now forwards:
   - `q_block_size`
   - `kv_block_pages`

2. `_do_tpu_ragged_paged_attention(...)` now accepts those args and passes explicit TPU kernel params:
   - `num_queries_per_block=q_block_size`
   - `num_kv_pages_per_block=min(kv_block_pages, pages_per_seq)`

Net effect: TPU path no longer depends on kernel auto block-size selection for these knobs.

## Validation Summary

### Historical Pre-Patch Baseline

- E1 FAIL:
  - Config: `config/sampler/sample_llama8b_multihost_real_1prompt_2048_reset_physical_round1_cleanup_none_noop_m7_kernel_on.yaml`
  - Log: `/tmp/levanter_run_m7_e1_1prompt_2048_kernel_on.log`
  - Failure class: prefill VMEM OOM.

- E8 FAIL:
  - Config: `config/sampler/sample_llama8b_multihost_real_1prompt_512_reset_physical_round1_cleanup_none_noop_m7_kernel_on_boundary512.yaml`
  - Log: `/tmp/levanter_run_m7_e8_1prompt_512_kernel_on_boundary512.log`
  - Failure class: prefill VMEM OOM.

### Post-Patch Validation (Current)

- E9 PASS:
  - Config: `config/sampler/sample_llama8b_multihost_real_1prompt_512_reset_physical_round1_cleanup_none_noop_m7_kernel_on_boundary512_kv8.yaml`
  - Log: `/tmp/levanter_run_m7_e9_1prompt_512_kernel_on_boundary512_kv8.log`
  - Result: full generation (`round_total_generated=512`).

- E10 PASS:
  - Config: `config/sampler/sample_llama8b_multihost_real_1prompt_2048_reset_physical_round1_cleanup_none_noop_m7_kernel_on_prefill1024_pages40_kv8.yaml`
  - Log: `/tmp/levanter_run_m7_e10_1prompt_2048_kernel_on_prefill1024_pages40_kv8.log`
  - Result: full generation (`round_total_generated=2048`, no OOM).

- E11 PASS (critical regression check):
  - Config: `config/sampler/sample_llama8b_multihost_real_1prompt_2048_reset_physical_round1_cleanup_none_noop_m7_kernel_on_prefill1024_pages40.yaml`
  - Log: `/tmp/levanter_run_m7_e11_1prompt_2048_kernel_on_prefill1024_pages40_afterpatch.log`
  - Result: full generation (`round_total_generated=2048`).
  - Note: this shape previously failed in E5 pre-patch.

- E14 PASS (exact original failing config, post-patch):
  - Config: `config/sampler/sample_llama8b_multihost_real_1prompt_2048_reset_physical_round1_cleanup_none_noop_m7_kernel_on.yaml`
  - Log: `/tmp/levanter_run_m7_e14_1prompt_2048_kernel_on_original_afterpatch.log`
  - Result: full generation (`round_total_generated=2048`, no OOM).

### Additional Stress/Characterization

- E12 PASS (`kv32`): `/tmp/levanter_run_m7_e12_1prompt_2048_kernel_on_prefill1024_pages40_kv32.log`
- E13 PASS (`q32, kv40`): `/tmp/levanter_run_m7_e13_1prompt_2048_kernel_on_prefill1024_pages40_q32_kv40.log`
- E15 PASS (`q128, kv80`): `/tmp/levanter_run_m7_e15_1prompt_2048_kernel_on_prefill4096_pages80_q128_kv80.log`

## Final Takeaways

1. M7 exit target is met: TPU kernel path now runs long generation in configs that previously OOM’d.
2. The practical fix is to explicitly wire and apply block-size controls on TPU kernel calls.
3. The pre-patch OOM was not a scheduler/reset bug (kernel-off controls remained stable).
4. Keep block-size controls explicit in config-backed TPU runs to avoid dependence on auto path behavior.

## Follow-Up (Optional)

1. Add a focused unit/integration test that asserts TPU path forwards explicit block sizes.
2. Prepare an upstream JAX minimal reproducer contrasting `None` auto path vs explicit block-size path on v5p.
