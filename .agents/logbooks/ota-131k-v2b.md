# OT-Agent 131K v2b: Research Logbook

## Scope
- Goal: Find a `v5p-32` Levanter recipe for 131K SFT that can run stably to at least the halfway point without OOM.
- Primary metric(s): no host-RAM or HBM OOM through long-run training; improved effective throughput by removing avoidable checkpoint overhead.
- Constraints: keep `max_seq_len=131072`; do not change away from the Levanter training path; prefer `v5p-32` over `v5p-256`.

## Baseline
- Date: 2026-03-30
- Issue: https://github.com/marin-community/marin/issues/3897
- Code refs:
  - `experiments/exp3897_sft_ota_131k_qwen3_8b.py`
  - `experiments/exp3897v3_sft_ota_131k_qwen3_8b_no_offload.py`
- Baseline findings:
  - Offload-based `v2` on `v5p-32` failed repeatedly with exit 137 during JAX checkpoint serialization.
  - No-offload `v3` on `v5p-32` was manually killed after 1h29m, not OOM-killed.
  - `v3` reached step 13 and survived multiple time-based checkpoints at 256 GB host RAM.

## EXP3897-V2B-001
- Date: 2026-03-30 UTC
- Hypothesis: `v5p-32` instability came from `ScanCheckpointPolicy(save_carries="offload")` plus Marin's 10-minute temporary checkpoint cadence, not from 131K training itself.
- Evidence gathered:
  - `uv run iris --config lib/iris/examples/marin.yaml job bug-report /kevin/iris-run-exp3897_sft_ota_131k_qwen3_8b-20260325-061745/train_lm --tail 120`
  - `uv run iris --config lib/iris/examples/marin.yaml job bug-report /kevin/iris-run-exp3897v3_sft_ota_131k_qwen3_8b_no_offload-20260326-055401 --tail 80`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs /kevin/iris-run-exp3897_sft_ota_131k_qwen3_8b-20260325-061745/train_lm --level info | tail -n 220`
- Results:
  - Offload `v2` saved a temporary checkpoint every ~10 minutes, which at ~224s/step meant roughly every 3 steps.
  - The fatal `v2` OOM at step 410 occurred inside `jax.experimental.array_serialization.serialization` while committing checkpoint data.
  - No-offload `v3` was explicitly `Terminated by user`, not OOM, after 1h29m. It had already completed several checkpoint commits successfully.
- Decision:
  - Build `v2b` from the no-offload recipe.
  - Disable time-based temporary checkpoints.
  - Disable interim HF exports.
  - Use a half-run stability target by default.

## Next
- Launch `exp3897v2b` on `v5p-32` with `num_train_steps=989`, `save_interval=None`, and no HF exports.
- Monitor for:
  - successful JIT and first train steps,
  - absence of checkpoint-related OOM,
  - absence of host-offload/XLA regressions.

## EXP3897-V2B-002
- Date: 2026-03-30 UTC
- Hypothesis: the `v2b` recipe can get through distributed init and HF model loading on `v5p-32` without reproducing the earlier host-memory failures.
- Command:
  - `uv run iris --config lib/iris/examples/marin.yaml job run --tpu v5p-32 --memory 256GB --region us-central1 -e WANDB_ENTITY marin-community -e WANDB_PROJECT marin -e WANDB_API_KEY ${WANDB_API_KEY} -e HF_TOKEN ${HF_TOKEN} -e MARIN_PREFIX gs://marin-us-central1 -e GIT_COMMIT ca02df261 --no-wait -- python experiments/exp3897v2b_sft_ota_131k_qwen3_8b.py`
- Job:
  - `/kevin/iris-run-exp3897v2b_sft_ota_131k_qwen3_8b-20260330-083952`
  - child: `/kevin/iris-run-exp3897v2b_sft_ota_131k_qwen3_8b-20260330-083952/train_lm`
- Results so far:
  - Initial child-job scheduling was delayed until a fresh `v5p-32` slice became available.
  - `train_lm` started at `08:44:10 UTC` on slice `marin-tpu-v5p-32-us-central1-a-20260330-0840-1dcab560`.
  - All 4 workers initialized `jax.distributed` successfully.
  - HF checkpoint loading began at `08:45:39 UTC` from `hf://Qwen/Qwen3-8B`.
  - HF shard restore completed and the trainer reached `Tracing train_step for jaxpr...` at `08:47:17 UTC`.
  - All workers finished tracing in about `104.7s` and lowered `train_step` to HLO in about `1.5s` by `08:49:07 UTC`.
  - No OOM, exit 137, or checkpoint-serialization failure has occurred during startup.
  - A non-fatal warning appears when Levanter tries to dump the config artifact because `save_interval=None` is not encodable by the current `draccus` timedelta hook.
- Decision:
  - Keep the run unchanged and monitor through compilation and first train steps.
  - Treat the config-artifact warning as follow-up cleanup, not a blocker for the stability experiment.

## EXP3897-V2B-003
- Date: 2026-03-30 19:29 UTC
- Hypothesis: upstream long-context implementations in LLaMA-Factory and MaxText will point to a better TPU/JAX path than host offload or TPU-only CE kernel swaps.
- Evidence gathered:
  - `curl -L -s https://raw.githubusercontent.com/hiyouga/LlamaFactory/main/src/llamafactory/model/model_utils/rope.py | sed -n '1,220p'`
  - `curl -L -s https://raw.githubusercontent.com/hiyouga/LlamaFactory/main/src/llamafactory/model/model_utils/longlora.py | sed -n '1,220p'`
  - `curl -L -s https://raw.githubusercontent.com/hiyouga/LlamaFactory/main/src/llamafactory/v1/plugins/model_plugins/parallelization/ulysses.py | sed -n '1,260p'`
  - `curl -L -s https://raw.githubusercontent.com/hiyouga/LlamaFactory/main/examples/v1/train_full/train_full_ulysses_cp.yaml | sed -n '1,220p'`
  - `curl -L -s https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/docs/guides/optimization/sharding.md | sed -n '250,390p'`
  - `curl -L -s https://raw.githubusercontent.com/AI-Hypercomputer/maxtext/main/src/maxtext/configs/base.yml | rg -n 'max_target_length|context_parallelism|sequence_parallelism|remat_policy|num_vocab_tiling|attention|ici_.*parallelism|per_device_batch_size|gradient_accumulation_steps'`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs /kevin/iris-run-exp3897v2c_sft_ota_131k_qwen3_8b_tp2-20260330-162457 --include-children --level info | rg 'Progress on:train|loss=' | tail -40`
  - `uv run iris --config lib/iris/examples/marin.yaml job logs /kevin/iris-run-exp3897v2d_sft_ota_131k_qwen3_8b_pallas_ce-20260330-162928 --include-children --level info | rg -n 'pallas_tpu|ExceptionGroup|Traceback|train_step' | tail -120`
- Results:
  - LLaMA-Factory extends long context mainly with RoPE scaling plus faster attention backends (`flash_attn`), LongLoRA's `shift_attn`, and optional Unsloth integration; newer `v1` code also has `cp_mode: ulysses` with `cp_size` for sequence/context parallel attention.
  - LLaMA-Factory's Ulysses path is an all-to-all-based sequence parallel implementation around attention, not a host-offload strategy.
  - MaxText treats long context as a sharding problem first: `context_parallelism`, `sequence_parallelism`, `tensor_parallelism`, `fsdp`, `remat_policy`, and `num_vocab_tiling` are all first-class config knobs.
  - MaxText's TPU guidance is especially relevant:
    - TPU context parallelism currently uses `context_parallel_strategy=all_gather`, not ring attention.
    - TPU sequence parallelism can be more efficient than context parallelism when there are enough KV heads, but is constrained by `tensor_parallelism * sequence_parallelism < kv_heads`.
  - For our model, `experiments/qwen3.py` shows `Qwen3-8B` has `num_kv_heads=8`, which means any MaxText-like SP design would need to stay under that head-budget.
  - Speed experiments so far:
    - `v2c` (`v5p-64`, `tensor_parallel_size=2`) completed 24 steps at about `212.5s/step`, only a marginal improvement over `v2b` (~`217.3s/step`).
    - `v2d` (`v5p-32`, `pallas_tpu` fused CE) failed at compile/autotune with `Fused CE autotune found no viable block-size candidates for pallas_tpu`.
- Decision:
  - Stop treating CE kernel swaps as the primary throughput path.
  - Prioritize Levanter work that is analogous to MaxText's long-context playbook:
    - sequence/context sharding of activations and attention,
    - better remat policy selection without host offload,
    - vocab/logits tiling if CE memory or compile pressure remains material.
  - Use LLaMA-Factory mainly as confirmation that successful long-context stacks rely on RoPE scaling plus attention/parallelism changes, not on naive larger slices alone.
