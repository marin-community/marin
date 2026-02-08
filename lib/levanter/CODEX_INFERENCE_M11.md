# CODEX Inference M11: Interleaved SimPO Train -> Host-DP Inference -> Train

Status: IMPLEMENTED (2026-02-08; code/config/tests complete, TPU runtime validation pending)

## Goal

Implement and validate an in-process workflow in `src/levanter/main/train_simpo.py`:
1. Train for 2 steps.
2. Pause training and run multi-host host-data-parallel inference for `128` prompts with `max_new_tokens=2048`.
3. Resume the same training run for 5 additional steps (total steps `= 7`).

## Work Log

- 2026-02-08: Created M11 execution plan in `CODEX_REFACTOR_KV.md`.
- 2026-02-08: Started implementation work:
  - extending inference-eval scheduling for exact step triggers,
  - adding host-data-parallel prompt sharding in `train_simpo.py`,
  - preparing M11 config derived from `config/simpo_ultrafeedback_llama3_8b_v5p_32.yaml`.
- 2026-02-08: Implemented core callback refactor in `src/levanter/main/train_simpo.py`:
  - added exact-step scheduling support (`eval_at_steps`) with `eval_every` fallback,
  - added `inference_mode: global_mesh | host_data_parallel`,
  - added deterministic host prompt sharding in callback mode `host_data_parallel`,
  - added host payload gather to leader for global metrics/samples in host-DP mode,
  - added per-host JSONL output writing per eval step (`host_data_parallel_output_dir`),
  - converted callback exception handling to propagate failures (no swallow).
- 2026-02-08: Added M11 run config:
  - `config/simpo_ultrafeedback_llama3_8b_v5p_32_m11_train2_infer128_train5.yaml`
  - key settings:
    - `trainer.num_train_steps: 7`
    - `inference_eval.eval_at_steps: [2]`
    - `inference_eval.inference_mode: host_data_parallel`
    - `inference_eval.synthetic_prompt_count: 128`
    - `inference_eval.max_new_tokens: 2048`

## Hypotheses

1. Existing callback/barrier scaffolding can support interleaving train/infer/train with exact-step scheduling.
2. Host-data-parallel prompt sharding inside the callback will prevent duplicate prompt work across hosts.
3. Step-scheduled callback (`eval_at_steps=[2]`) is sufficient for M11 without changing the trainer core loop.

## Pending

- Execute multi-host TPU runtime validation for the full M11 flow.
- Capture wall-clock and token-total metrics from a real M11 run.

## Validation (Code-Level)

Completed:
- `uv run pytest tests/test_train_simpo_inference_eval.py tests/test_simpo.py`
  - result: `14 passed`.
- `./infra/pre-commit.py --all-files`
  - result: all checks passed.
- Config decode sanity:
  - `simpo_ultrafeedback_llama3_8b_v5p_32_m11_train2_infer128_train5.yaml` decodes into `TrainSimpoConfig`.

Outstanding runtime validation:
- Multi-host TPU run for full M11 flow (step 2 inference trigger and step 7 completion) has not yet been executed in this local implementation pass.
