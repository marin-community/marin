# CODEX Inference M12: Periodic Host-DP Inference During 200-Step SimPO Training

Status: IMPLEMENTED (2026-02-08; code/config/tests complete, TPU runtime validation pending)

## Goal

Implement and validate a long-run workflow in `src/levanter/main/train_simpo.py`:
- total training steps: `200`
- run host-data-parallel inference every `50` steps
- inference workload per interval: `1024` prompts, `max_new_tokens=2048`

## Work Log

- 2026-02-08: Added M12 plan entry to `CODEX_REFACTOR_KV.md`.
- 2026-02-08: Started implementation with shared M11/M12 code path in `train_simpo.py`:
  - exact-step scheduling support,
  - host-data-parallel prompt sharding support,
  - prompt-source options suitable for large prompt counts.
- 2026-02-08: Implemented shared callback/runtime changes in `src/levanter/main/train_simpo.py`:
  - explicit step scheduling (`eval_at_steps`) for periodic inference checkpoints,
  - host-data-parallel execution path in training callback,
  - prompt-source additions (`prompts_path`, synthetic prompt generation),
  - per-step host JSONL output support and leader metric aggregation.
- 2026-02-08: Added M12 run config:
  - `config/simpo_ultrafeedback_llama3_8b_v5p_32_m12_train200_infer1024_every50.yaml`
  - key settings:
    - `trainer.num_train_steps: 200`
    - `inference_eval.eval_at_steps: [50, 100, 150, 200]`
    - `inference_eval.inference_mode: host_data_parallel`
    - `inference_eval.synthetic_prompt_count: 1024`
    - `inference_eval.max_new_tokens: 2048`

## Hypotheses

1. M11 and M12 can share the same callback machinery; only schedules and prompt counts differ.
2. Periodic inference at fixed boundaries (`50, 100, 150, 200`) is cleanly expressible with explicit step triggers.
3. Large prompt sets are best configured via generated prompts or file-based prompt lists instead of giant inline YAML lists.

## Pending

- Execute long multi-host TPU runtime validation for the M12 schedule.
- Capture end-to-end timings and per-interval inference metrics for steps `50/100/150/200`.

## Validation (Code-Level)

Completed:
- Shared callback/config tests:
  - `uv run pytest tests/test_train_simpo_inference_eval.py tests/test_simpo.py`
  - result: `14 passed`.
- Full repository gates:
  - `./infra/pre-commit.py --all-files`
  - result: all checks passed.
- Config decode sanity:
  - `simpo_ultrafeedback_llama3_8b_v5p_32_m12_train200_infer1024_every50.yaml` decodes into `TrainSimpoConfig`.

Outstanding runtime validation:
- Long multi-host TPU run (`200` steps with inference at `50/100/150/200`) has not yet been executed in this local implementation pass.
