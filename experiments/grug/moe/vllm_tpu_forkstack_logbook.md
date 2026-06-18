# GrugMoE TPU/vLLM Fork-Stack Forward-Port Logbook

Date: 2026-06-18

## Branches and SHAs

- Marin: `codex/grugmoe-forkstack-forward-port-20260618`
  - Base: `aa6a44e77`
  - Current: `2eb1c0a28d0cd1012d1a0a49c5c1152f17cdf25a`
  - Commits:
    - `e19252b29` Forward-port GrugMoE TPU vLLM support
    - `56a070012` Pin TPU routed experts fix
    - `2eb1c0a28` Pin vLLM routed experts scheduler fix
- tpu-inference: `codex/grugmoe-forkstack-tpu-20260618`
  - Base: `4c38590e29d33451f2b4375a78ad8f70cacd7b98`
  - Current: `94267fa1daa7779f4e41fa486d0b51a44b39c010`
  - Commits:
    - `2d5e1b49` Register GrugMoE for TPU vLLM resolution
    - `7a51b483` Add GrugMoE HF config aliases
    - `f57e481f` Adapt GrugMoE head to PR5 JAX layers
    - `103cbfdb` Add GrugMoE routing debug hook
    - `1448a897` Add GrugMoE forward debug probes
    - `79ca48f3` Mask GrugMoE dense attention by request
    - `94267fa1` Return TPU routed experts via vLLM output
- vLLM: `codex/grugmoe-forkstack-vllm-20260618`
  - Base: `04321f0f01b8e60d577a85a1b1af688e4433e7e0`
  - Current: `43e648985619749e7d1b232fbe408c7f5995d2d6`
  - Commit:
    - `43e648985` Ignore runner routed experts when disabled

## Local Validation

- Marin:
  - `python -m py_compile` on the GrugMoE model, vLLM parity, installed vLLM smoke, HF checkpoint export, and Marin export wrapper paths passed.
  - `env VLLM_TARGET_DEVICE=tpu uv lock --check` passed after both dependency pin updates.
  - `env -u PYTHONPATH VLLM_TARGET_DEVICE=tpu UV_LOCK_TIMEOUT=1200 uv run --locked --package marin-core --extra vllm --extra eval python -m experiments.grug.moe.installed_vllm_full_canary_smoke --help` passed.
  - `env -u PYTHONPATH VLLM_TARGET_DEVICE=tpu UV_LOCK_TIMEOUT=1200 uv run --locked --package marin-core --extra vllm --extra eval python -m experiments.grug.moe.vllm_tpu_parity --help` passed.
  - CPU component parity passed: native `GrugMoeMLP` matched Levanter `moe_mlp`.
  - CPU realistic small-diagnostic roundtrip passed with single-file and sharded HF artifacts, native tpu-inference load, routed expert IDs, logits, and 1-token greedy generation.
  - `./infra/pre-commit.py` passed ruff, black, license, Python AST, Markdown, trailing-whitespace, and EOF checks on touched files. Pyrefly failed on unrelated existing errors in `simple_evaluator.py` and `async_vllm.py`.
- tpu-inference:
  - `python -m py_compile tpu_inference/runner/tpu_runner.py tpu_inference/models/jax/grugmoe.py tests/models/common/test_model_loader.py` passed.
  - `git diff --check` passed.
  - `uv run ruff ...` could not run locally because the checkout environment did not provide ruff.
- vLLM:
  - `python -m py_compile vllm/v1/core/sched/scheduler.py tests/v1/core/test_scheduler.py` passed.
  - `git diff --check` passed.
  - `uvx ruff check vllm/v1/core/sched/scheduler.py tests/v1/core/test_scheduler.py` passed.
  - `uvx ruff format --check vllm/v1/core/sched/scheduler.py tests/v1/core/test_scheduler.py` passed.
  - Full pytest could not run locally: bare Python lacked `torch`, `numpy`, and `pytest`; `uv run` attempted to build vLLM and failed before tests because `CUDA_HOME` was unset.

## TPU Jobs

- Small diagnostic, initial:
  - Job: `/romain/grugmoe-forkstack-small-diagnostic-20260618-015457`
  - TPU: `v6e-4`, region `europe-west4`
  - Marin: `e19252b29`; tpu-inference: `79ca48f3`; vLLM: `04321f0f`
  - Result: failed after export/reference and vLLM logprob checks passed.
  - Localized failure: installed vLLM did not return `routed_experts`; runner-side debug had matching top-k routing, so the issue was routed-expert propagation from tpu-inference into vLLM outputs.
- Small diagnostic, rerun:
  - Job: `/romain/grugmoe-forkstack-small-diagnostic-rerun-20260618-020641`
  - TPU: `v6e-4`, region `europe-west4`
  - Marin: `56a070012`; tpu-inference: `94267fa1`; vLLM: `04321f0f`
  - Result: succeeded.
  - Key checks:
    - Manual-copy native reference matched Levanter hidden states, logits, and routed experts.
    - Sharded training-state export loaded in native tpu-inference and matched Levanter/manual-copy hidden states, logits, and routed expert IDs.
    - `score_continuation_generated_token_ids=[3045, 2536, 3106]`
    - `logprob_summary.max_abs_delta=0.0019140243530273438` with allowed delta `5.0`.
    - Routing summary had `top1_match_rate=1.0`, `ordered_topk_match_rate=1.0`, `mismatch_count=0`, and `routing_mismatch_locations=[]`.
    - `installed_path_result=works:full_canary_logprobs_and_routing`.
- Full canary, initial:
  - Job: `/romain/grugmoe-forkstack-full-canary-20260618-021336`
  - TPU: `v6e-4`, region `europe-west4`
  - Marin: `56a070012`; tpu-inference: `94267fa1`; vLLM: `04321f0f`
  - Result: failed in the serve phase after full-canary export/reference checks passed.
  - Passed before failure:
    - Manual-copy native reference matched Levanter hidden states, logits, and routed experts.
    - Manual-copy full-forward greedy generation matched Levanter for 3 steps.
    - Generated IDs were `[57524, 45040, 67859]`.
    - HF export saved 26 shards, 5.76 GB total, with max shard size about 525 MB.
  - Localized failure: vLLM server path received runner-supplied `routed_experts` while `enable_return_routed_experts` was disabled, and `AsyncScheduler` raised `AttributeError: 'AsyncScheduler' object has no attribute 'routed_experts_mgr'`. Fixed in vLLM `43e648985`.
- Full canary, rerun:
  - Job: `/romain/grugmoe-forkstack-full-canary-rerun-20260618-023108`
  - TPU: `v6e-4`, region `europe-west4`
  - Marin: `2eb1c0a28`; tpu-inference: `94267fa1`; vLLM: `43e648985`
  - Result: succeeded with one TPU preemption and automatic task restart.
  - Key checks:
    - Patched vLLM wheel built and loaded as `0.20.1rc1.dev993+g43e648985.tpu`.
    - Full-canary manual-copy native reference matched Levanter hidden states, logits, and routed experts.
    - Manual-copy full-forward greedy generation matched Levanter for 3 steps.
    - Reference generated IDs were `[57524, 45040, 67859]`.
    - HF export saved 26 shards, 5.76 GB total, with max shard size about 525 MB.
    - vLLM server generation returned HTTP 200 and `installed_path_result=works:full_canary_generate`.
    - Score path generated continuation IDs `[57524, 45040, 67859]`.
    - `logprob_summary.max_abs_delta=0.008803367614746094` with allowed delta `5.0`.
    - Routing summary had `top1_match_rate=1.0`, `unordered_full_match_rate=0.9752066115702479`, `ordered_topk_match_rate=0.9256198347107438`, `suspicious_mismatch_count=0`, and `low_margin_boundary_mismatch_count=3`.
    - `installed_path_result=works:full_canary_logprobs_and_routing`.

## Caveats

- The small diagnostic parity path is passing on real TPU.
- Full canary passed on real TPU after pinning the vLLM async scheduler fix.
- Remaining validation gap before landing is CI that covers the full vLLM test environment.
