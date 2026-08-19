---
topic: moe-tasktrove-v326
issue: https://github.com/marin-community/marin/issues/8449
description: Qwen3.6 and Gemma 4 TaskTrove v3.26 evaluation campaign
author: benjaminfeuer
---

# MoE TaskTrove v3.26: Task Logbook

## Current TL;DR

The 564-run campaign is in smoke validation. Harbor PR #83 supplies the Pi hosted-vLLM endpoint path. Four one-task smokes are being iterated until both models produce scored Terminus-2 and Pi output; the eight-job campaign queue remains closed.

## Scope

- Goal: run 141 TaskTrove v3.26 datasets with two models and two Harbor harnesses, materializing 300 tasks per cell.
- Primary metrics: pass@1 over scoreable trials, scoreable trial count, infrastructure exception count, decode throughput, and agent-timeout incidence.
- Constraints: 32,768-token context; 16,384-token output allowance; prior Qwen3-Coder rollout policy except measured MoE serving/concurrency changes; at most eight active Iris jobs; at least 271 scoreable trials per completed cell.
- DRI: Benjamin Feuer.
- Coordinating issue: https://github.com/marin-community/marin/issues/8449
- Campaign tracker: `/Users/benjaminfeuer/Documents/experiments/active/moe-data-quality/TRACKER.md`
- W&B: not used for Harbor evaluation jobs.
- Checkpoints: not applicable.

## Baseline

- Date: 2026-08-19
- Marin code: `be0f70b169` (`agent/moe-tasktrove-v326`, based on `origin/main`)
- OpenThoughts-Agent code: `c5cccf33d79511172ac910f36385007a123d5aa1` (`penfever/working`)
- Harbor code: `41f4320c0471ea3362a6d3160df8b6c75f0126f7` (`main`, includes PR #83)
- TaskTrove: `open-thoughts/TaskTrove@6ac7c547ee2a8108836887e6530eb7dddf02dd9a` (latest v3.26 revision at integration time)
- Qwen weights observed at HF revision: `995ad96eacd98c81ed38be0c5b274b04031597b0`
- Gemma weights observed at HF revision: `24548b62aa021d562695c04aaf7758a1ea47990b`
- Prior campaign: `/Users/benjaminfeuer/Documents/experiments/active/qwen3-coder-data-quality`
- Historical Qwen3.6 serving signal: decode occupied 99.5–99.8% of request time and AgentTimeoutError affected 40–76% of trials on a B200 profile; see https://echo.oa.dev/wiki/50.

## Decision Log

- `MDQ-001`: use the latest v3.26 TaskTrove commit, including the duplicate `verifier.env` repairs recorded at https://echo.oa.dev/wiki/182.
- `MDQ-002`: keep the prior 16,384-token output allowance. A smaller allowance confounded the historical Qwen3.6 comparison.
- `MDQ-003`: run Pi through CoreWeave controller ingress because its installed CLI executes inside Daytona; keep Terminus-2 on the worker-local vLLM endpoint.
- `MDQ-004`: require decode-rate and timeout evidence from smokes before fixing per-harness concurrency.
- `MDQ-005`: keep no more than eight campaign jobs in Iris states PENDING, BUILDING, or RUNNING.

## Entry Log

### 2026-08-19 13:49 UTC - Bootstrap campaign record

- Hypothesis: Qwen3.6-35B-A3B and Gemma-4-26B-A4B can complete the prior TaskTrove rollout contract on one H100x8 Iris worker per cell; TP/DP and Harbor concurrency may need model-specific adjustment.
- Commit Hash: pending initial logbook commit.
- Command: no launch command executed.
- Config: 141 datasets × 2 models × 2 harnesses; 300 tasks; context 32,768; output 16,384; eight active jobs.
- Result: Harbor PR #83 merged and local Harbor advanced to `41f4320c0471`. TaskTrove advanced from the tracker's older v3.26 pin `a9c9bd35cb4f` to `6ac7c547ee2a`; the tracker and manifest must be regenerated before launch.
- Interpretation: the prior Harbor blocker is resolved. Model registration, durable campaign artifacts, and four smokes remain.
- Next action: register Gemma serving, create the harness configs and v3.26 manifest, then dry-run and submit one-task smokes.

### 2026-08-19 14:08 UTC - Pin launch configuration and build manifest

- Hypothesis: Qwen TP2×DP4 and Gemma TP1×DP8 will use all eight H100s while preserving the prior 32k/16k rollout contract; four one-task smokes will determine whether either topology or 32-way Harbor concurrency needs adjustment.
- Commit Hash: OpenThoughts-Agent `6b57e1a31c` on pushed branch `penfever/working`; Harbor `41f4320c0471`.
- Command: dry-ran `artifacts/launch_cell.py <qwen36|gemma4> <terminus-2|pi> 1 --smoke --dry-run` for all four combinations. Planned mirror command: `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python -m scripts.iris.launch_mirror hf-to-gcs --cluster marin --tpu v6e-4 --priority interactive --no-wait --output-mode gcs --gcs-output-dir gs://marin-us-central2/tmp/ttl=14d/ot-agent/model-mirrors/benjaminfeuer --secrets-env <filtered-secrets> --repo Qwen/Qwen3.6-35B-A3B --repo google/gemma-4-26B-A4B --gcs-prefix gs://marin-models-us/ot-agent/models --job_name mdq-tasktrove-v326-model-mirrors`.
- Config: Terminus-2 and Pi share 32 concurrent trials, one attempt, 7,200-second agent timeout, 14,400-second verifier timeout, Daytona 2 CPU/4 GiB/4 GiB, and 32,768 input/16,384 output limits. Pi is pinned to 0.73.1 with model-specific `qwen-chat-template` or `chat-template` thinking format. Qwen uses TP2×DP4; Gemma uses TP1×DP8. Both retain the prior Qwen3-Coder server sampling defaults (temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05).
- Result: all four dry runs resolved controller ingress and exact pinned TaskTrove selectors. The manifest contains 141 datasets, 564 grid cells, and 169,200 attempts; seven datasets require deterministic round-robin expansion. Both models are absent from `gs://marin-models-us/ot-agent/models`.
- Interpretation: no further Harbor changes are required. Model mirroring is the only launch prerequisite remaining before the four smoke jobs.
- Next action: mirror both model repositories once, verify cache hits, then submit and monitor all four smoke jobs.

### 2026-08-19 14:07 UTC - Advance OTA Harbor runtime pin

- Hypothesis: the merged Harbor source alone is insufficient unless OTA's frozen worker environment and baked images resolve the same commit.
- Commit Hash: OpenThoughts-Agent `30aec355` on pushed branch `penfever/working`; Harbor `41f4320c0471`.
- Command: `uv lock --upgrade-package harbor`; `uv run pytest -q tests/unit/agents/installed/test_pi.py` in the Harbor checkout.
- Config: `uv.lock` and every `docker/Dockerfile.*` `HARBOR_COMMIT` pin now resolve `41f4320c0471ea3362a6d3160df8b6c75f0126f7`.
- Result: Harbor's Pi unit suite passed 22/22. The mirror worker had shown OTA's previous lock resolving pre-PR commit `772e20f7`; the new pushed OTA revision removes that drift before any eval launch.
- Interpretation: smoke workers built from `30aec355` will contain the hosted-vLLM Pi endpoint implementation. The already-running mirror job is unaffected because it does not execute Harbor.
- Next action: wait for `/benjaminfeuer/mdq-tasktrove-v326-model-mirrors`, confirm both regional cache hits, then launch the four smokes from OTA `30aec355`.

### 2026-08-19 14:40 UTC - Repair worker ingress argument wiring

- Hypothesis: the four smoke workers exited before vLLM because the outer Iris launcher forwarded controller-ingress arguments that the local eval parser did not register.
- Commit Hash: OpenThoughts-Agent `37131ce1` on pushed branch `penfever/working`.
- Command: `uv run pytest -q tests/eval/test_local_ingress_args.py tests/hpc/test_ingress_wiring.py`.
- Config: `EvalRunner.create_parser()` now registers the shared `record_literal`, `ingress_mode`, and `ingress_host` argument group already consumed by `LocalHarborRunner`.
- Result: the new regression test failed before the fix on `--ingress_mode controller --ingress_host https://iris.oa.dev`; the focused parser and ingress suites pass 33/33 after the fix. The original four smoke tasks used OTA `30aec355` and reached `run_eval.py`, where they exited on the unknown arguments before vLLM startup.
- Interpretation: the failure is an OTA launcher/worker interface mismatch, not a model, topology, CoreWeave, or Harbor endpoint failure. The original smoke attempts are invalid and must be replaced using the same job names from `37131ce1`.
- Next action: stop the retrying invalid smoke jobs, mark them `RETRY REQUIRED`, and relaunch all four from OTA `37131ce1`.

### 2026-08-19 14:44 UTC - Remove unsupported vLLM text-only flag

- Hypothesis: after repairing the worker parser, vLLM 0.16 failed before model load because `--language-model-only` is not available in that installed release.
- Commit Hash: OpenThoughts-Agent `775ce0a0` on pushed branch `penfever/working`.
- Command: parsed both campaign datagen configs with `parse_datagen_config`; reran `uv run pytest -q tests/eval/test_local_ingress_args.py tests/hpc/test_ingress_wiring.py`.
- Config: removed only `--language-model-only`; `limit_mm_per_prompt={"image":0,"video":0}` remains in the model registry, and every context, output, sampling, topology, and harness setting is unchanged.
- Result: both model configs parse with their intended TP/DP layouts and supported vLLM arguments; focused tests pass 33/33. The second smoke attempts reached `api_server.py` and failed on the unsupported flag before model loading.
- Interpretation: these attempts also contain no model or harness evidence. Removing the optional text-only optimization is the smallest compatible change for vLLM 0.16.
- Next action: stop the health-check loops from OTA `37131ce1`, mark all four smokes for retry, and relaunch from `775ce0a0`.

### 2026-08-19 14:48 UTC - Upgrade model-serving runtime

- Hypothesis: Qwen3.6 and Gemma 4 both require a newer model architecture registry than vLLM 0.16 with Transformers 4.57.3.
- Commit Hash: OpenThoughts-Agent `ffe7ba3b` on pushed branch `penfever/working`.
- Command: `uv lock --upgrade-package vllm --upgrade-package transformers`; `uv run --extra datagen` config-only `AutoConfig.from_pretrained` validation for both exact model IDs; focused ingress tests.
- Config: GPU datagen now resolves vLLM 0.19.1 and Transformers 5.15.1. Official Qwen3.6 guidance recommends vLLM >=0.19. All rollout and model topology settings remain unchanged.
- Result: Transformers recognizes `Qwen/Qwen3.6-35B-A3B` as `Qwen3_5MoeConfig` and `google/gemma-4-26B-A4B` as `Gemma4Config`; focused tests pass 33/33. The third smoke attempts failed before weight load because the older runtime did not recognize either architecture.
- Interpretation: a serving runtime upgrade is required for both requested models; this is not a Harbor code change. The next smoke attempts will provide the first evidence about weight loading and TP/DP viability.
- Next action: stop the old health-check loops, mark all four smokes for retry, and relaunch from OTA `ffe7ba3b`.

### 2026-08-19 14:55 UTC - Remove retired vLLM swap option

- Hypothesis: vLLM 0.19.1 recognizes both requested architectures but removed the `--swap-space` CLI option still supplied by the two new model profiles and datagen configs.
- Commit Hash: OpenThoughts-Agent `a08b0368` on pushed branch `penfever/working`.
- Command: regenerated `eval/configs/model_configs.yaml`; ran `uv run pytest -q tests/eval/test_model_registry_resolve.py tests/eval/test_local_ingress_args.py tests/eval/test_per_model_agent_kwargs.py`; dry-ran Qwen/Terminus-2 and Gemma/Pi launch paths.
- Config: removed `swap_space` only from the Qwen3.6 and Gemma 4 H100 eval profiles and campaign datagen configs. The Qwen GH200 variant remains unchanged. All rollout, topology, memory-utilization, context, output, and sampling settings remain unchanged.
- Result: the fourth smoke attempts reached the vLLM 0.19.1 parser and failed uniformly on `api_server.py: error: unrecognized arguments: --swap-space`, before weight loading. The focused suites pass 15/15, generated registry matches the canonical profiles, and dry-run transparency output no longer includes `swap_space`. All four invalid health-check loops were stopped.
- Interpretation: the runtime upgrade is active and resolved the architecture blocker. This failure is another OTA serving-profile mismatch, not a Harbor change or model/harness result.
- Next action: relaunch the four smokes from `a08b0368`, then require successful model health checks and numeric Harbor results before opening the campaign queue.

### 2026-08-19 15:02 UTC - Align GPU runtime and controller ingress dependencies

- Hypothesis: vLLM 0.19.1 expects the Transformers 5.5 generation, while OTA's open lower bound admitted Transformers 5.15.1; the GPU datagen extra also omitted the Iris client needed after a controller endpoint becomes healthy.
- Commit Hash: OpenThoughts-Agent `71586e0b` on pushed branch `penfever/working`.
- Command: resolved the GPU datagen lock with Transformers `5.5.3`; imported `iris.cluster.client.endpoint_client.EndpointClient` from `uv run --extra datagen`; loaded both exact HF configs; parsed the Qwen campaign datagen config; ran the focused 15-test eval suite.
- Config: GPU datagen now bounds Transformers at `5.5.3`, includes the already-pinned Marin Iris client, and uses the Triton GDN prefill backend for Qwen because the runtime image has no `nvcc` for FlashInfer JIT warmup. All rollout and topology settings are unchanged.
- Result: the fifth Gemma attempts failed before weight loading with `AmbiguousGlobalPerLayerAttributeError` on heterogeneous `head_dim` under Transformers 5.15.1. Transformers 5.5.3 recognizes both exact models and reads Gemma's 256/512 local/global head dimensions. Qwen TP2×DP4 loaded all 26 shards on every DP rank, used 9.93 GiB per GPU, completed compilation, and returned HTTP 200 from `/v1/models`; the run then failed while registering controller ingress because `iris` was absent. FlashInfer GDN warmup also reported missing `/usr/local/cuda/bin/nvcc`, so the explicit Triton backend avoids first-request autotuning risk.
- Interpretation: both model-serving paths have passed architecture parsing, and Qwen has passed full endpoint startup. The remaining failures are OTA environment composition, not Harbor code or model capacity.
- Next action: relaunch all four smokes from `71586e0b` and require scored results from the consistent runtime before opening the campaign queue.

### 2026-08-19 15:20 UTC - Split harness routing after valid Pi smokes

- Hypothesis: Pi needs controller ingress from its Daytona-hosted CLI, while Terminus-2 should use the direct co-located vLLM endpoint because its LLM client runs in the Iris worker process.
- Commit Hash: OpenThoughts-Agent `71586e0b`; experiment launcher changed locally without an OTA code change.
- Command: refreshed `artifacts/supervise.py`, stopped the two invalid Terminus-2 smoke jobs, and relaunched them under fresh `mdq-smk7-*` identities after removing controller-ingress arguments only from the Terminus-2 launch path.
- Config: Pi retains `--ingress_mode controller --ingress_host https://iris.oa.dev`; Terminus-2 receives the worker-local `http://127.0.0.1:8000/v1` endpoint metadata. Model, sampling, context, output, Daytona, topology, and concurrency settings are unchanged.
- Result: Qwen/Pi and Gemma/Pi each completed one scoreable trial. Both model servers started successfully under the sixth attempts. Terminus-2 requests sent through controller ingress returned HTTP 401, while the same capability routes worked for Pi. A corrected direct-endpoint Gemma run reached HTTP 200 locally, then its old run directory rejected the changed runtime lock; the replacement `mdq-smk7-*` identities avoid that stale invalid-smoke state.
- Interpretation: no additional Harbor change is required. The 401 was a harness-routing error, and the subsequent lock mismatch came from reusing an invalid smoke identity across a semantic endpoint change.
- Next action: require one scoreable result from each fresh Terminus-2 smoke, then open eight full-grid jobs.

### 2026-08-19 15:48 UTC - Pass the four-way smoke gate and open the grid

- Hypothesis: Gemma 4 base can serve Terminus-2 conversations with the canonical instruction-model chat template supplied explicitly, without a Harbor code change.
- Commit Hash: OpenThoughts-Agent `79a560b4` on pushed branch `penfever/working`; Harbor `41f4320c0471`.
- Command: launched `/benjaminfeuer/mdq-smk10-gemma4-t2-001-nl2bash-tasks-cleaned-oracle`, then ran `artifacts/campaign_loop.py --interval 30` after all four smoke records became scoreable.
- Config: `google/gemma-4-26B-A4B` uses the canonical `google/gemma-4-26B-A4B-it` chat template at `/app/eval/configs/gemma4_chat_template.jinja`. The requested base-model weights, 32,768-token context, 16,384-token output allowance, TP1×DP8 topology, and all rollout settings remain unchanged.
- Result: Qwen/Terminus-2, Qwen/Pi, Gemma/Terminus-2, and Gemma/Pi each completed one scoreable trial. The corrected Gemma/Terminus-2 endpoint loaded all eight data-parallel engines, returned HTTP 200, and completed with a numeric verifier result. The supervisor then submitted exactly eight full cells: dataset rows 1 and 2 across both models and both harnesses.
- Interpretation: the smoke gate passed with no Harbor code change. The earlier Gemma failure was missing tokenizer presentation metadata in the requested base checkpoint, resolved in OTA model configuration.
- Next action: keep at most eight full cells active, resume infrastructure failures in place, and require at least 271 numeric verifier outcomes before marking each cell complete.
