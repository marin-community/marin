# Iris Inference Service Pressure Test Findings

## Latest Status 2026-05-13 17:48 PT

- PR #5400 branch `design/iris_inference_service` is locally rebased on PR
  #5712 commit `323805384b7ac0e393f34a9864a0dc6b96050951`. Nothing was pushed.
- Local commits on top of #5712 now are:
  - `d5f779011` Add Iris vLLM inference service design
  - `a972bb54b` Document Iris inference pressure-test findings
  - `045c8ab49` Update Iris inference MVP pressure-test scope
  - `048192e25` Pass lm-eval generation params to harness
- The branch tracks `origin/main`, so `git status` reports ahead/behind against
  `origin/main` rather than against #5712. This is expected until the PR is
  retargeted or #5712 lands.
- `report.md` remains untracked and untouched. `tmp/` remains untracked.

## PR Update Status 2026-05-14 14:23 PT

- PR #5712 merged into `main` on 2026-05-14 at 17:03 UTC.
- PR #5400 has been updated:
  - title: `Design Iris inference service`
  - base: `main`
  - head: `8c652c5e3cde63ffc3e1ad5658058775d76181fe`
  - draft: true
  - mergeable: true
- The local branch was rebased onto current `origin/main`, dropping the old
  pre-merge #5712 commit from the stack.
- Current PR stack is four commits:
  - `572873f1a` Add Iris vLLM inference service design
  - `4250fc83a` Document Iris inference pressure-test findings
  - `277952a70` Update Iris inference MVP pressure-test scope
  - `8c652c5e3` Pass lm-eval generation params to harness
- Validation after rebase:
  - `./infra/pre-commit.py --all-files --fix`: passed
  - `uv run --package marin --extra eval --extra cpu pytest -q tests/evals/test_served_lm_eval.py -m 'not slow'`: 5 passed
- GitHub checks started after the push. At the first poll, several path-filter
  checks had succeeded/skipped and Marin lint/integration/unit plus CodeQL and
  ReadTheDocs were still running or pending.

## Successful MVP Smoke On Rebased Stack

- Iris job:
  `/romain/iris-inference-standard-smoke-20260514T003544Z`
- Result: `JOB_STATE_SUCCEEDED`, exit code 0, duration 147s, worker
  `marin-tpu-v6e-preemptible-4-us-east5-b-20260514-0026-237647e2-worker-0`.
- Command shape: `Qwen/Qwen3-0.6B` served by native vLLM on `v6e-4`, stock
  lm-eval `humaneval`, `limit=1`, `apply_chat_template=True`,
  `max_model_len=1024`, `max_num_batched_tokens=1024`, and
  `generation_params={"max_gen_toks": 128}`.
- GCS artifacts:
  - `gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_standard_smoke_20260514T003544Z/humaneval_0shot/Qwen__Qwen3-0.6B/results_2026-05-14T00-39-54.767392.json`
  - `gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_standard_smoke_20260514T003544Z/humaneval_0shot/Qwen__Qwen3-0.6B/samples_humaneval_2026-05-14T00-39-54.767392.jsonl`
- Local copied result:
  `tmp/iris_pressure_test_20260504/rebased_standard_smoke_results_20260514T003544Z.json`
- Result details: saved lm-eval config reports `num_fewshot: 0`,
  `generation_kwargs.max_gen_toks: 128`,
  `base_url: http://127.0.0.1:8000/v1/chat/completions`,
  `tokenizer_backend: huggingface`, and `tokenized_requests: false`.
- Metric from the one-sample smoke was `pass@1,create_test = 0.0`. This is not
  quality evidence; it is endpoint/lifecycle evidence.

## New Finding From The Rebased Smoke

- Marin's lm-eval harness was already carrying `EvaluationConfig.generation_params`
  into `ModelConfig`, but `LMEvaluationHarnessEvaluator` did not pass those
  params to lm-eval. As a result, stock HumanEval's task YAML
  `generation_kwargs.max_gen_toks: 1024` overrode the model default and caused
  vLLM to reject a 1024-output-token request against `max_model_len=1024`.
- Commit `048192e25` fixes that by passing
  `gen_kwargs=model.generation_params or None` to `simple_evaluate`.
- The scratch runner now keeps vLLM engine settings in `engine_kwargs` and puts
  the output cap in `generation_params`.
- The successful smoke confirms this is the right plumbing: lm-eval logged that
  `generation_kwargs: {'max_gen_toks': 128}` updated the task YAML, and the
  `/v1/chat/completions` request completed.

## Current Recommendation

- Open/update the RFC for review on the rebased branch with #5712 as its base
  while #5712 is open, or wait for #5712 to merge and then retarget to `main`.
- Do not block RFC review or first service implementation on #5672. The current
  validated MVP lane is standard HF model + vLLM on Iris + OpenAI chat
  completions + lm-eval generation.
- Treat 1e22 readiness and `mmlu_sl_verb_5shot` prompt-logprob scoring as
  follow-up compatibility gates. The pressure test still shows that
  completions/logprob scoring is the severe remaining eval-path risk.

## Refresh Status 2026-05-13

- PR #5400 is still open, draft, and unchanged since 2026-05-04. It has one
  commit, three changed files, no review comments, no review submissions, and a
  green ReadTheDocs status on commit `69569a5d268af83024cafc17410907f04cfe5900`.
- Local branch `design/iris_inference_service` is still at
  `69569a5d268af83024cafc17410907f04cfe5900`. It tracks `origin/main`, is
  ahead by 1 and behind current `origin/main` by 157 commits, and also matches
  `origin/design/iris_inference_service`.
- The local pressure-test doc edits are still uncommitted:
  `.agents/projects/iris_inference_service/design.md`,
  `.agents/projects/iris_inference_service/spec.md`, and
  `.agents/projects/iris_inference_service/research.md`. Current diff size is
  138 insertions and 17 deletions across those files.
- `report.md` remains untracked and untouched. `tmp/` remains untracked.
- Fresh Iris query for prefix
  `/romain/iris-run-run_existing_vllm_lm_eval` returned `[]`; no old pressure
  test jobs are listed now. Direct summary for
  `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030623` now returns
  `Job ... not found`, so the previously pending canonical job is no longer in
  the live controller job set.
- Current `origin/main` still pins `vllm-tpu==0.18.0`,
  `tpu-inference==0.18.0`, and `libtpu==0.0.38` in `lib/marin/pyproject.toml`.
- New relevant PR #5712 is open and mergeable. It bumps the TPU vLLM lane to
  `vllm-tpu==0.19.0`, `tpu-inference==0.19.0`, and `libtpu==0.0.39`, with
  focused local tests and a Grug TPU smoke reported in the PR body.
- New blocker #5672 is open for the Delphi/1e22 vLLM RPA scoped-VMEM failure.
  The latest comment reproduces the issue on
  `gs://marin-us-east5/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf/step-38234/`
  with the 0.19 stack. It fails during vLLM startup/precompile before server
  readiness with `RESOURCE_EXHAUSTED` in scoped VMEM for kernel
  `RPAm-p_256-bq_256_512-bkv_1024_512`.

### 2026-05-13 Revised Recommendation

- Yes, use #5712 to unblock. For local implementation and validation, rebase or
  branch from `codex/vllm-tpu-019` (`323805384b7ac0e393f34a9864a0dc6b96050951`)
  so the service work uses the intended `vllm-tpu==0.19.0`,
  `tpu-inference==0.19.0`, and `libtpu==0.0.39` lane.
- If PR #5400 is pushed while #5712 is still open, either retarget #5400 to
  `codex/vllm-tpu-019` or keep the branch local until #5712 lands. Rebasing on
  #5712 while leaving the PR base as `main` would make GitHub show #5712's
  dependency diff inside the design PR.
- Do not block the first inference-service implementation on #5672. Ship the
  service against a standard, vLLM-friendly HF model first. Use that milestone
  to validate lifecycle, readiness, proxy/broker behavior, OpenAI-compatible
  endpoint plumbing, cleanup, result upload, and normal CI contracts.
- Treat #5672 as a documented model-family limitation and follow-up, not a
  launch blocker. The service design should make clear that Delphi/1e22 support
  is a later compatibility target once vLLM can reach readiness on that model.
- Split acceptance into two tracks:
  1. MVP acceptance: standard model on Iris through vLLM, with at least one
     chat/generation eval and the service cleanup/diagnostics path passing.
  2. Canonical/Delphi acceptance: `1e22` readiness, then
     `mmlu_sl_verb_5shot` prompt-logprob scoring once `/v1/completions`
     logprobs work on TPU.
- Update PR #5400 docs to avoid presenting the old 1e22 MMLU milestone as the
  blocking MVP gate. Keep it as the pressure-test that exposed follow-up risks,
  and name the first shipped target as a standard model with known vLLM support.

## Current Status

- Started from branch `design/iris_inference_service`, which is ahead of `origin/main` by one commit.
- Existing untracked `report.md` is intentionally untouched.
- Relevant skills and Iris operational docs were read.
- Submitted canonical MMLU vLLM pressure-test attempt: `/romain/iris-run-run_existing_vllm_lm_eval-20260504-022544`.
- Stopped the original canonical job after the fallback exposed an lm-eval/vLLM import mismatch that would have affected the queued canonical run.
- Latest live job as of 2026-05-04T00:06:00-0400: patched canonical v5p-8 MMLU attempt `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030623`, still pending on capacity.
- Final recent-job scan found no other running pressure-test jobs; all other recent eval jobs are succeeded, failed, or killed.
- The spec's implementation files do not exist on this branch. The pressure test must use existing vLLM/Iris eval paths or scratch/manual-only code.

## Preliminary Constraints

- Team direction is vLLM-only served inference.
- Do not add Levanter/JAX OpenAI HTTP support.
- Do not restart or bounce any Iris cluster.
- Keep experimental code scratch/manual-only unless design docs need a local patch.

## Open Questions To Resolve

- What exact repo config or history identifies the 1e22 MoE checkpoint/model path?
- Does vLLM on TPU actually support the completion/logprob shape needed by `mmlu_sl_verb_5shot`, or does it fail as existing comments suggest?
- What tokenizer/chat-template args are required for a 5-shot HumanEval run against the target checkpoint?

## Findings So Far

- `RunningModel`/`OpenAIEndpoint` is already a real decoupling boundary. `run_lm_eval` consumes only `RunningModel` and maps to lm-eval `local-completions` or `local-chat-completions`.
- Existing `VllmEnvironment` is the smallest practical serving path. It starts native `vllm serve`, waits for `/v1/models`, and exposes a `/v1` base URL.
- Existing executor helper `evaluate_lm_evaluation_harness` can launch the evaluator on Iris with `["eval", "vllm", "tpu"]` dependency groups and `ResourceConfig.with_tpu("v5p-8")`.
- The documented 1e22 model candidate is `adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf` under `MARIN_PREFIX` from `experiments/exp1337_eval_suite.py`.
- Existing `exp1337` routes generation tasks through vLLM but routes `mmlu_sl_verb_5shot` through Levanter. Its comment says multiple-choice tasks currently do not work on TPUs through vLLM.
- The explicit MoE artifact exists at `gs://marin-us-central2/grug/moe-v7-1e22-d3200-v4-56ba43/checkpoints/step-77725/`, but it is Orbax/OCDBT-shaped and has no `config.json` or tokenizer config. It is not directly vLLM-loadable from the current `VllmEnvironment` path.
- The closest vLLM-loadable stand-in found is `gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf/step-38234/`, which has HF `config.json` and tokenizer metadata.
- Scratch runner: `tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py`. It is manual-only and uses the existing `lm_evaluation_harness` evaluator.

## Canonical MMLU Attempts

- Task: `mmlu_sl_verb_5shot`
- Model path: `gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf/step-38234/`
- Reason for stand-in: exact 1e22 MoE checkpoint has no HF/vLLM export in the discovered paths.
- Iris job id: `/romain/iris-run-run_existing_vllm_lm_eval-20260504-022544`
- Submit command artifact: `tmp/iris_pressure_test_20260504/submit_mmlu_vllm_iris.out`
- Result: killed before scheduling.
- What ran: the job was accepted by Iris and sat pending on v5p-8 capacity for multiple polls.
- What did not run: no worker assignment, no vLLM process, no OpenAI endpoint, no lm-eval requests.
- Scheduler signal before stop: insufficient currently-free memory and TPUs on matching v5p-8 workers plus autoscaler `tier_blocked` by quota-pool tier monotonicity.
- Capacity context: captured Iris snapshots show many v5p-8 jobs already running or pending. The canonical run is queued behind real cluster load. A small-model fallback on a less congested TPU shape can validate Iris/vLLM/lm-eval mechanics, but it cannot validate the canonical 1e22 MoE/v5p-8 milestone.

### Patched Canonical Attempt

- Iris job id: `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030623`
- Patch included: scratch-only `vllm.utils.get_open_port` import shim.
- Status: `JOB_STATE_PENDING` as of 2026-05-04T00:06:00-0400.
- Pending reason: still insufficient current v5p-8 memory, and the autoscaler state regressed to quota-pool tier blocking. No worker has been assigned.

## Fallback Mechanics Probe

- Task: `mmlu_sl_verb_5shot`
- Model path: `Qwen/Qwen3-0.6B`
- Tokenizer: `Qwen/Qwen3-0.6B`
- Accelerator: `v6e-4`
- First Iris job id: `/romain/iris-run-run_existing_vllm_lm_eval-20260504-024730`
- Patched rerun job id: `/romain/iris-run-run_existing_vllm_lm_eval-20260504-025615`
- Larger-context rerun job id: `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030231`
- Larger-context lm-eval-max-length rerun job id: `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030919`
- Purpose: validate the same real vLLM plus real lm-eval local-completions/loglikelihood path under Iris when the canonical v5p-8 run is capacity-blocked.
- Limitation: this does not validate the 1e22 MoE artifact, the v5p-8 scheduler shape, or production-scale throughput.
- First result: failed before HTTP scoring. Root cause was `ImportError: cannot import name 'get_open_port' from 'vllm.utils'` when importing `lm_eval.evaluator`; `lm_eval.models.vllm_causallms` expects an older/different vLLM utility surface.
- Scratch-only mitigation: `tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py` installs a local `vllm.utils.get_open_port` shim before running lm-eval.
- Second result: patched rerun `/romain/iris-run-run_existing_vllm_lm_eval-20260504-025615` reached real lm-eval and real `/v1/completions` scoring. It failed on the first loglikelihood request after retries with HTTP 500 from vLLM: `{"message":"32658","type":"InternalServerError","code":500}`.
- Context-control result: `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030231` also failed with the same `/v1/completions` HTTP 500 body `32658`. It still logged lm-eval `max_length (2047)`, so `max_model_len=4096` alone is not enough for lm-eval local-completions.
- Final MMLU context-control result: `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030919` set `max_model_len=4096`, `max_num_batched_tokens=4096`, and lm-eval `max_length=4096`. The context warning disappeared, but the first loglikelihood request still failed after retries with HTTP 500 from `/v1/completions`, body `{"message":"2701","type":"InternalServerError","code":500}`.
- Interpretation: this falsifies the hypothesis that the MMLU failure was only a context-length mismatch. The blocker is vLLM TPU `/v1/completions` logprob scoring.
- Logs: controller log RPC works and shows the runner config plus Marin eval startup; regular `iris job logs` still fails with `finelog.errors.StatsError: Not Found`.

## Fallback HumanEval Probe

- Task: `humaneval_5shot`
- Model path: `Qwen/Qwen3-0.6B`
- Tokenizer: `Qwen/Qwen3-0.6B`
- Accelerator: `v6e-4`, pinned to `us-east1-d` for the successful run.
- Successful Iris job id: `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030528`
- Purpose: validate the real lm-eval `local-chat-completions` generation path separately from the MMLU/loglikelihood path.
- Result: succeeded in 147s on `marin-tpu-v6e-preemptible-4-us-east1-d-20260504-0154-b4091c92-worker-0`.
- Evidence: result metadata reports `model_source: local-chat-completions`, `base_url: http://127.0.0.1:8000/v1/chat/completions`, `apply_chat_template=True`, and one generated HumanEval sample.
- Metric from limit-1 probe: `pass@1,create_test = 0.0`. This is not quality evidence; the important signal is endpoint/task compatibility.
- Few-shot caveat: the saved results config reports `num_fewshot: 0`, and the saved prompt is the standard HumanEval prompt rather than five external few-shot examples. This validates the generation endpoint, not the exact `humaneval_5shot` milestone wording.
- Warning: W&B sample logging failed on the custom `pass_at_k` metric object, but local result writing and GCS upload succeeded.
- Redundant HumanEval jobs `/romain/iris-run-run_existing_vllm_lm_eval-20260504-031005` and `/romain/iris-run-run_existing_vllm_lm_eval-20260504-031612` were stopped to avoid unnecessary resource use after the pinned run succeeded.

## Stopped Attempts

- Stopped `/romain/iris-run-run_existing_vllm_lm_eval-20260504-022544`, the original canonical v5p-8 attempt, because it was submitted before the lm-eval/vLLM import shim. If it had scheduled, it would likely have consumed v5p-8 only to fail before HTTP scoring.
