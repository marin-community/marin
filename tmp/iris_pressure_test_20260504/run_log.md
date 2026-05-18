# Iris Inference Service Pressure Test Run Log

Append-only log for the May 4, 2026 pressure test of the Iris vLLM-only inference service design.

## 2026-05-03T22:11:00-0400

- Command: `sed -n '1,220p' .agents/skills/agent-research/SKILL.md`
- Why: Read the matching long-running research workflow before starting the pressure test.
- Outcome: Succeeded.
- Learned: Use a reproducible logbook-style workflow, exact commands, scoped conclusions, and clear handoff artifacts. The user explicitly requested local scratch logs instead of creating an issue or pushing snapshots.

## 2026-05-03T22:11:00-0400

- Command: `sed -n '1,220p' .agents/skills/debug-infra/SKILL.md`
- Why: The task involves real Iris job execution and debugging.
- Outcome: Succeeded.
- Learned: Read `lib/iris/AGENTS.md` and relevant `lib/iris/OPS.md` sections; avoid cluster mutations and use Iris diagnostic commands.

## 2026-05-03T22:11:00-0400

- Command: `sed -n '1,180p' .agents/skills/babysit-job/SKILL.md`
- Why: The pressure test may submit and monitor Iris jobs.
- Outcome: Succeeded.
- Learned: Monitor by job id, config, and exact resubmit command; use `--tpu`, not `--reserve`, for TPU hardware; collect `job logs`, `job list --json`, and `job summary --json` for failures.

## 2026-05-03T22:11:00-0400

- Command: `sed -n '1,180p' .codex/plugins/cache/openai-curated/github/3c463363/skills/github/SKILL.md`
- Why: The task references GitHub PRs for design context and prototype evidence.
- Outcome: Failed; path was relative to the repo and did not exist.
- Learned: The GitHub plugin skill path lives under `/Users/romain/.codex/plugins/cache/...`, not under this repository.

## 2026-05-03T22:11:00-0400

- Command: `git status --short --branch`
- Why: Establish current branch and identify existing user/agent changes before touching files.
- Outcome: Succeeded; branch is `design/iris_inference_service...origin/main [ahead 1]` with untracked `report.md` and `tmp/`.
- Learned: Leave `report.md` alone. Use only the requested scratch directory under `tmp/`.

## 2026-05-03T22:13:00-0400

- Command: `sed -n '1,220p' /Users/romain/.codex/plugins/cache/openai-curated/github/3c463363/skills/github/SKILL.md`
- Why: Read GitHub plugin guidance for PR/prototype inspection.
- Outcome: Succeeded.
- Learned: Prefer the GitHub connector for structured PR metadata and patch inspection; use local `gh` only for gaps.

## 2026-05-03T22:13:00-0400

- Command: `sed -n '1,220p' lib/iris/AGENTS.md`
- Why: Follow Iris-specific repository instructions before using Iris.
- Outcome: Succeeded.
- Learned: Use Iris RPC/CLI paths, avoid raw HTTP for Iris APIs, and pass env vars to jobs through Iris mechanisms rather than relying on the submitter environment.

## 2026-05-03T22:13:00-0400

- Command: `sed -n '1,220p' lib/iris/OPS.md`
- Why: Read operational guardrails for job execution, diagnostics, and troubleshooting.
- Outcome: Succeeded.
- Learned: Normal job submission/log/summary commands are allowed; never run `iris cluster restart` without explicit approval; use `--memory`, `--tpu`, and structured job summaries for failure analysis.

## 2026-05-03T22:13:00-0400

- Command: `mkdir -p tmp/iris_pressure_test_20260504`
- Why: Create the requested scratch artifact directory.
- Outcome: Succeeded.
- Learned: The target directory is available for logs and small local artifacts.

## 2026-05-03T22:14:57-0400

- Command: `date '+%Y-%m-%dT%H:%M:%S%z'`
- Why: Capture the local timestamp format for subsequent log entries.
- Outcome: Succeeded; output was `2026-05-03T22:14:57-0400`.
- Learned: The local machine is still on May 3 in `America/New_York`; the test artifact path follows the requested May 4 UTC/date label.

## 2026-05-03T22:16:00-0400

- Command: `sed -n '1,260p' .agents/projects/iris_inference_service/design.md`
- Why: Read the top-level design under review.
- Outcome: Succeeded.
- Learned: The design explicitly requires vLLM-only served inference, keeps eval decoupled through `RunningModel`/OpenAI HTTP, and names the canonical pressure test as `mmlu_sl_verb_5shot` plus `humaneval_5shot` on the 1e22 MoE via Iris `v5p-8`.

## 2026-05-03T22:16:00-0400

- Command: `sed -n '1,280p' .agents/projects/iris_inference_service/spec.md`
- Why: Read the implementation contract and expected manual runner surface.
- Outcome: Succeeded.
- Learned: The spec expects new files that are not necessarily present yet: `lib/marin/src/marin/inference/iris_vllm.py`, `tests/evals/test_iris_vllm_inference.py`, and `scripts/iris/run_vllm_eval_pressure_test.py`.

## 2026-05-03T22:16:00-0400

- Command: `sed -n '1,240p' .agents/projects/iris_inference_service/research.md`
- Why: Read prior findings and prototype claims before choosing commands.
- Outcome: Succeeded.
- Learned: Current in-repo primitives are `VllmEnvironment`, `run_lm_eval`, and the `RunningModel` boundary; PR #5351 is evidence for broker/proxy shape but not expected to be merged directly.

## 2026-05-03T22:16:00-0400

- Command: `rg -n "vLLM|vllm|lm[-_ ]eval|lm_eval|mmlu_sl|humaneval|RunningModel|OpenAIEndpoint|1e22|1e22 MoE|MoE" . > tmp/iris_pressure_test_20260504/rg_vllm_lmeval.txt 2>&1`
- Why: Broadly locate existing code/configs for vLLM, lm-eval, target tasks, model references, and served-model types.
- Outcome: Succeeded but produced 1,880 lines including scratch files and very long `uv.lock` markers.
- Learned: Need to narrow follow-up searches by excluding `uv.lock`, `tmp/**`, and generated/vendor-heavy files.

## 2026-05-03T22:18:00-0400

- Command: `sed -n '281,620p' .agents/projects/iris_inference_service/spec.md`
- Why: Finish reading the spec, especially launcher and manual pressure-test contract.
- Outcome: Succeeded.
- Learned: Manual runner should default to `mmlu_sl_verb_5shot` via `LOCAL_COMPLETIONS` and `humaneval_5shot` via `LOCAL_CHAT_COMPLETIONS` with `apply_chat_template=True`, and should record Iris job ids, vLLM `/v1/models` id, output directory, request counts, throughput, and failure summaries.

## 2026-05-03T22:18:00-0400

- Command: `wc -l tmp/iris_pressure_test_20260504/rg_vllm_lmeval.txt`
- Why: Size the broad search output before inspecting it.
- Outcome: Succeeded; output was `1880`.
- Learned: The broad result is too noisy for direct reasoning.

## 2026-05-03T22:18:00-0400

- Command: `sed -n '1,220p' tmp/iris_pressure_test_20260504/rg_vllm_lmeval.txt`
- Why: Inspect the head of the broad search result.
- Outcome: Succeeded but displayed very long `uv.lock` lines.
- Learned: The first useful source hits were drowned by generated lockfile content; future inspection should use scoped searches.

## 2026-05-03T22:18:00-0400

- Command: `rg --files | rg '(^|/)(lm_eval|vllm|inference|evaluat|eval|iris).*\\.(py|md|yaml|toml)$' > tmp/iris_pressure_test_20260504/rg_relevant_files.txt 2>&1`
- Why: Build a smaller candidate file list for eval/vLLM/Iris code and docs.
- Outcome: Succeeded.
- Learned: Candidate file list is available for focused reading.

## 2026-05-03T22:21:00-0400

- Command: `wc -l tmp/iris_pressure_test_20260504/rg_relevant_files.txt`
- Why: Size the candidate file list before inspecting it.
- Outcome: Succeeded; output was `460`.
- Learned: The filtered list is still broad but usable for selecting known files.

## 2026-05-03T22:21:00-0400

- Command: `sed -n '1,240p' tmp/iris_pressure_test_20260504/rg_relevant_files.txt`
- Why: Inspect the most relevant candidate files.
- Outcome: Succeeded.
- Learned: Key paths are `lib/marin/src/marin/evaluation/lm_eval.py`, `lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py`, `lib/marin/src/marin/inference/vllm_server.py`, `tests/evals/test_served_lm_eval.py`, and `experiments/exp1337_eval_suite.py`.

## 2026-05-03T22:21:00-0400

- Command: `rg -n "mmlu_sl_verb_5shot|humaneval_5shot|1e22|MoE|moe|v5p-8|vllm|VllmEnvironment|RunningModel|OpenAIEndpoint|LmEvalAdapter|run_lm_eval" lib experiments scripts .agents --glob '!uv.lock' --glob '!tmp/**' > tmp/iris_pressure_test_20260504/rg_scoped_core.txt 2>&1`
- Why: Create a scoped core search output excluding scratch and lockfiles.
- Outcome: Succeeded; output written to `tmp/iris_pressure_test_20260504/rg_scoped_core.txt`.
- Learned: Relevant hits include the design docs, `experiments/exp1337_eval_suite.py`, `VllmEnvironment`, and served lm-eval code.

## 2026-05-03T22:21:00-0400

- Command: `ls lib/marin/src/marin/inference/iris_vllm.py tests/evals/test_iris_vllm_inference.py scripts/iris/run_vllm_eval_pressure_test.py`
- Why: Check whether the spec's proposed implementation files already exist.
- Outcome: Failed for all three paths with "No such file or directory".
- Learned: The design branch has docs only for the Iris vLLM service; no broker/proxy/worker implementation or canonical runner exists yet.

## 2026-05-03T22:22:00-0400

- Command: `wc -l tmp/iris_pressure_test_20260504/rg_scoped_core.txt`
- Why: Size the scoped search output.
- Outcome: Succeeded; output was `857`.
- Learned: Still too large for direct full inspection; use targeted searches.

## 2026-05-03T22:22:00-0400

- Command: `rg -n "mmlu_sl_verb_5shot|humaneval_5shot" lib experiments scripts .agents --glob '!tmp/**'`
- Why: Find exact canonical task aliases.
- Outcome: Succeeded.
- Learned: `mmlu_sl_verb_5shot` is defined as an alias in `experiments/exp1337_eval_suite.py`; `humaneval_5shot` is only in the design/spec, while the existing suite uses `humaneval_10shot`.

## 2026-05-03T22:22:00-0400

- Command: `rg -n "1e22|MoE|moe|v5p-8" lib experiments scripts .agents --glob '!tmp/**' --glob '!uv.lock'`
- Why: Find the 1e22 MoE/model path and v5p-8 usage.
- Outcome: Succeeded but produced a large MoE-heavy output.
- Learned: The strongest target-model hit is `experiments/exp1337_eval_suite.py`, whose `DELPHI_OPTIMAL_RUNS["1e+22"]` is `adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf`; `scripts/ops/storage/purge/protect_rules.json` also protects `grug/moe-v7-1e22-d3200%`, suggesting another possible 1e22 MoE artifact.

## 2026-05-03T22:22:00-0400

- Command: `rg -n "VllmEnvironment|RunningModel|OpenAIEndpoint|LmEvalAdapter|run_lm_eval" lib/marin/src/marin lib/fray/src/fray tests/evals experiments/evals --glob '!tmp/**' > tmp/iris_pressure_test_20260504/rg_scoped_symbols.txt 2>&1`
- Why: Focus on the served-model and vLLM symbols used by the design.
- Outcome: Succeeded.
- Learned: Symbol references are concentrated in `marin.inference.types`, `marin.evaluation.lm_eval`, `VllmEnvironment`, and served lm-eval tests.

## 2026-05-03T22:24:00-0400

- Command: `sed -n '1,220p' lib/marin/src/marin/inference/types.py`
- Why: Verify the `RunningModel`/`OpenAIEndpoint` boundary.
- Outcome: Succeeded.
- Learned: The eval side can remain decoupled: `RunningModel` only carries `OpenAIEndpoint` plus optional tokenizer.

## 2026-05-03T22:24:00-0400

- Command: `sed -n '1,220p' lib/marin/src/marin/evaluation/lm_eval.py`
- Why: Inspect the new served lm-eval runner.
- Outcome: Succeeded.
- Learned: `run_lm_eval` uses lm-eval API models `local-completions` and `local-chat-completions`, constructing `base_url` as `{endpoint.base_url}/completions` or `{endpoint.base_url}/chat/completions`.

## 2026-05-03T22:24:00-0400

- Command: `sed -n '1,260p' lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py`
- Why: Inspect the existing vLLM evaluator used by executor steps.
- Outcome: Succeeded.
- Learned: Existing evaluator starts `VllmEnvironment`, reads `/v1/models`, and routes lm-eval through `local-completions` or `local-chat-completions`; it stages remote tokenizers if tokenizer files exist under the remote model path.

## 2026-05-03T22:24:00-0400

- Command: `sed -n '1,520p' lib/marin/src/marin/inference/vllm_server.py`
- Why: Inspect vLLM lifecycle, readiness, mode selection, and engine kwargs handling.
- Outcome: Succeeded.
- Learned: Native mode is the supported Iris path; readiness is `GET /v1/models`; object-store paths default to `load_format=runai_streamer`; engine kwargs supported as CLI args include `load_format`, `max_model_len`, `gpu_memory_utilization`, and `max_num_batched_tokens`.

## 2026-05-03T22:24:00-0400

- Command: `sed -n '100,155p' experiments/exp1337_eval_suite.py`
- Why: Inspect canonical task/env definitions around the earlier search hit.
- Outcome: Succeeded.
- Learned: The suite uses vLLM-native env vars for generation, sets `RESOURCE_CONFIG = ResourceConfig.with_tpu("v5p-8")`, and defines `mmlu_sl_verb_5shot`.

## 2026-05-03T22:26:00-0400

- Command: `sed -n '520,920p' lib/marin/src/marin/inference/vllm_server.py`
- Why: Inspect native vLLM process startup and environment details.
- Outcome: Succeeded.
- Learned: Native startup runs `vllm serve <model> --trust-remote-code --host <host> --port <port> ...`, logs to a temp directory, and inherits canonical TPU/vLLM env defaults.

## 2026-05-03T22:26:00-0400

- Command: `sed -n '1,260p' experiments/exp1337_eval_suite.py`
- Why: Read the existing 1e22 eval suite in context.
- Outcome: Succeeded.
- Learned: The exact documented 1e22 model path is `adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf` under `MARIN_PREFIX`; generation tasks use vLLM, but `mmlu_sl_verb_5shot` is routed through `evaluate_levanter_lm_evaluation_harness`, not vLLM. The file comment says multiple-choice tasks currently do not work on TPUs via vLLM.

## 2026-05-03T22:26:00-0400

- Command: `sed -n '1,240p' tests/evals/test_served_lm_eval.py`
- Why: Inspect CI-safe tests for the served lm-eval contract.
- Outcome: Succeeded.
- Learned: Existing tests assert lm-eval sends scoring requests to `/v1/completions` with `echo=true`, `logprobs=1`, and `max_tokens=1`, matching the design's completion/logprob concern.

## 2026-05-03T22:26:00-0400

- Command: `sed -n '1,220p' tests/evals/openai_stub.py`
- Why: Inspect the deterministic OpenAI stub used by served lm-eval tests.
- Outcome: Succeeded.
- Learned: The stub implements `/v1/models`, `/v1/completions`, and `/v1/chat/completions` with the OpenAI shapes lm-eval expects.

## 2026-05-03T22:26:00-0400

- Command: `sed -n '500,610p' experiments/evals/evals.py`
- Why: Inspect neighboring eval helper patterns.
- Outcome: Succeeded.
- Learned: Evalchemy helpers are separate; the lm-eval harness path remains the smallest current vLLM/Iris path.

## 2026-05-03T22:27:00-0400

- Command: `rg -n "def evaluate_lm_evaluation_harness|def evaluate_levanter_lm_evaluation_harness|LMEvaluationHarnessEvaluator|EvaluationConfig" experiments/evals/evals.py lib/marin/src/marin/evaluation`
- Why: Locate helper definitions for eval step construction and runtime config.
- Outcome: Succeeded.
- Learned: `evaluate_lm_evaluation_harness` and `evaluate_levanter_lm_evaluation_harness` are both in `experiments/evals/evals.py`; runtime dispatch is in `lib/marin/src/marin/evaluation/run.py`.

## 2026-05-03T22:27:00-0400

- Command: `sed -n '1,190p' experiments/evals/evals.py`
- Why: Read the lm-eval executor-step constructor.
- Outcome: Succeeded.
- Learned: The existing vLLM evaluator requests dependency groups `["eval", "vllm", "tpu"]`, supports explicit child env vars, and takes `ResourceConfig.with_tpu("v5p-8")`.

## 2026-05-03T22:27:00-0400

- Command: `sed -n '1,220p' lib/marin/src/marin/evaluation/evaluation_config.py`
- Why: Inspect eval task/config shapes for a scratch/manual runner.
- Outcome: Succeeded.
- Learned: `EvalTaskConfig` supports a `task_alias`, but the vLLM evaluator passes `eval_task.name` to lm-eval and uses alias only in result path naming.

## 2026-05-03T22:27:00-0400

- Command: `sed -n '1,220p' lib/marin/src/marin/evaluation/run.py`
- Why: Inspect model path normalization and checkpoint discovery behavior.
- Outcome: Succeeded.
- Learned: `_normalize_model_path` can select `/hf` or parent HF roots; `discover_latest_checkpoint=True` calls `discover_hf_checkpoints(model_path)[-1]`.

## 2026-05-03T22:27:00-0400

- Command: `sed -n '1,240p' lib/marin/src/marin/evaluation/evaluators/evaluator.py`
- Why: Inspect `ModelConfig` fields passed to `VllmEnvironment`.
- Outcome: Succeeded.
- Learned: `ModelConfig` carries path, engine kwargs, generation params, and `apply_chat_template`; this is enough for scratch/manual vLLM eval execution.

## 2026-05-03T22:28:00-0400

- Command: `GitHub connector _fetch_pr repo=marin-community/marin pr=5351`
- Why: Inspect the closed prototype evidence PR read-only.
- Outcome: Succeeded.
- Learned: Prototype #5351 implemented local broker/proxy/worker routing and tests, but intentionally did not launch vLLM inside workers. It is useful design evidence but not enough for the real vLLM milestone.

## 2026-05-03T22:28:00-0400

- Command: `GitHub connector _fetch_pr repo=marin-community/marin pr=5400`
- Why: Confirm draft design PR metadata and branch state.
- Outcome: Succeeded.
- Learned: PR #5400 is open/draft and contains the three design docs only.

## 2026-05-03T22:28:00-0400

- Command: `git show --stat --oneline --decorate HEAD`
- Why: Confirm local branch content.
- Outcome: Succeeded.
- Learned: HEAD `69569a5d2` adds only `.agents/projects/iris_inference_service/{design.md,research.md,spec.md}`.

## 2026-05-03T22:28:00-0400

- Command: `git diff --stat origin/main...HEAD`
- Why: Confirm design PR diff scope.
- Outcome: Succeeded.
- Learned: The branch diff is the same three docs, 690 insertions.

## 2026-05-03T22:30:00-0400

- Command: `printenv MARIN_PREFIX`
- Why: Check local storage prefix context before resolving hardcoded model paths.
- Outcome: Exit code 1; `MARIN_PREFIX` is unset locally.
- Learned: Local `marin_prefix()` will fall back to `/tmp/marin`; do not rely on it for the model path.

## 2026-05-03T22:30:00-0400

- Command: `uv run python -c 'from rigging.filesystem import marin_prefix; print(marin_prefix())'`
- Why: Confirm `marin_prefix()` fallback on the submitter machine.
- Outcome: Succeeded; output was `/tmp/marin`.
- Learned: `InputName.hardcoded(...)` paths in `exp1337` only become GCS paths when `MARIN_PREFIX` is set or metadata is available in an Iris job.

## 2026-05-03T22:30:00-0400

- Command: `ls lib/iris/examples`
- Why: Identify available Iris cluster configs.
- Outcome: Succeeded.
- Learned: Production config is `lib/iris/examples/marin.yaml`; dev config also exists.

## 2026-05-03T22:30:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --help > tmp/iris_pressure_test_20260504/iris_job_run_help.txt 2>&1`
- Why: Verify current CLI flags before submitting.
- Outcome: Succeeded.
- Learned: `--tpu` and large memory/disk require `--enable-extra-resources`; `--extra` installs uv extras; `--reserve` would not attach TPU hardware.

## 2026-05-03T22:30:00-0400

- Command: `sed -n '1,180p' lib/iris/examples/marin.yaml`
- Why: Read the production Iris config for v5p pool and state location.
- Outcome: Succeeded.
- Learned: v5p-8 preemptible pools exist in `us-central1-a` and `us-east5-a`; controller state is in `gs://marin-us-central2/iris/marin/state`.

## 2026-05-03T22:31:00-0400

- Command: `sed -n '1,220p' tmp/iris_pressure_test_20260504/iris_job_run_help.txt`
- Why: Inspect captured job-run help.
- Outcome: Succeeded.
- Learned: The correct submit flags are available for a direct v5p-8 job.

## 2026-05-03T22:31:00-0400

- Command: `rg -n "extra|pip_dependency_groups|dependency|task image|docker_image|env_vars|EnvironmentSpec|job run" lib/iris/src lib/marin/src/marin/execution lib/fray/src/fray experiments/evals/evals.py | head -n 240 > tmp/iris_pressure_test_20260504/rg_iris_remote_deps_head.txt 2>&1`
- Why: Understand how Iris/Fray passes extras and env vars.
- Outcome: Succeeded.
- Learned: Fray converts dependency extras and env vars into Iris `EnvironmentSpec`.

## 2026-05-03T22:31:00-0400

- Command: `rg -n "remote\\(|pip_dependency_groups|env_vars|extra" lib/marin/src/marin/execution lib/fray/src/fray | head -n 240 > tmp/iris_pressure_test_20260504/rg_remote_head.txt 2>&1`
- Why: Find the remote execution wrapper.
- Outcome: Succeeded.
- Learned: `marin.execution.remote.remote` wraps functions into Fray jobs with `create_environment(extras=..., env_vars=...)`.

## 2026-05-03T22:31:00-0400

- Command: `rg -n "class InputName|def hardcoded|MARIN_PREFIX|marin_prefix" lib/marin/src/marin lib/rigging/src rigging experiments | head -n 200 > tmp/iris_pressure_test_20260504/rg_prefix_inputname.txt 2>&1`
- Why: Locate path-prefix resolution.
- Outcome: Failed partially because `rigging` is not a repo-root path.
- Learned: Re-run with `lib/*/src`.

## 2026-05-03T22:32:00-0400

- Command: `sed -n '1,260p' tmp/iris_pressure_test_20260504/rg_remote_head.txt`
- Why: Inspect remote/extras search hits.
- Outcome: Succeeded.
- Learned: The remote path uses Fray jobs and Iris `EnvironmentSpec`.

## 2026-05-03T22:32:00-0400

- Command: `sed -n '1,260p' tmp/iris_pressure_test_20260504/rg_iris_remote_deps_head.txt`
- Why: Inspect env/extras handling hits.
- Outcome: Succeeded.
- Learned: Iris child jobs inherit parent env vars/extras; direct CLI jobs can also specify `--extra`.

## 2026-05-03T22:32:00-0400

- Command: `rg -n "class InputName|def hardcoded|MARIN_PREFIX|marin_prefix" lib/marin/src lib/*/src experiments | head -n 200 > tmp/iris_pressure_test_20260504/rg_prefix_inputname2.txt 2>&1`
- Why: Re-run prefix resolution search with correct source roots.
- Outcome: Succeeded.
- Learned: `InputName.hardcoded` is resolved under the executor prefix; `marin_prefix()` uses env, metadata, then `/tmp/marin`.

## 2026-05-03T22:32:00-0400

- Command: `sed -n '1,220p' lib/marin/src/marin/execution/remote.py`
- Why: Read direct remote wrapper semantics.
- Outcome: Succeeded.
- Learned: Remote jobs get `ResourceConfig`, env vars, and pip dependency extras from the wrapper.

## 2026-05-03T22:32:00-0400

- Command: `sed -n '520,640p' lib/fray/src/fray/iris_backend.py`
- Why: Read Fray-to-Iris submit path.
- Outcome: Succeeded.
- Learned: Fray submits Iris jobs with converted resources, constraints, environment, and task image.

## 2026-05-03T22:33:00-0400

- Command: `sed -n '640,760p' lib/fray/src/fray/iris_backend.py`
- Why: Inspect actor group submission.
- Outcome: Succeeded.
- Learned: Actor groups submit a single Iris job with replicas and inherit resource device env defaults.

## 2026-05-03T22:33:00-0400

- Command: `sed -n '720,825p' lib/marin/src/marin/execution/executor.py`
- Why: Inspect `ExecutorStep` and `InputName` behavior.
- Outcome: Succeeded.
- Learned: `InputName.hardcoded(path)` is a prefix-relative path when resolved by the executor.

## 2026-05-03T22:33:00-0400

- Command: `sed -n '825,925p' lib/marin/src/marin/execution/executor.py`
- Why: Continue reading special path objects.
- Outcome: Succeeded.
- Learned: Direct hardcoded paths are not pipeline dependencies; this matters for the `exp1337` model path.

## 2026-05-03T22:33:00-0400

- Command: `sed -n '1,220p' tmp/iris_pressure_test_20260504/rg_prefix_inputname2.txt`
- Why: Inspect prefix resolution hits.
- Outcome: Succeeded.
- Learned: `marin_prefix()` resolution and `InputName.hardcoded` behavior match the model-path hypothesis.

## 2026-05-03T22:34:00-0400

- Command: `sed -n '120,170p' lib/rigging/src/rigging/filesystem.py`
- Why: Read exact `marin_prefix()` fallback logic.
- Outcome: Succeeded.
- Learned: Iris jobs in GCP can infer regional `gs://marin-<region>` from metadata, but the local submitter falls back to `/tmp/marin`.

## 2026-05-03T22:34:00-0400

- Command: `ls -la .marin.yaml .marin.yml marin.yaml 2> tmp/iris_pressure_test_20260504/ls_marin_config_stderr.txt`
- Why: Check for a repo-local env/config file that might set `MARIN_PREFIX`.
- Outcome: Exit code 1; no matching local files.
- Learned: No local `.marin.yaml` provides model-prefix env vars.

## 2026-05-03T22:34:00-0400

- Command: `git ls-files '*marin*.yaml' '*marin*.yml' | sed -n '1,200p'`
- Why: Find committed Marin configs.
- Outcome: Succeeded.
- Learned: No committed repo-root `.marin.yaml` exists; Iris configs are under `lib/iris/examples`.

## 2026-05-03T22:34:00-0400

- Command: `rg -n "marin-us-central|adamh-scaling-ladder-nemotron-optimal-1e\\+22|grug/moe-v7-1e22|1e\\+22-v5" experiments .agents docs scripts lib --glob '!tmp/**' --glob '!uv.lock' > tmp/iris_pressure_test_20260504/rg_model_path_candidates.txt 2>&1`
- Why: Find exact model path candidates and bucket context.
- Outcome: Succeeded.
- Learned: The explicit HF candidate appears at `experiments/exp1337_eval_suite.py:185`; storage protect rules mention `grug/moe-v7-1e22-d3200%`.

## 2026-05-03T22:35:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/grug/moe-v7-1e22-d3200*' > tmp/iris_pressure_test_20260504/gcloud_ls_grug_moe_1e22.txt 2>&1`
- Why: Metadata-only listing for the explicit 1e22 MoE artifact family.
- Outcome: Succeeded.
- Learned: Several `grug/moe-v7-1e22-d3200-*` runs exist; the v4 run has checkpoints.

## 2026-05-03T22:35:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf*' > tmp/iris_pressure_test_20260504/gcloud_ls_adamh_1e22.txt 2>&1`
- Why: Check whether the `exp1337` HF path exists.
- Outcome: Succeeded.
- Learned: HF steps exist at `step-10000`, `step-20000`, `step-30000`, and `step-38234`.

## 2026-05-03T22:35:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/checkpoints/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf*' > tmp/iris_pressure_test_20260504/gcloud_ls_checkpoints_adamh_1e22.txt 2>&1`
- Why: Test whether the same run lives under a `checkpoints/` prefix.
- Outcome: Failed with no matching objects.
- Learned: The `exp1337` path is rooted directly under the bucket, not `checkpoints/`.

## 2026-05-03T22:35:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/grug/moe-v7-1e22-d3200*/hf*' > tmp/iris_pressure_test_20260504/gcloud_ls_grug_moe_1e22_hf.txt 2>&1`
- Why: Check for HF exports under the explicit MoE artifact family.
- Outcome: Failed with no matching objects.
- Learned: The explicit MoE family does not expose an obvious HF root.

## 2026-05-03T22:36:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/grug/moe-v7-1e22-d3200-v4-56ba43/checkpoints/' > tmp/iris_pressure_test_20260504/gcloud_ls_grug_v4_checkpoints.txt 2>&1`
- Why: Identify the latest explicit MoE checkpoint.
- Outcome: Succeeded.
- Learned: The v4 MoE run has `checkpoints/step-77725/`.

## 2026-05-03T22:36:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/grug/moe-v7-1e22-d3200-v3-7a8de9/checkpoints/' > tmp/iris_pressure_test_20260504/gcloud_ls_grug_v3_checkpoints.txt 2>&1`
- Why: Compare adjacent MoE run checkpoint layout.
- Outcome: Succeeded.
- Learned: The v3 MoE run also has `checkpoints/step-77725/`.

## 2026-05-03T22:36:00-0400

- Command: `gcloud storage cat 'gs://marin-us-central2/grug/moe-v7-1e22-d3200-v4-56ba43/.executor_status' > tmp/iris_pressure_test_20260504/grug_v4_executor_status.txt 2>&1`
- Why: Check whether the v4 MoE run completed.
- Outcome: Succeeded.
- Learned: Executor status is `SUCCESS`.

## 2026-05-03T22:36:00-0400

- Command: `gcloud storage cat 'gs://marin-us-central2/grug/moe-v7-1e22-d3200-v4-56ba43/.artifact' > tmp/iris_pressure_test_20260504/grug_v4_artifact.txt 2>&1`
- Why: Inspect the artifact marker for the v4 MoE run.
- Outcome: Succeeded.
- Learned: Artifact marker is `null`.

## 2026-05-03T22:36:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf/step-38234/config.json' > tmp/iris_pressure_test_20260504/gcloud_ls_adamh_step38234_config.txt 2>&1`
- Why: Confirm the stand-in HF checkpoint has a `config.json`.
- Outcome: Succeeded.
- Learned: `step-38234` is vLLM/HF-shaped.

## 2026-05-03T22:37:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/grug/moe-v7-1e22-d3200-v4-56ba43/checkpoints/step-77725/config.json' > tmp/iris_pressure_test_20260504/gcloud_ls_grug_v4_step_config.txt 2>&1`
- Why: Check whether the explicit MoE checkpoint root is HF-shaped.
- Outcome: Failed with no matching objects.
- Learned: The explicit MoE checkpoint root is not an HF model directory.

## 2026-05-03T22:37:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/grug/moe-v7-1e22-d3200-v4-56ba43/checkpoints/step-77725/hf/config.json' > tmp/iris_pressure_test_20260504/gcloud_ls_grug_v4_step_hf_config.txt 2>&1`
- Why: Check for a nested HF export under the explicit MoE checkpoint.
- Outcome: Failed with no matching objects.
- Learned: No nested `hf/config.json` exists under the explicit MoE checkpoint.

## 2026-05-03T22:37:00-0400

- Command: `gcloud storage cat 'gs://marin-us-central2/grug/moe-v7-1e22-d3200-v4-56ba43/.executor_info' > tmp/iris_pressure_test_20260504/grug_v4_executor_info.txt 2>&1`
- Why: Inspect the explicit MoE run config.
- Outcome: Succeeded.
- Learned: The run is named `grug/moe-v7-1e22-d3200-v4`; it is a real MoE with `hidden_dim=3200`, `num_experts=64`, `num_experts_per_token=4`, `num_layers=32`, tokenizer `meta-llama/Meta-Llama-3.1-8B`.

## 2026-05-03T22:37:00-0400

- Command: `gcloud storage cat 'gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf/step-38234/config.json' > tmp/iris_pressure_test_20260504/adamh_step38234_config.json 2>&1`
- Why: Inspect the vLLM-loadable stand-in config.
- Outcome: Succeeded.
- Learned: The stand-in is HF-shaped with `architectures=["LlamaForCausalLM"]`, `model_type="qwen3"`, hidden size 3840, 37 layers, and tokenizer ids.

## 2026-05-03T22:37:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf/step-38234/tokenizer_config.json' > tmp/iris_pressure_test_20260504/gcloud_ls_adamh_step38234_tokenizer_config.txt 2>&1`
- Why: Confirm tokenizer metadata exists for the stand-in.
- Outcome: Succeeded.
- Learned: The stand-in has `tokenizer_config.json`, and the scratch runner also passes an explicit HF tokenizer id.

## 2026-05-03T22:38:00-0400

- Command: `rg -n "moe-v7-1e22-d3200|grug/moe-v7|d3200-v4|1e22-d3200" experiments .agents docs scripts lib --glob '!tmp/**' --glob '!uv.lock' > tmp/iris_pressure_test_20260504/rg_grug_moe_1e22_repo.txt 2>&1`
- Why: Look for repo references to the explicit MoE path.
- Outcome: Succeeded.
- Learned: The only repo reference is the storage protect rule; no evaluation/HF export config is checked in.

## 2026-05-03T22:38:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/grug/moe-v7-1e22-d3200-v4-56ba43/checkpoints/step-77725/*' > tmp/iris_pressure_test_20260504/gcloud_ls_grug_v4_step_top.txt 2>&1`
- Why: Inspect top-level contents of the explicit MoE checkpoint without downloading data.
- Outcome: Succeeded but noisy.
- Learned: The checkpoint is Orbax/OCDBT-like (`manifest.ocdbt`, `metadata.json`, many `d/` shards), not a vLLM HF directory.

## 2026-05-03T22:38:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/grug/moe-v7-1e22-d3200-v4-56ba43/checkpoints/step-77725/**/config.json' > tmp/iris_pressure_test_20260504/gcloud_ls_grug_v4_step_recursive_config.txt 2>&1`
- Why: Check for any recursive `config.json` under the explicit MoE checkpoint.
- Outcome: Failed with no matching objects.
- Learned: No HF config appears under the explicit MoE checkpoint.

## 2026-05-03T22:38:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/grug/moe-v7-1e22-d3200-v4-56ba43/**/tokenizer_config.json' > tmp/iris_pressure_test_20260504/gcloud_ls_grug_v4_recursive_tokenizer_config.txt 2>&1`
- Why: Check for any recursive tokenizer config under the explicit MoE run.
- Outcome: Failed with no matching objects.
- Learned: No HF tokenizer export appears under the explicit MoE run.

## 2026-05-03T22:40:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --help > tmp/iris_pressure_test_20260504/iris_job_list_help.txt 2>&1`
- Why: Verify status polling flags.
- Outcome: Succeeded.
- Learned: `job list` supports `--json`, `--state`, and `--prefix`.

## 2026-05-03T22:40:00-0400

- Command: `test -n "$HF_TOKEN"`
- Why: Check whether the submitter can auto-inject HF auth without printing secrets.
- Outcome: Exit code 1.
- Learned: `HF_TOKEN` is not set locally; this may matter only for private HF models, not for the GCS checkpoint or public tokenizer.

## 2026-05-03T22:40:00-0400

- Command: `test -n "$WANDB_API_KEY"`
- Why: Check whether WandB auth can be auto-injected without printing secrets.
- Outcome: Exit code 1.
- Learned: `WANDB_API_KEY` is not set locally; scratch runner sets `WANDB_MODE=offline`.

## 2026-05-03T22:40:00-0400

- Command: `test -n "$GOOGLE_APPLICATION_CREDENTIALS"`
- Why: Check whether local GCP auth uses an explicit credentials file.
- Outcome: Exit code 1.
- Learned: Local GCS access is likely via gcloud/user ADC rather than `GOOGLE_APPLICATION_CREDENTIALS`; Iris task service account should handle GCS.

## 2026-05-03T22:41:00-0400

- Command: `sed -n '1,180p' tmp/iris_pressure_test_20260504/iris_job_list_help.txt`
- Why: Inspect job-list help.
- Outcome: Succeeded.
- Learned: Prefix polling is the right status check for this job.

## 2026-05-03T22:42:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --state running > tmp/iris_pressure_test_20260504/iris_job_list_running_initial.json 2> tmp/iris_pressure_test_20260504/iris_job_list_running_initial.stderr`
- Why: Smoke-test Iris controller connectivity before submitting new work.
- Outcome: Succeeded.
- Learned: Controller tunnel works; there were 424 running jobs in the captured JSON.

## 2026-05-03T22:42:00-0400

- Command: `jq 'length' tmp/iris_pressure_test_20260504/iris_job_list_running_initial.json`
- Why: Count running jobs without reading the full JSON.
- Outcome: Succeeded; output was `424`.
- Learned: The cluster is busy, so v5p scheduling may require waiting.

## 2026-05-03T22:43:00-0400

- Command: `apply_patch` add `tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py`
- Why: Create a scratch-only runner using the existing vLLM lm-eval evaluator, with dry-run support.
- Outcome: Succeeded.
- Learned: The manual runner can target the existing `lm_evaluation_harness` evaluator without touching tracked implementation files.

## 2026-05-03T22:44:00-0400

- Command: `uv run python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --dry-run --task mmlu_sl_verb_5shot --model-name adamh-1e22-standin --model-path 'gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf/step-38234' --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/mmlu_sl_verb_5shot' > tmp/iris_pressure_test_20260504/local_dry_run_mmlu.txt 2>&1`
- Why: Validate the MMLU runner shape before submitting an Iris job.
- Outcome: Succeeded.
- Learned: MMLU maps to `apply_chat_template=False`, tokenizer `meta-llama/Meta-Llama-3.1-8B`, and `max_num_batched_tokens=1024`.

## 2026-05-03T22:44:00-0400

- Command: `uv run python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --dry-run --task humaneval_5shot --model-name adamh-1e22-standin --model-path 'gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf/step-38234' --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/humaneval_5shot' > tmp/iris_pressure_test_20260504/local_dry_run_humaneval.txt 2>&1`
- Why: Validate the HumanEval runner shape before submitting.
- Outcome: Succeeded.
- Learned: HumanEval maps to `apply_chat_template=True` and the chat-completions path in the existing evaluator.

## 2026-05-03T22:44:00-0400

- Command: `python -m py_compile tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py`
- Why: Syntax-check the scratch runner.
- Outcome: Succeeded.
- Learned: Scratch runner syntax is valid.

## 2026-05-03T22:44:00-0400

- Command: `git status --short --branch`
- Why: Check that only scratch/untracked files changed.
- Outcome: Succeeded.
- Learned: Branch still has untracked `report.md` and `tmp/`; tracked docs are unchanged at this point.

## 2026-05-03T22:45:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --enable-extra-resources --tpu v5p-8 --zone us-central1-a --cpu 32 --memory 128GB --disk 50GB --timeout 7200 --extra eval --extra vllm --extra tpu -e VLLM_ENABLE_V1_MULTIPROCESSING 0 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 -e VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION 1 -e VLLM_TPU_SKIP_PRECOMPILE 1 -e HF_ALLOW_CODE_EVAL 1 -e WANDB_MODE offline -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task mmlu_sl_verb_5shot --model-name adamh-1e22-standin --model-path 'gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf/step-38234' --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/mmlu_sl_verb_5shot' --limit 1 > tmp/iris_pressure_test_20260504/submit_mmlu_vllm_iris.out 2>&1`
- Why: Submit the smallest real MMLU vLLM-on-Iris attempt.
- Outcome: Succeeded; job id `/romain/iris-run-run_existing_vllm_lm_eval-20260504-022544`.
- Learned: The job was accepted with v5p-8, 32 CPU, 128GB memory, 50GB disk, eval/vllm/tpu extras, and a 2-hour timeout.

## 2026-05-03T22:46:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-022544 > tmp/iris_pressure_test_20260504/mmlu_job_list_001.json 2> tmp/iris_pressure_test_20260504/mmlu_job_list_001.stderr`
- Why: Check immediate state after submit.
- Outcome: Succeeded.
- Learned: Job is `JOB_STATE_PENDING`; scheduler reports insufficient currently-free v5p-8 memory and autoscaler is waiting for `tpu_v5p-preemptible_8-us-central1-a` workers to become ready.

## 2026-05-03T22:46:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 900 --include-children /romain/iris-run-run_existing_vllm_lm_eval-20260504-022544 > tmp/iris_pressure_test_20260504/mmlu_job_logs_001.txt 2>&1`
- Why: Try the babysit playbook log command.
- Outcome: Failed; this checkout's `iris job logs` has no `--include-children` option.
- Learned: Use direct `job logs` for this single-task direct job.

## 2026-05-03T22:46:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job logs --help > tmp/iris_pressure_test_20260504/iris_job_logs_help.txt 2>&1`
- Why: Verify available log flags.
- Outcome: Succeeded.
- Learned: `job logs` supports `--since-seconds`, `--tail`, and `--max-lines`, but not `--include-children`.

## 2026-05-03T22:28:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-022544 > tmp/iris_pressure_test_20260504/mmlu_job_list_parent_001.json 2> tmp/iris_pressure_test_20260504/mmlu_job_list_parent_001.stderr`
- Why: Parent-agent monitor while subagent continues; check whether the submitted MMLU vLLM Iris job has scheduled.
- Outcome: Succeeded; job remains `JOB_STATE_PENDING`.
- Learned: Scheduler still sees insufficient v5p-8 memory; autoscaler now reports `tier_blocked: 1 matching group(s) blocked by quota-pool tier monotonicity`.

## 2026-05-03T22:35:00-0400

- Command: `sed -n '1,220p' .agents/skills/babysit-job/SKILL.md`, `sed -n '1,220p' .agents/skills/debug-infra/SKILL.md`, `sed -n '1,180p' .agents/skills/agent-research/SKILL.md`
- Why: Reconfirm the monitoring and infrastructure-debug rules after context compaction.
- Outcome: Succeeded.
- Learned: Continue one monitor loop; pending capacity waits are not cluster failures and should not trigger cluster lifecycle actions.

## 2026-05-03T22:35:00-0400

- Command: `tail -n 80 tmp/iris_pressure_test_20260504/run_log.md`, `sed -n '1,240p' tmp/iris_pressure_test_20260504/findings.md`, `sed -n '1,220p' scratch/20260503-2246_monitoring_state.json`
- Why: Rehydrate current pressure-test state from local artifacts.
- Outcome: Succeeded.
- Learned: Active job is still `/romain/iris-run-run_existing_vllm_lm_eval-20260504-022544`; no tracked design files have been changed.

## 2026-05-03T22:35:00-0400

- Command: `find tmp/iris_pressure_test_20260504 -maxdepth 1 -type f -print`, `ls -lt tmp/iris_pressure_test_20260504`
- Why: Reconcile which diagnostic artifacts were saved before continuing.
- Outcome: Succeeded.
- Learned: Autoscaler, scheduler, second job-list, and job-summary artifacts exist locally; the run log had not yet recorded all of those diagnostic commands.

## 2026-05-03T22:35:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-022544`
- Why: Refresh MMLU pressure-test job status.
- Outcome: Succeeded.
- Learned: Job remains `JOB_STATE_PENDING`; pending reason is still insufficient currently-free v5p-8 memory and autoscaler tier blocking.

## 2026-05-03T22:35:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-022544`
- Why: Confirm per-task state and whether any task has started.
- Outcome: Succeeded.
- Learned: The single task is still pending, has no worker id, and has no runtime or memory usage yet.

## 2026-05-03T22:35:00-0400

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/findings.md`
- Why: Keep the compact findings file current after the refreshed status poll.
- Outcome: Succeeded.
- Learned: Current blocker is still v5p-8 capacity; no design-facing runtime evidence has appeared yet.

## 2026-05-03T22:42:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-022544 > tmp/iris_pressure_test_20260504/mmlu_job_list_monitor_003.json 2> tmp/iris_pressure_test_20260504/mmlu_job_list_monitor_003.stderr`
- Why: Parent-agent monitor while subagent continues; check if the canonical MMLU vLLM attempt has scheduled.
- Outcome: Succeeded; job remains `JOB_STATE_PENDING`.
- Learned: Pending reason is still scheduler capacity, specifically 128GB requested on v5p-8 with only 0.8GB free on matching workers plus autoscaler quota-pool tier blocking.

## 2026-05-03T22:51:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --help > tmp/iris_pressure_test_20260504/iris_job_run_help_monitor.txt 2>&1`; `uv run iris --config lib/iris/examples/marin.yaml job --help > tmp/iris_pressure_test_20260504/iris_job_help_monitor.txt 2>&1`; `uv run iris --config lib/iris/examples/marin.yaml job stop --help > tmp/iris_pressure_test_20260504/iris_job_stop_help_monitor.txt 2>&1`
- Why: Find safe fallback options if the submitted v5p-8 job remains queued.
- Outcome: Succeeded.
- Learned: `iris job run` supports `--preemptible`/`--no-preemptible`; `iris job stop` can terminate our own pending job if we need to avoid duplicate queued attempts.

## 2026-05-03T22:52:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --state pending > tmp/iris_pressure_test_20260504/iris_job_list_pending_monitor_001.json 2> tmp/iris_pressure_test_20260504/iris_job_list_pending_monitor_001.stderr`; `uv run iris --config lib/iris/examples/marin.yaml job list --json --state running > tmp/iris_pressure_test_20260504/iris_job_list_running_monitor_002.json 2> tmp/iris_pressure_test_20260504/iris_job_list_running_monitor_002.stderr`
- Why: Understand whether v5p-8 is globally busy or the submitted job shape is uniquely bad.
- Outcome: Succeeded.
- Learned: There are many v5p-8 jobs already running and queued. The canonical run is competing with real cluster load rather than failing at startup.

## 2026-05-03T22:54:00-0400

- Command: `jq '[.[] | .resources.device.tpu.variant] | group_by(.) | map({variant:.[0], count:length}) | sort_by(.variant)' tmp/iris_pressure_test_20260504/iris_job_list_pending_monitor_002.json`; same grouping for running jobs in `tmp/iris_pressure_test_20260504/iris_job_list_running_monitor_003.json`
- Why: Look for a less congested TPU shape to validate the same Iris/vLLM/lm-eval mechanics while the canonical v5p-8 run waits.
- Outcome: Succeeded.
- Learned: v5p-8 is heavily used; v6e-4 has running jobs but no pending backlog in the captured snapshot. A small-model v6e-4 fallback is a reasonable mechanics probe, but not a substitute for the canonical v5p-8 milestone.

## 2026-05-03T22:57:00-0400

- Command: `uv run python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --dry-run --task mmlu_sl_verb_5shot --model-name qwen3-0_6b-mechanics --model-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --tpu-type v6e-4 --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_mmlu_sl_verb_5shot' --limit 1`; same dry-run for `humaneval_5shot`.
- Why: Validate a fallback small-model mechanics probe before submitting it to Iris.
- Outcome: Succeeded.
- Learned: The fallback uses the same evaluator path, same real lm-eval task name, same vLLM environment, and a public Qwen tokenizer/model. It is a mechanics probe only.

## 2026-05-03T22:57:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --enable-extra-resources --tpu v6e-4 --cpu 32 --memory 128GB --disk 50GB --timeout 7200 --extra eval --extra vllm --extra tpu -e VLLM_ENABLE_V1_MULTIPROCESSING 0 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 -e VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION 1 -e VLLM_TPU_SKIP_PRECOMPILE 1 -e HF_ALLOW_CODE_EVAL 1 -e WANDB_MODE offline -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task mmlu_sl_verb_5shot --model-name qwen3-0_6b-mechanics --model-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --tpu-type v6e-4 --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_mmlu_sl_verb_5shot' --limit 1 > tmp/iris_pressure_test_20260504/submit_qwen3_v6e4_mmlu.out 2>&1`
- Why: Try to get real vLLM and real lm-eval to user code while the canonical v5p-8 attempt waits for capacity.
- Outcome: Succeeded; job id `/romain/iris-run-run_existing_vllm_lm_eval-20260504-024730`.
- Learned: Iris accepted the fallback v6e-4 job. Need monitor scheduling and logs.

## 2026-05-03T22:58:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-024730 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_list_001.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_list_001.stderr`; `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-024730 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_summary_001.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_summary_001.stderr`
- Why: Check whether the fallback mechanics probe scheduled.
- Outcome: Succeeded; job is `JOB_STATE_RUNNING`.
- Learned: The fallback started immediately on `marin-tpu-v6e-preemptible-4-us-east1-d-20260504-0155-11d43b66-worker-0`; now watch logs for vLLM and lm-eval behavior.

## 2026-05-03T22:58:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-022544 > tmp/iris_pressure_test_20260504/mmlu_job_list_monitor_004.json 2> tmp/iris_pressure_test_20260504/mmlu_job_list_monitor_004.stderr`
- Why: Keep monitoring the canonical v5p-8 attempt while fallback runs.
- Outcome: Succeeded; canonical job remains `JOB_STATE_PENDING`.
- Learned: Pending reason now includes insufficient TPUs on one matching worker and insufficient memory on eight matching workers, plus autoscaler quota-pool tier blocking.

## 2026-05-03T22:59:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job logs --tail --max-lines 240 /romain/iris-run-run_existing_vllm_lm_eval-20260504-024730 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_logs_001_tail.txt 2>&1`
- Why: Inspect fallback job logs.
- Outcome: Failed in the Iris/Finelog client with `finelog.errors.StatsError: Not Found`.
- Learned: Log indexing/fetching is not immediately available for this running task. Continue polling job state and retry logs later; do not treat this as a workload failure.

## 2026-05-03T23:01:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-024730 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_list_002.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_list_002.stderr`; `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-024730 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_summary_002.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_summary_002.stderr`
- Why: Check whether fallback is still running and consuming resources.
- Outcome: Succeeded; fallback remains running.
- Learned: The task has been running for ~147s with non-zero memory/CPU on the v6e-4 worker, so it reached the worker process. Logs are still unavailable through Finelog.

## 2026-05-03T23:01:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 600 --max-lines 300 /romain/iris-run-run_existing_vllm_lm_eval-20260504-024730 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_logs_002.txt 2>&1`
- Why: Retry fallback job log fetch without `--tail`.
- Outcome: Failed again with `finelog.errors.StatsError: Not Found`.
- Learned: Current blocker for detailed observations is log retrieval, not scheduling.

## 2026-05-03T23:03:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-024730 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_summary_003.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_summary_003.stderr`; `uv run iris --config lib/iris/examples/marin.yaml rpc controller get-task-logs --id /romain/iris-run-run_existing_vllm_lm_eval-20260504-024730 --include-children --max-total-lines 2000 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_controller_logs_full_001.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_controller_logs_full_001.stderr`
- Why: Diagnose fallback failure after the CLI log surface continued to fail.
- Outcome: Succeeded via controller log RPC.
- Learned: The fallback failed before any HTTP scoring request. `lm_eval` import imports `lm_eval.models.vllm_causallms`, which imports `vllm.utils.get_open_port`; the installed Iris vLLM package does not export that symbol. This is lm-eval/vLLM dependency skew, not an OpenAI HTTP contract failure.

## 2026-05-03T23:04:00-0400

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py`
- Why: Add a scratch-only compatibility shim for `vllm.utils.get_open_port` so real lm-eval can import under the current Iris vLLM package.
- Outcome: Succeeded.
- Learned: The workaround is local/manual only and should not be interpreted as a production implementation.

## 2026-05-03T23:05:00-0400

- Command: `python -m py_compile tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py`; dry-run the patched Qwen3/v6e-4 MMLU command.
- Why: Check the patched scratch runner before resubmitting.
- Outcome: Succeeded.
- Learned: The patched scratch runner is syntactically valid and preserves the intended command shape.

## 2026-05-03T23:06:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job stop /romain/iris-run-run_existing_vllm_lm_eval-20260504-022544 > tmp/iris_pressure_test_20260504/stop_unpatched_canonical_mmlu_001.txt 2>&1`
- Why: Avoid wasting a v5p-8 slot on the old unpatched canonical submission, which would hit the same lm-eval import failure if it scheduled.
- Outcome: Succeeded; job `/romain/iris-run-run_existing_vllm_lm_eval-20260504-022544` is `JOB_STATE_KILLED`.
- Learned: Need resubmit the canonical v5p-8 attempt only after the shim is included in the job bundle.

## 2026-05-03T23:06:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --enable-extra-resources --tpu v6e-4 --cpu 32 --memory 128GB --disk 50GB --timeout 7200 --extra eval --extra vllm --extra tpu -e VLLM_ENABLE_V1_MULTIPROCESSING 0 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 -e VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION 1 -e VLLM_TPU_SKIP_PRECOMPILE 1 -e HF_ALLOW_CODE_EVAL 1 -e WANDB_MODE offline -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task mmlu_sl_verb_5shot --model-name qwen3-0_6b-mechanics --model-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --tpu-type v6e-4 --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_mmlu_sl_verb_5shot_shim' --limit 1 > tmp/iris_pressure_test_20260504/submit_qwen3_v6e4_mmlu_shim_001.out 2>&1`
- Why: Rerun the small mechanics probe with the lm-eval/vLLM import shim included.
- Outcome: Succeeded; job id `/romain/iris-run-run_existing_vllm_lm_eval-20260504-025615`.
- Learned: Need monitor whether the job reaches vLLM HTTP scoring.

## 2026-05-03T23:09:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml rpc controller get-task-logs --id /romain/iris-run-run_existing_vllm_lm_eval-20260504-025615 --include-children --max-total-lines 1800 --tail > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_controller_logs_003.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_controller_logs_003.stderr`
- Why: Inspect the patched MMLU mechanics probe after it failed.
- Outcome: Succeeded.
- Learned: The shim worked; real lm-eval reached `/v1/completions`. The first stock loglikelihood request failed after retries with HTTP 500 from `http://127.0.0.1:8000/v1/completions`, body `{"error":{"message":"32658","type":"InternalServerError","param":null,"code":500}}`. lm-eval also warned that the 5-shot prompt exceeded max length 2047 and was left-truncated, so run one follow-up with `max_model_len=4096`.

## 2026-05-03T23:10:00-0400

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py`
- Why: Add optional `--max-model-len` to the scratch runner so vLLM and lm-eval can share a larger context limit.
- Outcome: Succeeded.
- Learned: This lets the next MMLU probe distinguish context-limit confounding from the vLLM logprob endpoint failure.

## 2026-05-03T23:12:00-0400

- Command: `uv run iris ... --max-model-len 4096 --max-num-batched-tokens 4096 ... > tmp/iris_pressure_test_20260504/submit_qwen3_v6e4_mmlu_4096_001.out 2>&1`
- Why: Submit the larger-context MMLU mechanics probe.
- Outcome: Failed before submission due a transient GitHub 502 while uv fetched `dupekit` release assets.
- Learned: Retry the same submit command; this was local dependency resolution noise.

## 2026-05-03T23:12:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --enable-extra-resources --tpu v6e-4 --cpu 32 --memory 128GB --disk 50GB --timeout 7200 --extra eval --extra vllm --extra tpu -e VLLM_ENABLE_V1_MULTIPROCESSING 0 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 -e VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION 1 -e VLLM_TPU_SKIP_PRECOMPILE 1 -e HF_ALLOW_CODE_EVAL 1 -e WANDB_MODE offline -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task mmlu_sl_verb_5shot --model-name qwen3-0_6b-mechanics --model-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --tpu-type v6e-4 --max-model-len 4096 --max-num-batched-tokens 4096 --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_mmlu_sl_verb_5shot_4096' --limit 1 > tmp/iris_pressure_test_20260504/submit_qwen3_v6e4_mmlu_4096_002.out 2>&1`
- Why: Retry the larger-context MMLU mechanics probe.
- Outcome: Succeeded; job id `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030231`.
- Learned: Need monitor whether the larger context changes the HTTP 500 behavior.

## 2026-05-03T23:18:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml rpc controller get-task-logs --id /romain/iris-run-run_existing_vllm_lm_eval-20260504-030231 --include-children --max-total-lines 2600 --tail > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_4096_controller_logs_004.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_4096_controller_logs_004.stderr`
- Why: Inspect the `max_model_len=4096` MMLU rerun after it failed.
- Outcome: Succeeded.
- Learned: It still failed with the same `/v1/completions` HTTP 500 body `32658`; lm-eval still reported `max_length (2047)`, so `max_model_len` alone does not adjust lm-eval's local-completions max length.

## 2026-05-03T23:19:00-0400

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py`
- Why: Add scratch-only `--lm-eval-max-length` so the next run can pass `max_length=4096` to lm-eval local-completions as well as `max_model_len=4096` to vLLM.
- Outcome: Succeeded and dry-run passed.
- Learned: The next MMLU run has engine kwargs `tokenizer`, `max_num_batched_tokens=4096`, `max_model_len=4096`, and `max_length=4096`.

## 2026-05-03T23:19:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --enable-extra-resources --tpu v6e-4 --cpu 32 --memory 128GB --disk 50GB --timeout 7200 --extra eval --extra vllm --extra tpu -e VLLM_ENABLE_V1_MULTIPROCESSING 0 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 -e VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION 1 -e VLLM_TPU_SKIP_PRECOMPILE 1 -e HF_ALLOW_CODE_EVAL 1 -e WANDB_MODE offline -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task mmlu_sl_verb_5shot --model-name qwen3-0_6b-mechanics --model-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --tpu-type v6e-4 --max-model-len 4096 --lm-eval-max-length 4096 --max-num-batched-tokens 4096 --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_mmlu_sl_verb_5shot_4096_lmeval' --limit 1 > tmp/iris_pressure_test_20260504/submit_qwen3_v6e4_mmlu_4096_lmeval_001.out 2>&1`
- Why: Test whether raising both vLLM and lm-eval context lengths changes the loglikelihood failure.
- Outcome: Succeeded; job id `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030919`.
- Learned: Need monitor for the HTTP result.

## 2026-05-03T23:20:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --enable-extra-resources --tpu v6e-4 --cpu 32 --memory 128GB --disk 50GB --timeout 7200 --extra eval --extra vllm --extra tpu -e VLLM_ENABLE_V1_MULTIPROCESSING 0 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 -e VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION 1 -e VLLM_TPU_SKIP_PRECOMPILE 1 -e HF_ALLOW_CODE_EVAL 1 -e WANDB_MODE offline -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task humaneval_5shot --model-name qwen3-0_6b-mechanics --model-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --tpu-type v6e-4 --max-model-len 4096 --lm-eval-max-length 4096 --max-num-batched-tokens 4096 --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_humaneval_5shot' --limit 1 > tmp/iris_pressure_test_20260504/submit_qwen3_v6e4_humaneval_001.out 2>&1`
- Why: Exercise the generative/chat-completions side of the canonical milestone shape while MMLU/loglikelihood is known to fail.
- Outcome: Succeeded; job id `/romain/iris-run-run_existing_vllm_lm_eval-20260504-031005`.
- Learned: Need monitor whether generation succeeds under real vLLM + real lm-eval.

## 2026-05-03T23:26:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-030919 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_4096_lmeval_summary_002.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_4096_lmeval_summary_002.stderr`; `uv run iris --config lib/iris/examples/marin.yaml rpc controller get-task-logs --id /romain/iris-run-run_existing_vllm_lm_eval-20260504-030919 --include-children --max-total-lines 3000 --tail > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_4096_lmeval_controller_logs_002.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_4096_lmeval_controller_logs_002.stderr`
- Why: Inspect the MMLU run with both vLLM `max_model_len=4096` and lm-eval `max_length=4096`.
- Outcome: Failed with exit 1 after reaching real lm-eval API requests.
- Learned: The context-length confound is controlled enough for the spec decision. This run still failed on the first `local-completions` loglikelihood request with `/v1/completions` HTTP 500, body `{"message":"2701","type":"InternalServerError","code":500}`. No `max_length (2047)` warning appeared in the inspected tail.

## 2026-05-03T23:27:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-031005 > tmp/iris_pressure_test_20260504/qwen3_v6e4_humaneval_summary_002.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_humaneval_summary_002.stderr`; `uv run iris --config lib/iris/examples/marin.yaml rpc controller get-task-logs --id /romain/iris-run-run_existing_vllm_lm_eval-20260504-031005 --include-children --max-total-lines 5000 --tail > tmp/iris_pressure_test_20260504/qwen3_v6e4_humaneval_controller_logs_004.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_humaneval_controller_logs_004.stderr`
- Why: Inspect the first HumanEval generation probe.
- Outcome: Job state was `JOB_STATE_KILLED` with reason `Terminated by user`; controller logs only showed runner startup and no lm-eval/vLLM HTTP request.
- Learned: This is not evidence about generation behavior. It did allocate a worker, then terminated before meaningful scoring. No local `job stop` command for this HumanEval job was found in the run log.

## 2026-05-03T23:16:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --enable-extra-resources --tpu v6e-4 --cpu 32 --memory 128GB --disk 50GB --timeout 7200 --extra eval --extra vllm --extra tpu -e VLLM_ENABLE_V1_MULTIPROCESSING 0 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 -e VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION 1 -e VLLM_TPU_SKIP_PRECOMPILE 1 -e HF_ALLOW_CODE_EVAL 1 -e WANDB_MODE offline -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task humaneval_5shot --model-name qwen3-0_6b-mechanics --model-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --tpu-type v6e-4 --max-model-len 4096 --lm-eval-max-length 4096 --max-num-batched-tokens 4096 --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_humaneval_5shot_rerun' --limit 1 > tmp/iris_pressure_test_20260504/submit_qwen3_v6e4_humaneval_rerun_001.out 2>&1`
- Why: Rerun HumanEval generation after the first attempt was killed before reaching scoring.
- Outcome: Succeeded; job id `/romain/iris-run-run_existing_vllm_lm_eval-20260504-031612`.
- Learned: Need monitor whether this second generation attempt reaches `local-chat-completions`.

## 2026-05-03T22:45:00-0400

- Command: `sleep 570`
- Why: Follow the babysit cadence after the job remained pending due to capacity wait.
- Outcome: Succeeded.
- Learned: No local monitor errors occurred during the wait.

## 2026-05-03T22:45:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-022544 > tmp/iris_pressure_test_20260504/mmlu_job_list_003.json 2> tmp/iris_pressure_test_20260504/mmlu_job_list_003.stderr`
- Why: Capture a fresh structured status snapshot after the wait.
- Outcome: Succeeded.
- Learned: Job remains `JOB_STATE_PENDING` with one pending task; pending reason still cites insufficient matching v5p-8 resources and autoscaler quota-tier blocking.

## 2026-05-03T22:45:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-022544 > tmp/iris_pressure_test_20260504/mmlu_job_summary_003.json 2> tmp/iris_pressure_test_20260504/mmlu_job_summary_003.stderr`
- Why: Confirm whether the task has ever received a worker allocation.
- Outcome: Succeeded.
- Learned: The task is still pending with no worker id, no duration, and zero recorded resource usage.

## 2026-05-03T22:45:00-0400

- Command: `jq '.[0] | {state, task_state_counts, pending_reason, failure_count, preemption_count}' tmp/iris_pressure_test_20260504/mmlu_job_list_003.json`, `jq '{state, task_state_counts, tasks}' tmp/iris_pressure_test_20260504/mmlu_job_summary_003.json`, `wc -c tmp/iris_pressure_test_20260504/mmlu_job_list_003.stderr tmp/iris_pressure_test_20260504/mmlu_job_summary_003.stderr`
- Why: Inspect compact status fields without loading full controller output into the conversation.
- Outcome: Succeeded.
- Learned: There are no failures or preemptions; this is still a pure scheduling/capacity wait.

## 2026-05-03T22:45:00-0400

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/findings.md`
- Why: Record the sustained pending state and the expanded pending reason.
- Outcome: Succeeded.
- Learned: The compact findings file now distinguishes scheduler capacity wait from design/runtime evidence.

## 2026-05-03T22:56:00-0400

- Command: `sleep 570`
- Why: Continue the single babysit loop after the job remained pending.
- Outcome: Succeeded.
- Learned: No local monitor errors occurred during the wait.

## 2026-05-03T22:56:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-022544 > tmp/iris_pressure_test_20260504/mmlu_job_list_004.json 2> tmp/iris_pressure_test_20260504/mmlu_job_list_004.stderr`
- Why: Capture the next structured status snapshot.
- Outcome: Succeeded.
- Learned: Job reached terminal state `JOB_STATE_KILLED`; task state is `killed`; controller reason is `Terminated by user`.

## 2026-05-03T22:56:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-022544 > tmp/iris_pressure_test_20260504/mmlu_job_summary_004.json 2> tmp/iris_pressure_test_20260504/mmlu_job_summary_004.stderr`
- Why: Check whether the killed job ever allocated a worker or produced task runtime signals.
- Outcome: Succeeded.
- Learned: The task was killed before worker assignment; it has no duration, no worker id, and no resource usage.

## 2026-05-03T22:56:00-0400

- Command: `jq '.[0] | {state, task_state_counts, pending_reason, failure_count, preemption_count}' tmp/iris_pressure_test_20260504/mmlu_job_list_004.json`, `jq '{state, task_state_counts, tasks}' tmp/iris_pressure_test_20260504/mmlu_job_summary_004.json`
- Why: Inspect compact terminal status fields.
- Outcome: Succeeded.
- Learned: This terminal state provides no vLLM or lm-eval evidence; it only proves the first material attempt was accepted, waited for capacity, then was killed before scheduling.

## 2026-05-03T22:58:00-0400

- Command: `tail -n 180 tmp/iris_pressure_test_20260504/run_log.md`, `find scratch -maxdepth 1 -type f -name '*monitoring_state.json' -print`, `find tmp/iris_pressure_test_20260504 -maxdepth 1 -type f -name '*fallback*' -print`
- Why: Reconcile active state after discovering `findings.md` had newer fallback-run information than the resume summary.
- Outcome: Succeeded.
- Learned: A fallback v6e-4 Qwen probe had already run, exposed `vllm.utils.get_open_port` dependency skew, and submitted shimmed job `/romain/iris-run-run_existing_vllm_lm_eval-20260504-025615`.

## 2026-05-03T22:58:00-0400

- Command: `find tmp/iris_pressure_test_20260504 -maxdepth 1 -type f -name 'qwen3_v6e4_mmlu*' -print`, `find tmp/iris_pressure_test_20260504 -maxdepth 1 -type f -name 'submit_qwen3*' -print`, `sed -n '1,220p' tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py`
- Why: Inspect fallback artifacts and confirm the scratch-only import shim in the runner.
- Outcome: Succeeded.
- Learned: The runner now patches `vllm.utils.get_open_port` locally before invoking the existing Marin evaluator.

## 2026-05-03T22:58:00-0400

- Command: `jq '{file:input_filename, job_id:.[0].job_id, state:.[0].state, task_state_counts:.[0].task_state_counts, pending_reason:.[0].pending_reason}' tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_list_001.json tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_list_002.json tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_list_003.json`, `cat tmp/iris_pressure_test_20260504/submit_qwen3_v6e4_mmlu_shim_001.out`, `jq '{file:input_filename, state, task_state_counts, tasks}' tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_job_summary_003.json`
- Why: Reconstruct first fallback outcome and shimmed resubmission id.
- Outcome: Succeeded.
- Learned: First fallback `/romain/iris-run-run_existing_vllm_lm_eval-20260504-024730` failed after ~182s with exit 1; shimmed fallback was submitted as `/romain/iris-run-run_existing_vllm_lm_eval-20260504-025615`.

## 2026-05-03T22:58:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-025615 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_job_list_001.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_job_list_001.stderr`
- Why: Capture current state of the shimmed fallback job.
- Outcome: Succeeded.
- Learned: Job is `JOB_STATE_RUNNING` with one running task.

## 2026-05-03T22:58:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-025615 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_job_summary_001.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_job_summary_001.stderr`
- Why: Confirm worker assignment and resource usage for the shimmed fallback.
- Outcome: Succeeded.
- Learned: The task is running on `marin-tpu-v6e-preemptible-4-us-east1-d-20260504-0155-11d43b66-worker-0` with non-zero CPU and memory.

## 2026-05-03T22:58:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml rpc controller get-task-logs --id /romain/iris-run-run_existing_vllm_lm_eval-20260504-025615 --include-children --max-total-lines 3000 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_controller_logs_001.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_controller_logs_001.stderr`
- Why: Fetch logs through controller RPC because regular `iris job logs` was failing via Finelog.
- Outcome: Succeeded.
- Learned: Logs show runner config and Marin eval startup; no HTTP scoring request yet in the captured excerpt.

## 2026-05-03T22:58:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 900 --max-lines 300 /romain/iris-run-run_existing_vllm_lm_eval-20260504-025615 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_job_logs_001.txt 2>&1`
- Why: Check whether the standard job log path recovered for the shimmed fallback.
- Outcome: Failed with `finelog.errors.StatsError: Not Found`.
- Learned: Use controller RPC logs for this pressure test; the job-log CLI issue is orthogonal to vLLM/lm-eval.

## 2026-05-03T22:58:00-0400

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/findings.md` and `scratch/20260503-2246_monitoring_state.json`
- Why: Make the active monitor state reflect the shimmed fallback job and record the canonical job stop.
- Outcome: Succeeded.
- Learned: Current handoff state now points to `/romain/iris-run-run_existing_vllm_lm_eval-20260504-025615`.

## 2026-05-03T23:15:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-025615 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_job_list_002.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_job_list_002.stderr`; `uv run iris --config lib/iris/examples/marin.yaml rpc controller get-task-logs --id /romain/iris-run-run_existing_vllm_lm_eval-20260504-025615 --include-children --max-total-lines 6000 > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_controller_logs_002.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_controller_logs_002.stderr`
- Why: Close out the patched MMLU mechanics probe.
- Outcome: Job failed with `RuntimeError: lm-eval failed`; controller logs were captured.
- Learned: The run reached real `/v1/completions` loglikelihood requests and failed with repeated vLLM HTTP 500 responses (`32658`).

## 2026-05-03T23:15:00-0400

- Command: `jq -r '.. | objects | select(has("data")) | .data' tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_controller_logs_002.json > tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_controller_logs_002_lines.txt`; `rg -n -i "ERROR|Traceback|Exception|HTTPError|500 Server Error|/v1/completions|logprob|logprobs|ValueError|RuntimeError|vllm" tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_controller_logs_002_lines.txt`; `tail -n 220 tmp/iris_pressure_test_20260504/qwen3_v6e4_mmlu_shim_controller_logs_002_lines.txt`
- Why: Extract the actionable MMLU failure from controller JSON.
- Outcome: Succeeded.
- Learned: lm-eval used `loglikelihood`, called `/v1/completions`, and failed after retries on HTTP 500. The first run also had a context-length warning, so a larger-context follow-up was warranted.

## 2026-05-03T23:15:00-0400

- Command: `python -m py_compile tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py`; `uv run python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --dry-run --task humaneval_5shot --model-name qwen3-0_6b-mechanics --model-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --tpu-type v6e-4 --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_humaneval_5shot_shim' --limit 1 > tmp/iris_pressure_test_20260504/local_dry_run_qwen3_v6e4_humaneval_shim.txt 2>&1`
- Why: Validate the scratch runner before a HumanEval chat/generation probe.
- Outcome: Succeeded.
- Learned: HumanEval maps to `apply_chat_template=True` with the same shimmed runner.

## 2026-05-03T23:15:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --enable-extra-resources --tpu v6e-4 --cpu 32 --memory 128GB --disk 50GB --timeout 7200 --extra eval --extra vllm --extra tpu -e VLLM_ENABLE_V1_MULTIPROCESSING 0 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 -e VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION 1 -e VLLM_TPU_SKIP_PRECOMPILE 1 -e HF_ALLOW_CODE_EVAL 1 -e WANDB_MODE offline -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task humaneval_5shot --model-name qwen3-0_6b-mechanics --model-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --tpu-type v6e-4 --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_humaneval_5shot_shim' --limit 1 > tmp/iris_pressure_test_20260504/submit_qwen3_v6e4_humaneval_shim_001.out 2>&1`
- Why: Submit a small HumanEval mechanics probe.
- Outcome: Succeeded; job id `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030417`.
- Learned: The job initially scheduled on a Europe worker because no zone was pinned.

## 2026-05-03T23:15:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job stop /romain/iris-run-run_existing_vllm_lm_eval-20260504-030417 > tmp/iris_pressure_test_20260504/stop_qwen3_v6e4_humaneval_europe_001.txt 2>&1`
- Why: Stop the accidental Europe-scheduled fallback before it could do unnecessary region/bandwidth work.
- Outcome: Succeeded.
- Learned: Resubmit HumanEval pinned to `us-east1-d`.

## 2026-05-03T23:15:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --enable-extra-resources --tpu v6e-4 --zone us-east1-d --cpu 32 --memory 128GB --disk 50GB --timeout 7200 --extra eval --extra vllm --extra tpu -e VLLM_ENABLE_V1_MULTIPROCESSING 0 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 -e VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION 1 -e VLLM_TPU_SKIP_PRECOMPILE 1 -e HF_ALLOW_CODE_EVAL 1 -e WANDB_MODE offline -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task humaneval_5shot --model-name qwen3-0_6b-mechanics --model-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --tpu-type v6e-4 --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_us_east1_humaneval_5shot_shim' --limit 1 > tmp/iris_pressure_test_20260504/submit_qwen3_v6e4_useast_humaneval_shim_001.out 2>&1`
- Why: Re-run HumanEval in the closer v6e zone already used by the MMLU mechanics probes.
- Outcome: Succeeded; job id `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030528`.
- Learned: The pinned HumanEval job scheduled on `us-east1-d`.

## 2026-05-03T23:15:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --enable-extra-resources --tpu v5p-8 --zone us-central1-a --cpu 32 --memory 128GB --disk 50GB --timeout 7200 --extra eval --extra vllm --extra tpu -e VLLM_ENABLE_V1_MULTIPROCESSING 0 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN 1 -e VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION 1 -e VLLM_TPU_SKIP_PRECOMPILE 1 -e HF_ALLOW_CODE_EVAL 1 -e WANDB_MODE offline -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task mmlu_sl_verb_5shot --model-name adamh-1e22-standin --model-path 'gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf/step-38234' --tokenizer meta-llama/Meta-Llama-3.1-8B --tpu-type v5p-8 --output-path 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/mmlu_sl_verb_5shot_shim' --limit 1 > tmp/iris_pressure_test_20260504/submit_canonical_mmlu_shim_001.out 2>&1`
- Why: Resubmit the canonical v5p-8 MMLU attempt with the import shim included.
- Outcome: Succeeded; job id `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030623`.
- Learned: The patched canonical attempt is accepted but pending on the same v5p-8 capacity blocker.

## 2026-05-03T23:15:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-030528 > tmp/iris_pressure_test_20260504/qwen3_v6e4_useast_humaneval_shim_job_list_002.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_useast_humaneval_shim_job_list_002.stderr`; `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-030528 > tmp/iris_pressure_test_20260504/qwen3_v6e4_useast_humaneval_shim_job_summary_002.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_useast_humaneval_shim_job_summary_002.stderr`; `uv run iris --config lib/iris/examples/marin.yaml rpc controller get-task-logs --id /romain/iris-run-run_existing_vllm_lm_eval-20260504-030528 --include-children --max-total-lines 6000 > tmp/iris_pressure_test_20260504/qwen3_v6e4_useast_humaneval_shim_controller_logs_001.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_useast_humaneval_shim_controller_logs_001.stderr`
- Why: Collect terminal status and logs for pinned HumanEval.
- Outcome: Succeeded; job completed with exit 0 in 147s.
- Learned: HumanEval successfully exercised lm-eval `local-chat-completions` against `/v1/chat/completions`.

## 2026-05-03T23:15:00-0400

- Command: `gcloud storage ls --recursive 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_us_east1_humaneval_5shot_shim/**' > tmp/iris_pressure_test_20260504/gcloud_ls_humaneval_qwen3_useast_output_001.txt 2>&1`; `gcloud storage cat 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_us_east1_humaneval_5shot_shim/humaneval_5shot/Qwen__Qwen3-0.6B/results_2026-05-04T03-08-08.644824.json' > tmp/iris_pressure_test_20260504/humaneval_qwen3_useast_results_001.json 2> tmp/iris_pressure_test_20260504/humaneval_qwen3_useast_results_001.stderr`; `gcloud storage cat 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_us_east1_humaneval_5shot_shim/humaneval_5shot/Qwen__Qwen3-0.6B/samples_humaneval_2026-05-04T03-08-08.644824.jsonl' > tmp/iris_pressure_test_20260504/humaneval_qwen3_useast_samples_001.jsonl 2> tmp/iris_pressure_test_20260504/humaneval_qwen3_useast_samples_001.stderr`
- Why: Verify HumanEval result artifacts were uploaded and inspect the saved metadata.
- Outcome: Succeeded.
- Learned: Results metadata confirms `model_source=local-chat-completions`, `base_url=http://127.0.0.1:8000/v1/chat/completions`, and `pass@1,create_test=0.0` for the limit-1 sample.

## 2026-05-03T23:15:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job stop /romain/iris-run-run_existing_vllm_lm_eval-20260504-031005 > tmp/iris_pressure_test_20260504/stop_redundant_humaneval_europe_031005_001.txt 2>&1`
- Why: Stop a redundant unpinned HumanEval job that scheduled in `europe-west4` after the pinned run had already succeeded.
- Outcome: Succeeded.
- Learned: Avoid further unnecessary region/bandwidth work.

## 2026-05-03T23:15:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-030919 > tmp/iris_pressure_test_20260504/mmlu_4096_lmeval_030919_job_list_002.json 2> tmp/iris_pressure_test_20260504/mmlu_4096_lmeval_030919_job_list_002.stderr`; `uv run iris --config lib/iris/examples/marin.yaml rpc controller get-task-logs --id /romain/iris-run-run_existing_vllm_lm_eval-20260504-030919 --include-children --max-total-lines 8000 > tmp/iris_pressure_test_20260504/mmlu_4096_lmeval_030919_controller_logs_002.json 2> tmp/iris_pressure_test_20260504/mmlu_4096_lmeval_030919_controller_logs_002.stderr`
- Why: Close out the context-controlled MMLU follow-up.
- Outcome: Succeeded; job failed with `RuntimeError: lm-eval failed`.
- Learned: Even with `max_model_len=4096` and lm-eval `max_length=4096`, `/v1/completions` loglikelihood fails with vLLM HTTP 500 (`2701`), so the blocker is not just the context-length warning.

## 2026-05-03T23:15:00-0400

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/findings.md` and `scratch/20260503-2246_monitoring_state.json`
- Why: Record final fallback MMLU/HumanEval results and point the handoff state at the still-pending canonical v5p-8 attempt.
- Outcome: Succeeded.
- Learned: Findings now distinguish canonical capacity blockage from fallback design evidence.

## 2026-05-03T23:16:00-0400

- Command: `sed -n '1,260p' .agents/projects/iris_inference_service/design.md`, `sed -n '1,280p' .agents/projects/iris_inference_service/spec.md`, `sed -n '1,260p' .agents/projects/iris_inference_service/research.md`
- Why: Read the design docs before patching them with pressure-test evidence.
- Outcome: Succeeded.
- Learned: The docs described the intended milestone but did not yet record the current vLLM logprob blocker, model artifact issue, or HumanEval success signal.

## 2026-05-03T23:16:00-0400

- Command: `apply_patch` update `.agents/projects/iris_inference_service/design.md`, `.agents/projects/iris_inference_service/spec.md`, and `.agents/projects/iris_inference_service/research.md`
- Why: Document the pressure-test findings locally for the draft design PR without pushing.
- Outcome: Succeeded.
- Learned: The tracked docs now state that chat/generation looks viable, completions logprob scoring is the primary blocker, the exact MoE artifact is not vLLM-loadable as found, and the current canonical v5p-8 attempt is pending capacity.

## 2026-05-03T23:16:00-0400

- Command: `./infra/pre-commit.py --all-files --fix > tmp/iris_pressure_test_20260504/pre_commit_all_files_fix_001.txt 2>&1`
- Why: Run the required repo check after editing tracked docs.
- Outcome: Succeeded.
- Learned: Ruff, Black, license headers, Pyrefly, Markdown, and other pre-commit checks passed.

## 2026-05-03T23:16:00-0400

- Command: `tail -n 120 tmp/iris_pressure_test_20260504/pre_commit_all_files_fix_001.txt`, `git status --short`, `git diff -- .agents/projects/iris_inference_service/design.md .agents/projects/iris_inference_service/spec.md .agents/projects/iris_inference_service/research.md`
- Why: Verify check output and summarize the local tracked diff.
- Outcome: Succeeded.
- Learned: Only the three design docs are modified among tracked files; existing untracked `report.md` and `tmp/` remain untracked.

## 2026-05-03T23:30:00-0400

- Command: `sleep 570`
- Why: Give the patched canonical v5p-8 MMLU attempt another babysit cadence after it remained pending.
- Outcome: Succeeded.
- Learned: No local monitor errors occurred during the wait.

## 2026-05-03T23:30:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-030623 > tmp/iris_pressure_test_20260504/canonical_mmlu_shim_job_list_003.json 2> tmp/iris_pressure_test_20260504/canonical_mmlu_shim_job_list_003.stderr`
- Why: Capture final canonical v5p-8 status snapshot.
- Outcome: Succeeded.
- Learned: Job remains `JOB_STATE_PENDING` with one pending task.

## 2026-05-03T23:30:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-030623 > tmp/iris_pressure_test_20260504/canonical_mmlu_shim_job_summary_003.json 2> tmp/iris_pressure_test_20260504/canonical_mmlu_shim_job_summary_003.stderr`
- Why: Confirm whether the canonical task has received a worker.
- Outcome: Succeeded.
- Learned: The canonical task has no worker id, duration, or resource usage yet.

## 2026-05-03T23:30:00-0400

- Command: `jq '.[0] | {job_id, state, task_state_counts, pending_reason, failure_count, preemption_count, error}' tmp/iris_pressure_test_20260504/canonical_mmlu_shim_job_list_003.json`; `jq '{state, task_state_counts, tasks}' tmp/iris_pressure_test_20260504/canonical_mmlu_shim_job_summary_003.json`
- Why: Inspect compact canonical status fields.
- Outcome: Succeeded.
- Learned: The pending reason is still insufficient matching v5p-8 memory/TPUs plus autoscaler quota-tier blocking.

## 2026-05-03T23:30:00-0400

- Command: `rg -n "Latest live|Status: `JOB_STATE_PENDING`|Active|Current" tmp/iris_pressure_test_20260504/findings.md`
- Why: Locate status lines before updating findings.
- Outcome: Failed partially because shell backticks in the pattern attempted command substitution; useful matching output still showed the stale timestamp.
- Learned: Escape backticks or avoid them in shell patterns.

## 2026-05-03T23:30:00-0400

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/findings.md`
- Why: Record the final canonical status timestamp.
- Outcome: Succeeded.
- Learned: Findings now report the patched canonical job as pending at 2026-05-03T23:30:00-0400.

## 2026-05-03T23:31:00-0400

- Command: `git status --short`
- Why: Final local worktree sanity check.
- Outcome: Succeeded.
- Learned: Tracked changes are limited to the three design docs; `report.md` and `tmp/` are untracked.

## 2026-05-03T23:31:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-03 > tmp/iris_pressure_test_20260504/recent_iris_run_eval_jobs_final.json 2> tmp/iris_pressure_test_20260504/recent_iris_run_eval_jobs_final.stderr`; `jq 'map({job_id, state, task_state_counts, pending_reason, error})' tmp/iris_pressure_test_20260504/recent_iris_run_eval_jobs_final.json`
- Why: Verify no unexpected recent eval jobs are still running.
- Outcome: Succeeded.
- Learned: Only `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030623` is still non-terminal, and it is pending capacity; recent fallback jobs are succeeded, failed, or killed.

## 2026-05-03T23:31:00-0400

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/findings.md`
- Why: Record that the final recent-job scan found no other running pressure-test jobs.
- Outcome: Succeeded.
- Learned: The only active external work left is the canonical v5p-8 job pending on capacity.

## 2026-05-03T23:18:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job stop /romain/iris-run-run_existing_vllm_lm_eval-20260504-031612 > tmp/iris_pressure_test_20260504/stop_redundant_humaneval_rerun_001.txt 2>&1`
- Why: Stop a redundant HumanEval rerun submitted before reconciling the already-successful pinned HumanEval run.
- Outcome: Succeeded.
- Learned: Active generation pressure testing is complete; no extra HumanEval jobs should remain running.

## 2026-05-03T23:19:00-0400

- Command: `gcloud storage ls 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_us_east1_humaneval_5shot_shim/**' > tmp/iris_pressure_test_20260504/qwen3_v6e4_humaneval_success_gcs_ls_001.txt 2>&1`; `gcloud storage cp ...results_2026-05-04T03-08-08.644824.json tmp/iris_pressure_test_20260504/qwen3_v6e4_humaneval_success_results.json`; `gcloud storage cp ...samples_humaneval_2026-05-04T03-08-08.644824.jsonl tmp/iris_pressure_test_20260504/qwen3_v6e4_humaneval_success_samples.jsonl`
- Why: Independently verify the successful HumanEval output artifacts and inspect the saved result/sample layout.
- Outcome: Succeeded.
- Learned: The uploaded artifacts include one aggregate results JSON and one samples JSONL. The sample proves `/v1/chat/completions` generation worked, but the prompt appears to be the standard HumanEval prompt rather than five external few-shot examples; result config reports `num_fewshot: 0`.

## 2026-05-03T23:21:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-030623 > tmp/iris_pressure_test_20260504/patched_canonical_mmlu_job_list_003.json 2> tmp/iris_pressure_test_20260504/patched_canonical_mmlu_job_list_003.stderr`; `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-031612 > tmp/iris_pressure_test_20260504/qwen3_v6e4_humaneval_rerun_summary_002.json 2> tmp/iris_pressure_test_20260504/qwen3_v6e4_humaneval_rerun_summary_002.stderr`
- Why: Check active work after stopping the redundant HumanEval rerun.
- Outcome: HumanEval rerun is killed; patched canonical MMLU remains pending.
- Learned: The only remaining live pressure-test job is `/romain/iris-run-run_existing_vllm_lm_eval-20260504-030623`, still blocked by v5p-8 capacity and autoscaler quota-pool tier monotonicity.

## 2026-05-03T23:44:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-030623 > tmp/iris_pressure_test_20260504/patched_canonical_mmlu_job_list_005.json 2> tmp/iris_pressure_test_20260504/patched_canonical_mmlu_job_list_005.stderr`; `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-030623 > tmp/iris_pressure_test_20260504/patched_canonical_mmlu_summary_004.json 2> tmp/iris_pressure_test_20260504/patched_canonical_mmlu_summary_004.stderr`
- Why: Continue babysitting the patched canonical v5p-8 attempt.
- Outcome: Still pending; no worker assignment yet.
- Learned: Scheduler still sees insufficient current v5p-8 memory/TPUs, but the autoscaler state changed to waiting for workers in `tpu_v5p-preemptible_8-us-central1-a` to become ready. This is progress from the earlier quota-tier blocker, so keep monitoring.

## 2026-05-03T23:55:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-030623 > tmp/iris_pressure_test_20260504/patched_canonical_mmlu_job_list_006.json 2> tmp/iris_pressure_test_20260504/patched_canonical_mmlu_job_list_006.stderr`; `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-030623 > tmp/iris_pressure_test_20260504/patched_canonical_mmlu_summary_005.json 2> tmp/iris_pressure_test_20260504/patched_canonical_mmlu_summary_005.stderr`
- Why: Continue babysitting the patched canonical v5p-8 attempt after the autoscaler began waiting for workers.
- Outcome: Still pending; no worker assignment yet.
- Learned: Autoscaler is still waiting for workers in `tpu_v5p-preemptible_8-us-central1-a` to become ready. Current matching workers still lack enough available memory.

## 2026-05-04T00:06:00-0400

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval-20260504-030623 > tmp/iris_pressure_test_20260504/patched_canonical_mmlu_job_list_007.json 2> tmp/iris_pressure_test_20260504/patched_canonical_mmlu_job_list_007.stderr`; `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-030623 > tmp/iris_pressure_test_20260504/patched_canonical_mmlu_summary_006.json 2> tmp/iris_pressure_test_20260504/patched_canonical_mmlu_summary_006.stderr`
- Why: Continue babysitting the patched canonical v5p-8 attempt after waiting for autoscaled workers.
- Outcome: Still pending; no worker assignment yet.
- Learned: Autoscaler state regressed to quota-pool tier blocking. The canonical run remains capacity-blocked and has produced no runtime evidence.

## 2026-05-03T23:32:00-0400

- Command: `apply_patch` update `.agents/projects/iris_inference_service/design.md`
- Why: Add the HumanEval `num_fewshot: 0` caveat to the design addendum so the docs do not overclaim exact `humaneval_5shot` acceptance.
- Outcome: Succeeded.
- Learned: `design.md`, `spec.md`, and `research.md` now all distinguish endpoint compatibility from exact HumanEval milestone acceptance.

## 2026-05-03T23:32:00-0400

- Command: `./infra/pre-commit.py --all-files --fix > tmp/iris_pressure_test_20260504/pre_commit_all_files_fix_002.txt 2>&1`
- Why: Re-run required repo checks after the tracked doc correction.
- Outcome: Succeeded.
- Learned: Full pre-commit entry point passed again.

## 2026-05-03T23:32:00-0400

- Command: `tail -n 80 tmp/iris_pressure_test_20260504/pre_commit_all_files_fix_002.txt`, `git status --short`, `git diff --stat`
- Why: Final check of validation output and local diff.
- Outcome: Succeeded.
- Learned: Tracked diff is three docs, 99 insertions and 1 deletion; untracked `report.md` and `tmp/` remain.

## 2026-05-04T14:37:00-0700

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name prompt-logprob-baseline-20260504T213715Z --enable-extra-resources --tpu v6e-4 --zone europe-west4-a ... -- python tmp/iris_pressure_test_20260504/run_direct_prompt_logprob_vllm.py ...`
- Why: Reproduce the MMLU prompt-logprob failure without lm-eval in the loop, on a non-`us-central1-a` worker zone.
- Outcome: Succeeded as a captured baseline; the runner exited 0 after preserving the HTTP response and vLLM logs.
- Learned: vLLM 0.18.0 on TPU returned HTTP 500 for `POST /v1/completions` with `echo=true`, `logprobs=1`, and `max_tokens=1`; server logs show `KeyError: 425` in `_create_completion_logprobs`.

## 2026-05-04T14:45:00-0700

- Command: `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name humaneval-5shot-qwen3-v6e4-ew4a-20260504T214501Z --enable-extra-resources --tpu v6e-4 --zone europe-west4-a ... -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task humaneval_5shot ... --limit 1`
- Why: Check whether a real lm-eval HumanEval generation run can succeed on Iris and whether explicit `EvalTaskConfig("humaneval", 5, task_alias="humaneval_5shot")` is honored.
- Outcome: Succeeded; uploaded result and sample artifacts under `gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_v6e4_euw4_humaneval_5shot_20260504T214501Z/`.
- Learned: The generation path works through `local-chat-completions` and `/v1/chat/completions`, but stock lm-eval saved `num_fewshot: 0` and `n-shot: 0`; exact `humaneval_5shot` needs a custom verified task/config or different milestone wording.

## 2026-05-04T14:50:00-0700

- Command: `apply_patch` update `.agents/projects/iris_inference_service/{design.md,spec.md,research.md}` and local issue-search notes.
- Why: Keep the design spec aligned with the new pressure-test evidence.
- Outcome: Succeeded.
- Learned: The spec now splits the MVP generation track from the MMLU prompt-logprob blocker and avoids claiming exact HumanEval few-shot support.

## 2026-05-13T16:20:00-0700

- Command: Read `.agents/skills/agent-research/SKILL.md`, `.agents/skills/debug-infra/SKILL.md`, `lib/iris/AGENTS.md`, and `lib/iris/OPS.md`.
- Why: Refresh the applicable workflow and Iris guardrails before touching old pressure-test state.
- Outcome: Succeeded.
- Learned: Continue treating this as a research/debug refresh; do not restart clusters; use read-only Iris job inspection for live state.

## 2026-05-13T16:22:00-0700

- Command: Search `/Users/romain/.codex/memories/MEMORY.md` for Iris/vLLM/lm-eval pressure-test context.
- Why: Recover what changed since the May 4 handoff without relying only on stale local files.
- Outcome: Succeeded.
- Learned: The relevant newer memory thread is the TPU/vLLM lane: prompt-logprob completions remain the eval-critical missing feature, and a 0.19 stack PR was prepared separately.

## 2026-05-13T16:24:00-0700

- Command: GitHub connector `get_pr_info`, `fetch_pr_comments`, `list_pull_request_reviews`, and `list_pr_changed_filenames` for PR #5400.
- Why: Check whether the design PR changed, received review, or moved out of draft.
- Outcome: Succeeded.
- Learned: PR #5400 is still open, draft, unchanged since 2026-05-04, with no comments or reviews and only the three design-doc files changed.

## 2026-05-13T16:25:00-0700

- Command: `git fetch origin --prune`; `git status --short --branch`; `git rev-list --left-right --count HEAD...origin/main`; `git diff --stat`.
- Why: Refresh remote refs and measure local drift before recommending branch work.
- Outcome: Succeeded.
- Learned: `origin/main` advanced to `13a0ef6e6`; local `design/iris_inference_service` is ahead 1 and behind 157, with uncommitted edits in the three design docs.

## 2026-05-13T16:26:00-0700

- Command: `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /romain/iris-run-run_existing_vllm_lm_eval > tmp/iris_pressure_test_20260504/refresh_20260513_lm_eval_jobs.json 2> tmp/iris_pressure_test_20260504/refresh_20260513_lm_eval_jobs.stderr`
- Why: Check whether any old pressure-test Iris jobs are still present or running.
- Outcome: Succeeded; output was `[]`.
- Learned: No old `iris-run-run_existing_vllm_lm_eval` jobs are listed in the live controller now.

## 2026-05-13T16:29:00-0700

- Command: `uv run iris --config lib/iris/examples/marin.yaml job summary --json /romain/iris-run-run_existing_vllm_lm_eval-20260504-030623 > tmp/iris_pressure_test_20260504/refresh_20260513_canonical_mmlu_030623_summary.json 2> tmp/iris_pressure_test_20260504/refresh_20260513_canonical_mmlu_030623_summary.stderr`
- Why: Directly refresh the last known pending canonical v5p-8 MMLU job.
- Outcome: Failed with `Job /romain/iris-run-run_existing_vllm_lm_eval-20260504-030623 not found`.
- Learned: The old pending canonical job is no longer retained in the live controller job set; it should not be considered active.

## 2026-05-13T16:29:00-0700

- Command: `git grep -n 'vllm-tpu\|tpu-inference\|libtpu' origin/main -- pyproject.toml uv.lock lib/marin/pyproject.toml lib/levanter/pyproject.toml`
- Why: Check whether the TPU vLLM dependency lane had already landed on `origin/main`.
- Outcome: Succeeded.
- Learned: Current `origin/main` still pins `vllm-tpu==0.18.0`, `tpu-inference==0.18.0`, and `libtpu==0.0.38`.

## 2026-05-13T16:30:00-0700

- Command: GitHub connector search and reads for PR #5712 and issue #5672.
- Why: Identify newer live work that supersedes or changes the design pressure-test next step.
- Outcome: Succeeded.
- Learned: PR #5712 is open and mergeable for the 0.19 TPU vLLM stack. Issue #5672 is open and now has a 1e22 repro on the 0.19 stack that fails before vLLM readiness with RPA scoped-VMEM OOM.

## 2026-05-13T16:31:00-0700

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/findings.md`.
- Why: Record the May 13 live refresh and next-step recommendation.
- Outcome: Succeeded.
- Learned: Findings now distinguish the stale May 4 pressure-test state from the current #5712/#5672 blockers.

## 2026-05-13T16:45:00-0700

- Command: GitHub connector `get_pr_info` and `get_commit_combined_status` for PR #5712; `git ls-remote origin refs/heads/codex/vllm-tpu-019 refs/heads/design/iris_inference_service`.
- Why: Re-check whether #5712 can be used as a base for unblocking the inference-service work.
- Outcome: Succeeded.
- Learned: #5712 is still open, mergeable, and headed at `323805384b7ac0e393f34a9864a0dc6b96050951`; `origin/design/iris_inference_service` remains at `69569a5d268af83024cafc17410907f04cfe5900`.

## 2026-05-13T16:47:00-0700

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/findings.md`.
- Why: Revise the recommendation after the user clarified that #5672 should not block first service shipment.
- Outcome: Succeeded.
- Learned: The recommended path is now to build and validate the service on top of #5712 with a standard vLLM-friendly model, while treating Delphi/1e22 and MMLU prompt-logprob scoring as follow-up acceptance tracks.

## 2026-05-13T17:18:00-0700

- Command: Read `.agents/skills/pull-request/SKILL.md`, `.agents/skills/agent-research/SKILL.md`, `.agents/skills/debug-infra/SKILL.md`, `.agents/skills/babysit-job/SKILL.md`; `git status --short --branch`; `git log --oneline --decorate -6`.
- Why: Resume after context compaction and confirm the branch, applicable playbooks, and local rebase state before touching live Iris jobs or tracked files.
- Outcome: Succeeded.
- Learned: Branch `design/iris_inference_service` is locally rebased onto `origin/codex/vllm-tpu-019` and has uncommitted RFC doc edits plus untracked `report.md` and `tmp/`.

## 2026-05-13T17:20:00-0700

- Command: Read `lib/iris/AGENTS.md`, `lib/iris/OPS.md`, current RFC doc diff, scratch runner, `tmp/iris_pressure_test_20260504/run_log.md`, and `tmp/iris_pressure_test_20260504/findings.md`.
- Why: Refresh Iris guardrails, current pressure-test conclusions, and the exact local runner code before resubmitting.
- Outcome: Succeeded.
- Learned: The scratch runner already defaults to `standard_humaneval_smoke`, `Qwen/Qwen3-0.6B`, and `max_gen_toks=128`; RFC docs split MVP standard-model service validation from 1e22/MMLU follow-up validation.

## 2026-05-13T17:27:00-0700

- Command: `uv run --package marin --extra eval --extra cpu python -m py_compile tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py > tmp/iris_pressure_test_20260504/rebased_runner_py_compile_20260513T1727.txt 2>&1`
- Why: Revalidate the scratch runner after the max-generation-token fix.
- Outcome: Succeeded with empty stdout/stderr artifact.
- Learned: The runner is syntactically valid under the package/env used for local eval tooling.

## 2026-05-13T17:27:00-0700

- Command: `uv run --package marin --extra eval --extra cpu python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --dry-run > tmp/iris_pressure_test_20260504/rebased_runner_dry_run_20260513T1727.txt 2>&1`
- Why: Confirm the runner shape before spending Iris capacity.
- Outcome: Succeeded.
- Learned: Dry-run config uses `standard_humaneval_smoke`, stock `humaneval` with `num_fewshot=0`, `apply_chat_template=True`, and `engine_kwargs` including `max_num_batched_tokens=1024` and `max_gen_toks=128`.

## 2026-05-13T17:27:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job summary --json /romain/iris-inference-standard-smoke-20260514T000934Z > tmp/iris_pressure_test_20260504/rebased_standard_smoke_summary_resume_20260513T1727.json 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_summary_resume_20260513T1727.stderr`
- Why: Reconfirm the previous live smoke result before resubmitting the fixed runner.
- Outcome: Succeeded; summary reports `failed`.
- Learned: The previous job failed with `RuntimeError: lm-eval failed`, matching the earlier logprob/output-token diagnosis rather than a scheduling issue.

## 2026-05-13T17:27:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --job-name iris-inference-standard-smoke-20260514T002749Z --enable-extra-resources --tpu v6e-4 --cpu 16 --memory 64GB --disk 100GB --timeout 2400 --extra eval --extra tpu --extra vllm -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task standard_humaneval_smoke --model-name qwen3-0.6b-standard-smoke --model-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --output-path gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_standard_smoke_20260514T002749Z --tpu-type v6e-4 --limit 1 --max-model-len 1024 --max-num-batched-tokens 1024 --max-gen-toks 128 > tmp/iris_pressure_test_20260504/rebased_standard_smoke_submit_20260514T002749Z.out 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_submit_20260514T002749Z.err`
- Why: Resubmit the MVP standard-model smoke with the corrected generation cap.
- Outcome: Succeeded; Iris accepted `/romain/iris-inference-standard-smoke-20260514T002749Z`.
- Learned: The job command now includes `--max-gen-toks 128`, avoiding the previous 1024-token completion request against a 1024-token context.

## 2026-05-13T17:28:00-0700

- Command: `apply_patch` update `scratch/20260513-1708_monitoring_state.json`.
- Why: Keep the resume/babysit state pointed at the latest corrected smoke job and exact resubmit command.
- Outcome: Succeeded.
- Learned: Monitoring state now records `/romain/iris-inference-standard-smoke-20260514T002749Z` and `restart_count=2`.

## 2026-05-13T17:29:00-0700

- Command: `./infra/pre-commit.py --all-files --fix > tmp/iris_pressure_test_20260504/pre_commit_all_files_fix_rebase_20260513T1729.txt 2>&1`
- Why: Run the required repo check after tracked RFC doc edits.
- Outcome: Succeeded.
- Learned: Ruff, Black, license headers, pyrefly, markdown, and general file checks all passed.

## 2026-05-13T17:29:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job list --json --prefix /romain/iris-inference-standard-smoke-20260514T002749Z > tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1729.json 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1729.stderr`
- Why: Check the corrected smoke shortly after submission.
- Outcome: Succeeded; job state was `JOB_STATE_PENDING`.
- Learned: Initial pending reason was no free v6e-4 TPUs plus autoscaler quota-pool tier blocking, but this later cleared.

## 2026-05-13T17:31:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job list --json --prefix /romain/iris-inference-standard-smoke-20260514T002749Z > tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1731.json 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1731.stderr`
- Why: Poll whether the pending v6e-4 job had acquired capacity.
- Outcome: Succeeded; job state was `JOB_STATE_RUNNING` with task state `building`.
- Learned: Iris assigned capacity without manual intervention.

## 2026-05-13T17:32:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job logs /romain/iris-inference-standard-smoke-20260514T002749Z --since-seconds 600 > tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1732.txt 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1732.stderr`
- Why: Inspect early task startup while keeping noisy logs in an artifact.
- Outcome: Succeeded.
- Learned: The task synced deps, activated the venv, and reached `running user command`.

## 2026-05-13T17:33:00-0700

- Command: `git add .agents/projects/iris_inference_service/design.md .agents/projects/iris_inference_service/research.md .agents/projects/iris_inference_service/spec.md && git commit -m "Update Iris inference MVP pressure-test scope" > tmp/iris_pressure_test_20260504/git_commit_rebase_docs_20260513T1733.txt 2>&1`
- Why: Checkpoint the tracked RFC doc edits locally after pre-commit passed.
- Outcome: Succeeded; created commit `045c8ab49`.
- Learned: Branch now has three local design commits on top of #5712; no push was performed.

## 2026-05-13T17:34:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job list --json --prefix /romain/iris-inference-standard-smoke-20260514T002749Z > tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1734.json 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1734.stderr`; `uv run iris --config lib/iris/config/marin.yaml job logs /romain/iris-inference-standard-smoke-20260514T002749Z --since-seconds 600 > tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1734.txt 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1734.stderr`
- Why: Check whether the corrected smoke reached real vLLM/lm-eval execution.
- Outcome: Succeeded; job state was `JOB_STATE_RUNNING` with task state `running`.
- Learned: Logs show TPU v6e-4 detection on `us-east5-b`, runner config with `max_model_len=1024` and `max_gen_toks=128`, and lm-eval startup.

## 2026-05-13T17:37:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job list --json --prefix /romain/iris-inference-standard-smoke-20260514T002749Z > tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1737.json 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1737.stderr`; `uv run iris --config lib/iris/config/marin.yaml job logs /romain/iris-inference-standard-smoke-20260514T002749Z --since-seconds 900 > tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1737.txt 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1737.stderr`
- Why: Continue monitoring the corrected smoke after vLLM/lm-eval startup.
- Outcome: Succeeded; job still running with the same early lm-eval logs.
- Learned: No terminal result yet at this poll.

## 2026-05-13T17:39:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job list --json --prefix /romain/iris-inference-standard-smoke-20260514T002749Z > tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1739.json 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1739.stderr`; `uv run iris --config lib/iris/config/marin.yaml job logs /romain/iris-inference-standard-smoke-20260514T002749Z --since-seconds 1200 > tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1739.txt 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1739.stderr`
- Why: Collect the terminal result for the second rebased smoke attempt.
- Outcome: Succeeded; job state was `JOB_STATE_FAILED`.
- Learned: lm-eval still requested 1024 output tokens. Passing `max_gen_toks=128` via `engine_kwargs` only set the model default; stock HumanEval task YAML `generation_kwargs.max_gen_toks: 1024` overrode it.

## 2026-05-13T17:40:00-0700

- Command: Read `lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py`; search `max_gen_toks`, `gen_kwargs`, and local-completions code in Marin and installed lm-eval; inspect lm-eval API model/task YAML signatures.
- Why: Find the exact argument path needed to override stock HumanEval generation kwargs.
- Outcome: Succeeded.
- Learned: lm-eval's `simple_evaluate` accepts `gen_kwargs`, `local-chat-completions` honors `max_gen_toks`, and stock HumanEval has `generation_kwargs.max_gen_toks: 1024`.

## 2026-05-13T17:42:00-0700

- Command: `apply_patch` update `lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py` and `tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py`.
- Why: Pass existing Marin `generation_params` into lm-eval `simple_evaluate(gen_kwargs=...)` and keep the scratch runner's output-token cap out of vLLM engine kwargs.
- Outcome: Succeeded.
- Learned: This is the smallest code-path fix that lets the eval config override task generation kwargs without changing lm-eval task files.

## 2026-05-13T17:43:00-0700

- Command: `uv run --package marin --extra eval --extra cpu python -m py_compile lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py > tmp/iris_pressure_test_20260504/rebased_generation_params_py_compile_20260513T1743.txt 2>&1`; `uv run --package marin --extra eval --extra cpu python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --dry-run > tmp/iris_pressure_test_20260504/rebased_generation_params_dry_run_20260513T1743.txt 2>&1`
- Why: Validate the evaluator/runner edit before resubmitting to Iris.
- Outcome: Succeeded.
- Learned: Dry-run now shows `engine_kwargs={'tokenizer': 'Qwen/Qwen3-0.6B', 'max_num_batched_tokens': 1024}` and `generation_params={'max_gen_toks': 128}`.

## 2026-05-13T17:35:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job run --no-wait --job-name iris-inference-standard-smoke-20260514T003544Z --enable-extra-resources --tpu v6e-4 --cpu 16 --memory 64GB --disk 100GB --timeout 2400 --extra eval --extra tpu --extra vllm -- python tmp/iris_pressure_test_20260504/run_existing_vllm_lm_eval.py --task standard_humaneval_smoke --model-name qwen3-0.6b-standard-smoke --model-path Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B --output-path gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_standard_smoke_20260514T003544Z --tpu-type v6e-4 --limit 1 --max-model-len 1024 --max-num-batched-tokens 1024 --max-gen-toks 128 > tmp/iris_pressure_test_20260504/rebased_standard_smoke_submit_20260514T003544Z.out 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_submit_20260514T003544Z.err`
- Why: Resubmit the MVP standard-model smoke after wiring `generation_params` into the lm-eval harness.
- Outcome: Succeeded; Iris accepted `/romain/iris-inference-standard-smoke-20260514T003544Z`.
- Learned: Workspace bundle still fit in the normal Iris bundle path; no push was required.

## 2026-05-13T17:36:00-0700

- Command: `apply_patch` update `scratch/20260513-1708_monitoring_state.json`.
- Why: Keep the babysit state pointed at the latest smoke job and exact resubmit command.
- Outcome: Succeeded.
- Learned: Monitoring state now records `/romain/iris-inference-standard-smoke-20260514T003544Z` and `restart_count=3`.

## 2026-05-13T17:36:00-0700

- Command: `./infra/pre-commit.py --all-files --fix > tmp/iris_pressure_test_20260504/pre_commit_all_files_fix_generation_params_20260513T1736.txt 2>&1`
- Why: Run the required repo check after the tracked evaluator change.
- Outcome: Succeeded.
- Learned: Ruff, Black, license headers, pyrefly, markdown, and general file checks passed.

## 2026-05-13T17:36:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job list --json --prefix /romain/iris-inference-standard-smoke-20260514T003544Z > tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1736.json 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1736.stderr`
- Why: Check scheduling state for the third smoke.
- Outcome: Succeeded; job was pending.
- Learned: Autoscaler was actively scaling up `tpu_v6e-preemptible_4-europe-west4-a`.

## 2026-05-13T17:39:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job list --json --prefix /romain/iris-inference-standard-smoke-20260514T003544Z > tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1739b.json 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1739b.stderr`; `uv run iris --config lib/iris/config/marin.yaml job logs /romain/iris-inference-standard-smoke-20260514T003544Z --since-seconds 600 > tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1739b.txt 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1739b.stderr`
- Why: Check whether scale-up completed and whether the corrected runner reached vLLM/lm-eval.
- Outcome: Succeeded; job was running.
- Learned: Logs show real v6e-4 worker assignment, runner config with `generation_params={'max_gen_toks': 128}`, and no 1024-token failure at startup.

## 2026-05-13T17:44:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job list --json --prefix /romain/iris-inference-standard-smoke-20260514T003544Z > tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1744.json 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_20260513T1744.stderr`; `uv run iris --config lib/iris/config/marin.yaml job logs /romain/iris-inference-standard-smoke-20260514T003544Z --since-seconds 1200 > tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1744.txt 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1744.stderr`
- Why: Collect terminal status and key logs for the third smoke.
- Outcome: Succeeded; job state was `JOB_STATE_SUCCEEDED`.
- Learned: lm-eval logged that CLI `generation_kwargs={'max_gen_toks': 128}` updated task YAML; `/v1/chat/completions` API request completed; runner elapsed time was 118.250s.

## 2026-05-13T17:45:00-0700

- Command: `uv run iris --config lib/iris/config/marin.yaml job summary --json /romain/iris-inference-standard-smoke-20260514T003544Z > tmp/iris_pressure_test_20260504/rebased_standard_smoke_summary_success_20260513T1745.json 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_summary_success_20260513T1745.stderr`
- Why: Capture structured task state, worker, and duration for the successful smoke.
- Outcome: Succeeded.
- Learned: Summary reports exit code 0, duration 147051ms, no preemptions, worker `marin-tpu-v6e-preemptible-4-us-east5-b-20260514-0026-237647e2-worker-0`.

## 2026-05-13T17:45:00-0700

- Command: `gcloud storage ls gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_standard_smoke_20260514T003544Z/** > tmp/iris_pressure_test_20260504/rebased_standard_smoke_gcs_ls_20260513T1745.txt 2>&1`
- Why: List the output artifact path for the successful smoke.
- Outcome: Failed locally before `gcloud` ran because zsh expanded the unquoted `**` glob.
- Learned: Quote GCS globs in zsh.

## 2026-05-13T17:46:00-0700

- Command: `gcloud storage ls 'gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_standard_smoke_20260514T003544Z/**' > tmp/iris_pressure_test_20260504/rebased_standard_smoke_gcs_ls_20260513T1746.txt 2>&1`
- Why: List the output artifact path for the successful smoke with proper shell quoting.
- Outcome: Succeeded.
- Learned: GCS contains the lm-eval results JSON and HumanEval samples JSONL under `humaneval_0shot/Qwen__Qwen3-0.6B/`.

## 2026-05-13T17:46:00-0700

- Command: `gcloud storage cp gs://marin-us-central2/tmp/ttl=7d/iris-pressure-test-20260504/qwen3_0_6b_standard_smoke_20260514T003544Z/humaneval_0shot/Qwen__Qwen3-0.6B/results_2026-05-14T00-39-54.767392.json tmp/iris_pressure_test_20260504/rebased_standard_smoke_results_20260514T003544Z.json > tmp/iris_pressure_test_20260504/rebased_standard_smoke_results_cp_20260513T1746.txt 2>&1`
- Why: Preserve a small local copy of the successful results JSON for inspection.
- Outcome: Succeeded; copied a 12K JSON file.
- Learned: Saved config records `num_fewshot: 0`, `generation_kwargs.max_gen_toks: 128`, `/v1/chat/completions`, Hugging Face tokenizer backend, and `pass@1,create_test = 0.0`.

## 2026-05-13T17:47:00-0700

- Command: `git add lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py && git commit -m "Pass lm-eval generation params to harness" > tmp/iris_pressure_test_20260504/git_commit_generation_params_20260513T1747.txt 2>&1`
- Why: Checkpoint the one-line tracked evaluator fix after pre-commit and live smoke validation.
- Outcome: Succeeded; created commit `048192e25`.
- Learned: The branch now has four local commits on top of #5712; no push was performed.

## 2026-05-13T17:48:00-0700

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/findings.md`.
- Why: Record the successful MVP smoke, the newly discovered generation-params plumbing issue, and updated recommendation.
- Outcome: Succeeded.
- Learned: Findings now identify the standard-model MVP lane as validated and keep 1e22/MMLU as follow-up compatibility gates.

## 2026-05-13T17:49:00-0700

- Command: `git status --short --branch`; `git log --oneline --decorate -8`; `uv run iris --config lib/iris/config/marin.yaml job list --json --prefix /romain/iris-inference-standard-smoke > tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_all_20260513T1749.json 2> tmp/iris_pressure_test_20260504/rebased_standard_smoke_list_all_20260513T1749.stderr`
- Why: Final sanity check of local branch state and standard-smoke Iris jobs.
- Outcome: Succeeded.
- Learned: Branch is ahead of `origin/main` by five commits with only untracked `report.md` and `tmp/`; standard-smoke jobs are terminal: latest succeeded, two earlier attempts failed, first zone-pinned attempt was killed.

## 2026-05-14T14:16:00-0700

- Command: GitHub connector `get_pr_info` for PR #5400 and PR #5712; `fetch_issue` for #5672; `git status --short --branch`; `git log --oneline --decorate -8`; `git fetch origin --prune`; `uv run iris --config lib/iris/config/marin.yaml job list --json --prefix /romain/iris-inference-standard-smoke > tmp/iris_pressure_test_20260504/status_standard_smoke_list_20260514T211627.json 2> ...`; ancestry/status checks.
- Why: Answer a fresh current-status request after #5712 may have changed.
- Outcome: Succeeded.
- Learned: PR #5712 merged on 2026-05-14T17:03:40Z. PR #5400 is still open, draft, and remote-stale at `69569a5d`. Local branch is `048192e25`, ahead of `origin/main` by five commits and behind by twenty-one after fetch. Latest standard smoke `/romain/iris-inference-standard-smoke-20260514T003544Z` remains succeeded; no standard-smoke job is running. Issue #5672 remains open.

## 2026-05-14T14:20:00-0700

- Command: Read `.agents/skills/pull-request/SKILL.md`; `git fetch origin --prune`; `git status --short --branch`; `git log --oneline --decorate -8`.
- Why: Start the requested PR update and refresh the branch after #5712 merged into `main`.
- Outcome: Succeeded.
- Learned: Local branch had no tracked dirty files and still included old #5712 head plus four local commits.

## 2026-05-14T14:20:00-0700

- Command: `git rebase --onto origin/main 323805384b7ac0e393f34a9864a0dc6b96050951 > tmp/iris_pressure_test_20260504/update_pr_rebase_onto_main_20260514T2120xx.txt 2>&1`
- Why: Drop the now-merged #5712 stack commit and replay only the inference-service commits onto current `origin/main`.
- Outcome: Succeeded.
- Learned: Branch now has four commits on top of `origin/main`: `572873f1a`, `4250fc83a`, `277952a70`, and `8c652c5e3`.

## 2026-05-14T14:21:00-0700

- Command: `./infra/pre-commit.py --all-files --fix > tmp/iris_pressure_test_20260504/update_pr_pre_commit_20260514T2121xx.txt 2>&1`
- Why: Run required repo checks after rebasing onto current `origin/main`.
- Outcome: Succeeded.
- Learned: Full pre-commit entrypoint passed after the rebase.

## 2026-05-14T14:21:00-0700

- Command: `uv run --package marin --extra eval --extra cpu pytest -q tests/evals/test_served_lm_eval.py -m 'not slow' > tmp/iris_pressure_test_20260504/update_pr_pytest_served_lm_eval_20260514T2121xx.txt 2>&1`
- Why: Run a focused eval-path test before pushing the PR update.
- Outcome: Succeeded.
- Learned: `tests/evals/test_served_lm_eval.py` passed 5 tests in 21.85s.

## 2026-05-14T14:22:00-0700

- Command: `git push --force-with-lease origin HEAD:design/iris_inference_service > tmp/iris_pressure_test_20260504/update_pr_push_20260514T2122xx.txt 2>&1`
- Why: Update PR #5400's rebased branch after replaying it onto current `origin/main`.
- Outcome: Succeeded.
- Learned: Remote branch moved from `69569a5d` to `8c652c5e3`.

## 2026-05-14T14:22:00-0700

- Command: GitHub connector `update_pull_request` for PR #5400.
- Why: Replace the stale PR title/body that still described 1e22/MMLU as the MVP gate.
- Outcome: Succeeded.
- Learned: PR #5400 is still draft, now titled `Design Iris inference service`, base `main`, head `8c652c5e3`, mergeable, with four commits and four changed files.

## 2026-05-14T14:23:00-0700

- Command: `gh pr view 5400 --json statusCheckRollup,mergeStateStatus,isDraft,headRefOid,title,baseRefName,url > tmp/iris_pressure_test_20260504/update_pr_gh_view_5400_20260514T2123xx.json 2> ...`
- Why: Monitor PR checks after pushing.
- Outcome: Succeeded.
- Learned: PR remains draft and GitHub merge state is blocked only because checks are still running/pending; several path-filter checks already succeeded or skipped.

## 2026-05-14T14:24:00-0700

- Command: `git status --short --branch && git log --oneline --decorate -5`; `gh pr view 5400 --json statusCheckRollup,mergeStateStatus,isDraft,headRefOid,title,baseRefName,url > tmp/iris_pressure_test_20260504/status_pr_5400_20260514T2124xx.json 2> ...`
- Why: Answer an interrupted-turn status request with a fresh local and GitHub check.
- Outcome: Succeeded.
- Learned: PR #5400 is draft on `main` at head `8c652c5e3`; local branch matches `origin/design/iris_inference_service` and has only untracked `report.md` and `tmp/`. All visible checks are success/skipped except `marin-integration` and `marin-unit`, which are still in progress.

## 2026-05-14T14:30:00-0700

- Command: Read `.agents/skills/writing-style/SKILL.md`, `.agents/skills/writing-style/references/issues.md`, `.agents/skills/writing-style/references/ai-writing-donts.md`, and `.agents/skills/file-issue/SKILL.md`.
- Why: Draft a local GitHub issue in Marin house style without posting it.
- Outcome: Succeeded.
- Learned: The draft should be self-contained, concise, evidence-backed, and kept local until explicitly approved.

## 2026-05-14T14:31:00-0700

- Command: Search memory registry for prior served-model/lm-eval RFC context.
- Why: Avoid losing relevant historical context while drafting the issue.
- Outcome: Succeeded.
- Learned: Prior served-lm-eval work used local Markdown artifacts and kept lm-eval optional, but the current bug is narrower: `generation_params` are accepted by Marin config and then dropped before `lm_eval.simple_evaluate`.

## 2026-05-14T14:31:00-0700

- Command: `gh issue list --repo marin-community/marin --state open --search "lm-eval generation_params max_gen_toks" --json number,title,state,url,labels --limit 20 > tmp/iris_pressure_test_20260504/issue_draft_duplicate_check_lmeval_generation_params.json 2> tmp/iris_pressure_test_20260504/issue_draft_duplicate_check_lmeval_generation_params.stderr`
- Why: Check whether an exact issue already exists before drafting a new one.
- Outcome: Succeeded.
- Learned: The search returned #4827, a broader served-lm-eval tracker; no exact open bug for dropped `generation_params` was found.

## 2026-05-14T14:32:00-0700

- Command: Inspect `lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py`, `lib/marin/src/marin/evaluation/evaluation_config.py`, and `lib/marin/src/marin/evaluation/run.py` on `origin/main`.
- Why: Confirm whether the failure is present on main and identify the missing handoff.
- Outcome: Succeeded.
- Learned: `EvaluationConfig.generation_params` is defined and copied into `ModelConfig`, but `lm_evaluation_harness_evaluator.py` on main does not pass it to `simple_evaluate`.

## 2026-05-14T14:32:00-0700

- Command: Inspect `tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1739.txt`, `tmp/iris_pressure_test_20260504/rebased_standard_smoke_logs_20260513T1744.txt`, and `tmp/iris_pressure_test_20260504/rebased_standard_smoke_results_20260514T003544Z.json`.
- Why: Pull concrete repro and fixed-run evidence into the issue draft.
- Outcome: Succeeded.
- Learned: The failing main-equivalent run sent `max_tokens=1024` despite a requested `max_gen_toks=128`; the fixed run records `generation_kwargs.max_gen_toks=128` and succeeded through `/v1/chat/completions`.

## 2026-05-14T14:33:00-0700

- Command: Create `tmp/iris_pressure_test_20260504/lm_eval_generation_params_issue_draft.md`.
- Why: Prepare a local-only issue draft for user review before posting.
- Outcome: Succeeded.
- Learned: The draft explains the bug, exact repro shape, expected behavior, current workaround/fix, duplicate check, and evidence artifacts.

## 2026-05-14T14:34:00-0700

- Command: `git status --short --branch`; `wc -l tmp/iris_pressure_test_20260504/lm_eval_generation_params_issue_draft.md tmp/iris_pressure_test_20260504/run_log.md tmp/iris_pressure_test_20260504/findings.md`; `tail -n 80 tmp/iris_pressure_test_20260504/run_log.md`; `sed -n '1,220p' tmp/iris_pressure_test_20260504/lm_eval_generation_params_issue_draft.md`; `date +%Y-%m-%dT%H:%M:%S%z`.
- Why: Verify local draft contents and pressure-test artifact state after context resume.
- Outcome: Succeeded.
- Learned: Draft is 47 lines, local branch remains ahead of `origin/main` by four commits, and only untracked `report.md` and `tmp/` are present.

## 2026-05-14T14:35:00-0700

- Command: `apply_patch` update `tmp/iris_pressure_test_20260504/run_log.md`.
- Why: Append the missing issue-draft workflow entries.
- Outcome: Succeeded.
- Learned: The pressure-test command log now records the local issue-draft work and verification pass.

## 2026-05-14T14:36:00-0700

- Command: Read `.agents/skills/writing-style/SKILL.md`, `.agents/skills/pull-request/SKILL.md`, `.agents/skills/design-doc/SKILL.md`, `.agents/skills/writing-style/reference-docs.md`, `.agents/skills/writing-style/ai-writing-donts.md`, and `.agents/projects/design-template.md`.
- Why: Apply the repo's writing, PR, and design-doc guidance before rewriting the RFC docs.
- Outcome: Succeeded; two attempted reads under `.agents/skills/writing-style/references/` failed because the files live directly under `.agents/skills/writing-style/`.
- Learned: The design doc should be close to a one-page decision doc; `research.md` carries evidence; `spec.md` carries contracts; the branch's PR body should stay aligned if pushed later.

## 2026-05-14T14:37:00-0700

- Command: `sed -n` over `.agents/projects/iris_inference_service/{design.md,spec.md,research.md}`; `wc -l ...`; `rg -n "Delphi|1e22|mmlu_sl_verb|MMLU|HumanEval|humaneval|#5672|#5712|v5p|canonical" ...`; `git status --short --branch`.
- Why: Measure the current RFC size and find model-specific detail before simplifying it.
- Outcome: Succeeded.
- Learned: The three docs totaled 853 lines; `design.md` contained multiple Delphi/1e22 and canonical-test references that should move out of the reviewer-facing design.

## 2026-05-14T14:38:00-0700

- Command: `apply_patch` replace `.agents/projects/iris_inference_service/design.md`, `.agents/projects/iris_inference_service/spec.md`, and `.agents/projects/iris_inference_service/research.md`.
- Why: Shorten and clarify the RFC while preserving design-skill structure.
- Outcome: Succeeded.
- Learned: The new split keeps `design.md` at the decision level, `spec.md` as API contracts, and `research.md` as evidence and pressure-test history.

## 2026-05-14T14:39:00-0700

- Command: `wc -l .agents/projects/iris_inference_service/{design.md,spec.md,research.md}`; `rg -n "Delphi|1e22" .agents/projects/iris_inference_service/design.md`; `sed -n` spot checks; `rg -n` for common AI-writing boilerplate; `git diff --stat`.
- Why: Self-review the rewritten docs for length, forbidden design mentions, and prose issues.
- Outcome: Succeeded.
- Learned: New docs total 609 lines; `design.md` is 108 lines and has no `Delphi` or `1e22` references.

## 2026-05-14T14:38:00-0700

- Command: `./infra/pre-commit.py --all-files --fix > tmp/iris_pressure_test_20260504/docs_clarity_pre_commit_20260514T1438.txt 2>&1`
- Why: Run the required repo check after editing tracked files.
- Outcome: Succeeded.
- Learned: Full pre-commit passed, including Ruff, Black, Pyrefly, markdown checks, and file hygiene checks.

## 2026-05-14T14:44:00-0700

- Command: `tail -n 80 tmp/iris_pressure_test_20260504/docs_clarity_pre_commit_20260514T1438.txt`; `git status --short --branch`; `wc -l ...`; `rg -n "Delphi|1e22" .agents/projects/iris_inference_service/design.md`; `git diff --stat`.
- Why: Final verification of the docs rewrite and check output.
- Outcome: Succeeded.
- Learned: Only the three RFC docs are tracked-dirty; `report.md` and `tmp/` remain untracked; pre-commit passed; design doc still has no Delphi/1e22 mentions.

## 2026-05-14T14:45:00-0700

- Command: `git diff --check -- .agents/projects/iris_inference_service/design.md .agents/projects/iris_inference_service/spec.md .agents/projects/iris_inference_service/research.md`; `rg -n "Delphi|1e22" .agents/projects/iris_inference_service/design.md || true`; `git diff --stat`; `git status --short --branch`.
- Why: Final whitespace and forbidden-term check before reporting back.
- Outcome: Succeeded.
- Learned: The tracked doc diff has no whitespace errors; `design.md` has no Delphi/1e22 mentions; the branch has only the three tracked doc edits plus untracked `report.md` and `tmp/`.

## 2026-05-14T14:46:00-0700

- Command: `gh pr view 5400 --repo marin-community/marin --json title,body,isDraft,headRefOid,url > tmp/iris_pressure_test_20260504/docs_clarity_pr_view_5400_20260514T1446.json 2> tmp/iris_pressure_test_20260504/docs_clarity_pr_view_5400_20260514T1446.stderr`; inspect saved JSON summary.
- Why: Check the current PR title/body after editing a branch associated with PR #5400.
- Outcome: Succeeded.
- Learned: PR #5400 is still draft at remote head `8c652c5e3`; the PR body remains a short vLLM-only summary and still names 1e22/logprob work as follow-up compatibility gates. No PR body update was made because these doc edits have not been pushed.
