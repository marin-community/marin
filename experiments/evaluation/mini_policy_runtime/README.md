# Mini840 SkyRL evaluation runtime

Run [Mini840 #10804](https://github.com/marin-community/marin/issues/10804) as an eval-only SkyRL job on RNO2A. The launcher uses the released TaskRolloutSpecs, Pi 0.87.0 for all seven agentic benchmarks, and the original Harbor verifiers. It does not train a model.

The datasets are model agnostic. This reference smoke launcher is deliberately specific to Qwen: it fixes the tokenizer revision, `qwen3_coder` tool parser and TP2 topology. Supplying another weight URI alone does not adapt those settings. In a separate MarinSkyRL run, load the same TaskRolloutSpecs with the evaluated model's tokenizer, chat template, supported thinking controls, parsers and serving topology, while applying the shared validation profile and native Harbor/Pi bindings below. The E2E evidence here covers Qwen; it does not validate every model/backend combination.

The Daytona eval account can build the embedded environment recipes directly. Named snapshots and an externally supplied `environment_snapshots` map are optional. The old `815c4b2c` runtime incorrectly required that map for native execution.

## Frozen inputs and execution profile

- Frozen initial Mini840 export: `open-athena/Marin-Mini-Eval-840@bae388a44fc0ba9ffa551a4c6347fbe03a9fefd7`, 840 unique tasks and 1,626 expanded rollouts.
- Frozen initial Micro210 export: `open-athena/Marin-Micro-Eval-210@4564ed081c316bc3231ae5c3a53a46b39248c819`, 210 unique tasks and 450 expanded rollouts. Micro is the first 210 Mini tasks.
- Grading wheels: `open-athena/Marin-Mini-Eval-Policy-Index@d5dd315d58f0f47514dd0a2208e90e2a48541328`, directory `runtime-v4/wheels`.
- Candidate: `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`, compiled vLLM, TP2, two H100s, automatic tool calling with `qwen3_coder`.
- Cohort: the launch `--model` key binds repo, revision, TP2 topology, tool parser, and default chat-template kwargs in `cohort.py`. Chat-aligned bases: `qwen35-35b-a3b` and `nemotron3-nano-30b-a3b` use `qwen3_coder`; `qwen3-30b-a3b` and `snowball-67b-a2b` use `hermes`; no family uses a serving-side reasoning parser, matching the frozen policy model configs.
- Context: 65,536 total tokens; maximum 16,384 output tokens per turn; thinking enabled. Pi reserves the output budget for compaction and retains 24,576 tokens. There is no cumulative episode-token cap.
- Trial limits: agent 1,800 seconds in both training and eval; verifier 300; setup 360; cleanup 60; whole attempt 2,460. Harbor owns the agent and verifier deadlines separately so an agent timeout still reaches verification. The outer job allows 24 hours for the whole cohort.
- Judge: the pinned [MiniMax-M3-MXFP8 configuration](../campaigns/eval-campaign-09-25-unlabeled/ablations/olympiadbench_judge/MiniMaxAI-MiniMax-M3-MXFP8-131k.yaml), 131,072 total context tokens, TP8 on eight H100s. The launcher co-hosts it for BixBench and OlympiadBench fallback.
- Tau3's original counterpart remains Together `openai/gpt-oss-120b` with low reasoning effort. Its LiteLLM identifier is `openai/openai/gpt-oss-120b`; LiteLLM consumes the first provider prefix.

The repaired datasets preserve task IDs, prompts, task packages, selection order, repetitions and grading rules. Repairs make the grading dependencies installable as wheels, include pinned NLTK resources for IFBench's read-only sandbox, remove unsupported local-machine resource requests, and bind judge credentials at runtime. BixBench's source task requests 20 GB, but its frozen policy config overrides it to 10 GB; the launcher preserves that override by environment source key. Other source budgets are preserved.

Historical rank correlations in #10804 describe the original policy and fitted predictors. They do not establish the quality or rank fidelity of this uniform execution profile.

Preparation reconstructs the repaired packages from the frozen initial exports above using the pinned grading wheels. The staged full Mini cohort has SHA256 `3f5e44e57cf4d1fad60ba7169490e2c7dcbd268d985cc1152f6b0891bf353d16`, identical to the corrected public export. The dataset cards link the corrected download revisions for direct RL integration.

## Prepare a fresh run

Use runtime commit `f1a9e9640ffffcd421187164ddc6cbc81105ce38`, which contains the native environment launch path, runtime repairs and result-audit command:

```bash
git clone --filter=blob:none https://github.com/marin-community/marin.git mini840-repro
cd mini840-repro
git checkout --detach f1a9e9640ffffcd421187164ddc6cbc81105ce38
```

Run the remaining commands from that repository root. Existing Iris, CoreWeave storage, Hugging Face, Daytona eval and Together access is required. See [installation](../../../docs/tutorials/installation.md), [CoreWeave access](../../../docs/tutorials/cloud-gpu.md), and [Iris authentication](../../../lib/iris/OPS.md#authentication-headless--ci). The GCP identity must already have access to the four secrets below; these instructions do not grant it. Before submitting any Iris job, export `CW_KEY_ID` and `CW_KEY_SECRET` for CoreWeave storage handoff, plus `DAYTONA_EVAL_API_KEY` and `TOGETHER_API_KEY` for task execution, from the approved credential store; do not put them in files committed to Git. Public dataset downloads need no HF write token. With existing GCP access, the credential roles can be resolved without a local secrets file:

```bash
export CW_KEY_ID=$(gcloud secrets versions access latest --project=hai-gcp-models --secret=cw-object-storage-key-id)
export CW_KEY_SECRET=$(gcloud secrets versions access latest --project=hai-gcp-models --secret=cw-object-storage-key-secret)
export DAYTONA_EVAL_API_KEY=$(gcloud secrets versions access latest --project=hai-gcp-models --secret=DAYTONA_EVAL_API_KEY)
export TOGETHER_API_KEY=$(gcloud secrets versions access latest --project=hai-gcp-models --secret=TOGETHER_API_KEY)
```

Choose a fresh artifact version, an output prefix that this run owns, and the cohort model key:

```bash
export MINI_VERSION=$(date -u +%Y.%m.%d.%H%M%S)
export MINI_PREFIX="s3://marin-rno2a/marin/experiments/mini840-reproduction/$MINI_VERSION"
export MINI_WHEELS=d5dd315d58f0f47514dd0a2208e90e2a48541328
export MINI_COHORT_MODEL=qwen36-35b-a3b
```

Stage the unchanged first task from each of the 21 benchmarks and the full expanded cohort. This CPU job also resolves a complete revision-pinned candidate weight mirror. `--model` binds a cohort entry from `mini_policy_runtime/cohort.py`: `qwen36-35b-a3b` is the validated runtime candidate, and the chat-aligned bases are `qwen35-35b-a3b`, `qwen3-30b-a3b`, `nemotron3-nano-30b-a3b`, and `snowball-67b-a2b`.

```bash
uv run --frozen iris --cluster marin job run --target-cluster cw-rno2a \
  --job-name "mini840-inputs-$MINI_VERSION" \
  --cpu 4 --memory 16GB --disk 32GB --enable-extra-resources \
  --timeout 1800 --priority batch -- \
  uv run --frozen python -m experiments.evaluation.mini_policy_runtime.stage_inputs \
    --wheels-revision "$MINI_WHEELS" --output-prefix "$MINI_PREFIX" \
    --model "$MINI_COHORT_MODEL"
```

The command waits for completion. Download its staging report and read the model URI:

```bash
uv run --frozen fsutil cp "$MINI_PREFIX/smoke-runtime-v4-$MINI_COHORT_MODEL-staged-inputs.json" ./mini840-inputs.json
export MINI_MODEL_URI=$(uv run --frozen python -c 'import json; print(json.load(open("mini840-inputs.json"))["model_uri"])')
```

Check native environment startup and Pi installation for the smoke's seven agentic task sources:

```bash
uv run --frozen iris --cluster marin job run --target-cluster cw-rno2a \
  --job-name "mini840-environments-$MINI_VERSION" \
  --cpu 4 --memory 16GB --disk 32GB --enable-extra-resources \
  --timeout 7200 --priority batch \
  -e DAYTONA_EVAL_API_KEY "$DAYTONA_EVAL_API_KEY" \
  -e TOGETHER_API_KEY "$TOGETHER_API_KEY" -- \
  uv run --frozen --with-editable ./vendor/harbor --with daytona==0.200.2 --prerelease allow \
    python -m experiments.evaluation.mini_policy_runtime.environment_preflight \
    --smoke --credential-env DAYTONA_EVAL_API_KEY \
    --data-uri "$MINI_PREFIX/smoke-runtime-v4" --output-prefix "$MINI_PREFIX" \
    --model "$MINI_COHORT_MODEL"
```

This starts environments from the published recipes and may build their images. It does not require 197 named snapshots. A failed preflight prevents GPU launch.

## Plan, launch and inspect

Inspect the launch plan first:

```bash
uv run --frozen --with-editable ./vendor/harbor --with daytona==0.200.2 --prerelease allow \
  python -m experiments.evaluation.mini_policy_runtime.launch \
  --version "$MINI_VERSION" --model "$MINI_COHORT_MODEL" --model-uri "$MINI_MODEL_URI" \
  --data-uri "$MINI_PREFIX/smoke-runtime-v4" \
  --environment-manifest "$MINI_PREFIX/smoke-native-environments.json" \
  --priority production
```

Repeat the same command with `--run` to submit. The coordinator co-hosts the eight-H100 judge, then starts the two-H100 Qwen evaluation job. Both run on RNO2A. Use a fresh version after a runtime commit change. `--judge-base-url` can reuse an existing compatible judge with `JUDGE_API_KEY` supplied separately.

Inspect the printed coordinator ID with `uv run --frozen iris --cluster marin job describe JOB_ID` and `job logs JOB_ID --tail --max-lines 40`. A successful coordinator alone does not establish E2E success. Check the final results:

```bash
uv run --frozen python -m experiments.evaluation.mini_policy_runtime.audit_results \
  --result-uri "s3://marin-rno2a/marin/checkpoints/mini840-e2e/$MINI_VERSION" \
  --expected-rollouts 21 --output mini840-smoke-audit.json
```

`all_verified` requires all expected rollouts to have final verification. The report distinguishes unverified exceptions from verified model rewards. Inspect native `agent/pi.txt` and original trial `result.json` under the run's private attempt root to confirm actual tool use. A mechanically verified zero is not sufficient if a counterpart failed: Tau3 must successfully start its conversation and exchange messages through MCP without provider authentication, model-access or transport errors. The raw trace exposed this problem even when Harbor returned a zero reward with no exception. Resolved runtime endpoint URLs can contain credentials; publish only sanitized evidence. The default private attempt root is `s3://marin-rno2a/tmp/ttl=14d/skyrl/marin/checkpoints/mini840-e2e/$MINI_VERSION/attempts`; the coordinator also logs it as `artifacts.attempts_root`. Download native evidence to inspect it locally:

```bash
export MINI_ATTEMPTS="s3://marin-rno2a/tmp/ttl=14d/skyrl/marin/checkpoints/mini840-e2e/$MINI_VERSION/attempts"
uv run --frozen fsutil rsync "$MINI_ATTEMPTS/trajectories/installed-agents" ./mini840-native-evidence
```

For each Tau3 attempt, find its task ID through `identity.json`, then inspect `trials/ATTEMPT_ID/agent/pi.txt` and `trials/ATTEMPT_ID/result.json`. This prints counts and local paths without printing credentials:

```bash
uv run --frozen python - <<'PYCODE'
import json
from pathlib import Path
for path in sorted(Path("mini840-native-evidence").glob("*/identity.json")):
    identity = json.loads(path.read_text())
    if not identity["task_id"].startswith("tau3-pi/"):
        continue
    for trace in path.parent.glob("trials/*/agent/pi.txt"):
        events = [json.loads(line) for line in trace.read_text().splitlines() if line.startswith("{")]
        calls = [event for event in events if event.get("type") == "tool_execution_end"]
        errors = sum(bool(event.get("isError")) for event in calls)
        print(identity["task_id"], "tool calls:", len(calls), "tool errors:", errors, "trace:", trace)
PYCODE
```

The smoke has one Tau3 attempt; the expanded Mini cohort has 41. Account for every attempt. If a row has tool errors, inspect its MCP responses and reject provider/authentication/transport failures before accepting its reward. A successful counterpart start plus a model that ignores the messaging instruction is a valid model failure, as in the corrected canary.

For full Mini840, repeat environment preflight without `--smoke` and with `--data-uri "$MINI_PREFIX/full-runtime-v4"`. It checks all 197 environment sources. Launch with that data URI, the resulting `full-native-environments.json`, and a fresh version. Audit with `--expected-rollouts 1626`. The 21-task smoke is execution-path validation, not a complete Mini840 score.

## Evidence

An independent reader reproduced the 21-benchmark smoke on RNO2A from issue #10804 and these public instructions, using a fresh unchanged clone at runtime `f1a9e9640ffffcd421187164ddc6cbc81105ce38`. Existing credentials were resolved through the documented Secret Manager roles; no private campaign files were used. Artifact version `2026.10.31.135401` completed and verified all 21 rollouts, with no unverified error types. Its result root is `s3://marin-rno2a/marin/checkpoints/mini840-e2e/2026.10.31.135401`.

All seven native agentic paths made real Pi tool calls and ran their original verifiers: 112 tool calls in total. Six generated-command errors were inspected; none masked provider or runtime failures. SWE-bench passed its original 16-test verifier. Tau3 started a real counterpart conversation and searched its knowledge base without MCP/provider errors; its zero came from ending with ordinary text instead of using the required messaging tool. DS-1000, BixBench and TB2 zeros were confirmed against their original array-shape, AIC and fitted-parameter failures. The launcher stopped its judge and broker, and all owned compute is terminal.

Native startup and Pi installation separately passed for all 197 environment sources. This checks environment readiness, not model quality on every task. Read-only verifier controls passed MATH500 reference grading and IFBench positive/negative grading. A deliberately incorrect OlympiadBench response successfully called the co-hosted MiniMax judge and received zero.

The pinned-vLLM regression passed on RNO2A, including rejection of an oversized exact token prefix. The SkyRL CPU suite passed 318 tests. Separate real Pi/Daytona controls demonstrated compact-and-retry after a context rejection and verification after an agent timeout with packaged-style turn timers and a delayed verifier. Two real CUDA/Ray actors saw exactly their two assigned GPU UUIDs. These controls are reproducible from `native_recovery_smoke.py` (`--mode context` or `--mode timeout`) and `gpu_allocation_smoke.py`.

Run the native controls on an Iris CPU task using the frozen SkyRL CPU environment (`--extra cpu --group harbor-test --no-group dev`), with the repository root on `PYTHONPATH` and the Daytona eval credential passed to the task. Run the GPU control in a privileged two-H100 Iris task with the same pinned SkyRL bootstrap environment as the launcher. See [the GPU incident](https://marina.oa.dev/echo/wiki/907) for allocation evidence.

Earlier pins are rejected as complete E2E validation: the independent `8636783665` smoke verified only 19/21 after context/session and timeout failures; a later real-Qwen smoke exposed validation of reconstructed text before exact token substitution and recorder turn clocks canceling Harbor verification. The repaired runtime validates the actual prefix with vLLM's normal budget checks, returns recoverable context errors to Pi, and leaves native phase deadlines to Harbor while retaining the whole-attempt ceiling.

Private native evidence, rejected-run evidence and control receipts are preserved outside temporary storage under `s3://marin-rno2a/marin/experiments/mini-eval-policy/e2e-20261031`. Publish only sanitized receipts because resolved endpoint URLs can contain credentials. These checks validate execution paths; no complete 1,626-rollout Mini840 or 450-rollout Micro210 quality score is claimed.
