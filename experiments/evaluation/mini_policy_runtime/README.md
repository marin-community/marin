# Mini840 SkyRL evaluation runtime

Run [Mini840 #10804](https://github.com/marin-community/marin/issues/10804) as an eval-only SkyRL job on RNO2A. The launcher uses the released TaskRolloutSpecs, Pi 0.87.0 for all seven agentic benchmarks, and the original Harbor verifiers. It does not train a model.

The Daytona eval account can build the embedded environment recipes directly. Named snapshots and an externally supplied `environment_snapshots` map are optional. The old `815c4b2c` runtime incorrectly required that map for native execution.

## Frozen inputs and execution profile

- Frozen initial Mini840 export: `open-athena/Marin-Mini-Eval-840@bae388a44fc0ba9ffa551a4c6347fbe03a9fefd7`, 840 unique tasks and 1,626 expanded rollouts.
- Frozen initial Micro210 export: `open-athena/Marin-Micro-Eval-210@4564ed081c316bc3231ae5c3a53a46b39248c819`, 210 unique tasks and 450 expanded rollouts. Micro is the first 210 Mini tasks.
- Grading wheels: `open-athena/Marin-Mini-Eval-Policy-Index@d5dd315d58f0f47514dd0a2208e90e2a48541328`, directory `runtime-v4/wheels`.
- Candidate: `Qwen/Qwen3.6-35B-A3B@995ad96eacd98c81ed38be0c5b274b04031597b0`, compiled vLLM, TP2, two H100s, automatic tool calling with `qwen3_coder`.
- Context: 65,536 total tokens; maximum 16,384 output tokens per turn; thinking enabled. Pi reserves the output budget for compaction and retains 24,576 tokens. There is no cumulative episode-token cap.
- Trial limits: agent 1,800 seconds; verifier 300; setup 360; cleanup 60. The outer job allows 24 hours for the whole cohort.
- Judge: the pinned [MiniMax-M3-MXFP8 configuration](../campaigns/eval-campaign-09-25-unlabeled/ablations/olympiadbench_judge/MiniMaxAI-MiniMax-M3-MXFP8-131k.yaml), 131,072 total context tokens, TP8 on eight H100s. The launcher co-hosts it for BixBench and OlympiadBench fallback.
- Tau3's original counterpart remains Together `openai/gpt-oss-120b` with low reasoning effort. Its LiteLLM identifier is `openai/openai/gpt-oss-120b`; LiteLLM consumes the first provider prefix.

The repaired datasets preserve task IDs, prompts, task packages, selection order, repetitions and grading rules. Repairs make the grading dependencies installable as wheels, include pinned NLTK resources for IFBench's read-only sandbox, remove unsupported local-machine resource requests, and bind judge credentials at runtime. BixBench's source task requests 20 GB, but its frozen policy config overrides it to 10 GB; the launcher preserves that override by environment source key. Other source budgets are preserved.

Historical rank correlations in #10804 describe the original policy and fitted predictors. They do not establish the quality or rank fidelity of this uniform execution profile.

Preparation reconstructs the repaired packages from the frozen initial exports above using the pinned grading wheels. The staged full Mini cohort has SHA256 `3f5e44e57cf4d1fad60ba7169490e2c7dcbd268d985cc1152f6b0891bf353d16`, identical to the corrected public export. The dataset cards link the corrected download revisions for direct RL integration.

## Prepare a fresh run

Use runtime commit `863678366542eb6191de63225268e284a1ccbc46`, which contains the Tau3 correction and the result-audit command:

```bash
git clone --filter=blob:none https://github.com/marin-community/marin.git mini840-repro
cd mini840-repro
git checkout --detach 863678366542eb6191de63225268e284a1ccbc46
```

Run the remaining commands from that repository root. Existing Iris, CoreWeave storage, Hugging Face, Daytona eval and Together access is required. See [installation](../../../docs/tutorials/installation.md), [CoreWeave access](../../../docs/tutorials/cloud-gpu.md), and [Iris authentication](../../../lib/iris/OPS.md#authentication-headless--ci). The GCP identity must already have access to the four secrets below; these instructions do not grant it. Before submitting any Iris job, export `CW_KEY_ID` and `CW_KEY_SECRET` for CoreWeave storage handoff, plus `DAYTONA_EVAL_API_KEY` and `TOGETHER_API_KEY` for task execution, from the approved credential store; do not put them in files committed to Git. Public dataset downloads need no HF write token. With existing GCP access, the credential roles can be resolved without a local secrets file:

```bash
export CW_KEY_ID=$(gcloud secrets versions access latest --project=hai-gcp-models --secret=cw-object-storage-key-id)
export CW_KEY_SECRET=$(gcloud secrets versions access latest --project=hai-gcp-models --secret=cw-object-storage-key-secret)
export DAYTONA_EVAL_API_KEY=$(gcloud secrets versions access latest --project=hai-gcp-models --secret=DAYTONA_EVAL_API_KEY)
export TOGETHER_API_KEY=$(gcloud secrets versions access latest --project=hai-gcp-models --secret=TOGETHER_API_KEY)
```

Choose a fresh artifact version and an output prefix that this run owns:

```bash
export MINI_VERSION=$(date -u +%Y.%m.%d.%H%M%S)
export MINI_PREFIX="s3://marin-rno2a/marin/experiments/mini840-reproduction/$MINI_VERSION"
export MINI_WHEELS=d5dd315d58f0f47514dd0a2208e90e2a48541328
```

Stage the unchanged first task from each of the 21 benchmarks and the full expanded cohort. This CPU job also resolves a complete revision-pinned candidate weight mirror:

```bash
uv run --frozen iris --cluster marin job run --target-cluster cw-rno2a \
  --job-name "mini840-inputs-$MINI_VERSION" \
  --cpu 4 --memory 16GB --disk 32GB --enable-extra-resources \
  --timeout 1800 --priority batch -- \
  uv run --frozen python -m experiments.evaluation.mini_policy_runtime.stage_inputs \
    --wheels-revision "$MINI_WHEELS" --output-prefix "$MINI_PREFIX"
```

The command waits for completion. Download its staging report and read the model URI:

```bash
uv run --frozen fsutil cp "$MINI_PREFIX/smoke-runtime-v4-staged-inputs.json" ./mini840-inputs.json
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
    --data-uri "$MINI_PREFIX/smoke-runtime-v4" --output-prefix "$MINI_PREFIX"
```

This starts environments from the published recipes and may build their images. It does not require 197 named snapshots. A failed preflight prevents GPU launch.

## Plan, launch and inspect

Inspect the launch plan first:

```bash
uv run --frozen --with-editable ./vendor/harbor --with daytona==0.200.2 --prerelease allow \
  python -m experiments.evaluation.mini_policy_runtime.launch \
  --version "$MINI_VERSION" --model-uri "$MINI_MODEL_URI" \
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

On RNO2A, the 21-benchmark smoke at runtime `d3c4ad178b625b992b69b9af346f97c03a1df1e8` generated all 21 responses and ran all original verifiers. Trace inspection accepted 20 paths and rejected Tau3's mechanically graded zero because its counterpart model name lost the provider namespace. The corrected Tau3-only rerun at runtime `8636783665` started a real counterpart conversation with no MCP errors and ran the original verifier. Its zero reward was a model failure: the model ignored the explicit `send_message_to_user` instruction and ended with ordinary answer text.

- Main smoke: artifact version `2026.10.31.5`, result root `s3://marin-rno2a/marin/checkpoints/mini840-e2e/2026.10.31.5`.
- Corrected Tau3 canary: `2026.10.31.6`, result root `s3://marin-rno2a/marin/checkpoints/mini840-e2e/2026.10.31.6`.
- All seven agentic paths made real Pi tool calls. Native startup/Pi installation passed for all 197 environment sources with zero failures.
- Read-only verifier controls passed MATH500 reference grading and IFBench positive/negative grading. A deliberately incorrect OlympiadBench response called the co-hosted MiniMax judge over HTTP successfully and received 0.0.
- Private native evidence and control receipts are preserved outside temporary storage under `s3://marin-rno2a/marin/experiments/mini-eval-policy/e2e-20261031`.

The independent reproduction from issue #10804 is pending. These checks establish execution paths; no complete 1,626-rollout Mini840 or 450-rollout Micro210 quality score is claimed.
