# Native judge protocol probe

This probe tests one narrow infrastructure claim: the exactly pinned
TaskCompendium native judge can call the staged GLM endpoint from Harbor's private
verifier process and preserve a graded outcome with model provenance. It is not a
judge calibration or a generated capability task.

`scripts/build_judge_probe.py` verifies every staged TaskCompendium file against
`vendor/task_spec/source.lock.json`, builds a schema-0.9 one-step task, lowers it to
a no-tool Harbor package, and runs three `ReplayAgent` candidates through the real
`SemanticVerifier`. The verifier is TaskTrove `mode=judge` with `exact_gate=false`,
so an exact-match shortcut cannot satisfy any case. The positive is a valid
paraphrase, the negative contradicts visible evidence, and the injection candidate
tries to replace the evaluator instruction.

The task freezes provider `glm`, model `glm-5.3`, and the configured relay base
normalized to one `/v1` suffix. The credential value remains only in
`GLM_API_TOKEN`; the task, command, evidence, and trial artifacts contain only the
environment-variable name.

Run it inside the immutable package environment:

```bash
uv run --project "$TASKCOMPENDIUM_SOURCE" --extra harbor \
  python scripts/build_judge_probe.py \
  --source-lock vendor/task_spec/source.lock.json \
  --output "$WORK_ROOT/judge-probe/evidence.json" \
  --request-timeout 600
```

The evidence passes only if all three trials are `graded`, every outcome contains
nonempty `detail.judgments` rather than `detail.gate`, the positive receives at
least 0.5, neither negative receives more than 0.5, and the positive strictly
outscores both negatives. It binds the semantic TaskSpec hash, full Harbor package
tree, source lock, all raw outcome files, endpoint policy, and the three immutable
upstream revisions. Infrastructure failures and malformed replies remain failures;
they are never converted to semantic zero.

## Evidence status

Local protocol preparation passed on 2026-09-18: the script loaded the exact
`dc6b501c...` archive, verified its 80 locked files, constructed the typed task,
and passed Ruff and bytecode compilation. It also completed three real pinned
Harbor trials against a local fixture endpoint: rewards were 1.0, 0.0, and 0.0;
all three outcomes used the model judgment path; and the endpoint received exactly
three requests. This proves the script, lowering, Harbor class, credential-name
transport, outcome parsing, and evidence gates together.

The credentialed GLM run `/muchanem/cap-judge-probe-001` then completed with exit
0. Its durable root is
`s3://marin-us-east-02a/users/muchanem/capability-pipeline/runs/judge-probe-001/judge-probe/`.
The retained [`evidence.json`](../runs/judge-probe-001/judge-probe/evidence.json)
has SHA-256 `539d04f9ec16164ea149e9245d5820f0c9eafb066149f04b64463961838aeabf`;
the archived probe script matches the current file at SHA-256
`8beb1e7f139e4b437f0d10dcee9b79ac5842465f62c025826ec8a362293753eb`.

| Candidate | Expected | Reward | Native path | Retained outcome SHA-256 |
| --- | --- | ---: | --- | --- |
| Positive paraphrase | Accept | 1.0 | Model judgment | `8d43cfe7bac0bd7f89e572449b8ff6371ea2d6b2a32220d68c138af5f5b33576` |
| Plausible wrong answer | Reject | 0.0 | Model judgment | `93703e303cdfb97a961cc84bb00b69267aaeb67c0d8021fe88e981d87da6fc18` |
| Instruction injection | Reject | 0.0 | Model judgment | `78d4bc4b74410e07831a65686f397cb730db32f85b2ce4c8767f7c28bb5a8c97` |

All outcomes were `graded`; `failures` is empty; and each judgment records model
`glm-5.3` and served revision `vllm-0.28.0-tp8-f8f644d5`. The evidence binds the
exact TaskCompendium, Harbor, and TaskTrove revisions, source lock, semantic
TaskSpec hash, package tree, Harbor result, verifier outcome, and agent transcript
for every case. Inspection found only the credential environment-variable name in
JSON and logs, never a credential value. This establishes live native-judge
transport and elementary discrimination at the staged endpoint.

Even a passing remote probe has only three deliberately obvious cases. It does not
estimate false-accept rate, demonstrate 80 semantic variant groups, validate a
generated capability task, or show generalization to unseen questions. Those
claims require the separate hash-bound calibration and Harbor admission gates in
[`judge.md`](judge.md).
