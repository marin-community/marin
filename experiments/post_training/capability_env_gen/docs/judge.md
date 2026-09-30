# Private judge integration and calibration

Published judge tasks use the native judge path in the pinned TaskCompendium
revision. They do not use `capability_pipeline.judge.judge`, whose anchored rubric
format is a separate experimental evaluator. TaskCompendium's Harbor verifier
constructs its judge client inside the private verifier process, reads the API key
from the configured environment-variable name, and preserves `infra_error` without
turning it into reward zero.

A generated judge step must use a typed TaskTrove verifier with `mode: "judge"`,
the pinned implementation revision, source-ontology parameters, and an explicit
`JudgeConfig`. A reference judge typically includes nonempty private `references`,
an `exact_gate`, and a zero-temperature policy with explicit model, provider, base
URL, size class, and sample count. `JudgeView` declares the exact transcript,
workspace files, and private reference-context resources the judge can see. Any
context resource has only the `verifier` or `oracle` role. The model-visible task
must not mention the judge, rubric, references, reward, calibration controls, or
credential name.

At the pinned revision, native judge semantics are limited to either one reference
comparison scored `0`, `0.5`, or `1`, or an equal-weight checklist of binary
criteria whose mean is the reward. The pre-judge gates are the source verifier's
generic constraints and optional exact match. Native judge mode cannot run a
task-authored executable precheck or implement arbitrary weights, penalties,
critical criteria, or disqualifier aggregation. Splitting an executable precheck
and judge across steps does not supply a conjunctive workaround: the pinned Harbor
lowering rejects multi-step `all_required_steps`; `mean` can dilute a failed gate,
and `final` discards it.

The pipeline has a mandatory composite-verifier extension for blueprints that need
executable checks plus native judging. The lowered package replaces its base
specification with an explicit unsupported marker and records exact TaskCompendium,
adapter, policy, configuration, and original-specification hashes. Unextended
TaskCompendium therefore fails closed. The configured adapter downloads the actual
declared submission and `JudgeView` files from the trial environment, runs each
private script in a fresh network-blocked Daytona verifier sandbox, injects the
machine results and declared private reference context into the native judge, and
then aggregates the components. Infrastructure, invalid-task, and extraction
outcomes remain ungraded. A configuration may claim only the aggregation features
implemented and tested by its pinned policy. Conditional section caps name a
machine trigger and threshold, exact judge-criterion indices, and a maximum
weighted fraction; the result records both raw and effective criterion vectors and
every applied cap. This is distinct from a flat penalty or overall gate.
Calibration evaluates the implemented contract and cannot repair a lossy
aggregation.

The runtime passes only `{"judge_api_key_env":"GLM_API_TOKEN"}` as Harbor verifier
configuration. The token value stays in the process environment and must never be
stored in `specification.json`, a resource, a prompt, a result, or a trace. Missing
credentials, request errors, and malformed judge replies are ungraded
infrastructure outcomes.

The controller stages a nonsecret `judge-policy.json` for builders and later binds
its digest into runtime attestation. It normalizes the configured relay root to a
single `/v1` suffix because the pinned client appends `/chat/completions`. The
TaskSpec provider, model, and base URL must exactly match that frozen policy.

Every judge task bundle also contains a controller-private
`judge-calibration.json`:

```json
{
  "schema_version": "taskcompendium-judge-calibration-v1",
  "specification_sha256": "sha256 of the exact specification.json bytes",
  "cases": [
    {
      "id": "oracle-001",
      "kind": "oracle",
      "source_family": "fixed-task-source",
      "variant_group": "acceptable-design-001",
      "design_label": "independent valid derivation",
      "expected_judge_path": "model",
      "step_index": 0,
      "candidate": "A labeled acceptable answer.",
      "workspace_files": {},
      "transcript": [],
      "expected_reward_range": [0.8, 1.0]
    }
  ]
}
```

The complete fixture needs at least 40 acceptable and 40 unacceptable semantic
variant groups and
must include `oracle`, `plausible_wrong`, `empty`, and `prompt_injection` strata.
Every case declares its `source_family`, `variant_group`, and `design_label`.
Paraphrases and closely related constructions share a variant group, so they do
not pad the effective sample. A fixed task can legitimately use one source family
throughout. Exact candidate duplicates are rejected, while grouping quality and
semantic nonduplication still need human audit.
Workspace evidence uses absolute paths already declared by the task. Labels must
come from an independent domain reviewer or executable evidence check, not the task
author or judge. The fixture stays outside agent-visible resources.

For a composite task, every case also declares
`"expected_machine_gate":"pass"` or `"fail"`. Workspace state is created by
optional Harbor replay `commands`; prefilled `workspace_files` and supplied
transcripts are rejected because they would bypass the real task environment. At
least 40 positive and 40 negative variant groups must both pass the deterministic
gates and reach the native model judge. At least one additional control must make a
declared deterministic gate fail; that case declares
`"expected_judge_path":"skipped_machine_gate"` and proves no model call occurs.
This keeps machine rejection evidence separate from model accuracy and stability.

Run the exact configured task judge through the pinned package with:

```bash
uv run -m capability_pipeline.cli judge-calibrate-task \
  --bundle task \
  --package harbor \
  --out judge-calibration \
  --taskcompendium-source "$TASKCOMPENDIUM_SOURCE" \
  --concurrency 64 \
  --repeats 3
```

For an ordinary native judge task, the command invokes TaskCompendium's real
`grade_attempt` and `OpenAIJudgeClient`. For a composite task it instead executes a
full replay Harbor trial through `CompositeSemanticVerifier`, including the real
task environment, downloaded evidence, isolated machine scripts, native model
call, and final composed reward. Each result must carry the exact adapter, policy,
and configuration hashes. Each case declares
whether it should reach the model, exact gate, or constraint gate. At least 40
positive and 40 negative variant groups must actually reach the
model; exact and constraint outcomes are descriptive only, and every
`plausible_wrong` case must exercise the model. Passing requires all
declared score ranges, a 95% lower bound of at least 0.85 for balanced accuracy, a
95% upper bound of at most 0.10 for false accepts, and a 95% lower bound of at
least 0.90 for exact repeated-decision agreement. Class-rate intervals use unique
variant groups, with every grouped case and repeat required to be correct;
repeat-level rates are descriptive. The agreement interval uses only groups whose
every case and repeat reached the model. For composite calibration, those groups
must also pass every deterministic gate. Exact, constraint, and machine-gated
outcomes cannot pad model-stability evidence. The report keeps per-stratum model
paths and machine-gate paths separate.

The pinned native client sends `model`, `messages`, and `temperature`. It does not
send `max_tokens` or GLM `chat_template_kwargs`. The bounded credentialed protocol
run documented in [`judge_probe.md`](judge_probe.md) produced parseable native
verdicts for one positive and two negative model-path cases within its contract
timeout. This establishes the staged transport, not the accuracy of a task judge.
The reported intervals describe performance conditional on this designed
task-specific collection. They do not estimate generalization to unseen questions
or source families.

Calibration is still not a runtime certificate. The same bundle must complete a
real Harbor lifecycle with the native semantic verifier, artifact-bound raw trial
evidence, a positive independent-solver case, and adversarial negatives. A judge
task stays `pending_runtime` unless both its calibration report and Harbor evidence
pass.
