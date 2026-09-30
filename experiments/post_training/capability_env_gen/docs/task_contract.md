# Generated-task contract

This document fixes the output contract for the synthetic capability-task pipeline. It is based on the current head of Marin PR [#9187](https://github.com/marin-community/marin/pull/9187), not on Marin `main`:

- TaskCompendium source: [`dc6b501c8604bcd2e3c20c1e9947679845fdfef8`](https://github.com/marin-community/marin/commit/dc6b501c8604bcd2e3c20c1e9947679845fdfef8), committed 2026-09-16. The normative prose is [`SPECIFICATION.md`](https://github.com/marin-community/marin/blob/dc6b501c8604bcd2e3c20c1e9947679845fdfef8/lib/taskcompendium/SPECIFICATION.md), the machine schema is [`task-spec-v0.9.json`](https://github.com/marin-community/marin/blob/dc6b501c8604bcd2e3c20c1e9947679845fdfef8/lib/taskcompendium/schema/task-spec-v0.9.json), and Python validation lives in [`models.py`](https://github.com/marin-community/marin/blob/dc6b501c8604bcd2e3c20c1e9947679845fdfef8/lib/taskcompendium/src/taskcompendium/models.py). PR #9187 is an open draft, so the pipeline must pin this commit and must not resolve the moving branch at run time.
- Harbor target: [`93147ea9e07b04ec8d2eb5afd2916386f1aacc69`](https://github.com/marin-community/harbor/commit/93147ea9e07b04ec8d2eb5afd2916386f1aacc69), as declared by TaskCompendium at the pinned head.
- TaskTrove verifier implementation: current revision [`b76d03131cd88bd9fc711dba206659027edba3a8`](https://github.com/marin-community/marin/commit/b76d03131cd88bd9fc711dba206659027edba3a8); the schema also accepts legacy revision `b2b68d8b0a770cdc0ab3903780172c4b3eea81b1` for imported examples.
- ShellSim integration revision: [`5674a9492c35ffe390a0d23b49c0a340b12beb30`](https://github.com/rjpower/shellsim/commit/5674a9492c35ffe390a0d23b49c0a340b12beb30), package version 0.1.0. TaskCompendium pins this revision in its bridge `Cargo.toml` and lockfile. Upstream `main` was [`930d0d409910b65abf9d1db5b0df348143774446`](https://github.com/rjpower/shellsim/commit/930d0d409910b65abf9d1db5b0df348143774446), version 0.1.6, when inspected on 2026-09-18. Do not substitute the moving upstream head.
- Published example corpus: [`open-athena/taskcompendium-spike`](https://huggingface.co/datasets/open-athena/taskcompendium-spike) at dataset revision `016d52e79b218677cfefba84e2fe0f97cb5e6e05`, containing 63 schema-0.9 specifications and 164 Harbor lowerings.

## Ownership boundaries

The pipeline must keep four layers distinct:

| Layer | Owns | Must not own |
| --- | --- | --- |
| Proposal | capability rationale, realistic task concept, suggested complexity and verification | fabricated build or validation success |
| `TaskSpec` | fixed task semantics, required initial state/capabilities, source provenance, private correctness contract | ShellSim/Docker choice, Harbor agent, model, sampler, retry policy, output wrapper |
| Rendering | one public submission convention per step | semantic task changes or private evaluation details |
| Target binding/lowering | concrete environment implementation, public tools, limits, Harbor package layout | model choice, agent policy, verifier semantics |

The synthesis handoff is the accepted proposal record:

```json
{
  "proposal": {"...": "the proposal schema defined by capability_pipeline/prompts.py"},
  "review": {"...": "the independent proposal review"},
  "proposal_hash": "sha256 of the canonical proposal",
  "provenance": {
    "catalog_source": {"path": "catalog.json", "sha256": "64 lowercase hex"},
    "capability_record": {"...": "the exact selected catalog record"},
    "capability_record_hash": "sha256 of the canonical capability_record"
  }
}
```

The synthesizer must treat `proposal_hash` as immutable input identity. A practical generated-task provenance mapping is `metadata.source.dataset = "synthetic-capability-proposals"`, `metadata.source.revision = <accepted-run manifest hash or immutable input revision>`, `metadata.source.row = <proposal_hash>`, and `metadata.source.importer_revision = <synthesis code revision>`. Every value must be concrete and nonempty; labels such as `latest`, an uncommitted branch name, or a prompt description are insufficient.

Builder session completion requires a valid, artifact-backed handoff. A zero
process exit is not completion. Substantial sessions persist small work units in
`.capability-progress/<session>.json` and write real artifacts incrementally.
Continuations reuse the same OMP transcript and existing workspace; they do not
restart the task. The controller stops after two consecutive attempts with no
builder-payload change and reports output-budget exhaustion when transcript
metadata contains a length stop. Progress-only files cannot satisfy this gate.
The recovery and evidence contract is detailed in
[`construction_continuation.md`](construction_continuation.md).

## Canonical `TaskSpec` 0.9

Each completed semantic task is one canonical JSON document. Its required top-level fields are:

| Field | Required contract |
| --- | --- |
| `schema_version` | Exactly `"0.9"` after canonical serialization. |
| `id` | Stable, nonempty identifier for this fixed task instance. |
| `steps` | Nonempty ordered array. Each step has model-visible `instructions` and a private tagged `verifier`; answer requirements, resources, and context requirement may use defaults. |
| `requirements` | Semantic capabilities and pinned initial state, independent of the runtime implementation. |
| `resources` | Shared resources with explicit visibility roles. Empty is valid. |
| `metadata.source` | Nonempty `dataset`, `revision`, `row`, and `importer_revision`. |
| `coverage_tags` | Sorted, unique semantic labels. Exactly one `shape:*` is the publication policy. Result wrappers do not belong here. |
| `difficulty` | `null` or a non-boolean integer from 1 through 10. Reviewed published tasks should have a value. |
| `success_policy` | `all_required_steps`, `final`, or `mean`. Harbor rejects `all_required_steps` for multi-step exports. |

Each step contains:

- `instructions`: a direct, natural task request. It may state public formats, behavior, tests, or file paths. It must not mention a grader, verifier, judge, reward, hidden tests, reference answer, or evaluation score.
- `answer_requirements.kind`: one of `text`, `literal`, `json`, `xml`, `csv`, or `final_state`. A rendering may add transport syntax but may not weaken this intrinsic form.
- `verifier`: one of `tasktrove`, `instruction_constraints`, `code_answer`, `predicted_action`, or `provider_state`. Verifier data is private.
- `resources`: step-local resources using the same visibility rules as shared resources.
- `context_requirement`: `instruction_and_workspace` or `prior_conversation`. Use the latter only when a preceding step produces required visible history.

The current semantic capabilities are `filesystem`, `shell`, and `process`. `WorkspaceState` may declare an immutable image digest, absolute work directory, setup commands, and non-overlapping additional roots. Non-default workspace state requires `filesystem`. Images must be a bare `sha256:<64 lowercase hex>` digest or a registry reference ending in `@sha256:<64 lowercase hex>`.

An `ActionInterface` declares a stateful provider surface using `name`, `version`, and a 64-hex `seed_sha256`. It is not a shell capability. Workplace is the only implemented provider family in this spike.

Resources have a normalized relative `path`, one or more unique roles, an `executable` bit, and either embedded bytes or `{kind:"reference", uri, sha256}`. JSON represents embedded bytes as base64. `agent` resources are public and materialized into the environment. `verifier` and `oracle` resources remain private; `oracle` may not also be `agent`. The same `(role, path)` may not appear twice across shared and step-local resources for a step.

Coverage tags use `competency`, `shape`, `subject`, `artifact`, `interaction`, `state`, and `context` namespaces. Tags are lowercase, sorted, and unique. Subject roots are restricted to `math`, `physics`, `chemistry`, `biology`, `medicine`, `computing`, `engineering`, `law`, `finance`, `business`, `government`, `social_science`, `humanities`, `education`, `arts`, `design`, and `media`; recurring specialties may use dotted suffixes. `result:json`, `result:xml`, and `result:file` are added only to lowering manifests.

Canonical bytes are the typed TaskSpec converted to built-ins, JSON-normalized, then encoded with sorted keys and compact separators. The task identity is `sha256(canonical_bytes)`. Parquet is an interchange option, but it must use TaskCompendium's exact Arrow schema and file metadata `taskcompendium.schema_version=0.9`; verifier unions are canonical JSON strings inside the Arrow step records.

JSON Schema validation alone is insufficient. Several invariants, including the difficulty range, digest syntax, path containment, tag ordering/root validation, role conflicts, runtime-verifier consistency, and workspace overlap checks, are implemented in the typed Python constructors/decoder. Publication must pass `taskcompendium.serialization.from_json` at the pinned revision and then reproduce the same canonical bytes and hash.

## Verification contract

`TaskTroveVerifier.mode` accepts `mcq`, `math`, `numeric`, `exact`, `json-schema`, `xml-elements`, `csv-columns`, `ifeval`, `reasoning-gym`, `stdio`, `pytest`, `junit`, `gotest`, `judge`, and `script`, but the current Harbor lowering deliberately rejects `reasoning-gym`, `junit`, `gotest`, and `stdio` with a special judge. The `parameters` object must parse through the pinned TaskTrove verifier ontology. Do not invent parameter layouts from the JSON schema's open object.

Verification types from `Task Generation.md` map as follows:

| Requested type | Contract | Current status |
| --- | --- | --- |
| Simple verifiable | Trusted answer modes such as `exact`, `mcq`, `math`, `numeric`, JSON/XML/CSV structure, or `instruction_constraints`. | Implemented. |
| Code verifiable | Executable `stdio`, `pytest`, or `script` verifier with a private `ContainerRuntime`; workspace tasks use `FinalState`, while answer-only code uses `CodeAnswerVerifier`. | Implemented for the stated subset. Never execute candidates on the host verifier process. |
| LLM-as-judge | `tasktrove` mode `judge` plus explicit `JudgeConfig` containing model/provider/base URL/size class and an explicit evidence view. | The pinned Harbor `SemanticVerifier` constructs `OpenAIJudgeClient` from a named credential environment variable. `test_harbor_judge_transport_preserves_outcome_and_provenance` proves transport and outcome retention. Publication still requires task-private calibration and repeatability controls against the staged GLM policy. |

Every attempt retains one of four semantic outcomes:

| Status | Reward |
| --- | --- |
| `graded` | Numeric; zero is a valid incorrect result. |
| `extraction_error` | `null` |
| `invalid_task` | `null` |
| `infra_error` | `null` |

Never convert extraction, task, or infrastructure failures to semantic score zero in stored attempt data. The demonstration SkyRL adapter archives all attempts and rejects a numeric batch containing an ungraded attempt.

## Rendering and Harbor integration

Choose exactly one rendering per semantic step:

- `AssistantFinal` with `plain`, `boxed_latex`, `json_path`, or `xml_path` extraction;
- `FileSubmission` with an absolute public path and extractor;
- `FinalState` with explicit submitted paths and optional exclusions;
- `FinalActionSubmission` with public function definitions for predicted actions. This predicts an action; it neither grants nor dispatches tools.

A successful Harbor lowering writes `specification.json`, `renderings.json`, `binding.json`, `task.toml`, `manifest.json`, `instruction.md` (or per-step instructions), and only the `agent` resources beneath environment inputs/step workdirs. Its manifest records the TaskSpec hash, source provenance, selected renderings, binding, lowering version `0.9`, Harbor revision, verifier-runtime identities, tags, difficulty, and step names.

Environment and launch compatibility is exact:

| Binding environment | Guaranteed semantic capabilities | Valid use |
| --- | --- | --- |
| `none` | none | Direct assistant response/final-action only, no agent resources or semantic environment requirement. |
| `shellsim` | `filesystem`, `shell` | Persistent lightweight simulated shell/VFS. It does not satisfy `process`. |
| `docker` | `filesystem`, `shell`, `process` | Pinned image workspaces and native process tasks. |
| `provider` | the single exactly matching `ActionInterface` | Stateful provider tools; cannot materialize workspace state. |

Tools are explicit and are not inferred from the environment. The implemented adapters accept at most one binding: a named shell function, a terminal harness interface, or a provider interface. A ShellSim shell function uses `tool_chat`; ShellSim terminal-harness bindings are replay-only. Docker terminal bindings can use replay, Terminus-2, or mini-SWE-agent. Ordered tasks work with TaskCompendium chat/tool/provider/replay adapters; native Terminus-2 and mini-SWE-agent are rejected because they do not retain one conversation across steps.

File submissions, final-state submissions, and public resources add a filesystem requirement during lowering. Final-state submission is reserved for executable workspace verification and requires intrinsic `final_state`. Executable answer-only code uses `CodeAnswerVerifier`, which writes the answer inside the private checker container. Paths must remain within the workdir or declared additional roots. Snapshot exclusions require Docker, a whole-workspace `paths:["."]` submission, and an image-overlay verifier runtime.

## ShellSim contract

The TaskCompendium integration does not use the current upstream `shellsim serve` protocol. It compiles a custom `taskcompendium-shellsim` bridge against ShellSim commit `5674a949...`. That bridge owns one persistent environment and accepts JSONL operations `init`, `exec`, `read`, `write`, `mkdir`, `list`, `walk`, `stat`, and `close`. Limits are cumulative across a trial. Shell working directory, variables, functions, VFS state, and resource usage persist between actions. Exhausted CPU/output terminates later execution, while files remain readable for result collection.

The bridge allows at most 16 MiB per request, 32 MiB per response, and 16 MiB raw per read. ShellSim's baseline resource model defaults to 10,000,000 CPU units, 64 MiB memory, 64 MiB disk, and 4 MiB output, but a Harbor `ShellSimEnvironment` binding lowers `max_steps` to the CPU limit and `max_output_bytes` to the output limit. Set these budgets explicitly for generated tasks when the defaults are not justified by a measured pilot.

Simulated programs cannot access the host filesystem, processes, ambient network, environment, or clock. Supported HTTP behavior is fixture-driven. Unsupported commands and syntax fail visibly; there is no fallback to the host. Python is a modeled, deliberately partial runtime and does not provide native extensions, arbitrary packages, compilers, or arbitrary machine code. Use Docker whenever reliable success requires the `process` capability, native binaries/extensions, package installation, or behavior outside the pinned simulator's tested surface.

Every ShellSim candidate must have an executable compatibility probe that runs its setup and representative verifier path against the pinned bridge. A prose claim that standard Unix or Python should work is not evidence.

## Compact artifact examples

This is a complete schema-0.9 simple-answer `specification.json`. The expected value is private because the full semantic record is private; the rendered instruction is the model-visible projection.

```json
{"coverage_tags":["artifact:numeric_answer","competency:quantitative_reasoning","shape:calculation","subject:math"],"difficulty":2,"id":"synthetic/arithmetic/0001","metadata":{"competencies":[],"source":{"dataset":"synthetic-capability-proposals","importer_revision":"synthesizer-git-sha","revision":"accepted-run-sha256","row":"proposal-sha256"},"task_shape":"answer"},"requirements":{"action_interfaces":[],"capabilities":[],"state":{"additional_directories":[],"image":null,"setup_commands":[],"workdir":"/app"}},"resources":[],"schema_version":"0.9","steps":[{"answer_requirements":{"kind":"text"},"context_requirement":"instruction_and_workspace","instructions":"A box contains 6 rows of 7 bolts. How many bolts are in the box? Return only the integer.","resources":[],"verifier":{"implementation_revision":"b76d03131cd88bd9fc711dba206659027edba3a8","kind":"tasktrove","mode":"exact","parameters":{"expected":["42"]}}}],"success_policy":"all_required_steps"}
```

The corresponding direct-chat artifacts are:

```json
[{"id":"plain","submission":{"kind":"assistant_final","extractor":{"kind":"plain"}},"version":"0.2","instruction_surface":"original"}]
```

```json
{"environment":{"kind":"none"},"tools":[]}
```

A lightweight terminal task declares ShellSim-independent semantics in `specification.json`:

```json
{
  "requirements": {
    "capabilities": ["filesystem", "shell"],
    "state": {"image": null, "workdir": "/workspace", "setup_commands": [], "additional_directories": ["/output"]},
    "action_interfaces": []
  },
  "steps": [{
    "instructions": "Read input.txt, count unique normalized values, and write the sorted counts to /output/counts.csv.",
    "answer_requirements": {"kind": "final_state"},
    "context_requirement": "instruction_and_workspace",
    "resources": [],
    "verifier": {
      "kind": "tasktrove",
      "mode": "script",
      "parameters": {"path": "check.py", "args": ["/output/counts.csv"], "timeout": 60.0},
      "runtime": {
        "kind": "container",
        "image": "registry.example/verifier@sha256:REPLACE_WITH_64_HEX_DIGEST",
        "timeout": 120.0,
        "revision": "b76d03131cd88bd9fc711dba206659027edba3a8",
        "workspace": {"kind": "empty"},
        "supervisor_python": "python3"
      },
      "implementation_revision": "b76d03131cd88bd9fc711dba206659027edba3a8"
    }
  }]
}
```

The omitted required top-level identity, metadata, resources, tags, difficulty, schema version, and success policy are the same kind of fields shown in the complete example. The verifier image placeholder must be replaced before typed validation. Its rendering and binding are separate:

The container verifier's `supervisor_python` must resolve inside its pinned
image to Python 3.12 or newer. The pinned verifier dependency closure includes
`numpy==2.5.3`, which cannot install under Python 3.11. The task author must
choose a compatible image and interpreter.

```json
[{"id":"workspace","submission":{"kind":"final_state","paths":["/output/counts.csv"],"excluded_paths":[]},"version":"0.2","instruction_surface":"original"}]
```

```json
{"environment":{"kind":"shellsim","workdir":"/workspace","max_steps":100000,"max_output_bytes":1048576,"setup_commands":[],"additional_directories":["/output"]},"tools":[{"kind":"shell","name":"shell","backend":"shellsim"}]}
```

A full container task changes only the semantic requirements and target binding. The semantic state must name the same immutable image as the binding:

```json
{"capabilities":["filesystem","shell","process"],"state":{"image":"registry.example/task@sha256:REPLACE_WITH_64_HEX_DIGEST","workdir":"/app","setup_commands":[],"additional_directories":[]},"action_interfaces":[]}
```

```json
{"environment":{"kind":"docker","image":"registry.example/task@sha256:REPLACE_WITH_64_HEX_DIGEST","workdir":"/app","setup_commands":[],"additional_directories":[]},"tools":[{"kind":"harness","interface":"terminal","backend":"docker"}]}
```

## Synthesis and publication gates

Generation status is evidence, not optimism. The synthesis adapter records `pending_build`, `pending_build_acceptance`, `pending_schema_validation`, `pending_judge_policy`, `lowered`, `pending_judge_calibration`, `controls_passed_pending_rollout`, `pending_runtime`, `runtime_controls_passed_pending_adversary`, `pending_solver_adjudication`, `pending_attack_adjudication`, `pending_adversary_retry`, `validated`, `pending_quality_review`, `failed`, or `quality_accepted`. Passing direct verifier controls is deliberately `controls_passed_pending_rollout`, and passing Harbor trials for builder-authored fixed controls is `runtime_controls_passed_pending_adversary`. An agent transcript, partial directory, plausible spec, timed-out run, or self-described attack is not `validated`. `validated` and the separate `runtime_validated` flag cover the runtime gate only. The controller exports only `quality_accepted` tasks after a fresh immutable semantic review; other records remain in the synthesis ledger with the last concrete error and artifact paths.

The final public binding must preserve the admitted solver environment exactly:
reasoning uses `none` with no tools, ShellSim uses `shellsim` tools, and container
uses Docker tools. Private evaluator containers and composed machine checks do
not grant tools to the solver. A different solver surface needs a new hash-bound
admission and cannot be approved by a construction self-report.

The minimum independent gates are:

1. Decode with the pinned TaskCompendium Python type, canonicalize, hash, decode again, and prove byte/hash stability.
2. Confirm exact accepted-proposal provenance and reject duplicate task IDs or proposal hashes.
3. Inspect model-visible instructions and every rendered variant for verifier/oracle leakage.
4. Prove resource-role isolation by inspecting the lowered public filesystem/package.
5. Validate one rendering per step, intrinsic answer preservation, path containment, environment capabilities/state, tool binding, and launch compatibility.
6. Run the private verifier per step against at least a known-correct submission, an empty or malformed submission, a plausible wrong answer, a task-specific shortcut or reward hack, and an induced infrastructure failure. Assert all four semantic outcome classes remain distinguishable where applicable. Every fixed control records its source author and category.
7. Run the real Harbor `Trial` lifecycle for the selected environment. For ShellSim, use the pinned bridge and exercise representative setup/tool/verifier behavior; for Docker, use the immutable task and verifier images.
8. Record model, target, environment, tools, verifier revisions, budgets, trace/transcript, and semantic outcome for pilot rollouts. Capability quality and difficulty require real attempts; package conformance alone is insufficient.

The built-in runtime, or an explicit `synthesize --runtime-runner` override, is trusted infrastructure separate from builder sessions. It receives `--package`, `--bundle`, `--controls`, and `--output`. Its evidence must bind the specification, full Harbor package tree, Harbor manifest, and controls by SHA-256; identify Harbor revision `93147ea...`; record a fresh network-blocked environment per trial; and reference hashed raw Harbor, oracle, solver, isolation, and adversary artifacts stored beside the evidence JSON. Every authored reference must first pass through the real grader. Positive control evidence must also come from an independent solver path with bounded, retained attempts; exhaustion stops at `pending_solver_adjudication` rather than weakening the task. Builder-authored fixed negatives remain labeled as authored controls; a separate attack agent and artifact are required for the independent adversarial gate. The controller rejects summary-only JSON and does not export the package when any binding or raw artifact is missing.

A retained runtime-control attempt with no verifier result and an identified
Daytona provider failure may be revalidated without another semantic builder
repair. This path is explicit: `--retry-infrastructure` requires a fresh
`capability-daytona-health-v1` receipt from a successful network-blocked
sandbox create/delete probe, including the final provider lookup showing the
sandbox is gone. The controller freezes that receipt, archives the original
status and raw runtime artifacts, and reruns every runtime gate against the
unchanged task. A graded failure, an unclassified exception, an old receipt, or
a failure outside the runtime-control gate cannot use this path. Another
provider failure remains a failed revalidation rather than triggering an
unbounded retry or construction repair.

A rewarded independent attack is reviewed at most once for a given immutable
task-and-evidence snapshot. A legitimate correct or partial outcome may clear
only the reward alarm through a hash-bound sidecar. Exploit, uncertainty, and
incomplete review remain `pending_attack_adjudication`; controller resume reuses
that durable result rather than sampling another reviewer. A new adjudication
requires a bounded construction repair that changes the frozen task or runtime
evidence, and prior receipts remain available to repair and quality review.

An attack agent that exhausts its retained bounded response attempts without a
candidate or semantic grade is `pending_adversary_retry`, not a rewarded-attack
adjudication and not a construction defect. The status binds its raw trial and
request-log hashes, attempt budgets, and failed strategies. Normal resume does
not spend semantic repair rounds on it; a later follow-up must run a fresh full
independent adversary suite against the unchanged package and retain all three
strategy outcomes.

The adversary transport has a typed, retained output policy. Its default is
`131072,remaining_context`: the first request is capped at 128K and a length
stop triggers exactly one request with OpenAI-compatible `max_tokens: null`,
meaning the server's remaining context. The observed GLM limit is 262,144 total
prompt-plus-output tokens, so a literal 256K output request is not valid for a
nonempty task prompt. `CAPABILITY_ADVERSARY_TOKEN_LIMITS` may select another
strictly increasing positive sequence below that total limit followed by
`remaining_context`; `CAPABILITY_ADVERSARY_REQUEST_TIMEOUT` is a finite,
bounded per-request value up to 3600 seconds. The runtime records the selected
policy and every actual request budget, finish reason, and usage. A legacy
terminal `failed` record may use `--retry-adversary` only when its exact three
old incomplete-adversary issues, all non-adversary controls, and raw truncated
trial evidence verify; the controller archives that historical state unchanged
and reruns the complete runtime suite without invoking construction repair.

A machine-check plus native-judge task uses the maintained
`taskcompendium-composite-verifier-v1` contract. Its private per-step checks run
in declared immutable images and are labeled gate, weighted criterion, or
weighted penalty. Judge weights and critical indices map to the native checklist
criteria. Optional conditional caps name one machine trigger and the judge
criterion indices whose combined weighted section is capped. The result detail
retains raw and effective judge criterion vectors, each machine result, applied
caps, failed gates/criticals, positive and penalty sums, and the denominator.
A failed deterministic gate skips the model call and forces zero; an
infrastructure, invalid-task, or extraction outcome never becomes numeric.
FileSubmission, FinalState, JudgeView files and transcript evidence are copied
using the pinned TaskCompendium path rules, and original private contexts plus
trusted machine results are presented to the native judge.

The pinned native judge's `samples` setting only repeats binary checklist calls
and averages them. It does not implement adjudication. When the admitted rubric
requires two judgments followed by a third on disagreement, the composite step
must declare `judge.consensus` with `mode: two_then_third`,
`initial_samples: 2`, `disagreement_tolerance: 0.0`, and
`resolution: median`; the native policy must also declare two samples. Any
initial criterion disagreement triggers exactly one additional full one-sample
judge pass. Result detail retains both initial judgments, disagreement indices,
the adjudicator judgments, resolved scores, judge-pass count, and actual model
call count. The fields are `native_grade_attempt_count` (`1` or `2`) and
`judge_vector_count` (`2` or `3` complete criterion vectors), and
`judge_call_count` (actual model completions: criteria times samples). Failure
or incomplete coverage in either pass remains ungraded.

Ordinal judge items use explicit `judge.anchor_groups`. Each group lists
non-overlapping, equal-weight binary criterion indices ordered from the lowest
to highest threshold. Thus two bits encode `0/1/2` and three encode `0/1/2/3`.
Every raw sample and the resolved vector must be monotone, such as `[1, 1, 0]`;
an incoherent sequence such as `[0, 1, 0]` is an invalid task and never sums to
an anchor score. Critical indices and machine gates apply to the threshold bits
needed for full-credit conditions, no-zero rules, and global disqualifiers.
Threshold `bN` means that the original ordinal anchor is at least N, using the
full original anchor meaning: b1 accepts anchors 1 and above, b2 accepts anchors
2 and above, and so on. A threshold must not silently strengthen an admitted
partial-credit anchor or reject an alternate implementation that anchor permits.
If the source defines only endpoint anchors, any intermediate anchor is a
proposed interpolation that needs a fresh proposal review; it must not be
described as an original preserved boundary.
When a deterministic gate fails, the result retains the machine evidence and
reward zero but no fabricated raw judge or anchor score. The judge is skipped;
only a criterion disagreement can trigger the third pass.

Composite exports are deliberately unreadable as ordinary schema-0.9 packages.
The valid semantic record is retained as `composite-specification.json`; the
ordinary `specification.json` is an unsupported-extension sentinel and the
manifest contains the exact mandatory adapter/config/policy and source-overlay
hashes. The pinned TaskCompendium verifier overlay rejects that marker. Its
patched Harbor runner reads the preserved record only for the exact declared
composite-verifier import and marker hashes, after which the composite verifier
rechecks the package. An old or unconfigured consumer therefore fails closed
instead of silently applying native mean reward.

Maintained Daytona `dt.py`, `dt.sh`, `validate_env.py`, `verify.py`, and `adapter.py` helpers may be staged into each proposal workspace for untrusted building and cold base/oracle validation. Their call log and validation records are build evidence. They do not replace the real Harbor lifecycle, independent solver, or adversarial gate required for `validated`.

The pinned package's conformance command is:

```bash
uv run --project lib/taskcompendium --extra harbor --group test \
  pytest -m harbor_conformance lib/taskcompendium/tests
```

## Intentional partial-credit controls

`controls.json` uses these exact class/category pairs: `positive/known_correct`,
`negative/plausible_wrong`, `negative/task_specific_shortcut`,
`negative/reward_hack`, `malformed/empty_or_malformed`, and
`partial/criterion_mutation`. A control ID may be descriptive (such as `gold`);
its category must use the corresponding enum above.

Use `class: "partial"` and `category: "criterion_mutation"` for a candidate that
fails a specific criterion while legitimately earning other credit. Supply
`partial_credit_reason` tied to the admitted rubric, `expect.status: "graded"`,
both reward bounds with `reward_max < 1`, and nonempty `expect.assertions` such as
`{"path":["detail","stdout","criteria","C7","passed"],"equals":false}`.
The controller traverses actual grading detail (decoding embedded JSON strings),
and rejects absent or mismatched values. Preserve raw grading artifacts. These
cases do not replace required whole-answer negative/shortcut controls, and may
not relabel critical failures to evade low-reward requirements. See
[partial control examples](partial_controls.md).

## Known limitations and unavailable evidence

- PR #9187 is a draft. Its current GitHub checks include a failing `agentic-lint` gate because the required label/review workflow was not completed; ordinary unit, integration, lint, docs, and CodeQL checks shown on the PR passed. Pinning avoids silent schema drift but does not turn the draft into a stable release.
- The pinned normative prose says LLM-as-judge is stubbed, but the pinned source implements `grade_judge_attempt`, constructs `OpenAIJudgeClient` inside `SemanticVerifier`, and has a Harbor transport test. This repository follows the executable source contract and records the prose mismatch. The native client sends model, messages, and temperature only; it does not set a token budget or GLM reasoning template arguments, so live policy calibration and timeout evidence are mandatory.
- The checked-in `lib/taskcompendium/examples/{poc,first_wave}` manifests and Parquet files at the PR head are stale schema 0.8 artifacts. The schema-0.9 reader rejects the checked-in POC Parquet. Re-running the pinned builders produces schema-0.9 JSON, Parquet, and Harbor packages successfully. Do not consume the checked-in 0.8 files as the final contract.
- The public Hugging Face dataset at revision `016d52e...` is accessible and reports successful schema-0.9 serialization/package-integrity checks. Its own validation explicitly does not establish live-model performance, training suitability, or external-service availability.
- The example verifier/task image `sha256:1e02be2c8950c47f6d8eef8bfd775abbd9c547c68d57ac6a887fd13b8ae7c8b7` is a local Docker image ID, not a portable registry reference. PR #9187 says it must be rebuilt on another machine. A no-tool live Harbor trial has passed locally. ShellSim uses the staged pinned bridge, and Docker uses the Daytona adapter; their publication status still depends on retained live pilot evidence for the generated task.
- Some imported provenance points at `s3://marin-us-east-02a/...`. Those objects were not fetched. The repository fixtures and public published dataset were sufficient to inspect and regenerate sample packages without asserting access to the private source corpus.
- MCP bundles, browser environments, reactive user simulators, and general interactive environments are design sketches only. Do not generate a `validated` task requiring them under schema 0.9.
- TaskCompendium's direct non-Harbor chat target and production SkyRL runner registration are future work. Harbor is the only implemented lowering target at this revision. The one-step training smoke described in the PR showed tool-call parsing and answer-extraction failures; it is not evidence of sustained training readiness.
