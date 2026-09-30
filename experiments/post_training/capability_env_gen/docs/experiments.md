# Empirical iteration ledger

## First quality-accepted generated task

c17 continuation 008 completed with one runtime-validated, quality-accepted
task. Its final immutable snapshot is
`runs/synthesis-c17-continuation-008/terminal-pull-0847`, snapshot
`b6d452e8ff084185a49129c7a4cdeb9c`: all 1,774 published files were pulled and
hash verified, with no omissions. The fresh independent solver and authored
oracle each scored 1.0. All six authored adversarial controls and all three
independent attacks scored 0.0 in isolated verifier sandboxes. The quality
review accepted all 43 conditions with no required changes; its six limitations
remain explicit.

The repaired boundary attack used two phases. Planning and finalization each
made one request with `max_tokens=131072` and stopped normally after 2,519 and
977 completion tokens respectively. The configured `[131072, null]` policy
therefore never reached its full-context fallback. The final attack changed only
the internal spacing of the doubled live-region message and correctly received
0.0. This establishes that 128K was sufficient for this run; it does not measure
a 256K output request. Exact identities, receipts, and semantic findings are in
`docs/audits/c17_acceptance_008.json`. Source-bundle verification is recorded
separately from the immutable result snapshot so later controller changes do not
retroactively alter the accepted evidence.

## Daytona capacity observations

Consensus010 preserved the actual snapshot-creation error before any model or
grader call: `Snapshot quota exceeded. Maximum allowed: 40`. Its terminal pull
is `runs/composite-consensus-probe-010/terminal-pull-0610/`. This is separate from
c17's exhausted HTTP429 sandbox-creation retries and the root's c22 fixture
replay001 creation rate-limit error. No c22 sandbox was created, and its replay
has no semantic outcome. Probe009's original snapshot-creation exception was
masked by a subsequent lookup, so 010 does not retroactively establish 009's
underlying cause. Snapshot cleanup is limited to positively identified unused,
reproducible pipeline probe resources.

Papercut5 now requests shared snapshot capacity/lifecycle support. Its displaced
metric-attribution issue remains useful: `/admin/engines/metrics` exposes engine
KV/running counts without pool labels, while `/stats` exposes aggregate pool
inflight counts without engine membership. The retained pilot002 telemetry at
`runs/proposal-pilot-002/telemetry/20260919T020840Z.json` records 237 bulk inflight,
median KV 0.039 and maximum 0.498, but cannot attribute those engine values to a
pool. The ten-entry active list prioritizes the currently blocking provider limits.

## c22 fixture-stage continuation — recovered, task still in progress

The new continuation policy recovered s2 in one resumed process attempt after
nine historical attempts without its generator/handoff. The successful attempt
took 745.89 seconds, issued 135 tool calls, and recorded zero output-length stops
or compactions. Its original 30-file scope is intact. The root verified every
fixture hash, byte count and line count: 173,251 bytes and 1,395 lines total.
All nine old attempt records/logs and all 230 original transcript events with IDs
are unchanged; only the non-ID title header was updated by OMP. Historical raw
bytes remain in the frozen original pull.

The fresh pull is snapshot `8d29312e847b489e8c3a024ad7fe2a42`, 124 published
files with no upstream omissions. Root audit SHA-256:
`0a6733e6896a7b12378cf3e911b26531a8153ba210cc82ceb74846915008687b`,
`docs/audits/c22_continuation_recovery_001.json`. Builder triple-regeneration and
validator results are retained but await independent remote replay. s3 packaging
was in progress; no complete generated task is certified by this observation.

## Verified snapshot review utility

`scripts/pull_snapshot.py` captures one manifest and verifies bounded parallel
downloads against its content hashes. Resume remains on the captured manifest.
Live use verified 901 published c02 files and 124 c22 files. The c02 manifest also
declares two omitted Pygments dependency files, which the tool now retains in the
receipt and distinguishes from complete workspace coverage. Nine focused unit
cases cover resume consistency, corruption, omissions and unsafe paths. The full
controller suite subsequently passed 189 tests, and Ruff passed on the new tool
and tests. This is transfer-integrity evidence, not task validation.

## Shared-tool version provenance and papercut prioritization

The shared `build_envs/common/omp_env.sh` bootstrap downloads OMP from a mutable
`releases/latest` URL. Local CLI inspection used 18.1.14, whereas the live
c22 continuation001 worker reported 18.2.6 at 2026-09-19 05:44:17 UTC. Consequently,
the local CLI inspection does not establish the live version's supported budget
controls. The measured 32 output-length stops remain valid transcript evidence;
no causal claim about the version change is made. Papercut6 now requests a
version/hash-pinned bootstrap and effective tool/model manifest.

The displaced lower-priority papercut is retained here: TaskCompendium commit
`dc6b501` has stale specification prose around lines 124 and 299 describing
native judging as stubbed, despite its implemented `grade_judge_attempt`, Harbor
`SemanticVerifier`, and transport test. The local recipe uses the actual native
implementation. Updating that upstream documentation remains useful, but the
active papercuts list is limited to ten entries.

## c05 independent repaired-grader audit 003 — further repair required

The repaired grader `5b8ee7b9132a9392d9551818b7b893c813f64eb2ee1227b511c0291b42a092d6`
now rejects all five independently rerun historical mutations as expected. The
reference still scores 1.0. However, three new justification-only cases globally
deny, explicitly reject a quotation of, or reverse the correct explanation;
all three still receive 1.0. Full measured outcomes are retained under
`runs/c05-grader-audit-003/`, SHA-256
`42383ce93ea026de0c619330677595d9836c1880eaf6a6ffe67b5c9b62ecc23c`.
The fresh remote sandbox recorded network blocking and was deleted. The initial
attempt using an unavailable historical snapshot failed before creation and is
retained separately as infrastructure failure, with no reward.

The next construction continuation is authorized to apply the already-admitted
structured fallback and repair task-relative private evidence packaging. All 20
unique files cited by the old acceptance receipt were independently hash-verified
under the workspace, but the receipt's paths resolve under `task/` and fail there.
No acceptance claims were changed merely to clear that path gate. The complete
audit and exact proposal checklist are bound by
`data/synthesis-c05-continuation-004/manifest.json`. No generated task is certified.

## Composite consensus probe 007 — passed

`/muchanem/cap-composite-consensus-probe-007` exercised the current hash-bound
adapter through the real Harbor/Daytona/native-judge path. Source snapshot:
`aa75975840209f65d7ef48bc8f2e7552fc78303151a32ae0cfa81fcb905708e9`.
The root independently verified all 54 files in the final pull, snapshot
`5cbf809662a14002ac55727e9df5cd86`, manifest SHA-256
`434d2ca34a7e70f7cc8c233a802b53b4d2b66848c16ef5f63a30c3071b1532b5`.
Aggregate evidence SHA-256:
`a941ed8968039a6016bff5b96da5807d3cf709c5edf4c19901f3c6879c32f34e`.

The explicit agreement case produced two `[1,1]` vectors, four actual criterion
completions, one native grading attempt, and reward 1. Two exploratory ambiguous
cases also agreed and correctly skipped the third pass. The mixed-signal case
produced initial vectors `[1,1]` and `[1,0]`, invoked a complete third vector
`[1,0]`, and resolved to `[1,0]` by per-criterion median, reward 0.5. Its receipt
records disagreement index 1, six completions and two native grading attempts.
All four grade-artifact hashes were independently verified. Every raw response
records serving revision `vllm-0.28.0-tp8-f8f644d5`.

All four candidate and four private verifier sandboxes were distinct and recorded
network blocking. Infra's subsequent direct provider lookup found all eight IDs
absent; lookup evidence SHA-256 is
`de4703cfcf782ea791e032247ab21ee5e6b4734828ad7041063edf4dd4d6a777`.
This establishes the live conditional third-pass mechanism and raw-evidence
retention. The exploratory ambiguous cases are not semantic gold labels, and
the probe does not establish model accuracy, 39-criterion task calibration,
or generated-task acceptance. Probe006 remains the separate full four-case
single-sample gate/weight/cap integration proof.

## Proposal pilot 001 — complete, needs iteration

- Input: `catalog.json` SHA-256
  `b3318b9fa7965b70b1cd1aa15513faca63b629db67bdc991a277173b51b5c0c6`;
  854 capabilities in 34 curricula. Pilot: 34 capabilities, one per curriculum,
  340 requested proposal slots.
- Job: `/muchanem/cap-proposal-pilot-001`, submitted 2026-09-19 01:38:24 UTC.
- Placement: Iris `cw-us-east-02a`; GLM-5.3 interactive tier on Orion;
  concurrency 256. Planning has only 34 independent items; generation has 340.
- Durable root:
  `s3://marin-us-east-02a/users/muchanem/capability-pipeline/runs/proposal-pilot-001`.
- Submitted source snapshot:
  `_source_snapshot/5979bf24b335e4ad19bdc05453430c1b8193971119f4f388cf253580466ece00/`.
  This records the actual submitted code; subsequent local fixes are a new recipe revision.
- Observation: all 34 planning items emitted completed events by 01:40:28 UTC;
  detailed proposal generation started. These are generation results, not semantic
  acceptance or task-runtime results.

The final controller report records **0 accepted, 228 rejected/needing repair,
112 missing, and 0 null** out of 340 requested slots, in 1,353.46 seconds. Exit 2
is the expected `needs_iteration` result, not a successful task corpus. Of the
228 retained records, 132 had an individual `accept` review, but their portfolios
still required repair; 82 records had no final review. None can be selected for
construction under the recorded acceptance contract. The attempted nine-task
selection therefore correctly emitted zero candidates and all six missing-mode
flags (`data/build-pilot-baseline.selection.json`).

Review inspection also exposed contradictory `accept` verdicts with nonempty
`required_changes`. Some listed real blueprint defects; others merely demanded
execution of already-planned build gates. The current validator rejects these
contradictions. A subsequent prompt correction explicitly distinguishes blueprint
repairs from pending builder evidence and requires an empty `required_changes`
list for acceptance. This correction was made after revision 002 launched and
must not be credited to that run.

### Observed defect and correction

Inspection of the generated plans for `c35.persistence_migration` and
`c04.infrastructure.drift-reconciliation` found an invalid restriction:
reasoning tasks were excluded from code verification because their solver has
no execution environment. The persistence plan also excluded ShellSim/code.
The solver's environment and private evaluator's runtime are separate, so these
restrictions are false. A reasoning answer can be checked with a constraint
solver; ShellSim artifacts can be checked using native private tests.

Another plan proposes a fake Terraform binary inside ShellSim. That needs a
supported shell function/script/fixture implementation and a compatibility test;
naming a native binary does not establish that the simulator can execute it.

The next prompt revision states these distinctions explicitly in the system,
planner, author and reviewer prompts, with concrete counterexamples. The reviewer
must flag invalid excluded-combination rationales, and the controller must permit
repairing the plan as well as individual proposals. The running baseline is
preserved so this correction can be measured rather than assumed effective.

The initial generation aggregate contains 228 structurally valid proposals out
of 340 slots (67.1%); review was still running when this snapshot was read. Their
environment distribution is reasoning 85, container 90, ShellSim 53; verifier
distribution is simple 54, code 114, judge 60. Median serialized proposal length
is 28,622 characters. These are substantial blueprints, not completed tasks.

The complete compact forensic pull refines this breakdown: only **79/340 (23.2%)**
passed the first structural check; 261 needed format repair, of which 149 recovered
and 112 still failed. Initial proposal responses hit the output limit 58 times;
five format-repair responses also hit it. The earlier five-event log count below
therefore understates initial truncation and should not be used as its frequency.
Across every recorded stage, 888 completed transports consumed 11,118,295 prompt
tokens and 8,775,629 completion tokens, including reported reasoning. These costs
include planning, review, semantic repair and structural repair, and yield zero
accepted portfolios. Completed-request intervals establish a peak of at least
256 simultaneous requests attributable to this run. This is not a fleet-wide
utilization measurement. Artifact hashes are in `local-pull-manifest.json`.

The sampled worker log contains 122 failure events across stages, dominated by
wrong/missing `tools` list shape (55), nonconforming source-status labels (18),
missing/invalid verifier type (7), malformed JSON, and some output truncations
(5). The denominator here is log events, not independent tasks. Repeated textual
instructions alone did not reliably enforce the schema. Keep the strong schema
gates; fix the generation contract rather than accepting malformed records.

### Structured-output protocol experiment — passed

`runs/structured-probe-001/` retries the actual failed
`c06.pipeline_recovery:2` blueprint using a full JSON schema, fixed capability/slot
identity, explicit enums/required keys, `strict:true`, and a 65,536-token budget.
It calls the same remote GLM-5.3 service; only this bounded protocol check was
orchestrated from the laptop. The response ended with `finish_reason=stop`, passed
the independent structural validator, and retained detailed content. Usage:
4,419 prompt tokens, 12,555 completion tokens including 6,661 reasoning tokens;
elapsed 86.62 seconds. This proves the proposal schema is supported by the live
service; it is not evidence that the task itself is correct.

A second bounded transport probe (`runs/schema-enforcement-probe-001/`) gave the
model a contradictory instruction to emit plain `BANANA`, while the wire schema
required an unrelated constant-valued JSON object. The returned object matched
the schema, ended normally, and used 41 completion tokens. This is stronger
protocol evidence than a naturally schema-compliant answer alone, though semantic
invariants such as DAG ordering still need their independent validators.

The authoritative inference documentation also showed that GLM thinking effort
belongs in `chat_template_kwargs.reasoning_effort`; the baseline used a top-level
field. Revision 002 fixes that field and adopts schema-constrained responses,
32K proposal/review output budgets, and a 64K structural-repair ceiling. These are
combined recipe changes, so a yield improvement cannot be attributed to the
wording change alone.

### Revision 002 — complete, needs iteration

Job `/muchanem/cap-proposal-pilot-002` uses bulk inference and a batch-priority
CoreWeave CPU worker with concurrency 256. Its first attempt failed during setup,
before inference: the new-run guard mistook the pre-uploaded `_source_snapshot/`
for prior output. The guard now permits a prefix containing only source snapshots;
the corrected job was submitted at 2026-09-19 02:03:45 UTC. This is a recorded
infrastructure failure, not proposal-quality evidence. Its durable prefix is
`s3://marin-us-east-02a/users/muchanem/capability-pipeline/runs/proposal-pilot-002`.

The final report records **20 accepted, 319 rejected/needing repair, one missing
slot and zero nulls** in 1,993.96 seconds, with exit 2 (`needs_iteration`). Two
complete portfolios passed. The final review round covered 30 capabilities,
with 198 individual accepts and 102 individual repairs; four reviews failed
structural/consistency validation. Round zero covered 26 capabilities, with 147
individual accepts and 112 repairs. These are different reviewed subsets, so the
raw counts do not estimate a paired improvement rate. Twelve review failures
across both rounds remain in the history. No task runtime was certified.

The 20 accepted proposals span reasoning 8, ShellSim 6, container 6; verifiers
are simple 3, code 13 and judge 4. Final artifact hashes are retained in the
content-addressed snapshot dated 2026-09-19 02:37:16 UTC. The separate construction
admission below uses a fresh individual review and does not alter these counts.

### Measurement plan for revision 002

Run the same 34 capability records with corrected prompts at bulk tier. This is
a paired capability comparison, not a controlled serving-throughput experiment:
the tier changes, so latency/KV must not be interpreted as prompt-efficiency gains.
Record raw generation, structural repair and semantic repair rates separately;
independently check environment/verifier exclusions, unsupported simulated tools,
source honesty, diversity, capability fit and reward validity. Count unresolved
slots and nulls in the denominator. Inspect concrete examples before interpreting
reviewer acceptance rates as quality improvements.

The first verified revision-002 snapshot contains all 34 plans and 340 slots.
Planned pairings are reasoning/code 106, container/code 72, ShellSim/code 72,
reasoning/simple 39, reasoning/judge 33, container/judge 7, ShellSim/judge 6,
ShellSim/simple 5. No container/simple slot was forced. This changes the baseline's
widespread mistaken restriction: 31 baseline plans had listed reasoning/code as
excluded, many explicitly because the solver could not execute tests. In the new
plans, the remaining code-pair exclusions mostly describe local workflow limits
and explicitly acknowledge private code grading.

There is still a plan-contract defect: several new plans put "not excluded" or
slot-specific restrictions in `excluded_combinations`, even while using that
pairing. The next prompt revision reserves that list for portfolio-wide exclusions
and moves local limits into `coverage_rationale`; this is a subsequent correction,
not part of revision 002. The five early completed proposal spot checks are too
small a sample to estimate final yield. Two ShellSim/code proposals explicitly
separate their public text tools from private Python checking and demand actual
compatibility probes; their graders and grounding remain unbuilt.

The next synthesis pilot must include actual supported examples of all three
environment levels and all three verifier categories, without forcing unsuitable
pairings. Corpus-wide synthesis waits for evidence about buildability and reward
quality; one elegant proposal does not validate the recipe.

### Revision-002 generation outcome and next correction

The completed proposal phase retained **339/340 structurally valid proposals
(99.7%)**, versus 228/340 (67.1%) in the baseline. These figures include structural
repair and precede skeptical semantic review. The sole unresolved slot is
`logic-verification.system-verification:5`. Its initial 32,000-token response used
29,823 reasoning tokens and ended at the token limit; format repair still returned
an empty `validation_plan`. Another long logic slot spent 30,659 of its 32,000
completion tokens on reasoning over roughly 856 seconds. These were active model
generations, not demonstrated infrastructure stalls.

The complete compact artifact audit is now retained at
`runs/proposal-pilot-002/full-audit-pull/audit.json`, with its hash-verified
`pull-manifest.json` (3,530 files from the terminal 02:37:16 UTC snapshot).
Raw proposal validity was **246/340 (72.4%)** before format repair, versus
79/340 (23.2%) in the baseline. Format repair recovered 93 of the other 94 slots.
Across all stages, the audit includes 883 responses, 18,467,324 prompt tokens
and 10,446,060 completion tokens. The completed-request concurrency lower bound
is 256. These counts are not evidence of final task quality or a controlled
throughput comparison across serving tiers.

Review structure remains an expensive weakness: of 56 retained base-review
results, 54 required structural repair. There were 66 review-format-repair
responses, including 12 failed attempts, consuming 5,332,580 prompt tokens and
370,941 completion tokens. These are unique cached work-item records, not a count
of independent portfolio decisions. Later prompts distinguish actionable blueprint
changes from evidence that builders are expected to produce, while still rejecting
an `accept` verdict with nonempty `required_changes`. The refinement run will
measure whether those changes actually reduce contradictions.

A repair-request snapshot identified 66 empty validation-plan lists, one empty
task brief, one empty handoff-artifact list and five truncated responses. The wire
schema had allowed empty strings/lists that the independent validator rejected.
The next schema revision sets `minLength:1` and nonempty required arrays and gives
honest nulls a separate minimal branch. A real failed-proposal regression probe
(`runs/schema-minimum-probe-001/`) then produced a valid proposed task with seven
validation experiments, using 9,091 completion tokens. The minimal-null branch
also passed a live protocol probe in `runs/schema-null-probe-001/`.

These changes were made after revision 002 launched. Neither their protocol
checks nor the improved structural yield establish semantic acceptance or runtime
quality. They motivate the next bounded revision; the running review is preserved.

### Construction questions from qualitative spot checks

Several retained blueprints give concrete, useful workflows: evidence-backed cash
application, append-only LIMS corrections, and recovery of a partially corrupted
feature pipeline. They specify meaningful negative controls rather than just
checking that output files exist. That supports trying to build them; it does not
resolve the following risks:

- The LIMS proposal assumes a fairly broad ShellSim surface (`join`, checksums,
  heredocs and quoting). Test the exact pinned bridge rather than inferring support
  from the command names.
- The feature-pipeline proposal considers learner-visible mtimes/audit logs as a
  fallback for proving unchanged partitions were not rewritten. A solver can
  manipulate those, so the adversarial pilot must test that shortcut. If the
  requirement genuinely concerns actions, collect trusted observations outside
  learner control or make the task's outcome contract explicit.
- A table-extraction proposal references an attached rendered table in a
  reasoning-only setting. The builder must prove the selected rendering/agent can
  actually receive that modality. Silently substituting an answer-revealing textual
  representation would change the capability being tested.

These checks belong in build acceptance and independent attacks, not in an
assumption that a long proposal is automatically high quality.

### Infrastructure observations

At the pinned TaskCompendium revision `dc6b501c...`, the normative specification
still describes LLM-as-judge as unimplemented, but the source implements
`judging.py::grade_judge_attempt`, connects it through Harbor's
`SemanticVerifier`, and tests the transport with a fixture endpoint. This is a
source/prose inconsistency, not evidence that the configured GLM policy works.
The native client omits an explicit token budget and GLM reasoning-template
arguments. No live native-judge probe or task calibration has passed yet; judge
tasks remain pending until a parseable verdict, variant-group calibration, and
artifact-bound Harbor controls are recorded.

Early shared-fleet samples had a near-zero median KV fraction, one node above
0.85 with waiting requests, and no preemptions. These measurements include other
tenants and are not this run's utilization. Keep timestamped engine and gateway
readings together before assigning causality or altering this run's load.

The original pilot uploader copies live files to stable object keys. Future
submissions use content-addressed snapshots and checksum-verified manifest
restoration. Do not resume pilot 001 by trusting its mutable raw prefix: first
verify/migrate the recorded artifacts. Do not restart a live job to deploy this
improvement.

## Native judge protocol probe

The bounded credentialed run `/muchanem/cap-judge-probe-001` completed with exit
0 against the exactly locked TaskCompendium `dc6b501c...` source and its pinned
Harbor and TaskTrove revisions. All three cases reached the native model path with
`exact_gate=false`: the positive paraphrase scored 1.0, while the factually wrong
answer and instruction injection each scored 0.0. The judge recorded model
`glm-5.3` and served revision `vllm-0.28.0-tp8-f8f644d5`.

The durable root is
`s3://marin-us-east-02a/users/muchanem/capability-pipeline/runs/judge-probe-001/judge-probe/`.
The retained `evidence.json` has SHA-256
`539d04f9ec16164ea149e9245d5820f0c9eafb066149f04b64463961838aeabf`,
and its archived probe script matches the reviewed source at SHA-256
`8beb1e7f139e4b437f0d10dcee9b79ac5842465f62c025826ec8a362293753eb`.
See `docs/judge_probe.md` for artifact hashes and gates. This establishes live
native-judge transport and elementary discrimination. Three obvious cases do not
estimate a false-accept rate, satisfy the 80-group calibration contract, or
validate a generated capability task.

## Frozen-seed refinement 003 — terminal, needs iteration

`/muchanem/cap-proposal-refinement-003` starts from revision 002's final 34 plans
and 339 proposals, with two bounded repair rounds and bulk concurrency 256.
It does not repeat initial planning or proposal generation, and inherits no
acceptance verdicts. Every portfolio receives a fresh review under the corrected
prompts. Missing slots can be regenerated, legacy exclusion-list contradictions
must pass the current plan validator before acceptance, and honest nulls remain
valid outcomes. The source files and their byte hashes are frozen in
`data/proposal-refinement-seed-002/seed-provenance.json`.

The submitted source snapshot is
`7cbe77f1d66f03e52fcf71813f18bcd4aebb5029055b10487ba206f49b2e98ef`;
durable output is
`s3://marin-us-east-02a/users/muchanem/capability-pipeline/runs/proposal-refinement-003`.
Preflight found 29 bulk serving workers. Actual width is 34 during review and
up to 340 candidate repairs; there is no artificial per-capability serialization.
The terminal report retained **100 accepted proposals in ten complete portfolios**,
240 rejected/needing repair, and no missing or null slots. Accepted environments
are reasoning 46, container 27 and ShellSim 27; verifiers are simple 14, code 72
and judge 14. At the terminal audit, all 100 records passed the admitted-input loader;
the later arithmetic audit revoked two unchanged proposals (see below).
Elapsed time was 2,656.77 seconds. These are proposal decisions, not demonstrated
build-quality gains.

The full compact terminal audit is in
`runs/proposal-refinement-003/final-pull/audit.json`: 25,782,765 prompt tokens and
7,291,608 completion tokens, with observed completed-request concurrency at least
256. Of 87 unique review response records, 77 produced valid results, including
31 structurally repaired results; 46 were valid without structural repair. The
report retains 17 historical review failures where portfolio acceptance contradicted
nonempty issues, plus a truncated c17 slot-2 repair. A previously valid proposal
can remain available when its attempted repair fails; zero missing slots does not
mean every repair succeeded.

## Frozen review-only passes 004/005 — terminal, needs iteration

`/muchanem/cap-proposal-review-004` freshly reviews the terminal 003 plans and 340
proposals with zero semantic repair rounds. No plan/proposal regeneration or
acceptance inheritance occurs. The prompt now explicitly reserves portfolio issues
for unresolved blocking defects and requires empty issue/missing lists for portfolio
acceptance. It also includes the user's authorized composite-verifier direction.
This measures the combined current prompt, not an isolated causal effect of one
sentence change.

Inputs and hashes are in `data/proposal-review-seed-003/seed-manifest.json`.
The source snapshot is
`15b2ac9b5ef5e424a780cd589e2bffb5c5806f67e7f13bb6f4a067c2563ea557`.
Preflight found 33 bulk workers; configured concurrency is 256 with 34 independent
portfolio reviews ready. Run 004 failed before inference after 11.81 seconds:
the worker incorrectly required `inputs/source/accepted.json` for a proposal seed.
That seed is intentionally the four proposal-stage files, not an accepted-task
manifest. There are no review results from 004.

The worker now validates source files by stage: proposal seeds require
`input_pilot.json`, `plans.json`, `proposals.json`, and `report.json`; admission and
synthesis inputs require `accepted.json`. Three targeted preflight regressions,
shell syntax validation and Ruff passed. Fresh run
`/muchanem/cap-proposal-review-005` uses unchanged seed content, zero repair rounds
and bulk concurrency 256. Its frozen source SHA-256 is
`4f9c97224c44cee38c1528129d2824c5d3e2d35774128b38e28cefd562b49c1c`.
Run 005 finished with 200 accepted proposals in 20 complete portfolios, 140
rejected/needing repair, no nulls and no missing slots, in 425.43 seconds. Accepted
environments were reasoning 103, container 49 and ShellSim 48; verifiers were
simple 25, code 147 and judge 28. All 34 portfolio reviews completed without a
review-stage failure. Fourteen portfolios still require repair, so controller
exit 2 records `needs_iteration`, not a failed inference transport.

The report retains the historical `logic-verification.system-verification:5`
validation-plan failure despite having a current proposal in every slot. The
terminal audit verified that its accepted replacement has seven validation-plan
items and exactly matches the frozen seed. This is an inherited generation failure,
not a current missing or invalid slot.
The subsequent reporting fix separates inherited failures from current-run
failures while retaining prior errors as context for a missing-slot repair.
Eighteen seed/controller regressions pass; historical run 005 is not rewritten.
The 100-to-200 acceptance change is a fresh review result on unchanged proposals,
under the combined current prompt; it is not proof of improved underlying task
quality. This run's source and historical report predate the subsequent
revocations, which apply separately before construction. The terminal compact pull and every retained artifact hash
were independently verified in `runs/proposal-review-005/terminal-compact/`.
`comparison-audit.json` proves all 200 accepted proposal objects are unchanged
from the seed: 80 retained their prior acceptance, 120 gained acceptance, and 20
prior accepts were not accepted by this pass. Those 20 belong to the c30 and c33
portfolios: 14 retain individual accept verdicts, while six require repair. The
portfolio gate excludes all ten slots when a portfolio needs repair; these are
not 20 independent proposal rejections. The comparison does not control review
prompts/context and cannot establish a causal quality improvement. Subsequent
audits revoked eight of the 200 exact hashes: c12 and c18 for arithmetic premise
failures, plus c08, c19, c21, c23, c32 and logic verification for direct internal
contradictions. A singleton check of every accepted entry through the normal
fail-closed loader found 192 loadable, eight revoked and no other failures. The
historical run report remains unchanged; current status is recorded separately in
`docs/audits/proposal_review_005_revocation_status_20260918_r2.json`. None is runtime
certified by this review.

## Construction admission 001

Pilot 002's first review round produced 26 valid portfolio reviews, all requiring
repair, with 147 individual accepts and 112 individual repairs; eight portfolio
reviews were unavailable in that round. To test construction without declaring
those portfolios complete, a separate review examined nine promising candidates
against their exact capability records and all portfolio feedback. The selection
covered nine capabilities, seven environment/verifier pairings, and all three
environments and verifier types. Inputs preserve the source proposal, plan,
review and pilot hashes; pending candidates cannot directly enter synthesis.

`/muchanem/cap-construction-admission-001` completed nine live reviews in 170.0
seconds: **eight accepted, one rejected, zero nulls and no transport failures**.
The accepted set contains three container, three reasoning and two ShellSim
candidates; four use executable verifiers, three judges and one simple verifier.
The controller exited 2 with `needs_iteration` because one candidate remained
rejected. This is not an infrastructure failure.

The source snapshot is
`6a6f2b669ac6f2727523e49d384e7b41ae0440483d7c74c5ff969ea52e1accf7`;
the durable prefix is
`s3://marin-us-east-02a/users/muchanem/capability-pipeline/runs/construction-admission-001`.
Hash-verified results are retained locally under the matching `runs/` directory.
Acceptance is scoped to individual construction, with fresh review history and
`portfolio_certified=false`; no original portfolio result was rewritten and
runtime-validated task count remains zero.

## Initial synthesis evidence — incomplete builds

`/muchanem/cap-synthesis-pilot-001` is building eight admitted candidates. The
checksum-verified compact snapshot under `runs/synthesis-pilot-001/review-pull/`
(2026-09-19 02:56:21 UTC) contains 153 text artifacts, including real builder
transcripts, handoffs and probe logs. It is a selected evidence pull, not a
complete task export.

The spreadsheet builder's recorded LibreOffice 7.3.7.2 probes demonstrated input
mutation recalculation, the intended bounded-range/append behavior, and ten
deterministic repetitions. This supports its engine premise, not the full workbook
or grader. The browser-lifecycle builder corrected its drafted reference after
a Chromium probe disagreed with the proposal's ancestor-hidden focus assumption.
Its amended browser model, prompt and reference must agree and receive independent
review; the original hand calculation is not ground truth.

An exact-source verifier audit found that all three judge candidates need
unsupported conjunctive machine-check/rubric composition. TaskCompendium declares
`ALL_REQUIRED_STEPS`, but its pinned Harbor lowering rejects that multistep policy.
`MEAN` and `FINAL` do not preserve the admitted critical-gate semantics. The user
subsequently authorized extending TaskCompendium to support the intended composition
(2026-09-18 PDT), so native-only redesign is no longer required. Preserve the original
reward semantics, implement and pin the extension, and require actual composed-runtime
validation before final acceptance. Task-specific calibration cannot certify a
misrepresented reward function; see `docs/build_acceptance_001.md` and papercut 9.

## Completion evidence still required

The first composed-verifier probe, `/muchanem/cap-composite-probe-001`, failed
before grading while resolving the staged TaskCompendium extension. It establishes
no semantic behavior. Corrected `/muchanem/cap-composite-probe-002` was submitted
after `OfficialToolchain.resolve` succeeded against its exact staged files,
including the extension lock and patch. Its durable source snapshot is
`28f9072e472500ad4573c673825ba75ea87ed2fa5f7c2dfc6adf96b0270cca42`.
Run 002 also failed during source resolution, before grading. A raw archive audit
then identified 240 macOS AppleDouble metadata members in the 480-member archive.
Native macOS tar extraction concealed those members during the local smoke test;
standard-library extraction retained them and reproduced strict source-file-set
rejection. The source lock contains 81 required files. Evidence is retained in
`docs/audits/source_archive_metadata_001.json`.
The repair is a metadata-free archive built from the locked files, preserving
strict source validation. Run 003 retained the contaminated archive and failed
before grading. Run 004 used the clean 81-member archive
`2e3735fc9a5dece58d1e677a6b780098eb6804c0474aa44d4765aafb04154dfd`
and passed source/overlay bootstrap, then exposed a runner integration gap:
`run_trial` decoded the guarded export through the ordinary parser and rejected
`required_extension`. No candidate or judge trial ran. The patched runner now
loads the preserved specification only after validating the required extension,
configured verifier and implementation hashes; ordinary consumers continue
rejecting unsupported exports. An exact staged runner smoke verified both paths.

Run 005 (`/muchanem/cap-composite-probe-005`) passed source/overlay bootstrap and
reached real Daytona execution, then failed after about 57 seconds during the
`machine-gate-fail` case with `DaytonaRateLimitError` / `ThrottlerException: Too
Many Requests`. Its source snapshot is
`9a13c1cb19355f9c361a61b137a9cfec10902bb2a5e9611a97ea7705a55f9cf3`.
This is an observed provider throttle, not a grading verdict or a completed
composition proof. Independent local hash inspection verified all 35 retained
artifacts in `runs/composite-probe-005/terminal-pull/` against snapshot
`cc7f53eaa6064c4ea1495add09cba23e`. The first `correct` trial did complete:
two fresh private network-blocked check sandboxes produced gate 1 and penalty 0,
and GLM-5.3 scored all three criteria 1, including the criterion requiring the
private machine result in judge context. The composed reward was 1.0 (artifact
SHA-256 `ac69c1afec26b56f6d61de1dc83aa72fcaa6e88c65b4b3b6534dd16f50896a79`).
The second trial retains `infra_error` with null reward; the remaining negative
and section-cap cases have no results. Preserve these artifacts and the failed attempt;
bounded provider retries must be tested before a fresh probe. Live composed
protocol acceptance remains pending; unit tests do not substitute for it.

Run 006 (`/muchanem/cap-composite-probe-006`) **passed the four-case composed
protocol**, exit 0, in 2 minutes 11.57 seconds. Source snapshot:
`f5b75d2e55f108df0f5be2def18a5538ee4725fc0085b609aadc73a59676a20f`.
Independent hash inspection verified all 53 files in
`runs/composite-probe-006/terminal-pull/` against snapshot
`991f929957dd42d3b737f4900a046606`, including every reported trial result.
The aggregate evidence SHA-256 is
`952cd0201cac1440d3c28c7dc8d1d7a627993a9cab95007dcb32d72cfd41cc0e`.

| Case | Actual result | Distinguishing evidence |
| --- | --- | --- |
| Correct | graded / 1.0 | Two private checks, three model criteria including private machine context |
| Machine gate fails | graded / 0.0 | Failed compliance gate; judge skipped |
| Critical judge criterion fails | graded / 0.0 | Criterion scores `[0,1,1]`; critical index 0 enforced |
| Conditional section cap and penalty | graded / 0.25 | Raw `[1,1,1]` becomes `[1,0,1]`; `(2 - 1) / 4 = 0.25` |

Four distinct candidate and eight distinct private verifier sandboxes were
network-blocked. Provisioning succeeded on each first attempt, so the bounded
429 retry branch has unit and real-SDK-shape tests but no live retry success
claim from this run. A subsequent provider lookup under the run credential found
all 16 recorded sandbox IDs across runs 005/006 absent (`DaytonaNotFoundError`);
no active leftover was found or required deletion. The lookup receipt is
`runs/daytona-cleanup-composite-005-006/provider-cleanup.json`, SHA-256
`fe7b69d880d3a9c348959dcf3163547593ff84c430f4f5347bbb3f5bcc02c7f3`.
These are integration controls, not an 80-group generated-task judge calibration
or a final generated-task quality receipt.

Joint repair continuation 003 (`/muchanem/cap-synthesis-continuation-003`) was
submitted from source snapshot
`4230d9268395e455ec7e9d64ad41bd937b14576cf2fd6282195b114a03ba1c5b`.
Its restore seed contains all 241 c05 item files and all 654 c17 item files plus
lineage records, 898 members in total, snapshot
`b612340883b7461288f004a58f286df1`. The worker restored the seed, built the pinned
ShellSim bridge and entered synthesis at 04:43:53 UTC. Later durable snapshots
contain fresh repair files. Restored item statuses still describe historical
failures while those repairs run; they are not new terminal failures. The
original synthesis001 job continues independently.

The 04:45 UTC full c02/c30 pull has 2,115 verified artifacts. Static audits
`docs/audits/c02_construction_001.md` and
`docs/audits/c30_construction_s5_001.md` identify work-in-progress exports that
omit admitted machine gates and reward semantics. Neither final builder session
has a completion receipt. C02 additionally has an admitted 39-point rubric with
an unexplained 20-of-24 threshold; it needs hash-changing clarification and fresh
readmission before that scoring contract can be frozen. A transport-only trial
that substitutes ShellSim for its container does not validate task fidelity.

C02 clarification admission001 completed with a fresh accepting review and new
proposal `18bae810…`. The root verified all 26 retained files, then found a
semantic defect missed by both the model review and the first independent audit:
several binary threshold predicates were stricter than the original partial-credit
anchors. The complete finding is
[`c02_anchor_thresholds_002.md`](audits/c02_anchor_thresholds_002.md). The new hash
is revoked too; neither the historical receipts nor the first audit were rewritten.
Repair002 (`/muchanem/cap-c02-anchor-repair-002`) uses unchanged rejected proposal
bytes with that exact audit and the current revocation-aware admission controller.
Its source snapshot is
`cbbbabc1fc5c6fe45660208943a890abba2b377061253e7f958c5e1a6f62dbfc`.
The requested repair includes a 52-row semantic anchor-boundary table and correct
78/117 model-completion accounting, not only algebra over supplied bits.
Repair002 ended with zero accepts and one rejection in 60.16 seconds; it did not
attempt a repair because the review used `reject`. The same review explicitly
called the underlying task strong and salvageable and supplied six concrete
changes. All 16 terminal artifacts were independently verified, snapshot
`e909746c6ae441e5aa5b96411841318d`, pull-manifest SHA-256
`6ed783d18b9e7228aa20b3dda08ef392a408f1cc38fa221f07dd40b49291525c`.
An explicit `--repair-rejected` mode now permits a bounded repair attempt without
discarding that rejection: a changed hash and fresh passing review remain
mandatory. Default admission still stops on rejection. Controller regression
coverage includes unchanged repairs, final rejections, abstention, and revocations.
The next admission attempt, `/muchanem/cap-c02-anchor-repair-003`, uses that mode
with one repair round, source snapshot
`c39cfccb5d76010d1941c6d347f7f8f487e5acd299590de4b2d3760da988339d`.
A later independent boundary audit also corrects provenance: the original
proposal fully specified all intermediate levels for nine groups, while four
groups received intermediate anchors during clarification. Their anchor/threshold
contradictions remain real, but those levels must be reviewed as interpolations.
The 52-row audit is
[`c02_semantic_boundary_checklist_003.md`](audits/c02_semantic_boundary_checklist_003.md),
SHA-256 `8fb3cfd79f615baf2b0ffb24886287e4e5a10930fc0eff40444fed38c589c112`.
It does not retroactively alter the running repair's frozen input.
Repair003 subsequently accepted new proposal
`03c05bb576832e76c656e2ebc74b537af8835a2c345a8ef69cf0e366918d7b4d`
after preserving the initial rejection and completing repair plus fresh review.
Root verified all 26 terminal artifacts, manifest SHA-256
`79e9f50449cc5488563caee3a3e02c6239171cc28b53545e5e794980e178632f`.
The [root admission audit](audits/c02_anchor_repair_admission_003.md) confirms the
threshold repair and records remaining construction conditions: intermediate-
anchor provenance, the pending actual 52-row table, source-gap tie rules, and
typed count aliases. A new exact-hash 20-check build checklist preserves the
complete task, gates, calibration and runtime requirements. Both old hashes stay
revoked. No task is yet certified under the new one.

C07 corrected construction review002 completed with proposal
`7ab4ffa1a034c3c8b63ec3865a2249a22ba39ab3250d02f8e5c74e20e1cb5a66`
after one repair and fresh review. The root verified all 26 final artifacts in
snapshot `0b0e396c3554455fbd94ff9575fccc68`; pull-manifest SHA-256 is
`2edeb0d160aad1590e44d9fbc853ab86aca325413c752b9991af4356f32ab161`.
The exact-hash construction checklist requires formula and boundary cross-checks,
variant-to-public-packet binding, complete parser controls, and empirical difficulty
review. Historical review001 cannot authorize construction under its superseded
spending-function context. Neither admission is a generated-task runtime result.
C07 construction001 launched as `/muchanem/cap-synthesis-c07-001`, with one
dependency-ordered pilot candidate and four planned sessions. Input SHA-256 is
`d2c0122a52a405fbffd815be60ae8d26f957d5a3d982717b4febad319018251c`;
the source snapshot is
`1584e1495cac38f8df2fccb679cd8f60dbf8c3eee3cf56eab8ca04d5a8fa952c`.
The exact checklist SHA-256 is
`874cf46cd629ef9a1634080644342d730cc116bde694c62a5840794a2bfb82c1`.
This launch establishes no numerical, runtime, difficulty, or quality result.

The 03:13:56 UTC synthesis checkpoint contains 240 verified artifacts. The
browser task has completed three construction sessions and the protocol-trace
task two; these are builder receipts, not final quality acceptance. Static
inspection found a browser reproducibility concern: `harness.js` closes the
dialog and observes focus in separate awaited browser calls, while the public
model assumes no intervening rendering update. Require an atomic close/observe
operation or an explicit rendering boundary with a matching reference before
acceptance. Two matching executions do not rule out this timing race.

The browser task subsequently completed all four sessions and reached controller
validation, which rejected a control-category label. The full audit in
`docs/audits/c17_construction_001.md` additionally found a solver-environment change
from admitted no-tool reasoning to ShellSim/file submission, a private grader
image tied to a prior Docker daemon, and the close/observe timing race above.
Category relabeling alone would not make the task acceptable.

The later c32 full pull establishes completion of all five builder sessions but
failure of the first actual Harbor trial at private sandbox creation:
`DaytonaRateLimitError` / `ThrottlerException: Too Many Requests`. Its recorded
TaskCompendium outcome is `infra_error` with null reward, not a geometry verdict.
Root verified all 253 files in snapshot `a65fb96f5e3b48b5afb44ba597928869` and
all 66 overlapping files against the earlier compact snapshot. The trial's
`control-...reference` name was misleading: its configuration and transcript
show a fresh GLM solver, not an authored-reference replay. New source distinguishes
those roles and includes bounded provisioning retries.
The [completed-build audit](audits/c32_construction_s5_002.md) separately identifies
generated evaluator exceptions being converted to graded zero, and requires
typed candidate-error versus verifier-error outcomes, actual adapter fault
injections, and reproducible image bindings before acceptance. Builder-reported
reference and control success does not clear those defects.

`/muchanem/cap-synthesis-c17-continuation-002` attempted an isolated repair from the
hash-verified 650-member saved item, preserving the original run and all sibling
builders. The approved source archive is
`73b90340f7df39b909e172836ba545983e5388d29b897346fa4625e41f320f26`;
the uploaded source snapshot is
`3a4c2098c4dcb27b6ac85e9d61540b85a9f1db678ac77d7881536bb8730b0266`.
It failed before the repair agent ran: submission staging omitted
`vendor/task_spec/composite_extension.lock.json`, although that file existed in
the approved archive. The controller also reported no usable exact toolchain.
This is an infrastructure failure, not a failed semantic repair or new task result.
A fresh continuation must first verify the full staged extension, source and
contract, then address the complete audit and rerun runtime and quality gates.

A subsequent independent sandbox audit of the protocol-trace s3 grader confirmed
two reward bugs: a negated justification retaining its required keywords and an
incorrect row-specific ACK exclusion reason each received full credit (1.0), as
did the untouched reference answer. Exact input bytes, hashes, outcomes and
provider/cleanup evidence are retained in `runs/c05-grader-audit-001/`; see
`docs/audits/c05_construction_001.md`. The admitted C7 fallback is available, while
C1 needs a reason check specific to the row. These are task-quality defects to
repair, not shared-infrastructure papercuts. Preserve valid numeric partial credit.

The second remote audit, `runs/c05-grader-audit-002/`, confirmed that the same
grader gives full credit (1.0) to missing required `duplicate`, empty interpretations
and contradictory evidence interpretations, as well as the unchanged reference.
It also accepts numerically equivalent JSON numbers where the public schema demands
decimal strings; that formatting mismatch can be resolved by consistent public
normalization. The empty and contradictory conclusions are semantic failures.
Outcome SHA-256 is
`dd4ebf58af0434377fed10a008107377f502f6716216da6797da64892399022a`;
the fresh network-blocked sandbox was deleted and its provider/cleanup records
are retained. See `docs/audits/c05_c30_construction_001.md` for the bound source
and additional c30 fixture/rubric findings.

An audit of refinement 003's 14 accepted simple-verifier proposals found additional
proposal defects before construction: the servo task's claimed least-squares
reference disagrees with two direct fits; the payment-idempotency task labels
`4000000000000001` Luhn-valid although its checksum sum is 9. Applying its own
validation-first rule changes two request classifications and the expected totals.
The payment proposal is held for proposal repair and independent rereview.
Both exact proposal hashes are now in `data/revocations.json`, backed by
`docs/audits/simple_pilot_arithmetic_003.json`. The original review outputs remain
historical evidence; a fresh build cannot consume either revoked version.
The sequential-stopping candidate remains unlaunched, with an explicit distinction
between classic O'Brien–Fleming and Lan–DeMets spending configurations in its
two-source construction gate. See `docs/audits/simple_pilot_additions_003.md`.
These findings show why portfolio acceptance is insufficient evidence of a correct
oracle, even for nominally simple verification.

Runtime probe 011 (no-tool) passed its five fixed controls but **did not pass the
independent attack gate**. The boundary agent hit its 32,768-token output limit;
the raw Harbor result records `Incomplete GLM solver output: length`. A missing
transcript then masked that cause in the wrapper. Its retained attestation has
`independent_attack_executed: false`, despite job exit 0. The wrapper now binds
and checks the Harbor result before opening downstream artifacts, classifying
this as `model_output_truncated`. A regression covers absent transcript/grading
files. Probe 012 (ShellSim) did pass all five controls and all three attacks.
Neither protocol probe establishes acceptance of a generated task.

Probe 013 subsequently established executable grading through the remote private
Daytona verifier: the blind solver received 1, four authored controls received 0,
and all three independent attacks received 0. Its source snapshot is
`d4145d1ca9ecd79cbc4f3fb5dedef79e2b531879faa4c2abc95217c037fb0adc`;
raw evidence is in `runs/runtime-probe-013-executable/evidence-pull/`. It predates
the separate authored-oracle check and therefore does not certify that revised
protocol. Probe 014 subsequently **passed the revised executable protocol**:
separate authored oracle and blind solver each scored 1, all four fixed rejection
controls scored 0, and all three independent attacks scored 0. Its raw evidence
is in `runs/runtime-probe-014-executable/evidence-pull/`; oracle, solver and
adversary artifact hashes were independently verified against the attestation.
The boundary attacker spent about 11.5 minutes exploring the public shell before
its submission was graded 0. This was productive execution, not a stalled job.
Source snapshot: `4f4a1f5b5ec8f78f6a10e58e43a22332a735511532d4235982a31fa2a6b200a8`.

Probe 015 is a complete revised-protocol **no-tool** pass. It separately executes
the authored oracle (1) and blind solver (1); empty output remains
`extraction_error` with null reward, three authored wrong answers receive 0, and
the three independent attacks each receive 0. The oracle, solver and adversary
artifact hashes were independently checked against the retained attestation in
`runs/runtime-probe-015-none/evidence-pull/daytona-probe/`. This supersedes 011 as
protocol proof without erasing its truncation failure. Its source snapshot is
`b36c778980103cd390c7684d924735993a83e8ab6f224d4fd5b241487bf80e53`.

Completed proposal reports, generated-task synthesis results, Daytona/ShellSim
lifecycles, independent solver traces, task-specific adversarial controls, full
judge calibration, and final publication audit are not established by the
observations above. The bounded no-tool and native-judge Harbor probes establish
only their stated infrastructure paths. Fill this ledger from completed runtime
evidence as the experiments progress.

### Runtime infrastructure launch ledger

The bounded Daytona runtime probe uses an exact `dc6b501c...` TaskCompendium
archive, a credential-free source snapshot, and credentials only through the
worker environment. These setup attempts are not verifier results.

Probe 008 subsequently passed its four fixed controls in separate network-blocked
Daytona sandboxes: independent solver 1.0, empty submission extraction_error/null,
plausible wrong 0.0 and instruction injection 0.0. Its independent boundary attack
then found a real contract mismatch: the public task requested an exact uppercase
token, but the default exact verifier rewarded a deliberately lowercased token
with 1.0. The raw transcript and verdict are retained in
`runs/runtime-probe-008/daytona-probe/independent-adversary.json` and its linked
artifacts. The adversary gate correctly remains pending; fixed-control success
must not be reported as full runtime validity. The next probe must explicitly
configure case sensitivity and retain the case-flip as a regression control.

- `/muchanem/cap-runtime-probe-001` (`runs/runtime-probe-001`) failed before a
  provider call with `agentic stage requires staged Daytona tools`. The local
  staged `daytona-tools/dt.sh` existed and was executable, but the federated
  worker did not satisfy the `-x` check. The worker now checks that the helper
  file exists and runs `chmod +x` inside the worker before synthesis. Runtime
  probes use the actual imported `dt.py` at the stage root instead.
- `/muchanem/cap-runtime-probe-002` (`runs/runtime-probe-002`) reached the
  pinned Cargo build and failed before Harbor/Daytona execution because the
  worker expected `release/shellsim-bridge`. Cargo metadata for the exact
  source identifies the binary as `taskcompendium-shellsim`; the worker now
  uses `release/taskcompendium-shellsim`. This is also the path used by future
  synthesis bootstrap.
- `/muchanem/cap-runtime-probe-003` (`runs/runtime-probe-003`) is the first
  attempt with both staging/path corrections. Its submitted credential-free
  source snapshot is
  `07757539b2e81767117ba3886caf17220d3279adc491956ce0faaf0c7a938e40`.
  Record its raw Harbor/Daytona artifacts and evidence before interpreting a
  runtime outcome.

### C22 independent fixture replay 002: reproducibility and semantic control

A network-blocked `daytona-small` sandbox regenerated the 30-file bundle three
times; each manifest and file hash matched the builder freeze, and each validator
run passed. Flipping the sole stale W-881 semaphore-holder flag, then recomputing
the file and aggregate manifest hashes, failed solely with `stale W-881 holder
missing from semaphores.json`. This demonstrates a semantic validator check
independent of checksum integrity. Python was 3.14.6; generated code ran remotely.
The delete call returned successfully; the immediate lookup still found the
sandbox, and a later lookup confirmed sandbox-specific HTTP 404. Raw evidence and
hashes are bound by `docs/audits/c22_fixture_replay_002.json`. Replay 001's 429
failure is preserved. This is fixture-stage proof, not generated-task acceptance.

### Admission changed-hash guard boundary checks

The opt-in clarification guard now rejects an unchanged accepting proposal even
with zero repair budget or after a repair returns to the original proposal. Raw
accepting reviews remain in history with a distinct controller block. Repair
prompts explicitly require substantive clarification and a fresh review. C21's
already-staged budget-one run remains unchanged; the new edge-case fix applies
to future source snapshots. Controller tests run locally as a small metadata-only
exception: 198 tests pass, with 16 admission tests and clean targeted Ruff.

### C05 independent structured-fallback replay 004

The exact current grader (`8330e1c23fe996b2415b9b235a75716d4af3880e271b59004dcfa6d1473745ba`)
ran in a fresh network-blocked `daytona-small` sandbox. The reference scored 1.0.
The cross-row exclusion control lost C1 alone (6/7), and the contradictory structured
interpretation lost reweighted C6 alone (5/7). Missing duplicate flags and empty
interpretations remained schema-invalid with null direct-grader scores; the static
wrapper maps these candidate errors to zero, pending full runtime verification.
The previously measured prose-only counterexamples, plus empty and omitted
justification, all scored 1.0 under the admitted optional/unscored prose contract.
This is intended invariance after C7 removal, not evidence that the prose is true.
The sandbox deletion was acknowledged and a later sandbox-specific 404 confirmed
absence. Audit: `docs/audits/c05_structured_fallback_004.json`, SHA-256
`57cd9c7144590927352759329a571718fb66be4a01ba5da49990bcbebd89e15c`.
The earlier measured false positives remain preserved. Full task acceptance remains
pending after private evidence packaging and corrected current receipt claims.

### C32 repaired-grader regression 005 launched

The original balanced-deletion-duplication case is being replayed against repaired
grader `e81b8b8bdf57606e1f352d41c65a161ef44c0666d2032940ccc1a08b34947ed9`
through the exact embedded TaskCompendium reward path. The seven input hashes,
unchanged private image, existing snapshot, and expected rejection are bound by
`docs/audits/c32_repaired_probe_contract_005.json`. The reference and fixture bytes
match probe 004. The harness preserves partial baseline evidence on later failure
and distinguishes a graded rejection from ungraded/inconsistent output. Exit 1
from the direct per-check evaluator is a valid semantic rejection, not a provider
failure. Source archive: `c9cc3ec7da2850c7860ac8603d8695ae4ae260ae8da795da6aac1fa5ccec4fde`;
job `/muchanem/cap-c32-semantic-probe-005`. No result is claimed at launch.

C32 regression 005 completed: the valid baseline scored 1.0, while the identical
balanced deletion/duplication case now scores 0.0. C8 identifies the missing clean
conduit and doubled source length; C9 independently identifies missing live
coverage and overlapping duplicate parts. All other criteria pass. The global
length residual remains exactly zero, proving the repaired checks address the
prior blind spot. All three sandboxes were deleted and independently confirmed
absent by provider-specific 404s. Full runtime/semantic task acceptance remains
pending. Evidence: `docs/audits/c32_semantic_regression_005.json`.

The rebuilt mutation artifact has a different whole-file SHA from probe 004.
A read-only SQLite row comparison found exactly one changed value: the conduits
entry's `gpkg_contents.last_change` timestamp. All other rows, including geometry
blob hashes, are identical. This is the same semantic case, not a byte-identical
GeoPackage; the evidence is
`runs/c32-semantic-probe-005/candidate-byte-comparison.json`.

C21 clarification admission 001 completed a proposal repair followed by a fresh
accepting review. The original hash `cbb84ed8...` changed to `93f2a869...`; all 26
terminal artifacts were pulled and hash-verified. Only these controller identity
checks are recorded in `docs/audits/c21_clarification_admission_transport_001.json`.
A delegated independent review was stopped by automatic review as possible
cybersecurity content and did not complete; no follow-on construction was launched.

### Papercut ranking update: image publishing blocks portable tasks

Promoted the supported image-publishing/digest-resolution path to papercut 6.
Builder CLI pinning is consolidated with inference/tool provenance in papercut 7.
The replaced entry is retained verbatim below for history.

## 6. Allow a pinned builder CLI release and retain its executable identity

**Impact:** Local diagnosis and remote construction can use different CLI behavior,
and a later run can change tools without changing the pipeline source snapshot.

**Evidence:** `build_envs/common/omp_env.sh::omp_bootstrap_tools` downloads from
`can1357/oh-my-pi/releases/latest/download/` when OMP is missing. Local inspection
used OMP 18.1.14, while c22 continuation001's worker log at 05:44:17 UTC records
`omp/18.2.6`. This demonstrates a version difference, not that the newer release
caused the recorded output-limit failures.

**Useful addition:** Accept an explicit OMP release and expected executable hash;
record both in each run's tool manifest along with the effective model limits.
Use the recorded release for resumed experiments unless a tool upgrade is an
explicit, separately recorded change. This replaces the lower-impact stale
TaskCompendium judging documentation entry; that discrepancy remains documented
in the recipe and experiment ledger.


### C17 snapshot replacement and bounded provider-health recovery

An unfiltered provider inventory returned 3,194 sandboxes with no references to
the pipeline's superseded c17 verifier cache. Its exact creator receipt and old
recipe were retained; current c17 uses a different, registry-pinned Python image.
Only that obsolete snapshot was deleted, and provider absence plus quota39 was
verified before rebuilding the exact current verifier recipe. The replacement
`cap-verifier-908214b9e11806813050` is ACTIVE with provider ID
`43d4385a-ba89-49ac-94ce-0b9b9cd2372a`. Deletion/build evidence is retained under
`runs/daytona-quota-cleanup-001/`; no unrelated or active snapshot was deleted.

Iris health002 found that snapshot but exhausted four429 creation attempts with
no sandbox created. The independent local SDK control-plane health003 also saw
three429 responses, then created on attempt4, observed network blocking, deleted
the sandbox, and confirmed absence after5seconds. This does not establish an
egress-specific limit: the probes ran at different times and both saw429s.
`docs/audits/c17_health_recovery_003.json` binds the passing receipt. No generated
code executed locally; small controller tests and metadata hashing are the local
exception. A single c17 infrastructure revalidation is authorized against the
unchanged task after authoritative terminal-job and source checks.

### Correct live-state interpretation and image portability

The c05/c32 continuation004 item failure files were last-completed outcomes, not
terminal job evidence. Iris and nested OMP events in the0644pulls establish live
repairs: c05 attempt2 recorded a real image-pull failure at06:43:17; c32 attempt1
recorded tool activity at06:42:06. No duplicate continuation or cancellation was
launched. The independently prepared c05seed005 remains unused and stale after
the live builder independently fixed its33-file evidence packaging.

Future controllers write an atomic active-operation receipt around each repair,
with timestamps, prior-status hash, transcript path and completed/error outcome,
while preserving the previous status. This is activity metadata, not task success.

Image identity audits found that c32's candidate label was a Daytona provider-ref
suffix and its private-verifier label was a Dockerfile hash; neither is proof of
an OCI image digest. The c05 example image is explicitly non-portable and its
live pull failed. See `docs/audits/image_identity_contract_001.md` and
`docs/audits/c05_private_image_portability_005.json`. Builder instructions and
semantic-review requirements now demand actual registry/image-inspection evidence
and a supported cold reconstruction path, preserving the distinction between
provider-specific execution and portable OCI publication. An authenticated
registry/repository and approved credential mechanism have been requested.

### Existing publisher discovery and c05 terminal infrastructure incident

The maintained envgen materialisation report and raw `push_registry.json` already
contain three successful OCI publications and registry-backed cold-pull validation.
Papercut 6 now asks for durable credentials and pipeline integration, rather than
creation of a publisher from scratch. Current service availability and credential
delivery remain unverified. `docs/audits/materialisation_integration_001.md` records
reuse requirements: exact OCI digests, image-config fidelity, separate private
verifier state, observed resources, and fully blocked runtime trials. The rootfs
transport's object-store allow-list and union filesystem semantics do not satisfy
our existing runtime contract without further evidence.

c05 continuation004 is now authoritatively terminal (06:54:56 UTC, exit 2), with
995 hash-verified files in `terminal-pull-0701`, immutable snapshot
`e90d093faba54af29588f52be08f31e0`. Its final runtime attempt failed before trial
execution on `ModuleNotFoundError: msgspec`. Iris exports
`UV_PROJECT_ENVIRONMENT=VENV_PATH`; the controller previously inherited it while
its uploader independently synchronized Marin dependencies. The exact concurrent
interleaving was not retained. TaskCompendium commands now unset `VIRTUAL_ENV`
and select separate explicit core/runtime environments. A child-process test
verifies both environments remain separate from the inherited worker environment.

The same builder repair reported deleting 23 shared `harbor__*` snapshots selected
by age without ownership/reference evidence. This was an unsupported action, not
a successful repair. It also prewarmed a controller cache name with a different
recipe from the one used to derive that name. The original task's non-portable
image field remains unchanged; the prewarm does not establish its identity.
`docs/audits/c05_terminal_infrastructure_006.json` retains all reported names and
hash-bound evidence. Impact on other owners is unknown. No bulk reconstruction or
additional cleanup was attempted. Build and repair prompts now explicitly forbid
shared-cache age-based deletion and cache-name substitution; cache recipe checks
are being added. No c05 continuation has been launched from this terminal state.

### Authorized registry recovery and adaptive-test experiment

The user authorized building a registry, including a CoreWeave public IPv4 if
needed, with CW S3 as an interim artifact store. The existing CoreWeave registry
was healthy, so no new public endpoint was necessary. A dedicated publisher
credential was placed in Secret Manager and existing access preserved. The
CoreWeave transport probe rejected anonymous access and verified manifest,
config and layer bytes by authenticated digest readback. See
`docs/audits/registry_transport_001.json` and `registry_health_001.json`. This is a
benign transport fixture, not publication or acceptance of a generated task.

The new idea in `other_thoughts.md` proposes LLM adaptation of fixed behavioral
tests to a solver-chosen interface. `docs/adaptive_test_verifier.md` freezes a
bounded comparison of full test rewriting and adapter generation for unchanged
tests, measuring both correct-interface acceptance and behavioral mutation
detection. Missing assertions and skipped tests are failed adaptations. Actual
remote measurements are still pending. Dynamic multi-step tasks remain sequenced
after the basic POC.

### C17 revalidation004: infrastructure fixed, adversary output incomplete

The source-frozen c17 infrastructure revalidation completed with fresh successful
oracle/positive controls and zero rewards on the completed malformed, fixed
negative and independent injection/shortcut trials. The isolated runtime imports
and exact provider cache recipe checks worked. The boundary adversary then
exhausted both permitted GLM output attempts (32,768 and 65,536 reasoning tokens,
both `finish_reason=length`); no truncated answer was graded. The task remains
unaccepted. Final evidence is hash-verified under
`runs/synthesis-c17-continuation-004/terminal-pull-0717/`. The next work separates
incomplete adversary generation from task defects and retains both raw attempts;
there is no justification for changing the task semantics from this result.

### OCI integrity component

`capability_pipeline/oci_artifact.py` now validates complete compressed bytes and
gzip CRC/trailer, rejects trailing or concatenated members, enforces expansion
bounds, computes diff_id, preserves explicit source image configuration and
constructs deterministic OCI metadata. Thirteen focused synthetic-byte tests
cover valid and corrupt inputs plus configuration/readback integrity. This is
not yet a production publisher or proof that a tar archive is safe/private or
that a generated image is runnable. Registry-worker integration and task gates
remain separate.

### User-directed adversary budget expansion

The user challenged the 64K adversary cap. That was a controller policy, not a
measured model ceiling. Read-only inspection on September 19 at 07:25:43 UTC found
all 37 returned running GLM engine pods configured with `--max-model-len 262144`.
The current c17 boundary request used 3,212 prompt tokens, leaving 258,932 tokens
within that total context. `/v1/models` itself exposes only the model ID. The
source and sanitized live evidence are retained in
`docs/audits/glm_output_budget_metadata_001.json`, with historical builder logs
explicitly separated from the current controller trial.

Adversary policy now requests 131,072 output tokens followed, on length
exhaustion, by the server's remaining-context budget (`max_tokens: null`), with
sufficient request time. Previously exhausted 32K/64K attempts remain evidence;
the next c17 run should not repeat those budgets. A literal 262,144-token output
request would exceed total context once the nonempty prompt is included. No
higher-budget c17 completion has been measured yet.

c17 continuation 005 stopped before inference when worker restore membership
failed, although the independently retained source archive verified all 1,296
seed members locally. Continuation 006 was submitted with a hash-bound opaque
seed archive and a unique staging directory. Its source snapshot is
`8f9df40122bfae918a1d5564b2cc4ac9b85c266dece996b7b7444372851b4a90`.
At 07:59:20 UTC Iris reported the exact job
`/muchanem/cap-synthesis-c17-continuation-006` running with its task building.
This is launch evidence, not an adversary completion or task verdict.

### Streaming registry transport proof

The maintained `oci_registry.py` now composes validated gzip/rootfs metadata with
streaming upload and full digest/size readback. CoreWeave job
`envreg/cap-oci-publisher-001` completed September 19 at 07:47:45 UTC. Its benign
3,146,889-byte compressed layer crossed the streaming chunk boundary and produced
the immutable fixture manifest
`sha256:2dbef89c367d6536c088c64c891eed928ce8eec5eeab1f068442fa650fbdd0f3`
in `capability-infra/streaming-oci-smoke`. The exact library/script sources,
receipt, and verified cleanup are in `docs/audits/registry_streaming_*_001.json`.
The temporary Secret, ConfigMap and Job were deleted; the fixture remains.
Twenty-two focused OCI integrity/transport tests passed. This proves transport
for synthetic data, not generated-image privacy, boot behavior, provider mapping,
or task acceptance. No generated task image has yet been published.

### Captured-image content gate and c32 rebuild

Provider preflight found the reused c32 candidate snapshot carried an older
README even though its fixtures and entrypoint matched. A newly named snapshot,
`cap-harbor-c32pub-779fe036dd998023c787`, was built from the frozen latest public
inputs. Its exact provider recipe and all nine ready-file hashes passed, including
the rule-8 README in both baked and deployed locations. PostgreSQL was SQL-ready;
PGDATA was a separate mount. Owned probe cleanup was verified.

The capture implementation now explicitly excludes provider file mounts; directory
filesystem boundaries alone were insufficient evidence for those files. Candidate
and generic private-verifier captures completed with tar/gzip exit 0 and verified
owned sandbox absence. Their content-addressed CW S3 objects are recorded in
`docs/audits/c32_capture_execution_001.json`, bound to approved capture plan
`9b06aace1cf6ae91af65cab08617bffad30bb0299d725eb4cd75fb80ad981502`.
No registry publication or cold-pull success is inferred from capture.

Actual CW archive review then rejected both plan001 captures for a provider-state
entry under `/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs`.
Downloaded bytes matched the capture receipts; these are content-boundary
failures, not transport corruption. Failed archive evidence is retained under
`runs/c32-archive-review-001/results`. No registry upload occurred. The next
capture must explicitly exclude provider-state trees as well as file mounts.
Meanwhile construction004 reached terminal state on a measured Daytona aggregate
CPU-limit error before the oracle ran. Its final immutable snapshot
`d8cc34b743414bbfae7209a6321c37f4` also contains a later rule-5 clarification in
the public README, so plan002 must use that terminal source; plan001 remains
historical evidence and its candidate is not the final task image.

`rootfs_review.py` and `image_publication.py` now inspect a private copy of the
captured archive before invoking the maintained registry publisher. They verify
compressed integrity, reviewed public-file hashes, task payload boundaries,
provider/runtime-state exclusions, source identities, implementation hashes and
cleanup evidence. Twenty-eight focused tests passed, including stale public
content, hidden private payloads, file-bind residue, archive ambiguity, source
drift and prevention of publication after a failed gate. These are controller
unit tests on synthetic bytes, run locally; task code remains remotely executed.

### C22 native verdict failures beyond missing newlines

The fresh partial c22 snapshot `d406a95165c14d81a42a11debbb829e8` retains
99 completed trial results from the builder's aborted 24-way calibration:
eight graded and 91 infrastructure errors. Native malformed-verdict evidence
contains 55 replies without a score token and 32 with an inline terminal token.
The lower-concurrency campaign also exhibits both modes. The hash-bound result
audit is `docs/audits/c22_native_judge_protocol_002.json`. Native transport did not
retain usage or finish reasons, so truncation remains unproven. The existing
newline-only repair cannot resolve missing tokens; bounded protocol re-asks are
being added without retrying based on awarded score or treating exhausted
protocol failures as zero reward. No calibration or task acceptance is claimed.

### Portable image cold-boot gate prepared

`scripts/probe_published_task_image.py` now binds the exact approved plan,
registry host, repository, manifest digest, publication receipt and reviewed
ready-file hashes before provider access. It reconstructs a FROM-only snapshot
and uses two owned, network-blocked sandboxes to check SQL/PostGIS readiness,
file identity and database/workspace reset, including a mutation in only the
first sandbox. Registry reachability and owned-sandbox deletion are explicit
gates. Receipts retain the failing stage without exposing provider exception
text. A successful probe still leaves full task gates pending.

Eleven controller tests on synthetic receipts/stub providers passed, covering
foreign registry/repository, digest/content/plan mismatches, reachable networking
and unverified cleanup. No task image execution is inferred from these tests.
After supervisor-aware verifier integration, the full local controller suite
passed **351 tests**, with one optional pinned integration test skipped; Ruff
passed. Generated task code continues to run remotely.

### c32 published images pass fresh reconstruction and reset

Both c32 images were published with rootfs review and layer/config/manifest
readback (`docs/audits/c32_image_publication_execution_002.json`). Initial cold
checks exposed two infrastructure outcomes: shared snapshot capacity at 40/40,
and Daytona dropping an inherited ENTRYPOINT in a FROM-only reconstruction.
Ownership-verified obsolete caches were removed; unrelated snapshots were left
in place. A fresh source-snapshot control demonstrated normal PostgreSQL startup.

The credential-free published-image catalog now binds exact raw OCI config and
manifest hashes and restates ENTRYPOINT/CMD in provider recipes. With those
recipes, both images passed two fresh blocked-network boots, PostgreSQL/PostGIS
readiness, reviewed file hashes, mutation/reset and registry unreachability. All
four sandboxes were deleted with absence verified. The aggregate audit is
`docs/audits/c32_cold_pull_execution_003.json`. Passed cold caches were subsequently
removed after zero-reference checks to free slots for normal runtime reconstruction;
the immutable OCI images and all provider mapping receipts remain retained.

Root independently validated `data/c32-portable-runtime-003`: nine allowed
image-pointer/derived-manifest/producer-validation changes, 298 protected hashes,
and repair attempts 1 and 2 retained. New specification SHA-256 is
`17ca60e596916892404501fe6dd9dc8af3aef310b6a6191be516f303d2c1df17`.
Fresh full runtime/quality revalidation is required; cold boots do not imply task
acceptance. The integrated local controller suite now passes **368 tests**, with
one optional integration test skipped; Ruff and shell syntax checks pass.

### Continuation seed collision corrected

The first two c07 semantic-repair launches collided with the worker's newly
created `worker.log` while restoring historical run metadata. The old restorer
copied earlier members before failing, and final sync could then publish those
partial restore results. A post-failure manifest with 1,736 members did not prove
that the output prefix pre-existed the launch. No semantic repair ran.

The restorer now checks every destination before copying any member; a regression
proves that a late collision leaves no earlier member behind. The reviewed new
seed relocates eight run-level artifacts under `historical-metadata/`, retains
all original bytes and explicitly declares the two historical omitted reports.
The corrected c07 `semantic-repair-003` launch has a one-round cap, with zero
previous repair rounds consumed. The intended-discrimination gate stays binding.

### Repeated evaluation and fixed-workspace replay

The c17 pilot audit remains valid, but the broader matrix lacks repeated trials,
resource measurements and split evidence (`c17_recipe_matrix_gap_009.json`). The
new evaluator freezes three fresh runtime suites, forces one solver attempt per
positive per suite, retains failures, and requires at least two solver successes.
It preserves declared noncritical partial-credit ranges and freezes explicitly
zero-reward controls separately. It does not claim a provider seed, complete
critical-negative coverage, resource conformance or training admission. Raw
payload inventories, source/input hashes and each attempt's receipt are retained.

The c17 evaluation-001 plan was superseded before launch when controller fixes
changed its fingerprint. Evaluation-002 passed a 44-member transport round trip
and staged-controller validation, then launched three concurrent attempts as
`/muchanem/cap-c17-repeated-evaluation-002`. The source snapshot is
`89f2bac8818c99931033cdcc1e0a4f15bb73e13714771546b8f8a216ba37369a`;
`docs/audits/c17_repeated_evaluation_launch_002.json` records the plan and input
identity. No live repeated-evaluation success is claimed at launch.

Evaluation-002 subsequently failed before grading in all three attempts: the
private Daytona verifier lacked `CAPABILITY_DAYTONA_TOOLS`. The complete final
snapshot retains three infrastructure errors with null rewards, as recorded in
`docs/audits/c17_repeated_evaluation_failure_002.json`. The evaluator now requires
a frozen, hashed `dt.py` input for Daytona candidates or private verifiers and
derives the child helper setting from that input. A fresh plan is required.

c32's first actual migrated runtime booted both canonical images, but its authored
reference got zero because replay ignored `controls[].workspace` and submitted no
output file. That was a controller omission, not a task/image repair. The new
Docker replay path materializes only the declared fixed submission, checks paths,
records file/directory provenance, and verifies reconstructed file hashes. Payload
chunks stay below 32 KiB to avoid shell argument limits; empty directories survive.
Commands-only controls remain supported; unsupported transcript or non-Docker
workspace replay fails explicitly. The unchanged c32 bundle was relaunched as
`/muchanem/cap-synthesis-c32-image-migration-revalidation-002`.

The integrated suite passed **396 tests**, with one optional integration test
skipped; Ruff and shell syntax passed. Those controller checks are separate from
the live task outcomes.

### c22 complete trial inventory still fails strict calibration

Snapshot `c30a2ba5ba264baea09a5e68996de82b` contains all 252 trial records: 250
graded and two ungraded positive infrastructure errors. Six critical tamper/delete
repeats received exactly zero and skipped judging. Ten semantic-negative repeats
received 3/14, above their frozen 0.15 maximum. Binary negative separation alone
therefore cannot establish calibration success.

An independent audit attributes the extra 2/14 to injection-resistance credit,
which is plausibly legitimate even in an otherwise wrong causal answer. The
preregistered universal negative range did not accommodate that orthogonal credit;
the injection criterion also varied across repeated grades. No ranges were widened
and no result was relabeled into a pass. The concrete error analysis is retained in
`docs/audits/c22_calibration_252_error_analysis_003.json` for the next review.

### c07 repair did not demonstrate the intended discrimination

Semantic-repair-003 finished one construction repair, then failed controller
attestation checks before runtime validation or independent quality review. The
twelve mismatches concern newly emitted default-Python snapshot-recipe fields;
they are a separate controller defect from task difficulty.

All nine retained blind pilots produced exact-gold decision sequences, and none
entered the intended peeking trap. One response failed extraction because of a
bold-marked label; that formatting failure does not demonstrate the targeted
reasoning error. The current version fails its admitted discrimination requirement.
One semantic repair remains, but it should only fund a substantive, preregistered
redesign with fresh blind pilots. No lower-bar approval or acceptance is claimed.
See `docs/audits/c07_terminal_disposition_005.json`.

### September 21 terminal audit and infrastructure recovery

c17 evaluation-003 completed three reference grades, three first-attempt solver
grades, and eighteen authored negative/malformed grades with the expected rewards.
Eight independent attacks received zero, but the third boundary attack received
1.0 for conflicting duplicate `activeElement` names. The decoder kept the final
value and discarded the contradictory one. The complete final snapshot is
`fdb579bc0d1e4ad28b0235c620a03ef9`; the matrix correctly remains `needs_review`.
The global repair history contains two consumed rounds. This version is rejected
under expanded evaluation; no third round was allocated and no attack was
resampled to replace the result. Historical pilot acceptance remains unchanged.
See `docs/audits/c17_duplicate_member_disposition_003.json`.

c22 also finished unaccepted. Its complete 5,083-file snapshot preserves both
failed calibration campaigns and all six runtime attempts. The completed semantic
failure in attempt 3 cannot be erased by later replications. The final build record
keeps its judge-boundary gate failed. The terminal audit additionally flags prose
claiming all attacks graded in attempts 5 and 6, though each contains an execution
failure with null reward; see `docs/audits/c22_terminal_disposition_002.json`.

c32 revalidation-003 passed its reference, independent solver, six authored
negative/malformed controls, and two independent attacks. Boundary finalization
stopped on HTTP 400 before grading. The trace approached the configured context
budget, but the old transport discarded the response body, so the precise cause
is unproven. The controller now records a safe error category and body digest;
only an explicit context-length rejection may use the already configured
remaining-context fallback, preserving the entire conversation. Other HTTP 400s
remain terminal. Revalidation-004 uses the unchanged task and exhausted repair
history with this transport fix.

c02's 49.3 MB continuation archive exceeded Iris's workspace limit. The new
transport stores the unchanged archive at a content-addressed CW S3 key and
verifies its digest before existing seed-member validation. An initial worker cwd
bug failed before download; continuation-002 uses absolute paths and an actual
Marin working directory. Neither transport failure consumed a task repair.


## 2026-09-21: expanded catalog and unattended stage chain

The [ShellSim fixed-grading infrastructure audit](audits/shellsim_regrade_probe_003.json)
records two authored remote fixtures: native deterministic grading passed 20
fixed cells; a SCRIPT verifier passed two captured authored submissions and 20
fresh private Daytona grades, including actual workspace replay. Both complete
S3 snapshots were hash-verified and their raw evidence independently classified
without artifact issues. This validates the recipe's ShellSim grading path as
infrastructure only. It supplies no GLM-authored catalog task acceptance, no
adversarial robustness verdict, and no unattended completion claim.

The expanded catalog strictly ingests 1,999 capabilities across 45 curricula; no
capability IDs overlap the old source. A deterministic 45-subject cohort freezes
450 proposal slots and relevant incoming progression context. The new `generate`
worker composes proposal review/repair and construction, preserves stage receipts
and repair budgets, and reports all-slot accounting separately from accepted yield
and training-ready quality. Fresh actionable failures and validated graded exploits
now enter GLM construction repair without a separate submission. Integrated local
validation: 469 tests passed, one skipped; Ruff and shell syntax passed. These are
controller tests, not evidence of generated-task quality.

The [first live cohort](audits/unattended_catalog_launch_001.json) was submitted at
16:59:06Z with concurrency 256; the worker entered `generate` at 17:00:04Z. The
[c02 continuation](audits/c02_continuation_launch_003.json) was submitted at
16:58:37Z and entered `synthesize` at 16:59:27Z after its repaired seed transport.
Both jobs were authoritatively running at 17:00:16Z. No completion is claimed.

[c32 revalidation004](audits/c32_image_migration_revalidation_terminal_004.json)
finished with oracle and independent solver rewards 1.0, all six authored negative
or malformed controls 0.0, and three independent attack rewards 0.0. Its independent
quality review still requested repair/evidence, including the unproved resource
envelope and repeated-evaluation conditions. Both construction repair rounds remain
consumed. No context-400 or fallback occurred in this run; the previous HTTP400 cause
therefore remains unproved. Successful runtime gates do not override quality review.
