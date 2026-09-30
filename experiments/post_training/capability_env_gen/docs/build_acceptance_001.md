# Construction admission 001: blocking build evidence

This document turns every unresolved construction condition in
`runs/construction-admission-001/accepted.json` into a blocking evidence gate.
Construction admission allows a builder to start; it does not certify the full
portfolio, runtime, task semantics, or publication. The admitted manifest has
SHA-256 `9be6a34469af6281e2b6f7f93abbe41441f22000b2655abfa1c0a2fdce5f99e7`.
Every record is rooted in catalog SHA-256
`b3318b9fa7965b70b1cd1aa15513faca63b629db67bdc991a277173b51b5c0c6`.

Each build must materialize a private `build-acceptance.json` with this shape:

```json
{
  "schema_version": "capability-build-acceptance-v1",
  "identity": {
    "capability_id": "cNN.example",
    "slot": 1,
    "proposal_hash": "64 lowercase hex characters",
    "capability_record_hash": "64 lowercase hex characters",
    "catalog_sha256": "64 lowercase hex characters"
  },
  "checks": [
    {
      "id": "stable-check-id",
      "state": "passed",
      "claim": "the exact condition established",
      "artifacts": [{"path": "relative/path", "sha256": "64 lowercase hex characters"}],
      "summary": "measured result, including counts and thresholds"
    }
  ]
}
```

Every checklist ID below must occur exactly once with state `passed`, nonempty
hash-bound artifacts, and measured results. A builder assertion, transcript, or
agent handoff is not evidence. If a check cannot pass, apply the proposal's stated
fallback or abandonment rule and rerun affected checks; any change to task
semantics, environment, verifier, or pass boundary requires fresh semantic review.

After construction, an independent reviewer must compare the public task, private
answer/oracle, verifier, controls, and runtime evidence with the exact proposal,
source capability, admitted review issues, and this checklist. The reviewer records
artifact paths and hashes and must reject informational findings that were merely
restated rather than resolved. Runtime validation and portfolio certification remain
separate gates.

## Common native-judge gate

This gate applies to `c02`, `c22`, and `c30`. Their proposal controls define useful
strata but contain only three or four positives and four or five negatives, so they
are not calibration fixtures.

The pinned native verifier alone is narrower than the proposals' two-stage designs:
it cannot run a task-authored campaign, citation resolver, hash checker, weighted
rubric, penalty, or critical/disqualifier aggregation. These tasks may use the
mandatory composite-verifier extension, which runs pinned private machine checks in
fresh network-blocked Daytona sandboxes, injects their trusted results and declared
private context into the native judge, and applies a hash-bound aggregation policy.
The lowered package must fail closed when that extension is unavailable.

- `judge-semantic-lowering`: before fixture authoring, prove that the exact pinned
  TaskSpec and Harbor path implements the complete admitted reward semantics. If a
  separate executable stage is required, retain end-to-end proof that the mandatory
  extension downloads the real task evidence, executes the check, gives the native
  judge trusted per-attempt results, and computes the admitted final reward. Bind
  the exact TaskCompendium revision, raw specification, adapter, policy, and config
  hashes. A schema-valid pair of independently runnable verifiers does not pass.
  Machine or judge infrastructure failures must remain ungraded; deterministic or
  critical failures must not be diluted by native-judge credit.

Ordinary schema-0.9 aggregation remains insufficient: pinned Harbor rejects
multi-step `all_required_steps`, `mean` dilutes failed gates, and `final` discards
the precheck. The authorized composite extension is the supported construction
path only when its exact policy represents the accepted proposal. Grounding and
fixtures may proceed earlier, but 80-group calibration must execute the final
composed Harbor path. Calibration cannot repair a lossy representation.

- `judge-fixture`: provide at least 40 acceptable and 40 unacceptable distinct
  semantic variant groups. Closely related paraphrases share a group and cannot
  increase the effective sample. Repeats are descriptive and cannot increase it.
- `judge-model-path`: at least 40 positive and 40 negative groups must actually
  reach the pinned TaskCompendium native model judge. Exact and constraint-gate
  outcomes are reported separately. Every plausible-wrong case must use the model
  path. Each label has an independent domain or executable-evidence basis.
- `judge-strata`: include acceptable alternate wording and reasoning, boundary
  positives, plausible substantive errors, partial-but-fluent answers, empty
  answers, and prompt injection. Model-path cases must still satisfy any
  deterministic precheck needed to isolate the semantic judge decision.
- `judge-statistics`: require every grouped case and repeat to be correct. Compute
  Wilson intervals over semantic groups, with balanced-accuracy lower bound at
  least 0.85, false-accept upper bound at most 0.10, and repeated-decision agreement
  lower bound at least 0.90. The agreement interval also excludes exact and
  constraint-gated groups. State that the estimates are conditional on this fixed
  task-specific case collection.
- `judge-runtime`: bind the fixture to the exact `specification.json` bytes and
  record the frozen provider/model/base policy, native judgment path, report hash,
  and raw artifacts. The actual task must also pass its independent solver and
  adversarial Harbor trials; the three-case infrastructure probe is not a task
  calibration.

## c02.flaky_test_repair, slot 10 — container / judge

- Proposal: `d5aa409874d71bb4ea4f46bcc7627020ccc8d09948524b2255b22cd29d36c683`
- Capability record: `e219cb6e1b7e782c06761089d990c4939b4c89c1b687c27d7bb55607557b5890`

Blocking checks:

- `c02-plugin-contract`: record exact pytest, pytest-randomly, and pytest-xdist
  versions and executable probes for shuffle scope, `--randomly-seed`,
  `-p no:randomly`, `worker_id`, and `PYTEST_XDIST_WORKER`; record the selected
  fallback if any probe fails.
- `c02-baseline-consistency`: machine-generate all baseline logs. In the 20-run
  default shuffled sample, `test_ledger_archive` must fail 6–12 times alongside
  the two timing/concurrency tests; it must fail 0/20 with randomization disabled.
  The larger measured bands must be: mechanism a 40–75%, d 5–40%, b 30–60% under
  shuffled seeds and 0% disabled, and c 30–70% under `-n 4` and 0% serial.
- `c02-behavioral-controls`: the reference repair must complete a fresh 120-run
  green campaign, catch all four product mutants, pass slow-lateness, fail fast on
  never-ready, preserve a clean `src/` diff, and agree with the claimed campaign
  logs. Sleep inflation, retry masking, product edits, and a fabricated report must
  fail their intended gates.
- `c02-judge-boundary`: use the common native-judge gate. Model-path variants must
  emphasize report/evidence decisions that survive the machine prepass: correct
  alternative explanations and prose as positives; plausible misattribution,
  inconsistent evidence claims, assertion-weakening rationales, and fluent but
  unsupported conclusions as negatives. Machine-rejected shortcut controls do not
  count toward model-path groups.
- `c02-verifier-composition`: prove the private campaign, four mutants, source
  freeze, latency probes, anti-shortcut scans, and log-consistency checks execute as
  reward-bearing gates before or alongside the native judge. A report-only native
  checklist is not equivalent to the admitted verifier.
- `c02-budget`: measure the solver campaign and evaluator prepass on target
  hardware against the proposal's 45- and 60-minute limits.

## c05.analysis.protocol_trace, slot 10 — reasoning / code

- Proposal: `dda40ab415c300145629200646ac826eb1dc3824422ac7b4b4c583fd5fe04a35`
- Capability record: `ee342c257691d113cc4d097751261b5382233b7ee1ba2fcf146965b1ecda851a`

Blocking checks:

- `c05-pass-contract`: freeze an explicit aggregate pass rule in the public rules
  and grader; do not leave per-criterion scores without a pass boundary.
- `c05-row-domain`: state exactly which ACK rows are required, including the
  treatment of SYN-ACK and FIN,ACK, and prove the schema, reference answer, and
  public rules agree.
- `c05-estimator-and-timer`: derive the estimator with exact fractions before
  applying tolerances, and state timer stop-on-full-acknowledgment and
  start-on-new-data semantics. The private seed-B invariant check must cover them.
- `c05-grader-discrimination`: the reference answer scores 100%; at least 25
  single-defect mutations each fail the intended criterion; strict schema lint
  passes; a blind second implementation derived only from public inputs matches
  every reference field within tolerance.
- `c05-free-text`: demonstrate deterministic handling of C7 across paraphrases or
  apply the admitted fallback that removes C7 and reweights structured C6 before
  freezing the pass rule.
- `c05-justification-contradiction`: run the exact built grader against the measured
  contradiction form that contains every expected token and number while denying
  the required TSecr/ACK facts. It must lose C7 credit. If a bounded deterministic
  check cannot distinguish affirmation from negation, apply the admitted fallback:
  remove C7, fold its weight into structured C6, update the published score math,
  and rerun all discrimination controls. Preserve legitimate partial credit for
  numeric work that is correct despite a weak explanation.
- `c05-row-specific-exclusion`: bind each excluded row to its actual exclusion
  reason and expected state transition. A response that supplies a valid reason
  belonging to a different row, including labeling advancing ACK 253 as stale or
  nonadvancing, must lose the corresponding credit. Record per-row mutation
  results rather than accepting the presence of any exclusion keyword anywhere in
  the submission.
- `c05-public-schema-parity`: enforce the published `schema.json` in the actual
  reward path. In particular, evidence items must carry an explicit boolean
  `duplicate`, `interpretation` must be nonempty, and estimator values must use the
  published representation. Mutations that omit the false-valued `duplicate` or
  empty or contradict the evidence interpretations must not receive full reward.
  Semantically equivalent numeric JSON may be normalized, but if the verifier
  permits it then relax the public schema and instructions to say so rather than
  calling decimal strings the exact required format. Bind the exact public-schema
  and verifier hashes to the measured outcomes.

## c17.widget_lifecycle_accessibility, slot 8 — reasoning / simple

- Proposal: `570d4c881fc8d741cb2481c112ba49de5c88a0848cc4b2bc5939d4590fc6964e`
- Capability record: `b1735a172f01cc85cc2ab6dc4cb6cdc22fc4f510b9e77df318b3533a2b8b1698`

Blocking checks:

- `c17-browser-authority`: run both implementations twice in pinned headless
  Chromium and capture all 60 graded fields. Resolve active-element, click-focus,
  ARIA, and live-region assumptions from the observed trace or explicitly update
  the public browser model and regenerate the reference.
- `c17-atomic-close-snapshot`: implement the public no-render close observation as
  one synchronous in-page operation: trigger close and capture the graded DOM,
  focus, ARIA, and live-region snapshot before returning control to the harness.
  Do not place a separate awaited browser evaluation, animation frame, timer, CDP
  round trip, or implicit render opportunity between close and capture. Retain a
  pinned-Chromium trace that identifies the exact JavaScript boundary and proves
  the recorded state comes from that atomic operation.
- `c17-pass-policy`: add a grader README that records the intentional all-or-nothing
  60-field boundary and the semantic JSON/trim-only normalization rules.
- `c17-control-suite`: reference and harmless formatting variants pass; all seven
  negative controls fail at their expected first mismatch; every one-field mutation
  fails; repeated grading is byte-identical. Include case-only mutations of event
  IDs, element IDs, ARIA string values, and live-region text, and require them to
  fail wherever the public contract says values are exact. Record the pinned
  verifier normalization parameters so no default case folding weakens the task.
- `c17-ambiguity-pilot`: classify a fresh solver's misses as ambiguity or difficulty.
  Any ambiguity requires a wording/reference revision and complete rerun.

## c21.spreadsheets.calculate, slot 9 — container / code

- Proposal: `cbb84ed8e089542d2ead41b282e6c3d32abff30b4c6597cc13a03a3ae446d8a3`
- Capability record: `e931c90117f5ece3a6c701d55efbf66b8aec74157c05c1d3634b59a74946c484`

Blocking checks:

- `c21-engine-probe`: before fixture construction, prove that the selected pinned
  engine recomputes an openpyxl-mutated input and includes appended rows, with
  10/10 deterministic results. Use the documented fallback engine or abandon the
  design if neither works.
- `c21-minimality`: compute the actual defective formula-cell set in
  `answer_key.json`; derive the allowed edit bound from that cardinality. Replace
  the likely-invalid cap of 12 and ensure the 40-cell negative is genuinely broader
  than every accepted minimal repair.
- `c21-code-only-rubric`: implement mutation-scenario coverage as deterministic
  static or instrumented checks with recorded traces. No human “code inspection”
  may influence this code verifier's reward.
- `c21-oracle-and-variant`: require exact clean-workbook agreement with an
  independent oracle, all internal controls true, masked seeded defect behavior,
  held-out constant-cost divergence under a public legitimate propagation
  invariant, and five deterministic regenerations.
- `c21-roundtrip-and-controls`: prove mutation round trips introduce no unrelated
  value changes. The reference scores 100%; cached-only, hardcoded, over-broad, and
  pasted-value controls each lose exactly their intended criterion in two identical
  runs. Complete the clean-image leakage audit and the 10-minute rehearsal.

## c22.workflow_runtime_diagnosis, slot 9 — shellsim / judge

- Proposal: `e2ede7f26d29648534eebd99a1cfa8ae2564c1d7760299dd5b07a6170e1dfdb4`
- Capability record: `14b09dfb622ebdde2f0fcd96fa2637d46128a9180240d33b1eea1c7c593fef65`

Blocking checks:

- `c22-fixture-truth`: regenerate the 30-file bundle three times byte-identically;
  prove every ground-truth claim maps to an artifact, every wrong-order trap is
  contradicted by a named line, and the 90-second clock skew is consistent.
- `c22-shellsim-contract`: execute the exact exploration/write workflow in the
  pinned ShellSim. `grep -n`, required `awk`, citation line semantics, and writing
  `postmortem.md` must work with zero unsupported commands, or the documented
  line-prefix/core-command fallback must be applied and retested.
- `c22-citation-and-integrity`: gold citations resolve 100%, fabricated citations
  reject 100%, 20 perturbations match expected outcomes, and solve-session bundle
  hashes remain unchanged.
- `c22-judge-boundary`: use the common native-judge gate. Model-path positives
  should include independently phrased correct causal chains and legitimate
  citation styles. Model-path negatives must retain valid citations/prechecks while
  varying duplicate-vs-leak ordering, root-cause attribution, contradiction
  handling, confident decoy acceptance, partial narratives, and embedded-instruction
  following. Invalid citations and bundle tampering are separate machine gates and
  cannot pad native negative groups.
- `c22-verifier-composition`: prove regenerated-bundle hash checks and the citation
  resolver are reward-bearing deterministic gates in the published runtime. The
  native judge may consume their trusted result only if the runtime produces and
  binds it; static builder-authored context cannot stand in for a per-attempt check.
- `c22-realism`: retain a human ground-truth review and an independent dry run. If
  the solver reaches the correct chain in under 15 commands, increase realistic
  noise and rerun all dependent fixture, citation, and judge checks.

## c30.growth.performance_reporting, slot 5 — reasoning / judge

- Proposal: `e7a143b517fe913295831e736a2fcd99cd7874b9d47234a7b5bf1d1cae3b50df`
- Capability record: `007babdcdab144d10c13b5b8858ccffd7376d4a8a266274009c4b99602de432c`

Blocking checks:

- `c30-arithmetic-authority`: independently recompute every defect-key value and
  explain every numeric string in the visible memo. Record email complete-week
  ROAS as exactly `63700 / 16800 = 3.791666...` before display rounding. Resolve
  the visible source conflict where Tables 1-2 sum to `199,800 USD` gross paid
  revenue while Table 4 sums to `197,800 USD`; either repair the fixture or state a
  solver-visible authority/reconciliation rule and add controls for a solver that
  correctly flags the `2,000 USD` gap. A private authority choice is insufficient.
  Treat weekly deduplicated reach as a weekly metric: compare the four-week
  averages `120,000` and `174,000` (`+45%`) with the sampling caveat rather than
  presenting the sums `480,000` and `696,000` as deduplicated reach totals.
- `c30-window-semantics`: the visible 3.2 blended claim must explicitly span all
  three channels, making email's 14-day revenue window incompatible with the
  weekly spend windows. A paid-only blend is computable and cannot support this
  critical criterion.
- `c30-rubric-freeze`: freeze exact weights, total points, critical criteria,
  tolerances, false-positive penalties, and pass boundary in hashed artifacts.
  Resolve the temporal-versus-causal and NA gross-treatment boundaries in anchors.
- `c30-native-mapping`: retain a criterion-by-criterion mapping from the admitted
  weighted 0/1/2 rubric, four critical must-pass rules, disqualifiers, capped
  false-positive penalties, and register-section cap to the mandatory composite
  verifier. Paired native binary criteria may represent 0/1/2 anchors only if the
  weighted result is exact on every anchor. A machine penalty implements -1 per
  false positive capped at -4; a separate conditional cap zeros exactly the
  register-section indices above three false positives. Critical or deterministic
  gate failures force overall zero and cannot be averaged into a passing mean.
- `c30-judge-boundary`: use the common native-judge gate. All model-path cases must
  pass quote/format/length prechecks. Positive groups vary complete, good-faith,
  and stylistically distinct correct audits. Negative groups independently vary
  causal overclaim, refund/FX arithmetic, attribution-window blending, definition
  mismatch, unsupported generalization, false positives, out-of-menu action, and
  injection, including fluent near-boundary combinations.
- `c30-release`: run the discrimination pilots, verify synthetic glossary phrases
  are not copied from public sources, and bind material, defect-key, rubric, judge,
  and calibration hashes. Any fixture edit invalidates calibration.

## c32.geometry_topology_repair, slot 7 — container / code

- Proposal: `75adb802f475a87007fcaba979b78e8d6671c393270ad54b358f7dc30e42cf33`
- Capability record: `344ee46242ab144ad89bf763492870bcd902e2cf7d0afc7d7b03e26296f366b5`

Blocking checks:

- `c32-defect-inventory`: define whether the 0.300 m and 0.305 m cases are inside
  the stated 25 endpoints. Run an explicit nearest-candidate-gap scan proving that
  endpoints above 0.300 m are exactly the designed unresolved-dangle set, with no
  unintended terminals.
- `c32-determinism`: rebuild three times under pinned images. Record byte hashes
  when supported and canonical full-precision dumps as the timestamp-independent
  authority; document which deterministic representation is the release gate.
- `c32-reference-generality`: two independently implemented correct solutions pass
  all checks after reset. Seven bad outputs fail only their intended checks. Tighten
  the locked-feature negative description so it names endpoint motion rather than
  harmless collinear vertex injection.
- `c32-geometry-boundaries`: retain measured evidence for 0.2999/0.3000/0.3001 snap
  behavior, 1e-7 noding offsets under 1e-6 tolerance, length-identity mutation
  sensitivity, round-trip precision, and the measured Hausdorff budget.
- `c32-budget`: reference workflow stays below 90 minutes and evaluator below 10
  minutes on the declared 2 CPU / 2 GB target.

## c35.persistence_migration, slot 6 — shellsim / code

- Proposal: `bbeac26a2dfb5b328a30a0e274f35764d5f4b3c1362f4434db443e602b8498df`
- Capability record: `e0a89dcfd9d4167ba4a67a4d39640c8be51eb68e257a8f4b87ffb234b46e96ae`

Blocking checks:

- `c35-shellsim-contract`: the first build gate must prove fixture commands reread
  solver edits, a 100+ line heredoc round-trips byte-identically, crash-at exits 137
  with the intended partial temp state, `sha256sum` agrees, and visible checks expose
  the buggy script. If any fails, apply the documented container fallback and seek
  fresh semantic review of the environment change.
- `c35-iteration-usability`: demonstrate three edit/run cycles and a fresh boot in
  under two minutes. The calibrated solve must locate all five defects from public
  evidence and reach all-green in 60–150 minutes; editing mechanics must not dominate.
- `c35-harness`: reference passes G1–G9 identically three times under five minutes;
  the buggy script exposes every defect class; five single-defect mutants produce a
  diagonal discrimination matrix; hardcoding, naive-gate, and reject-all controls
  fail as designed.
- `c35-crash-policy`: repeat the seeded SIGKILL loop three times. If it is unstable,
  apply and record the admitted fallback that removes this secondary loop while
  retaining the deterministic crash matrix; rerun dependent controls.
- `c35-leakage`: search the visible repository and simulator image for hidden
  markers, seeds, and reference identifiers and retain zero-hit evidence.

## Admission provenance audit

All eight current entries passed an independent local integrity audit:

- each proposal hash equals the canonical digest of its proposal;
- each capability record hash equals the canonical digest of the full pilot record,
  and that record and catalog source exactly match `data/pilot.json`;
- construction context and provenance exactly match `input_candidates.json`;
- the last admission-history proposal hash and review equal the accepted values;
- source proposal hashes point to the actual admission inputs; and
- the four recorded pilot-002 source-file hashes match the retained files.

The controller contract still needs a hardening change: `synthesis.load_accepted`
currently verifies only the top-level accepting review and admission state/scope.
It should also recompute the history linkage described above and require
`portfolio_certified` and `runtime_certified` to remain false at construction
admission. Without those checks, later metadata edits could bypass the evidence
chain even though this retained run is internally consistent.
