# Snowball RL curriculum: v0 pool and curation pilot

Status: proposal, 2026-10-19. Audience: Marin RL and data teams.

Tracking: [v0 pool #10314](https://github.com/marin-community/marin/issues/10314),
[10k pilot #10315](https://github.com/marin-community/marin/issues/10315), and
[design review #10316](https://github.com/marin-community/marin/issues/10316).

Build two artifacts in parallel: a training pool targeting 100,000 distinct tasks
for a fast Snowball RL iteration, and a fixed 10,000-task diagnostic sample for
developing quality review, semantic annotations, and curriculum sampling. Start
v0 from Ben's Snowball source families. Use the pilot to decide how to expand and
reweight them. Neither artifact is materialized yet; source quotas alone do not
establish runnable or admitted tasks.

Here, an environment means one TaskSpec task instance with its public interaction
contract and private grader contract. It does not mean a distinct sandbox image
or a generated rollout.

## Inputs and evidence

The [RL Data Atlas](https://applets.marina.oa.dev/a/fb11c931-5861-4878-8bb5-a964d652b45f/)
publication contains 4,998,910 converted task instances: 4,997,251 training
instances across 142 sources, and 1,659 evaluation instances. These are conversion
counts before curation admission or this proposal's duplicate and holdout checks.

| Input | Location |
| --- | --- |
| Converted publication | `s3://marin-us-east-02a/marin/rl-semantics/publication-008e95e5/2026.10.15.1/release.json` |
| Reviewed production catalog | `s3://marin-us-east-02a/marin/rl-data/catalog/production` |
| Structural-compliance pool | Separate historical pool; its TaskSpecs, revision, and admission evidence must be pinned before selection |
| Ben's release evidence | [Snowball v0.1 report, #10305](https://github.com/marin-community/marin/issues/10305) |
| Historical mixture | [Pinned `mixture-v2.json`](https://github.com/marin-community/marin/blob/2aa96b638e880f72b0f2a9c4f5bb4ce94bd60dcd/experiments/post_training/smartie/mixture-v2.json) |

Follow each publication source's `curation_root` to converted TaskSpec Parquet.
Resolve the production catalog's per-source artifacts to `final/part-*.parquet`
for admitted rows. Freeze resolved artifact identities and manifest hashes;
`production` is a mutable catalog pointer. Record whether each selected task
appears in both populations and retain the existing admission decision.
Structural tasks are not established as a source in this publication. Apply the
same training, holdout, contract-identity, and admission checks to their separate
pool; do not infer eligibility from the historical training export.

Ben's exact historical release has 69,487 training tasks, including 57,364
structural tasks. Its four NVIDIA components contribute 13,023 converted
instances in the larger publication. Thus the 100,000-task target requires an
explicit extension of his source selection. His release supplies no calibrated
task difficulty labels. The historical training export also needs validation
against the latest RolloutEngine API; the native API port tracked in
[#10190](https://github.com/marin-community/marin/pull/10190) is a dependency.

## Effort 1: v0 training pool

### Selection

1. Freeze input snapshots, current runtime revision, selection seed, and source
   allowlist. Preserve Ben's calendar, competitive coding, Reasoning Gym, STEM,
   and structural-compliance families. Enumerate every extension by source ID
   and its family in the selection configuration; exclude unlisted sources.
2. Restrict to admitted training tasks. Exclude evaluation sources, declared
   holdouts, and known benchmark overlaps. Record the matching rule and its
   coverage; a train label alone does not establish absence of contamination.
3. Group exact task contracts using the existing TaskCompendium identity
   machinery. Keep one deterministic representative and retain all source
   aliases. Distinct private grader contracts must remain distinct. Repeated
   public text can share model inference without collapsing task identity.
4. Allocate 100,000 slots without replacement. Start from the historical intended
   family weights: one sixth each for calendar, competitive coding, Reasoning
   Gym, and STEM; one third for structural compliance. Within an expanded
   family, allocate proportional to the square root of eligible source counts.
   Cap sources at availability and redistribute within their family first. If a
   whole family is exhausted, redistribute its remaining slots across families
   with capacity in proportion to their original weights. Record the resulting
   family shares.
5. Within each source, cover available difficulty strata. Use native difficulty
   or generator settings first; otherwise use a versioned manual source/family
   prior with `easy`, `medium`, `hard`, or `unknown` and its rationale. Allocate
   equally across nonempty strata initially, capping and redistributing within
   the same source. Preserve `unknown` as a stratum.
6. Select tasks deterministically from stable identities, shuffle exposure, and
   export TaskSpecs plus selection sidecars. Record requested and actual counts
   by family, source, and difficulty. Report any shortfall without duplicates.

These labels are sampling priors. A long prompt, expensive grader, or low judge
quality score does not by itself establish learning difficulty. V0 fixes the
pool and its initial sampling weights; it does not impose an easy-to-hard
training schedule.

### Release gate

Load the exported pool through the latest RL stack. Exercise roughly 100 distinct
tasks, covering every selected execution/grader contract, with additional cases
if 100 cannot cover them. Check reset, public observation, action handling,
termination, grader invocation, reward interpretation, and resource access.
Retain traces and failures. Quarantine unsupported contracts and regenerate the
allocation before calling the pool runnable. A successful contract smoke does
not establish model quality or training benefit.

## Effort 2: fixed 10,000-task curation pilot

Sample converted training tasks before curation admission so the pilot can
compare existing admission with the proposed filter. Apply evaluation/holdout
exclusion and exact contract grouping first. Keep runtime compatibility as a
separate diagnostic field so unsupported contracts remain visible.

For each source with eligible count \(N_s\), allocate half the sample equally
across sources and half proportional to \(\sqrt{N_s}\). Cap small sources,
redistribute, and round by largest remainder with source-ID tie breaking.
Select deterministically and record inclusion probabilities. The initial
count-only allocation covers 142 sources with 40–480 tasks each; recompute it
after eligibility checks. Freeze the resulting 10,000 input identities and do
not refill rejected tasks. Report every stage against that fixed denominator.

### Processing stages

1. **Quality decision on Ortet.** Produce `keep`, `reject`, or `uncertain`, reason
   codes, and evidence. Review missing context, inconsistent references,
   answer leakage, ambiguous requirements, and agreement between task and
   verifier. Inspect bounded private grading evidence only in the authorized
   review packet. Preserve existing mechanical/resource checks and executed
   grader controls; model judgment does not replace admission.
2. **DeepSeek Flash annotation on Ortet.** For kept tasks, generate a 15–40-word
   operational summary, domain/skill tags, interaction mode, and provisional
   difficulty with rationale and confidence. Describe the required knowledge
   and operation. Inputs contain public instructions and public answer/tool
   contracts; exclude solutions, private references, and verifier resources.
3. **Harrier summary embeddings.** Reuse the existing [curation embedding
   pipeline](https://github.com/marin-community/marin/blob/205558e1a7fc3dd705b1132691798b09dc1f254c/experiments/post_training/task_curation/README.md#task-and-curriculum-embeddings),
   pinned checkpoint, retrieval instruction, pooling, normalization, and
   FineStore caches. Key annotations by full request/model revision and vectors
   by rendered text plus embedding specification. Keep task identity separate
   from public-content identity. Preserve curriculum matching as a diagnostic.
4. **Curriculum sampling.** Cluster summary vectors, then inspect source ×
   difficulty × semantic-cluster coverage. Use cluster coverage to propose v1
   quotas and reduce repeated operations within large sources. Embedding
   similarity alone is neither a rejection rule nor a difficulty measurement.

The Ortet decision-model and DeepSeek Flash service IDs, immutable revisions,
request limits, and output schemas must be pinned before inference. They remain
unverified. Measure tokens, latency, failures, and cost on a small initial batch
before processing the full pilot. Persist successful requests and bounded retry
outcomes; retain unresolved failures as explicit rows.

### Pilot evaluation

Audit examples stratified by source and `keep`/`reject`/`uncertain`, with a panel
size set before reviewing outputs. Record human disagreement and concrete error
types. Report quality outcomes, existing-admission disagreement, annotation
coverage, runtime compatibility, and cluster coverage by source. Keep excluded,
failed, and unresolved counts visible.

Run repeated attempts on a roughly 100-task Snowball panel covering source and
provisional difficulty. Use measured success rates and execution cost to check
the priors. This panel supports coarse calibration; it cannot calibrate every
source or establish training improvement. Training comparisons need the same
held-out evaluation protocol and budget.
Before execution, pin the Snowball checkpoint, tokenizer, runtime revision,
attempt count, decoding settings, token/turn/time limits, and pass definition.
Separate infrastructure failures from model failures and retain per-attempt
results against that frozen protocol.

## Artifacts and rollout

Each release retains an immutable input manifest, source/family allowlist,
eligibility and duplicate audits, quotas, selected TaskSpecs, annotation
sidecars, model/prompt revisions, and aggregate reports. Join records on source,
task identity, source locator, and contract digest; avoid lossy joins on task ID
alone. Keep private reviewer packets and credentials out of shared summaries
and team-facing reports. Large task/vector payloads stay in object storage.

| Phase | Completion evidence |
| --- | --- |
| V0 selection | Reconciled counts, exact allowlist, held-out exclusions, reproducible selection |
| V0 runtime gate | Latest-stack contract smoke and documented quarantines |
| Pilot freeze | Exactly 10,000 distinct eligible input identities, or explicit availability shortfall |
| Pilot inference | Pinned service bindings, complete stage accounting, retained request evidence |
| V1 proposal | Audited quality errors, measured difficulty diagnostics, proposed source/cluster weights |

After v0 is runnable, use online Snowball success, cost, and learning-progress
signals to adjust sampling weights. Retain exploration weight and snapshot each
sampler change. Online adaptation, a full-corpus filter, and any training launch
are subsequent experiments; this proposal does not claim their results.
