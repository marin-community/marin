# Curriculum review rubric

Score the curriculum by its usefulness for training. Encyclopedic completeness is not a goal. State the evidence
for each judgment and name changes that would apply beyond the current subject.

## Review modes

Use this detailed rubric while developing the generator and review procedure on sampled subjects. It diagnoses
section boundaries, hierarchy, probes, and evidence failures. Learning-prerequisite edges use a separate pass after
the capability taxonomy is stable.

Routine scale-out uses one high-reasoning review call for one complete subject curriculum version. Sol/high is the
current reference. The call reads the subject guideposts, curriculum, and a small evidence summary, then returns a
score, status, evidence confidence, blockers, and the highest-risk sections. It does not spawn a reviewer per section
or edge. A repair creates a new curriculum version and receives another complete-subject review. The reviewer may
request targeted blinded Luna probes when their result could change a finding.

1. **Coverage:** The capability sections jointly cover the important behaviors represented by the subject guideposts,
   discovery tasks, and permitted evaluation probes. Record plausible omissions; do not add a section for every
   observed task. In-distribution evaluation examples are held-out probes. Out-of-distribution evaluations contribute
   metadata only. The review must account for every guidepost, discovery task, and applicable held-out probe with a
   primary section or an exclusion rationale. Keep guidepost, discovery, and evaluation evidence counts separate,
   and record malformed or underdetermined examples rather than treating them as positive support. If an area has no
   applicable held-out probes, report insufficient evaluation evidence rather than inferring coverage.
2. **Observable outcomes:** Each capability says what a model can do after learning it. A topic name, tool name,
   answer format, or dataset source is not an outcome. A `group` is hierarchy only: it has a scope but is not a
   trainable outcome, task-assignment target, prerequisite, or probe target.
3. **Boundaries:** Includes, excludes, and examples distinguish neighboring capabilities. Overlap is acceptable only
   when the capabilities train meaningfully different behavior. For a task that could fit two capabilities, identify
   the decisive operation or knowledge that determines its primary home. A task's output format, delivery wrapper,
   or narrative subject does not establish membership when its decisive operation belongs to another subject.
4. **Learnable progression:** A capability parent's outcome contains its child outcomes; a group parent's scope
   contains its descendants. Children narrow that parent outcome or scope. The entry probe is the smallest
   self-contained exercise of its capability, and the representative probe exercises the full outcome. Flag an entry
   that already requires several unmodeled capabilities or an entry-to-representative jump that introduces a distinct
   operation family. Cross-capability learning prerequisites are generated after capability boundaries are stable by
   `learning_progression.md`. That pass implements the original epsilon condition: mastery of A materially raises the
   chance of solving some recurring family of entry-level B tasks. It does not require A to be necessary for every B
   task. Subject generation leaves embedded `prerequisites` arrays empty. Test containment by substituting every
   child's entry and representative probe into the parent's outcome. When a cross-child workflow is trainable but
   does not contain each child outcome, make the shared parent a group and add the workflow as a sibling synthesis
   capability.
5. **Mutual self-confidence:** Mastery of one representative task in a capability should predict success on most
   other representative tasks in that capability after changing surface form, data, and tools that are not part of
   the claimed outcome. Test composite capabilities with a counterexample pair drawn from different included operation
   families. Split the capability when the pair requires distinct central operations, tools, or bodies of knowledge;
   merge capabilities when mastery transfers and the distinction would not change sampling or evaluation. Compare
   tasks under the same knowledge regime: supplied context, broadly expected background, or specialist closed-book
   recall. A blocking split recommendation must cite two concrete, instantiable task probes, explain why their central
   operations do not transfer, and state how the split changes training sampling or evaluation. Different tools or
   subfields alone do not justify a split. When the central operation transfers but required background knowledge
   varies, declare the knowledge regime as a sampling and evaluation facet. Language direction, jurisdiction,
   protocol family, and
   scientific model family are examples. Declare reusable dimensions in `sampling_facets`; concrete task values stay
   in task annotations outside the curriculum. A task set is a capability plus fixed facet values. Hold those values
   fixed for the mutual-confidence test, sample the intended value range instead of narrowing to the one observed
   example, and split the capability when changing the facet also changes the central operation. A facet is not a
   waiver from the distant-pair test: compare tasks from its most
   operationally distant values. Split when changing the value introduces a different solver loop, state transition,
   or evaluation contract rather than different knowledge or data for the same operation.
   Record an operation signature for each distant probe: input representation, central transformation, output
   artifact, and correctness contract. Shared topic vocabulary or workflow context does not establish transfer when
   one of those elements changes materially. Common failure cases include classification versus regression,
   construction versus interpretation, paired versus unpaired inference, and deterministic mismatch analysis versus
   intermittent-mechanism diagnosis.
6. **Generation probes:** Every capability has one `entry` instruction and one `representative` instruction. Groups
   have neither. The entry probe is the smallest self-contained exercise of the capability. The representative probe
   exercises the full outcome and differs in substance from the entry probe. Internal-capability representative probes exercise
   cross-child synthesis. Both must be instantiable as tasks. When the outcome is transformation, explanation,
   revision, or documentation rather than domain expertise, the probe supplies the facts, interface, measurements,
   or completed solution it operates on.
   Solutions and correctness contracts are outside the curriculum. An internal node remains a capability only when
   it supports a natural cross-child synthesis task. A routing menu or omnibus bundle becomes a group rather than a
   capability.
   Execute each probe on paper before acceptance. Reject undefined factors, missing geometry or state, contradictory
   premises, trivial optima, and requests whose result cannot be checked from the supplied facts. This check validates
   the task instruction only; verifier design remains outside the curriculum.
7. **Mapping:** Assignment is multi-label across curriculum graphs and single-ranking within each selected graph.
   Each graph declares one membership projection. `subject_domain` is the broad body of knowledge or tool environment
   required by the decisive operation; it excludes narrative subject and answer format. `task_mechanic` is a
   domain-neutral operation family such as implementation, repair, supplied-context extraction, calculation,
   explanation, recall, or tool-state mutation. It excludes domain nouns and source identity. The hardest operation
   and required operations rank capabilities only after graph selection. Groups are never mapping targets. Use
   separate mapping anchors from generation probes. When held-out evidence is available, members should outrank
   close non-members and tasks that exercise the same capability should retrieve the same capability. Calibrate
   membership independently per graph; scores from
   different projections are not directly comparable. Treat low similarity and unstable neighbors as evidence about
   annotation quality, anchors, boundaries, or missing coverage. Do not add source-specific rules or per-task
   exceptions. Mapping is diagnostic rather than a curriculum acceptance gate. Roughly 70% reasonable placement is
   sufficient for initial scale-out when human reviewers find the curriculum boundary coherent; improve task keys,
   anchors, and embeddings separately.
8. **Adaptive size:** Capability count and depth follow the preceding criteria. There is no target density per macro
   area. Flag broad catch-alls when mutual self-confidence is implausible, and flag distinctions that would not change
   sampling, transfer, or evaluation.

## Routine score

The single-call reviewer scores five dimensions:

| Dimension | Points |
| --- | ---: |
| Coverage | 25 |
| Mutual self-confidence | 25 |
| Local progression | 25 |
| Observable boundaries | 15 |
| Probe quality and parsimony | 10 |

`Local progression` scores parent containment, self-contained entry probes, and coherent entry-to-representative
difficulty within each capability. Catalog-level learning prerequisites receive their own proposal and review and do
not contribute to this 100-point subject score.

The total is a comparison aid. `pilot_ready` requires at least 85 points and no structural blockers. Any failed gate
at 70 points or above yields `revise`; a score below 70 yields `regenerate`. Pervasive structural defects must reduce
the affected dimension scores instead of overriding the status thresholds.

Those statuses describe review maturity. They do not require a breadth-first catalog build to keep regenerating one
subject until it passes. After the documented one-repair stopping rule, the canonical versioned baseline may include
a `revise` or `regenerate` graph provisionally when it is schema-valid and every inventory guidepost has an explicit
home. The wave report must retain the exact score, blockers, blind-fit misses, and evidence limitations. Provisional
inclusion supplies a complete object to improve in later waves; it does not convert the review status to
`pilot_ready`.

Evidence confidence is reported separately. Sparse evidence identifies where to sample tasks or rerun review. A
structurally coherent graph can enter the versioned catalog with low evidence confidence. Put evidence limitations in
findings and recommended changes. Reserve blockers for structural defects. This keeps curriculum quality distinct
from the maturity of the current task inventory and lets later evidence drive targeted revisions.

The routine reviewer must inspect every capability section's most distant permitted task pair, every embedded
prerequisite array for emptiness, every internal-capability representative for coherent synthesis, and every
guidepost for coverage. It must not infer one capability per guidepost. High confidence requires direct or held-out
evidence reaching every capability. A complete structural scan with sparse task evidence is medium confidence;
missing evidence for major branches is low confidence.

After any repair, repeat the complete-subject review. A local check that the named blocker disappeared cannot detect
coverage loss, new catch-all sections, or shifted sibling boundaries.

## Independent blind-fit diagnostic

Alongside the holistic review, freeze `max(24, 2 * guidepost_count)` tasks generated by a separate Sol/high agent that
has the subject definition but no access to the curriculum. A second independent agent classifies each task against
the curriculum as `exact`, `ambiguous`, `gap`, or `invalid`. Report
`(exact + defensible ambiguous) / (total - invalid)` and all four counts. Partial coverage by several sections is a
gap, not an ambiguity.

This rate measures sampled coverage and boundary legibility. It does not measure mutual self-confidence or epsilon
continuity and is not added to the 100-point score. During calibration, a repeated operation-family gap is blocking.
Repeated means at least two gap tasks with the same normalized central operation, or one gap in a guidepost for which
the curriculum has no capability. The fit judge proposes repeated operation-family groupings and the operator
confirms them from the task text. Cross-artifact promotion validation detects the single-uncovered-guidepost case from
hidden task metadata and holistic guidepost accounting. Do not set a permanent numeric threshold until results from
varied subjects establish a useful operating range. Keep the task set fixed across repairs, and keep blind-fit results
hidden from the holistic reviewer.

## Optional Luna probes

For a boundary diagnostic, the subject reviewer generates 8–16 concrete tasks concentrated on close siblings and
coverage gaps. Hide the intended section, ask Luna for one capability or `out_of_scope`, and accept an alternative
only with a recorded rationale. About 70% reasonable placement is sufficient during initial scale-out. Curriculum-derived
tasks are a weak diagnostic because their wording tends to mirror the section definitions; held-out TaskTrove and
permitted in-distribution evaluation examples provide stronger evidence.

Model-training experiments are outside this curriculum workflow. During development or a periodic audit, use Luna
failures as a rough proxy on concrete tasks instantiated from the two witness-family sketches attached to each
sampled learning-prerequisite edge:

- failing a prerequisite probe is a floor condition and leaves the edge empirically inconclusive;
- passing both prerequisites and failing both entries rejects epsilon continuity;
- passing both prerequisites and at least one entry is positive epsilon evidence;
- solving every probe is a ceiling condition and leaves only the structural judgment; and
- representative success above entry success flags probe ordering for review.

Findings and reproducible examples govern revisions. Passing the structural or Luna audit does not establish learning
transfer.
