# Curriculum review rubric

Score the curriculum by its usefulness for training. Encyclopedic completeness is not a goal. State the evidence
for each judgment and name changes that would apply beyond the current subject.

## Review modes

Use this detailed rubric while developing the generator and review procedure on sampled subjects. It diagnoses
section boundaries, prerequisite edges, probes, and evidence failures.

Routine scale-out uses one high-reasoning review call for one complete subject curriculum version. Sol/high is the
current reference. The call reads the subject guideposts, curriculum, and a small evidence summary, then returns a
score, status, evidence confidence, blockers, and the highest-risk sections. It does not spawn a reviewer per section
or edge. A repair creates a new curriculum version and receives another complete-subject review. The reviewer may
request targeted blinded Luna probes when their result could change a finding.

1. **Coverage:** The leaves jointly cover the important capabilities represented by the subject guideposts,
   discovery tasks, and permitted evaluation probes. Record plausible omissions; do not add a section for every
   observed task. In-distribution evaluation examples are held-out probes. Out-of-distribution evaluations contribute
   metadata only. The review must account for every guidepost, discovery task, and applicable held-out probe with a
   primary section or an exclusion rationale. Keep guidepost, discovery, and evaluation evidence counts separate,
   and record malformed or underdetermined examples rather than treating them as positive support. If an area has no
   applicable held-out probes, report insufficient evaluation evidence rather than inferring coverage.
2. **Observable outcomes:** Each section says what a model can do after learning it. A topic name, tool name, answer
   format, or dataset source is not an outcome.
3. **Boundaries:** Includes, excludes, and examples distinguish neighboring sections. Overlap is acceptable only
   when the sections train meaningfully different behavior. For a task that could fit two sections, identify the
   decisive operation or knowledge that determines its primary home. A task's output format, delivery wrapper, or
   narrative subject does not establish membership when its decisive operation belongs to another subject.
4. **Learnable progression:** Every node is a trainable capability. A parent's outcome contains its child outcomes;
   children narrow that broader capability. Prerequisites are necessary, and every representative task in the
   dependent section requires their full outcomes. The entry task adds the smallest new operation or concept to the
   prerequisites. A learner that has mastered the prerequisites must have a non-trivial chance of solving it. Flag an
   edge when the easiest dependent task still requires several unmodeled capabilities. For every edge, record why the
   full prerequisite outcome is necessary, where the dependent representative uses it, and the single new operation
   in the entry task. If the dependent task remains solvable when a completed prerequisite artifact is supplied to a
   learner who cannot produce it, that is pipeline ordering rather than capability dependence. Internal-node
   representative tasks require coherent cross-child synthesis around one natural artifact, question, or decision;
   concatenating unrelated child deliverables does not qualify. For sampled edges, write a second prerequisite
   representative, dependent entry, and dependent representative. An independent reviewer checks both variants
   against prerequisite-to-entry transfer, material added capability, and the completed-artifact counterfactual. One
   failing variant blocks the edge. Passing this structural audit does not establish training transfer.
5. **Mutual self-confidence:** Mastery of one representative task in a section should predict success on most other
   representative tasks in that section after changing surface form, data, and tools that are not part of the claimed
   outcome. Test composite leaves with a counterexample pair drawn from different included operation families. Split
   the section when the pair requires distinct central operations, tools, or bodies of knowledge; merge sections when
   mastery transfers and the distinction would not change sampling or evaluation. Compare tasks under the same
   knowledge regime: supplied context, broadly expected background, or specialist closed-book recall. A blocking
   split recommendation must cite two concrete, instantiable task probes, explain why their central operations do not
   transfer, and state how the split changes training sampling or evaluation. Different tools or subfields alone do
   not justify a split. When the central operation transfers but required background knowledge varies, declare the
   knowledge regime as a sampling and evaluation facet. Language direction, jurisdiction, protocol family, and
   scientific model family are examples. Hold the facet fixed for the mutual-confidence test, cover the intended
   facet values instead of narrowing to the one observed example, and split the section only when changing the facet
   also changes the central operation.
6. **Generation probes:** Every section has one `entry` instruction and one `representative` instruction. The entry
   probe isolates the prerequisite-to-section delta. The representative probe exercises the full outcome and differs
   in substance from the entry probe. Internal-node representative probes exercise cross-child synthesis. Both must
   be instantiable as tasks. When the outcome is transformation, explanation, revision, or documentation rather than
   domain expertise, the probe supplies the facts, interface, measurements, or completed solution it operates on.
   Solutions and correctness contracts are outside the curriculum. An internal node remains a trainable section only
   when it supports a natural cross-child synthesis task. A routing menu or omnibus bundle is not synthesis; flatten
   or restructure that node under the current schema.
7. **Mapping:** Assignment is multi-label across curriculum graphs and single-ranking within each selected graph.
   Each graph declares one membership projection. `subject_domain` is the broad body of knowledge or tool environment
   required by the decisive operation; it excludes narrative subject and answer format. `task_mechanic` is a
   domain-neutral operation family such as implementation, repair, supplied-context extraction, calculation,
   explanation, recall, or tool-state mutation. It excludes domain nouns and source identity. The hardest operation
   and required operations rank sections only after graph selection. Use separate mapping anchors from generation
   probes. When held-out evidence is available, members should outrank close non-members and tasks that exercise the
   same capability should retrieve the same section. Calibrate membership independently per graph; scores from
   different projections are not directly comparable. Treat low similarity and unstable neighbors as evidence about
   annotation quality, anchors, boundaries, or missing coverage. Do not add source-specific rules or per-task
   exceptions. Mapping is diagnostic rather than a curriculum acceptance gate. Roughly 70% reasonable placement is
   sufficient for initial scale-out when human reviewers find the curriculum boundary coherent; improve task keys,
   anchors, and embeddings separately.
8. **Adaptive size:** Section count and depth follow the preceding criteria. There is no target density per macro
   area. Flag broad catch-alls when mutual self-confidence is implausible, and flag distinctions that would not change
   sampling, transfer, or evaluation.

## Routine score

The single-call reviewer scores five dimensions:

| Dimension | Points |
| --- | ---: |
| Coverage | 25 |
| Mutual self-confidence | 25 |
| Progression and epsilon continuity | 25 |
| Observable boundaries | 15 |
| Probe quality and parsimony | 10 |

The total is a comparison aid. `pilot_ready` requires at least 85 points, no blockers, and medium or high evidence
confidence. A score from 70 through 84, or any blocker, yields `revise`. A score below 70 or pervasive structural
failure yields `regenerate`.

The routine reviewer must inspect every leaf's most distant permitted task pair, every prerequisite edge under the
completed-artifact counterfactual, every internal-node representative for coherent synthesis, and every guidepost for
coverage. It must not infer one leaf per guidepost. High confidence requires direct or held-out evidence reaching
every leaf and sampled prerequisite edges. A complete structural scan with sparse task evidence is medium confidence;
missing evidence for major branches is low confidence.

After any repair, repeat the complete-subject review. A local check that the named blocker disappeared cannot detect
coverage loss, new catch-all sections, or shifted sibling boundaries.

## Optional Luna probes

For a boundary diagnostic, the subject reviewer generates 8–16 concrete tasks concentrated on close siblings and
coverage gaps. Hide the intended section, ask Luna for one leaf or `out_of_scope`, and accept an alternative only with
a recorded rationale. About 70% reasonable placement is sufficient during initial scale-out. Curriculum-derived
tasks are a weak diagnostic because their wording tends to mirror the section definitions; held-out TaskTrove and
permitted in-distribution evaluation examples provide stronger evidence.

Model-training experiments are outside this curriculum workflow. During development or a periodic audit, use Luna
failures as a rough proxy on two prerequisite representatives, two dependent entries, and two dependent
representatives per sampled edge:

- failing a prerequisite probe is a floor condition and leaves the edge empirically inconclusive;
- passing both prerequisites and failing both entries rejects epsilon continuity;
- passing both prerequisites and at least one entry is positive epsilon evidence;
- solving every probe is a ceiling condition and leaves only the structural judgment; and
- representative success above entry success flags probe ordering for review.

Findings and reproducible examples govern revisions. Passing the structural or Luna audit does not establish learning
transfer.
