# Curriculum review rubric

Score the curriculum by its usefulness for training. Encyclopedic completeness is not a goal. State the evidence
for each judgment and name changes that would apply beyond the current subject.

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
   knowledge regime: supplied context, broadly expected background, or specialist closed-book recall.
6. **Generation probes:** Every section has one `entry` instruction and one `representative` instruction. The entry
   probe isolates the prerequisite-to-section delta. The representative probe exercises the full outcome and differs
   in substance from the entry probe. Internal-node representative probes exercise cross-child synthesis. Both must
   be instantiable as tasks. When the outcome is transformation, explanation, revision, or documentation rather than
   domain expertise, the probe supplies the facts, interface, measurements, or completed solution it operates on.
   Solutions and correctness contracts are outside the curriculum.
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
   exceptions.
8. **Adaptive size:** Section count and depth follow the preceding criteria. There is no target density per macro
   area. Flag broad catch-alls when mutual self-confidence is implausible, and flag distinctions that would not change
   sampling, transfer, or evaluation.

A high score requires no blocking issue in coverage, boundaries, progression, mutual self-confidence, or generation.
The numeric score is a comparison aid; findings and reproducible examples govern revisions.
