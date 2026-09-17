# Curriculum review rubric

Score the curriculum by its usefulness for training. Encyclopedic completeness is not a goal. State the evidence
for each judgment and name changes that would apply beyond the current subject.

1. **Coverage:** The leaves jointly cover the important capabilities represented by the subject guideposts and
   discovery tasks. Record plausible omissions; do not add a section for every observed task.
2. **Observable outcomes:** Each section says what a model can do after learning it. A topic name, tool name, answer
   format, or dataset source is not an outcome.
3. **Boundaries:** Includes, excludes, and examples distinguish neighboring sections. Overlap is acceptable only
   when the sections train meaningfully different behavior. For a task that could fit two sections, identify the
   decisive operation or knowledge that determines its primary home.
4. **Learnable progression:** Every node is a trainable capability. A parent's outcome contains its child outcomes;
   children narrow that broader capability. The internal node's sample tasks require a coherent cross-child synthesis
   so they do not duplicate leaf sampling. Prerequisites are necessary,
   and every representative task in the dependent section must require the full prerequisite outcome. Put narrower
   assumptions in the task instead of adding a broad edge. Adjacent steps are small enough that partial success is
   plausible; do not infer transfer merely from a plausible edge.
5. **Useful granularity:** A section is broad enough to support varied task generation and narrow enough that
   success on one representative task predicts success on the others. Split or merge only when this improves a
   training or evaluation decision.
6. **Generation probe:** The two sample instructions actually exercise the claimed outcome, differ in substance,
   and can be instantiated as tasks. An internal node's probes must exercise its cross-child synthesis rather than
   duplicate one child. Solutions and correctness contracts are outside the curriculum.
7. **Mapping:** When held-out mapping evidence is available, tasks that exercise the same capability should
   retrieve the same section, while close non-members should expose real boundary errors. Treat low similarity and
   unstable neighbors as evidence about anchors or missing coverage. Do not add source-specific rules.
8. **Size:** The hierarchy uses no more sections or levels than needed for the preceding criteria. Flag both broad
   catch-alls and distinctions that would not change sampling, generation, or evaluation.

A high score requires no blocking issue in coverage, boundaries, progression, or generation. The numeric score is
a comparison aid; findings and reproducible examples govern revisions.
