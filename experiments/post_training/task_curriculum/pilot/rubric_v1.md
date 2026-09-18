# Curriculum review rubric

Score the curriculum by its usefulness for training. Encyclopedic completeness is outside scope. State the evidence
for each judgment and name changes that would apply beyond the current subject.

1. **Coverage:** The leaves jointly cover the important capabilities represented by the subject guideposts and
   discovery tasks. Record plausible omissions; do not add a section for every observed task.
2. **Observable outcomes:** Each section says what a model can do after learning it. A topic name, tool name, answer
   format, or dataset source is not an outcome.
3. **Boundaries:** Includes, excludes, and examples distinguish neighboring sections. Overlap is acceptable only
   when the sections train meaningfully different behavior.
4. **Learnable progression:** Children refine their parent, prerequisites are necessary, and adjacent steps are
   small enough that partial success is plausible. Do not infer transfer merely from a plausible edge.
5. **Useful granularity:** A section is broad enough to support varied task generation and narrow enough that
   success on one representative task predicts success on the others. Split or merge only when this improves a
   training or evaluation decision.
6. **Generation probe:** The two sample instructions actually exercise the claimed outcome, differ in substance,
   and can be instantiated as tasks. Their solutions and correctness contracts are outside the curriculum.
7. **Mapping:** When held-out mapping evidence is available, tasks that exercise the same capability should
   retrieve the same section, while close non-members should expose real boundary errors. Treat low similarity and
   unstable neighbors as evidence about anchors or missing coverage. Source-specific rules are prohibited.
8. **Size:** The hierarchy uses no more sections or levels than needed for the preceding criteria. Flag both broad
   catch-alls and distinctions that would not change sampling, generation, or evaluation.

A high score requires no blocking issue in coverage, boundaries, progression, or generation. The numeric score is
a comparison aid; findings and reproducible examples govern revisions.
