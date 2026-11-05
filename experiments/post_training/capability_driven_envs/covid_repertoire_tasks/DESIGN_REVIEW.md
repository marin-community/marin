# Task-design review and refocus log

This log records design decisions made after inspecting generated tasks. Keeping
this step explicit prevents a technically successful generator from silently
becoming a low-value training corpus.

## 2026-11-04: first vertical slice

### What we did

1. Identified the Stephenson COVID-19 multi-omics cohort and its canonical public
   accession, `E-MTAB-10026`.
2. Pinned the local TheisLab-derived TCR MuData convenience object by byte size
   and SHA-256.
3. Extracted a minimal donor-by-cell-type table containing cell counts, callable
   primary VJ chains, and TRAV1-2 counts.
4. Generated 25 deterministic tasks across aggregation, median, prevalence,
   comparison, and ranking operations.
5. Packaged each result as an ordinary `TaskSpec` with a hidden
   `StructuredExactSpec` answer.
6. Ran the extractor on the real 743 MB object and generated all 25 tasks.
7. Eyeballed one task from every family together with its hidden answer.

### What worked

- The complete source-to-task path was reproducible and inexpensive.
- Every answer was derivable from solver-visible material.
- The prompts fixed denominators, support filters, aggregation, rounding, output
  schema, and tie behavior.
- Correct JSON scored 1.0 and wrong JSON scored 0.0 without an LLM judge.
- The extraction preserved donor-level replication while removing source donor
  identifiers.

### What the review found

The first tasks required multiple rows, but almost no biology. A solver could
treat `MAIT`, `TRAV1-2`, and the cell-state labels as arbitrary strings. They are
useful plumbing tests, not the task library we want to demonstrate.

That distinction is now part of the campaign contract:

```text
biological setting + arithmetic != biological reasoning
```

### Refocus

The showcase tasks will use an open-book pattern:

```text
short cited biology card
+ multiple related observations
+ quantitative analysis
+ constrained biological conclusion
+ explicit limitation
→ objectively graded structured answer
```

The first new families are:

- **annotation triangulation:** use donor-consistent TRAV1-2 enrichment to assess
  a proposed MAIT annotation while recognizing that receptor usage is not proof;
- **analysis-unit diagnosis:** compare pooled-cell and donor-level summaries and
  select donors as the biological replicate.

The original arithmetic families remain as low-level controls.

## Target library: 50 reviewed tasks

Fifty is a design target, not a quota that the generator must fill at any cost.
The intended composition is:

| Lesson | Target | Required evidence |
| --- | ---: | --- |
| MAIT annotation triangulation | 8 | receptor usage, coverage, donor consistency, comparator states |
| Clonotype expansion | 10 | donor/state clone sizes, severity, expansion support |
| Diversity and sampling depth | 8 | donor-level richness/evenness, read/cell support |
| Replication and confounding | 8 | pooled versus donor estimates, coverage and covariates |
| B-cell maturation | 8 | isotype and V-gene mutation summaries |
| QC and sensitivity | 8 | alternate support/filter thresholds and stability |

The exact counts may change after source inspection. A family with insufficient
independent evidence receives fewer tasks rather than synthetic paraphrases.

## Acceptance checklist for each showcase task

A reviewer must be able to answer yes to all of these:

1. **Sufficient material:** all needed facts and data are solver-visible.
2. **Multi-evidence:** the answer requires multiple rows, groups, or evidence
   fields rather than lookup of one value.
3. **Biology:** the task applies a biological fact from a cited reference card or
   a clearly identified closed-book concept.
4. **Reasoning:** the solver must connect the quantitative result to a biological
   conclusion or methodological choice.
5. **Calibration:** the answer states an important limitation or rejects an
   overclaim.
6. **Objective contract:** numerical and categorical fields have one defensible
   answer under a frozen decision policy.
7. **No leakage:** the background material explains concepts and policies without
   containing the instance's computed answer.
8. **Nondegenerate:** support is sufficient, distractors are plausible, and the
   answer is not invariant across every generated instance.
9. **Distinctness:** the task differs in scientific evidence or reasoning, not
   only wording.
10. **Split safety:** every task from this cohort remains in the same dataset
    split.

## Review procedure

After generation, render a private review sheet:

```bash
PYTHONPATH=. uv run --project lib/taskforge python \
  experiments/post_training/capability_driven_envs/covid_repertoire_tasks/render_review.py \
  --tasks /tmp/covid-repertoire-tasks-v1/tasks.jsonl \
  --output /tmp/covid-repertoire-tasks-v1/private-review.md
```

Reviewers fill the checklist and mark each task `keep`, `revise`, or `reject`.
The private sheet contains hidden answers and is never included in solver-visible
resources or a public release.

## Next extraction work

The compact TRAV1-2 table cannot support all 50 target tasks. Before scaling:

1. derive donor-by-cell-state clonotype-size summaries from the pinned TCR object;
2. derive donor-level diversity and coverage tables;
3. join only the clinical fields needed for predefined severity/outcome tasks;
4. obtain and pin the corresponding BCR object for isotype/mutation lessons;
5. reproduce these derived objects from `E-MTAB-10026` rather than relying on the
   convenience LaminDB artifact;
6. complete the source-data rights audit before publishing capsules.

Each derived table gets its own schema, extraction test, source digest, and
human-reviewed lesson specification.
