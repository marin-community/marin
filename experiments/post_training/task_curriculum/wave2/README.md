# Six-area curriculum scale-out

## Result

This wave generated and independently reviewed curricula for six additional competency-chart areas. The repaired
curricula contain 233 sections, ranging from 20 for Knowledge Application & QA to 64 for Databases & Data
Engineering. The three pilot curricula bring the diagnostic catalog to 292 sections across nine areas.

The earlier estimate of 15–25 sections per macro is not supported. Software Engineering, Databases & Data
Engineering, and Data Analysis & Visualization alone contain 149 reviewed sections. A mechanical extrapolation of
the six-area mean would produce about 1,300 sections across 34 areas. Section count must follow mutual self-confidence
and prerequisite continuity, and overlapping curriculum graphs must first be separated from a single-label taxonomy.
A working envelope of roughly 1,000–2,000 sections is more credible than 500–900 and
still comfortably below the 10,000-group constraint.

The generation and review loop improved every area. It did not make every curriculum ready for training. One repaired
curriculum had no remaining structural blocker; the other five retained two to four concrete boundary, transfer, or
probe defects. Review scores summarize the pass, but blockers govern the next revision.

| Area | Sections v1 → v2 | Review score v1 → v2 | Remaining blockers |
| --- | ---: | ---: | ---: |
| C01 Software Engineering | 30 → 36 | 42 → 64 | 2 |
| C06 Databases & Data Engineering | 50 → 64 | 55 → 73 | 3 |
| C07 Data Analysis & Visualization | 42 → 49 | 52 → 68 | 3 |
| C23 Technical Writing & Documentation | 23 → 22 | 47 → 86 | 0 |
| C25 Knowledge Application & QA | 16 → 20 | 52 → 68 | 4 |
| C27 Finance, Accounting & Audit | 31 → 42 | 43 → 68 | 4 |

## Method

Each generator received one macro area from the September 3, 2026 competency chart, 16 source-free TaskTrove
discovery tasks, their curriculum-independent Luna semantic keys, the common rubric, and a maximum depth of four.
The prompt imposed no section target. Every section had to provide an entry task that adds the smallest capability
beyond its prerequisites and a representative task for the full outcome.

Independent reviewers then checked every prerequisite edge, internal-node synthesis task, and composite-leaf transfer
claim. Their first pass found omitted capabilities, discovery tasks routed by output format even when the decisive
operation differed, broad sections whose task families did not transfer, and prerequisites that merely described
pipeline order. A repair pass produced a second version and a fresh reviewer audited it. This report records the
scores and remaining blockers; the canonical catalog contains the reviewed result.

The repaired rubric now requires:

- an evidence ledger that accounts for every guidepost, discovery task, and applicable held-out evaluation probe;
- a decisive-operation containment test, so JSON, file delivery, or narrative topic cannot create a curriculum home;
- a three-part audit for every prerequisite edge: full-capability necessity, use in the representative task, and the
  single new operation in the entry task;
- an artifact-dependence falsification: receiving a completed upstream artifact is not mastery of the upstream skill;
- a counterexample pair across the operation families in every composite leaf;
- one natural artifact, question, or decision for internal-node synthesis; and
- supplied facts or completed analyses when a probe is testing transformation, explanation, revision, or writing.

These are structural proxies for the proposal's learning criteria. They make the epsilon-difficulty claim falsifiable:
a learner that masters the prerequisites should have a non-zero chance on the dependent entry probe. They do not
measure training transfer. The next empirical test must generate small task sets on both sides of selected edges and
measure model outcomes or learning curves.

## Evaluation-policy evidence

Issue #9193 is useful as a second coverage angle, with a strict access boundary. This wave used 15 individual examples
from four in-distribution evaluations—HumanEvalPlus, MBPPPlus, MMLU-Pro, and FinanceBench—as held-out coverage and
routing probes. The generator did not see them. AAII 4.3 remained metadata-only; no individual question or score was
inspected or used to model tasks.

The held-out probes exposed real omissions: standalone code synthesis in C01, factual QA in C25, and financial-
statement extraction, ratio, and variance analysis in C27. They also prevented irrelevant expansion: a consumer-
activism question and a generic autoregressive-process question were not treated as finance coverage. No held-out
example applied directly to C06 or C23, so those branches have insufficient evaluation evidence.

Eval samples should remain a review and routing holdout. They should not become the curriculum skeleton, task text,
or a substitute for TaskTrove and educational guideposts. Out-of-distribution evals contribute metadata only until
the policy permits more.

## Global mapping experiment

The mapping sample contains 48 TaskTrove tasks disjoint from discovery, eight selected for each wave-two cohort, plus
the 15 evaluation probes. Luna annotated semantic keys in batches of eight. A separate Luna pass assigned blind
reference labels from model-visible text after task IDs and order were anonymized. `text-embedding-3-small` embedded
each task key once and 292 section anchors; the SQLite cache was reused across all comparisons.

The source cohorts were much less pure than their selection labels suggested. All eight C01 tasks belonged in C01,
but only six C06, one C07, one C23, five C25, and zero C27 tasks belonged to their nominal area. The remaining tasks
were mostly generic QA or structured extraction; two were outside the nine-curriculum catalog. Keyword and source
filters therefore cannot replace task-level routing.

The flat global baseline retrieved an acceptable section first for 15 of 61 in-scope tasks, within the top three for
19, and within the top five for 22. The 15 evaluation probes did better at 8, 10, and 11 respectively. The baseline
was especially revealing for C25: the blind labeler placed 26 tasks under Knowledge Application & QA, while the
semantic key's `subject` field named chemistry, law, physics, or another knowledge domain. None retrieved its exact
C25 leaf in the baseline top five.

This is partly an evaluation-design failure. The nine-area catalog mixes overlapping domain curricula such as finance
and databases with practice curricula such as question answering and technical writing. A schema-extraction question
can legitimately receive both a database/data label and a supplied-context QA label. Forcing one global primary
section made the blind reference labels artificially exclusive; the absence of ambiguous labels is not evidence that
the catalog is disjoint.

A second Luna annotation split task mechanic from subject domain. Using only the mechanic projection improved the
global result to 18/61 top one, 30/61 top three, and 31/61 top five. Naively mixing domain scores for domain curricula
and mechanic scores for practice curricula reached only 13/61, 18/61, and 26/61 because the two projections were not
calibrated and exact section anchors remained weak. The experiment supports a faceted, multi-label assignment
architecture; it does not support another embedding threshold or source-specific rule.

## Architecture after this wave

The pipeline remains four separable stages:

1. **Evidence inventory.** A versioned broad inventory supplies guideposts. Small, source-diverse TaskTrove samples
   and permitted held-out evals expose missing coverage and bad boundaries.
2. **Curriculum generation and review.** One-off agents generate a graph for one area. Independent reviews apply the
   shared rubric; only recurring failures change the rubric. Checked-in outputs and reviews are the durable artifacts.
   This workflow does not require a production model API.
3. **Task annotation.** Luna reads each task once in batches of 8–16 and emits a content-addressed semantic key. The
   next key version must separate `subject_domain`, `task_mechanic`, hardest operation, required operations, and answer
   form. Each projection is embedded and cached once.
4. **Assignment.** A catalog entry declares which projection a curriculum graph consumes. Retrieval first selects
   zero or more applicable graphs, then ranks sections independently inside each graph. Output is multi-label with an
   assigned, ambiguous, or out-of-scope status per graph. A curriculum edit re-embeds changed anchors and reruns matrix
   ranking; it does not reannotate tasks.

Generation probes and assignment anchors should also separate. Entry and representative probes test learnability and
task generation. Mapping anchors need several representative phrasings of the section's decisive operation. Reusing
the two generation probes as the only anchors produced poor section-local retrieval even when the correct subject was
known.

## Consistent weaknesses

The reviewers repeatedly found four problems:

1. Broad leaves grouped tasks that shared an answer form or umbrella domain but required different central operations.
   Examples include direct calculations versus scheduling, sales-tax rules versus capitalization rules, and time-
   weighted returns versus currency attribution.
2. Prerequisite lists often encoded workflow order. If a dependent task can consume a completed forecast, chart, or
   validation report without being able to produce it, the upstream capability is not a prerequisite.
3. Internal-node tasks often concatenated one child deliverable from each branch instead of exercising a coherent
   cross-child decision.
4. Evidence is uneven. C23 has no applicable held-out eval probe, 31 of C27's 34 leaves lack direct eval evidence, and
   several discovery cohorts were dominated by tasks outside their nominal area.

The repaired curricula still exhibit these issues in narrower places. C01 retains a broad standalone/system synthesis
boundary; C06 has schema-validation/data-quality and record-integration/linkage collisions; C07 retains broad table,
experimental-design, longitudinal, and survival leaves; C25's closed-book factual QA leaf spans unrelated specialist
knowledge; and C27 retains composite tax, performance, factor-risk, and disbursement leaves.

## Wave-three disposition

Wave three completed the multi-label relabeling, split semantic keys into domain, mechanic, and operation projections,
tested graph and section anchors separately, and audited six prerequisite edges. The reviewed curricula are in
`../curriculum.yaml`; the current architecture and results are summarized in the parent README.

Task-level JSON and JSONL files are data products rather than source configuration, so they are not retained in the
repository. Future annotations, anchors, vectors, calibration labels, and assignments belong in versioned Parquet
artifacts in GCS, with curated public snapshots on Hugging Face when useful.
