# Task curriculum experiment history

This file summarizes the completed experiments that shaped the current schema, rubric, prompts, and cross-domain
catalogs. Detailed task-level inputs and outputs remain in the cited external artifacts rather than the repository.

## Three-area pilot

The first experiment generated and independently reviewed Shell, Mathematics, and Office curricula from the TaskTrove competency chart and eight discovery tasks per area. Reviews rose to 86, 95, and 88. Embedding retrieval found an acceptable section first for 16/21 in-scope tasks and in the top three for 19/21, but three out-of-scope Shell tasks overlapped the in-scope similarity range. The experiment established the operational-domain semantic key, exposed duplicated internal nodes and workflow-order prerequisites, and showed that a single cosine threshold could not provide a null assignment.

## Wave two: six-area scale-out

Wave two added Software Engineering, Databases, Data Analysis, Technical Writing, Knowledge QA, and Finance. The repaired graphs contained 233 sections, bringing the nine-area catalog to 292, and disproved the early estimate of 15–25 sections per macro. Fifteen answer-hidden evaluation-policy examples and 48 fresh TaskTrove tasks exposed missing capabilities and showed that domain and practice graphs overlap. Flat global retrieval found an acceptable section first for 15/61 in-scope tasks; splitting task semantics into subject domain and task mechanic improved the mechanic-only result to 18/61 and motivated faceted, multi-label routing.

## Wave three: routing and prerequisite audit

Wave three relabeled 63 tasks with 81 valid graph memberships, including 16 multi-label tasks. Declared per-graph projections reached 0.627 micro average precision, while blind graph anchors raised it to 0.832; with the correct graph supplied, bootstrap section anchors placed an acceptable section first for 30/81 memberships and in the top three for 47/81. A six-edge Luna audit rejected requirements-to-design-proposal and journal-entry-to-ledger-posting under the completed-artifact counterfactual. The result fixed the two-stage domain/mechanic routing architecture and kept mapping outside the curriculum promotion gate.

## Wave four: stricter whole-subject review

Wave four generated NLP, Computational Science, and Security curricula from guideposts plus eight TaskTrove examples each. A lightweight reviewer scored the drafts 91, 83, and 88, but a reviewer that challenged distant task pairs, every prerequisite, parent synthesis, and evidence breadth rescored them 77, 79, and 83. One repair pass remained provisional. The experiment added decisive-operation evidence filtering, concrete counterexample pairs for boundary findings, parent-containment checks, and the rule that different tools or subfields alone do not justify a split.

## Wave five: independent generation and review

Wave five used isolated high-reasoning generation and review roles for NLP, Computational Science, and Security. Initial scores were 90, 87, and 87 with one blocker each; repaired versions scored 79, 83, and 90, leaving only Security ready. Blinded Luna placed all 36 curriculum-derived boundary tasks, yet structural review still rejected two subjects. Translation and PDE repairs showed that knowledge regimes need explicit sampling facets and that closing one coverage gap can create a non-transferable omnibus capability.

## Wave six: groups and sampling facets

Wave six introduced hierarchy-only groups and reusable sampling facets, then regenerated NLP and Computational Science. NLP reached 92 with eight capabilities, three groups, and one language-direction facet. Computational Science required three reviews, moving from 81 to 86 to 92 as distinct solver loops were split into 32 capabilities, seven groups, and ten facets. Luna placed 12/12 tasks for every version, confirming that placement clarity does not establish mutual self-confidence or epsilon continuity.

## Wave seven: reproducible blind fit

Wave seven added strict models and cross-artifact validation for blind tasks, fit judgments, evidence accounting, reviews, and promotion. Software Testing, Robotics, Legal, and Bioinformatics finished at holistic scores of 92, 92, 95, and 91 with blind fits of 24/24, 23/24, 23/24, and 24/24. The same frozen 24-task sets were reused across repairs, which demonstrated that strong fit can coexist with defective prerequisites and probes. The evidence is Loom artifact `curriculum-wave7-evidence`, revision 1, SHA-256 `49fc55b3fd0a82ca1d2e62b4435ebfa28ebadfd101382a97b0c5d70779e73e6d`.

## Wave eight: evidence-aware scale-out

Wave eight added six C-series subjects using 16 TaskTrove discovery tasks and 24 curriculum-blind tasks per subject, plus permitted answer-hidden MMLU examples for three subjects. The resulting 22-subject catalog covered 107/151 guideposts with 697 nodes. Repairs split construction from diagnosis and supplied missing probe inputs, while reviews kept evidence confidence separate from structural readiness. Four final repairs were not re-reviewed, so their prior scores remained provisional. The evidence is Loom artifact `curriculum-wave8-evidence`, revision 1, SHA-256 `52b9a5907765ca21602dfebef4f4782910f93d6d16b714899105f275fb53dd75`.

## Wave nine: complete C-series baseline

Wave nine added the remaining 12 C-series areas. The retired 34-root catalog contains 959 nodes, including 854 capabilities, and covers all 151 guideposts. One new subject was `pilot_ready`; the other 11 retained documented blockers after bounded repairs. Recurrent failures were noun-based grouping, under-specified probes, composite blind tasks, and sparse non-code evidence. The immutable catalog is `s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.18-c1821f8f67c1/curriculum.yaml`; Loom artifact `curriculum-wave9-evidence`, revision 4, has SHA-256 `db23ae13e147cc9eac4193274902ec8ec6ad00f451f915006f52621e7724d0c7`.

## Wave ten: cross-domain v1

Wave ten replaced the practice-heavy C-series roots with 45 D-series subjects spanning academic, professional, service, sports, safety, transport, and practical domains. All subjects received one generation, blind-task, review, fit, and bounded-repair cycle. The selected catalog contains 2,252 nodes, including 1,853 capabilities, and fits 993/1,072 valid frozen tasks (92.6%); its mean holistic score is 81.3, three subjects pass promotion, and seven retain blocking repeated gaps. The canonical catalog is `s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.18-72a763b98f9e/curriculum.yaml` with SHA-256 `72a763b98f9ecf7f8f598b788c4f59e7ace213c01a30b403768a8f8f16f55382`; its evidence archive SHA-256 is `cd127af6afbffc188e94359d2995ad833e8a54ce8d0962dc036592e0d02cfc74`.

## Source-guided v2

A five-subject survey tested public course sequences, textbook exercise families, and professional standards as
direct generator inputs. Source-first graphs improved sampled fit from 99/120 to 106/120 but reduced the paired
holistic mean from 84.4 to 71.6 by copying course groupings and sequence into capabilities and prerequisites. The
immutable evidence is
`s3://marin-us-east-02a/marin/task-curriculum/experiments/2026.09.19-source-survey-v2-148820c9ecae/`.

## Bounded source-audit v3

V3 applied source evidence as bounded patches to D01, D08, D17, D31, and D41. It was preferred to v1 and source-first
v2 in all five anonymous comparisons, with a mean holistic score of 93.4. A separate D02 repair used 16 discovery and
16 holdout TaskTrove tasks to add seven command-environment capabilities; the final branch fit 30/32 tasks and all 16
holdout tasks. The immutable catalog and evidence are
`s3://marin-us-east-02a/marin/task-curriculum/experiments/2026.09.19-bounded-source-audit-v3-f0edd8ce5141/`.

## Production bounded repair: cross-domain v2

The production pass started from v3 and attempted one bounded repair on 42 roots, leaving D08, D17, and D31
unchanged. Anonymous same-call review preferred 41 repairs and tied D32, whose baseline was retained. The paired mean
rose from 78.36 to 92.81. The selected catalog contains 2,405 nodes: 1,999 capabilities and 406 groups. Twenty-four
subjects are `pilot_ready`; 21 are preserved as provisional rather than recursively optimized. Frozen fit is
1,001/1,080, but is explicitly a reused diagnostic rather than a fresh holdout because the earlier baseline repair
had seen an earlier fit summary.

A separate routing audit placed all 24 practical TaskTrove shell, repository-workspace, and bug-repair tasks in D02.
Twenty-one mapped to one complete capability; three distinct gaps did not justify another catch-all. Evaluation-policy
and AAII metadata found subject homes across all 45 roots, while ten entries were correctly treated as reusable task
mechanics rather than subjects. The canonical catalog is
`s3://marin-us-east-02a/marin/task-curriculum/catalogs/2026.09.19-2578589fb0de/curriculum.yaml`, with SHA-256
`2578589fb0de23b179765e6eba999730c7117b9f083140b266ccc391dde9685d`. Its evidence archive SHA-256 is
`f9d495bcf5afd3b2ae01eb8de60abc9c99e00abfbcd8e90709cce7424d50d74a`.
