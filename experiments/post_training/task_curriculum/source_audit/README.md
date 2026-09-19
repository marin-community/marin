# Bounded source-audit v3

V3 applies source and real-task evidence as small patches to the cross-domain v1 catalog. It does not regenerate a
subject from source headings. In a five-subject comparison, v3 was preferred over v1 and source-first v2 in every
anonymous review. Its mean holistic score was 93.4, compared with 85.8 for v1 and 72.0 for v2 in the same calls. Blind
fit was 107/120 for v3, 106/120 for v1, and 111/120 for v2. The broader v2 graphs fit four more sampled tasks than v3,
but their catch-all sections, false prerequisite edges, and weaker probes reduced structural quality.

The experimental 45-subject v3 catalog replaces D01, D02, D08, D17, D31, and D41. It has 2,285 nodes: 1,885
capabilities and 400 groups. V1 remains canonical because D01 and D41 retain local structural blockers and D02 keeps
unrelated baseline blockers.

## Method

The bounded repair workflow is:

1. Freeze the current catalog, subject inventory, detailed rubric, evidence brief, and evaluation tasks.
2. Give the repair role v1 plus a compact list of source-supported omissions and review defects. Hide v2 and all blind
   tasks.
3. Require the smallest patch justified by a named operation-family gap or mutual-confidence failure. Preserve every
   unaffected section exactly. Treat course headings and course order as evidence, not section or prerequisite
   definitions.
4. Return changed sections only. Merge them mechanically into the baseline and verify the unchanged-section and
   changed-section partitions. This avoids asking a model to reproduce a large unchanged graph.
5. Present v1, v2, and v3 under anonymous labels to one fresh high-reasoning reviewer. Reuse the same frozen blind
   tasks in a separate fit judgment. A wide capability that absorbs tasks can improve fit while failing the holistic
   mutual-confidence test, so neither signal substitutes for the other.
6. Publish the inputs, patches, evaluator inputs, results, logs, and hashes under a new immutable experiment prefix.
   Promote a later catalog only after unresolved blockers are dispositioned.

The five source-audit subjects were Mathematics & Statistics (D01); Mechanical, Aerospace & Marine Engineering
(D08); Medicine & Clinical Care (D17); Education & Learning Sciences (D31); and Sports, Exercise & Recreation (D41).
Each repair started from v1 and a frozen source brief produced by the v2 survey. The repair role could not read v2 or
the blind tasks.

| Subject | Holistic v1 / v2 / v3 | Blind fit v1 / v2 / v3 | V3 status |
| --- | ---: | ---: | --- |
| D01 Mathematics & Statistics | 80 / 73 / 88 | 16 / 21 / 16 | revise; one geometry split remains |
| D08 Mechanical, Aerospace & Marine Engineering | 88 / 76 / 96 | 23 / 24 / 23 | pilot ready |
| D17 Medicine & Clinical Care | 90 / 65 / 96 | 23 / 23 / 24 | pilot ready |
| D31 Education & Learning Sciences | 84 / 72 / 96 | 24 / 24 / 24 | pilot ready |
| D41 Sports, Exercise & Recreation | 87 / 74 / 91 | 20 / 19 / 20 | revise; four overlap boundaries remain |

These scores are paired within one three-way review call. They differ from the v2 report's earlier paired scores
because model judgments vary across calls. Compare candidates within a row and run, not absolute scores across runs.

## Why source-first v2 lost

V2 improved coverage by turning course and professional outlines into complete subject graphs. The same move degraded
four structural dimensions. Across the original v2 comparison, coverage rose by 2.0 points on average, while mutual
self-confidence fell by 5.4, progression and epsilon continuity fell by 5.8, observable boundaries fell by 1.8, and
probe quality fell by 1.8.

The recurrent causes were concrete:

- Course headings grouped operations that require different solver loops and correctness contracts. Examples include
  row reduction with eigenspaces, yielding with fracture, pathology with ECG interpretation, and quantitative studies
  with ethnography.
- Course order became prerequisite edges even when a completed upstream artifact removed the dependency. Every seven
  new D08 edges failed that counterfactual.
- Public outlines named exercises but did not supply enough literal data for executable probes. V2 therefore emitted
  placeholder-dependent or underdetermined tasks.

V2's higher blind fit is still useful evidence. A broad capability often absorbs more tasks. The holistic reviewer
then checks whether mastery transfers across that capability. V3 keeps source-supported additions while restoring
operation-level boundaries.

## D02 real-task repair

The v2 TaskTrove audit found four shell and tool tasks with no capability home. V3 expanded that check to 32 real
Unix and SuperUser tasks. A curriculum-blind curation role identified five repeated families: interactive command-line
configuration, execution-environment resolution, file and archive operations, pipeline and result handling, and
process startup and orchestration. The repair saw 16 discovery tasks. Sixteen holdout tasks remained hidden.

The first patch split two overbroad additions after holistic review. The final branch contains seven capabilities:
interactive configuration, environment resolution, file-tree mutation, archive transfer, record transformation,
stream and exit-status control, and process orchestration. V1 fit 0/32 on this deliberately gap-focused sample. V3
fit 30/32 overall and 16/16 on the unseen holdout. The two remaining gaps were a mixed tmux-control and file-editing
request and a Windows browser-instance dispatch request. The repair rejected a generic computer-use catch-all.

The complete D02 score rose from 70 to 81. Four remaining blockers belong to pre-existing D02 sections, including
graph traversal, runtime-memory facets, generative systems, and several probes. The new command-environment branch had
no blocker after refinement.

## Recommendation

Use the bounded repair pattern for later catalog revisions. Full three-source research is most useful for
practice-heavy, low-confidence, or repeatedly unmapped subjects. A brief source audit or real-task sample is enough
when it finds no coherent missing family. Keep blind fit as a coverage signal and the holistic rubric as the boundary
and progression gate.

Before promoting v3, split the D01 geometry catch-all, consolidate the four D41 overlaps, and disposition the inherited
D02 blockers. The current evidence does not measure model learning or prerequisite transfer. Luna failure thresholds
and mapping behavior remain rough diagnostics; this work does not propose small-model training.

## Artifacts

- Experimental catalog: `s3://marin-us-east-02a/marin/task-curriculum/experiments/2026.09.19-bounded-source-audit-v3-f0edd8ce5141/curriculum.yaml`
- Catalog SHA-256: `faa15da8458147bb5415b9213fc76f95963215a25ad889d3b9215687cae9993b`
- Comparison SHA-256: `f0edd8ce5141a4dc01dbeb26dbd55fdd0b438db9fd10d74e3b7ab2e4c8916730`
- Evidence SHA-256: `25ff4b75fb7b08e6e69fa4b8fd29f6f371b84a435b35af8253ed3cb49e0482c2`
- Viewer: [revision 4](https://applets.marina.oa.dev/a/67f69132-2ef4-4c9e-b8b5-77cabd126442/v/4/)
- Model: `gpt-5.6-sol`, high reasoning effort, provider-controlled sampling

The evidence archive contains the immutable inputs, repair briefs, patch outputs, anonymous candidate graphs, reviews,
fits, D02 discovery and holdout material, schemas, one-off runners, logs, and a file-level hash manifest.
