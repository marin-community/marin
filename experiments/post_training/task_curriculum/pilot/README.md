# Three-area curriculum pilot

## Result

The one-off generation and review loop improved all three curricula, but mapping is not ready to assign TaskTrove
rows confidently. The final reviews reached 86 for Shell, 95 for Mathematics, and 88 for Office. Across 21 blind
in-scope mapping tasks, cached embeddings retrieved an acceptable section first for 16 and within the top three for
19. Three Shell tasks were labeled out of scope; their similarity scores overlapped the in-scope scores, so a
single distance threshold cannot identify a null assignment.

Each cohort has eight tasks, and the Office cohort uses one highly coherent calendar source. The results describe
these pilot fixtures only.

## Method

The broad inventory came from the September 3, 2026 TaskTrove competency chart. C03 Shell & Systems
Administration, C14 Mathematics & Formal Reasoning, and C21 Office Productivity & Documents were chosen to cover
procedural, formal, and stateful work. One-off agents received the selected inventory area, the rubric, two hierarchy
levels, two sample instructions per section, and eight discovery tasks. Independent agents reviewed each result.

The first review round found the same two structural problems in different subjects:

- internal nodes duplicated leaf task distributions instead of training cross-child synthesis;
- prerequisites described useful background rather than a capability required by every representative dependent
  task.

`rubric.md` now makes those tests explicit. It also requires a routing rule for overlapping sections and preserves
the original mutual-predictiveness test for leaf granularity. The first-round rubric is retained as
`rubric_v1.md`.

| area | review sequence | final checked curriculum | remaining weakness |
|---|---:|---|---|
| C03 Shell & Systems Administration | 78 → 86 | `C03/curriculum_v2.json` | process/resource and authentication leaves still combine weakly predictive operations |
| C14 Mathematics & Formal Reasoning | 72 → 79 → 89 → 95 | `C14/curriculum_v4.json` | integer enumeration and structural proof remain broad; discovery evidence covers few leaves |
| C21 Office Productivity & Documents | 84 → 88 | `C21/curriculum_v2.json` | calendar placement and calendar lifecycle/state restoration may warrant separate leaves |

Mathematics needed a third generation pass because defining internal nodes as synthesis-only made their children no
longer refinements of the parent. The revised rule is: the parent outcome contains the child outcomes, while the
parent's sample tasks specifically exercise cross-child synthesis. Version 4 only repairs two under-specified logic
sample instructions found by the version 3 reviewer; the final reviewer found no new defect in those changes.

## Mapping evaluation

Luna annotated each mapping cohort in one eight-task batch. Each annotation stores the content hash, model,
`semantic-key-v1` prompt identity, and a curriculum-independent key. Initial Office annotations incorrectly treated
narrative details such as nursing or warehouse work as the subject. The corrected definition uses the operational
domain—the work the solver performs—so all eight became calendar scheduling and state management without consulting
the curriculum.

`text-embedding-3-small` embedded semantic keys and section anchors. Each section has one anchor per sample task,
combining subject name, section name, outcome, and task instruction. Section similarity is the maximum cosine over
its anchors. NumPy computes similarities in bounded task-row batches; the SQLite cache is keyed by embedding text
and model. Blind reference labels were created without viewing semantic keys or mapping results.

| area | in scope | acceptable top 1 | acceptable top 3 | out of scope |
|---|---:|---:|---:|---:|
| C03 | 5 | 2 | 5 | 3 |
| C14 | 8 | 6 | 6 | 0 |
| C21 | 8 | 8 | 8 | 0 |
| total | 21 | 16 | 19 | 3 |

Office confirms that a coherent mechanic can map cleanly on fresh rows: the eight held-out calendar tasks were
selected from the 3.57 GiB Clean release after excluding all discovery paths, and all eight retrieved the scheduling
leaf first. This does not test the other nine Office leaves.

Shell's top-three result is more encouraging than its top-one result. Both storage tasks retrieved the correct
branch but confused a synthesis parent with its leaves, and a routing task ranked the correct leaf second behind its
parent. The three out-of-scope tasks had top similarities between 0.403 and 0.476, inside the
0.361–0.484 range for in-scope Shell tasks. A per-curriculum cosine cutoff cannot separate them.

The two Mathematics misses expose a curriculum boundary rather than a source issue. Extremal coloring and
constant-intersection set-family tasks embed near combinatorial enumeration, while the blind labels route their
requested maximum plus upper-bound certificate to optimization or structural proof. The mapping procedure needs
either anchors that express the decisive operation more directly or a curriculum revision that makes this
cross-cutting boundary reproducible.

## Artifacts

Each area directory contains the generated curricula and independent reviews. The final mapping inputs and results
are:

- `mapping_manifest.jsonl`: TaskTrove source and path identities for the eight mapping tasks;
- `task_annotations.jsonl`: Luna semantic keys and content/model identities;
- `mapping_references.jsonl`: blind acceptable-section labels;
- `mappings.jsonl`: top-three candidates, cosine scores, curriculum version, annotation identity, and embedding
  model.

The source release is `s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9/`. C21 holdouts use SHA-256 ordering
over `curriculum-mapping-holdout-v1\0 + source + \0 + path` after excluding discovery paths. Exact source/path rows
are retained for all cohorts, so the selection can be reconstructed without checking in raw tasks or the Parquet.

## Next evaluation

1. Build one candidate index across several curricula plus reviewed null cases. Calibrate similarity and margin only
   after freezing balanced member, close-neighbor, ambiguous, and out-of-scope fixtures.
2. Test whether embedding decisive-operation anchors or restricting assignment to leaves improves Shell top-one and
   the Mathematics extremal boundary. Do not add source-specific rules.
3. Expand discovery and mapping evidence to unsupported leaves before splitting broad sections. Review findings
   propose hypotheses; transfer has not been measured.
4. Once key and assignment behavior stabilizes, annotate TaskTrove in 8–16-row model batches, persist keys by task
   content and annotation identity, and embed each key once in sharded artifacts. Curriculum revisions then rebuild
   only section anchors and assignments.
5. Generate complete TaskSpecs and correctness contracts separately. The two instructions per section are cheap
   curriculum probes. Verifier design and task-validity measurement remain separate.
