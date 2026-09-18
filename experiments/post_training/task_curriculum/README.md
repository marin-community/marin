# Task curriculum

This experiment turns a broad subject inventory into trainable curricula and maps tasks onto reviewed sections. It
keeps curriculum design separate from task correctness:
TaskCompendium owns model-visible task semantics and private verifier contracts; a curriculum describes observable
capabilities, boundaries, examples, and prerequisites.

`curriculum.yaml` is the canonical catalog. It contains 16 reviewed curriculum graphs and 493 globally unique nodes:
464 trainable capabilities and 29 organizational groups. The Markdown reports under `pilot/` and `wave2/` preserve
the experimental evidence; runtime assignment loads only the YAML catalog.

## Inputs

- `subject_inventory.json` is the September 3, 2026 TaskTrove competency chart reduced to 34 macro areas and 151
  micro guideposts. Its labels help find omissions; generated sections must still satisfy the curriculum rubric.
- `rubric.md` defines the cross-subject review criteria.
- `workflow.md` is the resumable operating procedure, including the exact generator, blind-task, fit-judge, holistic-
  review, and optional Luna prompts.
- A small set of model-visible TaskTrove tasks supplies concrete discovery evidence. In-distribution examples listed
  by the evaluation policy are held-out coverage probes. Out-of-distribution evaluations contribute domain and task
  format metadata only. Source names, solutions, and verifier implementations are excluded from generation.
- TaskCompendium's subject, competency, task-shape, artifact, context, interaction, and state facets are optional
  vocabulary for describing task semantics.

## Curriculum iteration

One-off agent runs produce curricula and reviews. Their reports record the evidence, and reviewed changes are promoted
into the canonical YAML. For each selected macro area:

1. Give a high-reasoning generator the inventory area, rubric, a maximum tree depth, and representative tasks. Before
   emitting the curriculum, it enumerates operation families, tests the most distant permitted pair for every
   capability, and audits proposed prerequisites with the completed-artifact counterfactual. Every section includes
   an `entry` probe that isolates the smallest prerequisite delta and a `representative` probe for the full outcome.
2. Validate the JSON with `Curriculum` and `Curriculum.check_generation_contract`.
3. Give the curriculum and rubric to an independent high-reasoning reviewer. The reviewer returns a score, verdict,
   concrete findings, and proposed rubric changes in one call for the complete subject curriculum.
4. When an uncertain boundary or difficulty claim could change the review, the reviewer may generate focused tasks
   and send them to a blinded Luna model in batches of 8–16. These diagnostics are optional and do not replace the
   whole-subject judgment.
5. Compare findings across subjects. Mutual self-confidence and prerequisite continuity are blocking criteria even
   when the numeric score is high. Revise the rubric or generation instructions only for problems that recur, then
   generate another version.

In parallel with curriculum generation, a curriculum-blind Sol/high agent generates
`max(24, 2 * guidepost_count)` diverse tasks from the subject definition. A separate judge reports how many have an
exact home, an ambiguous but complete home, a coverage gap, or are invalid. This blind-fit X/Y is a sampled coverage
diagnostic reported separately from the holistic score. Freeze the task set across curriculum repairs.
During initial calibration, a systematic operation-family gap blocks promotion; `workflow.md` defines the complete
independence and reporting rules.

The hierarchy has two node kinds. A `capability` is a trainable outcome and the only kind that may receive task
assignments, declare prerequisites, or carry entry and representative probes. A `group` is an organizational scope;
it has no outcome, probes, or prerequisite edges. Use a group when a parent would otherwise be a routing menu or an
artificial bundle of child deliverables. Capabilities may still contain narrower capabilities when their
representative probe exercises a natural cross-child synthesis.

Reusable knowledge dimensions belong in a capability's `sampling_facets`. A facet declares an axis such as language
direction, jurisdiction, or scientific-model family; its concrete value is recorded with task annotations outside
the catalog. The effective task set is a capability plus fixed facet values. This makes mutual self-confidence
testable under one knowledge regime without creating one curriculum node per language pair or jurisdiction.

The checked-in `pilot/` and `wave2/` reports summarize the one-off results. Promote reviewed changes into
`curriculum.yaml`; do not add another runtime curriculum or check in task-level experiment artifacts.

The detailed rubric is a development tool for a small subject sample. Routine scale-out uses its compact five-part
score in one high-reasoning call per curriculum version. Sampled Luna placements and failures provide optional
boundary and difficulty evidence; this workflow does not run model-training experiments.

## Task mapping

Task mapping uses a curriculum-independent semantic key and two routing stages:

```text
model-visible TaskSpec -> semantic key -> cached domain, mechanic, and operation embeddings
curriculum graph       -> declared membership facet -> zero or more graph candidates
selected graph         -> operation embedding       -> capability candidates within that graph
```

The key records `subject_domain`, `task_mechanic`, requested result, hardest operation, required operations, and
answer form. `subject_domain` is the broad knowledge or tool environment required by the decisive operation.
Narrative content and output format do not belong in it. `task_mechanic` is a domain-neutral operation family such as
implementation, repair, supplied-context extraction, calculation, explanation, recall, or tool-state mutation. The
hardest and required operations rank capabilities after graph selection. Groups never receive assignment anchors or
task mappings.

Annotate each task once in model batches of 8–16. Persist the task-content hash, annotation prompt version, and model.
Do not include curriculum labels in the prompt. `cli.py` caches each embedding by text and model, derives bootstrap
anchors from curriculum outcomes and includes, accepts separate assignment anchors, and ranks with NumPy in bounded
row batches. A curriculum edit re-embeds changed anchors and reuses task vectors. A semantic-key schema change requires
reannotation; an embedding-model change requires re-embedding.

Run a pilot mapping with:

```bash
uv run python -m experiments.post_training.task_curriculum.cli \
  --annotations /tmp/task-annotations-000.jsonl \
  --catalog experiments/post_training/task_curriculum/curriculum.yaml \
  --assignment-anchors /tmp/assignment-anchors.jsonl \
  --cache /tmp/curriculum-embeddings.sqlite \
  --embedding-model text-embedding-3-small \
  --top-k 3 \
  --output /tmp/C14-mappings.jsonl
```

The output contains a membership score and section candidates for every graph. Thresholds are calibrated independently
per graph on a frozen member, close-neighbor, overlap, and out-of-scope sample; scores from distinct projections are
not comparable. The mapper has no source-specific rules or per-task exceptions. Mapping is a diagnostic signal, not a
curriculum blocker; roughly 70% reasonable placement is adequate for initial scale-out.

For the full TaskTrove run, materialize the approximately 4 GB clean release and shard annotation and embedding work.
Keep these artifacts outside the curriculum YAML:

- semantic keys by task-content hash, annotation schema, prompt, and model;
- task vectors by semantic-key hash and embedding model;
- assignment anchors and anchor vectors by catalog and embedding model;
- graph calibration policies by catalog, anchor set, and labeled fixture; and
- assignments by task vector, catalog, anchor set, and calibration policy.

Use Parquet shards in GCS for the internal working set. A curated public snapshot can be published to Hugging Face.
The local SQLite cache is only a single-node development cache.

## Wave-three findings

The frozen regression fixture contains 63 tasks and 81 independently labeled graph memberships. Fourteen tasks have
two valid graph labels and two have three, confirming that a single primary curriculum is ill posed. With corrected
facet annotations, the declared per-graph projection reached 0.627 micro average precision for membership. Applying
`subject_domain` to every graph reached 0.574; applying `task_mechanic` to every graph reached 0.421. The declared
projection retrieved at least one valid graph first for 51/63 tasks and retrieved all valid graphs in the top three
for 45/63. C03 and C21 had no positive examples, so this is a regression fixture rather than a threshold-calibration
set. Eight blind mapping anchors for each of the two practice graphs raised micro average precision to 0.832. With
those graph anchors, at least one valid graph ranked first for 56/63 tasks and all valid graphs appeared in the top
three for 57/63.

Section-local retrieval remains the larger weakness. With bootstrap outcome/include anchors and the correct graph
given, an acceptable section appeared first for 30/81 memberships and in the top three for 47/81. Assignment anchors
generated blind for all C23 and C25 sections reduced top-three retrieval to 43/81, so they were rejected. Section
anchors must be derived or selected against a calibration split and evaluated on a fresh holdout before full-corpus
mapping.

A six-edge prerequisite audit generated two prerequisite representatives, two dependent entries, and two dependent
representatives per edge. Batched Luna review passed the transfer and added-capability checks on all six. It rejected
two edges under the completed-artifact counterfactual: requirements-to-design-proposal and journal-entry-to-ledger-
posting. The canonical catalog removes those edges and supplies the upstream artifacts in the dependent probes.

## Wave-four procedure test

Wave four generated curricula for NLP & Language Technology, Computational Science & Simulation, and Security &
Adversarial Analysis from their guideposts and eight TaskTrove examples each. The first evidence classifier produced
topic-driven false positives: safety-refusal tasks appeared to be security work, shell parsing appeared to be NLP,
and conceptual physics appeared to be simulation. The discovery rule now classifies the behavior required from the
answer and requires an actual computational model or method for computational-science membership.

The first lightweight review prompt scored the three generated curricula 91, 83, and 88 and assigned high confidence.
Manual inspection found that result too permissive. A revised prompt challenged every leaf's most distant task pair,
every prerequisite edge, internal-node synthesis, and confidence from evidence breadth. It rescored the same versions
77, 79, and 83, all `revise`. One repair pass produced 7, 21, and 12 sections; fresh single-call reviews scored them
80, 81, and 83 with medium confidence and remaining blockers.

The recurring defects were one-leaf-per-guidepost generation, broad specialist leaves, workflow-order prerequisites,
and parent probes that bundled independent child deliverables. The stricter reviewer also tended toward recursive
splitting based only on different tools or subfields. A blocking split must now cite two concrete probes with distinct
central operations and explain how the split changes sampling or evaluation. None of the three provisional curricula
was promoted during wave four.

## Wave-five scale-out test

Wave five used Sol/high for one complete-subject generation and one independent complete-subject review of NLP,
computational science, and security. Generation first enumerated operation families, challenged every leaf with a
counterexample pair, and audited candidate prerequisites. The first reviews scored 90, 87, and 87. Each found one
blocker. Targeted repairs then scored 79, 83, and 90. Security passed with no blockers and is promoted into the
canonical catalog.

The two failed repairs expose scale-out problems. Restricting translation to the one observed English-to-Chinese
direction restored mutual confidence but lost guidepost coverage. Knowledge-dependent capabilities need an explicit
sampling and evaluation facet, such as language direction, when the central operation transfers but background
knowledge does not. Adding one general-PDE leaf closed a coverage gap but combined elliptic, diffusion, reaction, and
wave operations and created ambiguous boundaries with transport simulation. A repair must receive another complete
subject review; closing the named blocker is insufficient.

The independent reviewers generated 12 boundary tasks per subject and hid their intended sections from Luna. Luna
placed all 36 into an accepted capability or justified alternative. These curriculum-derived probes confirm that the
named boundaries are legible, but they are easier than held-out data and do not override structural blockers.
In-distribution examples permitted by the [evaluation policy](https://github.com/marin-community/marin/issues/9193)
should supply a second held-out coverage view. Out-of-distribution evaluations contribute metadata only under that
policy.

These failures motivated the capability/group distinction and structured sampling facets added in wave six.

## Wave-six schema and repair test

Wave six added hierarchy-only groups and reusable sampling facets, then regenerated NLP and computational science.
The NLP curriculum uses eight capabilities, three groups, and one `language_direction` facet. Its independent review
scored 92 with no blockers. The computational-science curriculum needed three complete reviews. Scores moved from
81 to 86 to 92 as facets that concealed different solver operations were replaced by separate capabilities. The
final graph contains 32 capabilities, seven groups, and ten facets.

The C15 repairs split conservative-flux updates from incompressible pressure-velocity coupling, fixed-Hamiltonian
eigensolution from nonlinear self-consistency, linear finite-element solution from nonlinear incremental equilibrium,
and ordinary diffusion integration from stiff split reaction-diffusion. This yields 32 capabilities for one macro
area. The rubric sets density from transfer and epsilon continuity; it has no fixed per-subject target.

Blinded Luna placement scored 12/12 for NLP and for all three C15 versions. The two failed C15 versions therefore
show that placement clarity does not establish mutual self-confidence or entry-to-representative continuity. The
independent whole-subject review remains the acceptance gate. A sampling facet must also pass a distant-value test;
it cannot hide a different solver loop, state transition, or evaluation contract.

Both final reviews have medium confidence because the sampled tasks reach only part of each graph and no applicable
held-out evaluation examples were available. In-distribution examples permitted by the evaluation policy should be
added as coverage evidence during scale-out. They need not become curriculum nodes or source-specific mapping rules.

## Wave-seven reproducibility and blind fit

Wave seven added strict output models for blind tasks, fit judgments, evidence accounting, and holistic reviews.
`validation.py` checks curriculum versions, frozen task IDs, guidepost representation, capability references, and
evidence ledgers across one subject run. `workflow.md` records the complete role prompts and evidence boundaries.

Each of four subjects used 24 tasks generated from the subject inventory by an agent that could not read the
curriculum. The fit judge received only task IDs, instructions, and the candidate curriculum. The same frozen task
sets were reused after repairs.

| Subject | Final version | Nodes/capabilities | Initial score and E/A/G/I | Final score | Final E/A/G/I | Fit |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Software QA & Testing | `wave7-10` | 23/19 | 89; 8/0/16/0 | 92, pilot ready | 24/0/0/0 | 24/24 |
| Robotics & Embodied Systems | `wave7-4` | 28/25 | 81; 15/0/9/0 | 92, pilot ready | 23/0/1/0 | 23/24 |
| Legal & Compliance | `wave7-2` | 20/16 | 86; 18/0/6/0 | 95, pilot ready | 23/0/1/0 | 23/24 |
| Bioinformatics & Life-Science Computing | `wave7-5` | 59/51 | 83; 18/0/6/0 | 91, pilot ready | 24/0/0/0 | 24/24 |

E/A/G/I denotes exact, ambiguous, gap, and invalid. Every final systematic-gap list is empty. The roles used
`gpt-5.6-sol` at high reasoning effort with prompt versions `curriculum-generator-v1`, `blind-tasks-v1`,
`blind-fit-v1`, and `holistic-review-v1`; sampling parameters were provider controlled. The consolidated evidence is
Loom artifact `curriculum-wave7-evidence`, revision 1, branch scope `zooq8ec9`, SHA-256
`49fc55b3fd0a82ca1d2e62b4435ebfa28ebadfd101382a97b0c5d70779e73e6d`, associated with session channel
`s2uyqg34`. It contains the manifest, frozen blind tasks, final curricula, design audits, fit judgments, and holistic
reviews. Wave seven calibrated the procedure; the current prompts in `workflow.md` include the resulting corrections
and are the normative inputs for the next wave.

The initial software-testing score and fit rate show that a high holistic score can coexist with sampled coverage
gaps. An intermediate software-testing version reached 24/24 fit and scored 81 because its prerequisite and probe
structure remained defective. Blind fit therefore stays separate from the score. The final robotics and legal misses
are isolated composite tasks; neither repeated across an operation family nor exposed an uncovered guidepost.

Repairs repeatedly split capabilities whose input, transformation, output, or correctness contract differed. Other
failures were capability parents that did not contain their children, probes with missing executable inputs, and
design audits that retained superseded edges. The generator and reviewer prompts now require operation signatures,
parent-containment checks, and execute-on-paper probe checks. Output validation rejects stale review versions and
invalid cross-artifact references.

The catalog now covers 16 of 34 macro areas and 77 of 151 guideposts. Its 493 nodes imply about 970 nodes at full
guidepost coverage if the observed density stays constant, although subject density varies from eight capabilities
for NLP to 51 for bioinformatics. Mutual self-confidence and epsilon continuity determine density; no per-subject
section target applies. All four reviews have medium confidence because wave seven used no task-level discovery or
held-out evaluation questions.

## Provenance

- [TaskTrove competency coverage audit](https://storage.googleapis.com/marin-public/benjaminfeuer/tasktrove-competency-coverage/2026.09.03/index.html)
- [Coverage issue #8879](https://github.com/marin-community/marin/issues/8879)
- [Evaluation policy #9193](https://github.com/marin-community/marin/issues/9193)
- [TaskCompendium PR #9187](https://github.com/marin-community/marin/pull/9187)
