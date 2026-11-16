# OpenResearcher offline research preparation

The TRAIN environment uses these dataset releases:

| Input | Revision | Dataset license |
| --- | --- | --- |
| `OpenResearcher/OpenResearcher-Dataset` | `1edca18c5d7c9c105762da4f3c5c2b83855b3c6e` | MIT |
| `OpenResearcher/OpenResearcher-Corpus` | `8367703babe6e080899a48782f365c284c32b668` | ODC-By |

Run the [binding license gate](../../../../../docs/references/task-curation.md#stage-0-binding-license-gate)
before preparing either input. The upstream GitHub implementation and released
index are excluded because they lack licenses. Build a fresh BM25 index over
the complete approved corpus using this directory's `build_index` entrypoint.
Keep the large inputs, index and intermediate files in regional Iris CPU jobs.

`stage` streams pinned release files into regional storage and verifies their
SHA inventory. `extract_questions` reads only question, answer and source-row
identifiers, deduplicating exact question-and-answer pairs across synthesis
seeds. Conflicting answers remain visible in a separate receipt and block
grounding until resolved. Released synthesis
trajectories never become actor conversation history.

The index verifier compares every stored document with the original corpus
SHA inventory, checking document count, identity, uniqueness and text. Its
receipt seals a manifest of the uploaded index files. Grounding checks every
local index file against that manifest before starting a judge.

## Ground the complete question population

The `ground` entrypoint binds the source and index receipts, their SHA values,
the retrieval runtime's package freeze, and a pinned Qwen3 judge deployment.
The matcher contract in `grounding_contract.py` records the prompts, output
schemas, generation settings and retention rule. Execution failures fail the
artifact; unsupported content and evidence-budget limits receive explicit
drop decisions.

Each structured judge request permits at most three attempts for responses that
reach the 4,096-token output limit or fail schema validation. Retries use the same
request and generation settings and preserve every response. Exhaustion fails
the artifact.

For each question, the screen enumerates required clues and reasoning hops,
searches the approved index, and checks literal document quotations for every
hop. A separate request to the same pinned judge receives the question and
evidence quotations, excluding the gold answer and the gold-derived requirement
claims. Its
answer must match the gold answer under the MIT BrowseComp answer-equivalence
prompt and parser reused by the task grader. A separate audit checks all original clues, evidence links,
answer support and ambiguity. The private receipt retains retrieval results,
judge requests/responses and every keep/drop decision.

The grounding judge uses temperature 0.7 with its model revision, serving
configuration and backend job identities recorded. SAMPLE uses the exact V3
reviewer contract from the shared
[`sample_qc.py` procedure](../../sample_qc.py), with the profile defined in
[`full_qc.py`](../../full_qc.py).

## Continue through admission

The versioned `openresearcher-bm25:v1` provider exposes `search` and
`get_document`. Search returns five hits with 512-token snippets. Its state
pins the implementation, corpus, tokenizer and index manifest. The gold
answer belongs only in private verifier resources.

`prepare` creates the source's QUICK artifact after grounding and physical image
QA. Its [`PreparationConfig`](prepare.py) binds the stage, question, grounding and benchmark
hash receipts by SHA, the retrieval state, two immutable image digests and a
fresh local work directory. It rechecks both dataset licenses before reading
questions. Every original question must have exactly one completed grounding
decision; retained question and answer bytes must match their original rows.
Exact BrowseComp and BrowseComp-Plus question matches are excluded again and
recorded in `heldout-drops.json`. The benchmark hash manifest uses whitespace
collapse followed by Unicode case folding; it supplements Atlas decontamination.
The benchmark JSON has `normalizer` set to
`whitespace-collapse-then-unicode-casefold-v1` and a `questions` object mapping
SHA256 hashes of normalized questions to their benchmark source references.
Construct `state` with [`OpenResearcherState`](../../../../../lib/taskcompendium/src/taskcompendium/openresearcher.py),
using the verified index manifest SHA and the implementation/tokenizer revisions
in [`runtime.py`](runtime.py). The fixed corpus revision, top-five search,
512-token snippets and `get_document` setting must match the physically tested
image's `/opt/openresearcher/state.json`.

```bash
uv run python -m experiments.post_training.task_curation.datasets.openresearcher.prepare \
  --config pinned-preparation.json \
  --artifact-root s3://marin-us-east-02a/marin \
  --output s3://marin-us-east-02a/marin/rl-data/openresearcher/conversion/NEW_VERSION/receipt.json \
  --run
```

The artifact contains native TaskCompendium task directories, `tasks.parquet`
and standard `normalize/*.parquet` rows. Actor prompts contain the question;
private verifier resources retain the gold answer, original source-row references
and the MIT answer-equivalence grader. Task IDs depend only on the public
question. QUICK marks the population unreviewed and unadmitted.

Follow the [source procedure](../../../../../docs/references/task-curation.md#source-procedure)
for census, exact deduplication, the full shared lexical scan and evaluation
heldout checks, private publication/readback, final-pin qualification, SAMPLE,
retention and admission. BrowseComp and BrowseComp-Plus questions must remain
outside TRAIN. The released gold answers supply private grading references;
they do not establish executable source oracles. Use the documented
[no-oracle retention procedure](../../../../../docs/references/task-curation.md#sources-without-oracle-solutions).
