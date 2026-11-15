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

For each question, the screen enumerates required clues and reasoning hops,
searches the approved index, and checks literal document quotations for every
hop. A separate request to the same pinned judge receives the question and
evidence quotations, excluding the gold answer and the gold-derived requirement
claims. Its
answer must match the gold answer under the source's answer-equivalence
prompt and parser. A separate audit checks all original clues, evidence links,
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

After grounding, follow the [source procedure](../../../../../docs/references/task-curation.md#source-procedure)
for census, exact deduplication, the full shared lexical scan and evaluation
heldout checks, private publication/readback, final-pin qualification, SAMPLE,
retention and admission. BrowseComp and BrowseComp-Plus questions must remain
outside TRAIN. The released gold answers supply private grading references;
they do not establish executable source oracles. Use the documented
[no-oracle retention procedure](../../../../../docs/references/task-curation.md#sources-without-oracle-solutions).
