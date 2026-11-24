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

Grounding retrieves complete original document bodies over a local JSON-line
stream. Responses larger than the subprocess reader's buffer limit are drained
in chunks without truncating documents; evidence is token-bounded only after
the complete response arrives. EOF before the terminating newline and malformed
JSON still fail the artifact.

## Ground the complete question population

The `ground` entrypoint binds the question and index receipts, their SHA values,
the retrieval runtime's package freeze, and a pinned Qwen3 judge deployment.
The matcher contract in `grounding_contract.py` records the prompts, output
schemas, generation settings and retention rule. Execution failures fail the
artifact; unsupported content and evidence-budget limits receive explicit
drop decisions.

Each structured judge request permits at most three attempts for responses that
reach the 4,096-token output limit or fail schema validation. Retries use the same
request and generation settings and preserve every response and its token usage.
When the last attempt is truncated, the question receives a `technical_exclusion`
decision with `keep=false` and reason `grounding_incomplete_exhausted`. These
exclusions may cover at most 1% of the complete question population (61 of 6,102);
exceeding that limit fails the artifact. Conversion independently checks the
same limit and rejects any technical exclusion marked for retention. Exhausted
malformed responses and transport or backend failures still fail the artifact.
Only `grounding_incomplete_exhausted` drops count toward this cap.

A complete JSON requirements array longer than eight receives an explicit budget
drop before nested query validation. Other malformed structured responses keep
the same three-attempt retry and failure behavior. The final receipt reports
`drop_reason_counts`; conversion checks these counts against the sealed
question-level decisions.

For each question, the screen enumerates required clues and reasoning hops,
searches the approved index, and checks literal document quotations for every
hop. A separate request to the same pinned judge receives the question and
evidence quotations, excluding the gold answer and the gold-derived requirement
claims. Its
answer must match the gold answer under the MIT BrowseComp answer-equivalence
prompt and parser reused by the task grader. A final request to the same pinned
judge audits all original clues, evidence links, answer support and ambiguity.
A question is retained only when grounding, independent answering and audit
checks pass. The private receipt retains retrieval results,
judge requests/responses and every keep/drop decision.

The grounding judge uses temperature 0.7 with its model revision, serving
configuration and backend job identities recorded. SAMPLE uses the pinned V4
reviewer contract from the shared
[`sample_qc.py` procedure](../../sample_qc.py), with the profile defined in
[`full_qc.py`](../../full_qc.py).

## Continue through admission

[`recipe.sources()`](recipe.py) declares the frozen preparation inputs in the
configured source inventory. Constructing this declaration performs no regional
reads. `curate` checks the dataset and corpus license receipts before creating
the QUICK artifact graph; `prepare` rechecks both before reading question data.
The `OpenResearcher:OpenResearcher-Dataset` entry in the
[retained-release registry](../retained_release_receipts/) selects the private
TaskCompendium pin and binds its final census and static retention. The
[license](../license_receipts/) and [SAMPLE](../sample_qc_receipts/) registries
bind that pin's permissions and content result.
The [source procedure](../../../../../docs/references/task-curation.md#source-procedure)
defines admission and default-view eligibility.

The versioned `openresearcher-bm25:v1` provider exposes `search` and
`get_document`. Search returns five hits with 512-token snippets. Its state
pins the implementation, corpus, tokenizer and index manifest. The gold
answer belongs only in private verifier resources.

The declared actor image contains `/opt/openresearcher/state.json` and the
complete pinned BM25 index at `/opt/openresearcher/index/bm25`. The image-local
provider reads both paths with external network access disabled. The grader
uses the smaller runtime base; that base contains the Python dependencies and
tool code. Keep the actor bound to the full-index image when preparing tasks
for the image-local provider.

The full-index actor passed image-local provider QA in an Iris gVisor sandbox
on `cw-rno2a`. DATA Daytona rejected its 55.19 GB snapshot against the account's
20 GB maximum on 2026-11-21. Use a backend that accepts the full image.

In a real sandbox, require the parsed `state.json` to equal the TaskSpec's
`environment_requirements.tool_providers.retrieval.initial_state`,
then call `search` and `get_document` through the image-local provider. Verify
the complete returned document text against the pinned corpus. The grounding
JSON-line replay described above tests a separate transport path.

Before `prepare`, record image-local provider QA in the intake report. Its
evidence must bind the actor image digest, image-local tool runtime, complete
index-file manifest and the two production-provider probes above. `prepare`
creates the source's QUICK artifact
after grounding and this QA. Its [`PreparationConfig`](prepare.py) binds the stage,
question, grounding and benchmark hash receipts by SHA, the retrieval state,
immutable actor and grader image digests and a
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
actor image's state file.

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
question. This preparation output records no content oversight or admission.

Follow the [source procedure](../../../../../docs/references/task-curation.md#source-procedure)
for census, exact deduplication, the full shared lexical scan and evaluation
heldout checks, private publication/readback, final-pin qualification, SAMPLE,
retention and admission. BrowseComp and BrowseComp-Plus questions must remain
outside TRAIN. The released gold answers supply private grading references;
they do not establish executable source oracles. Use the documented
[no-oracle retention procedure](../../../../../docs/references/task-curation.md#sources-without-oracle-solutions).

## Hybrid retrieval provider

The versioned `openresearcher-hybrid:v1` provider searches the same approved
corpus with BM25 plus a dense index, fused by reciprocal rank fusion. The
dense side embeds every full corpus document with the pinned,
permissively licensed `Qwen/Qwen3-Embedding-8B` (Apache-2.0; receipt in
`embedding_model_license.json`) — the same encoder family, dimensionality and
pooling the synthesis environment used — storing raw fp16 shards in-region
and building a flat inner-product index whose docid order is verified
against the corpus shard order. Search encodes each query with the same
pinned encoder and the model card's documented instruction format,
retrieves 25 BM25 and 25 dense candidates, fuses the two lists with
`rrf_fuse` (k=60), and returns the top five hits with full documents. The
tool contract, hit shape, snippet budget and grounding contract are
unchanged; only the provider version differs. See COMPARISON.md next to the
task brief for the full original-implementation comparison.

The `embedding` entrypoint embeds the complete corpus in an Iris GPU job (GPU
work runs in the pinned dense-embedding venv through `embedding_worker`;
`--mode stats` first measures the corpus token distribution that fixes the
document token cap).
`build_dense_index` verifies every vector and docid part against the
embedding manifest and the approved corpus, then seals the flat index.
`hybrid_index` binds the BM25 index, dense index and embedding receipts into
one versioned artifact whose state is the `OpenResearcherHybridState`
contract. `pilot_sample` selects the fixed 300-drop/100-kept pilot sample
from the sealed v12 decisions, and `hybrid_ground` runs the same sealed
screen against the hybrid provider for the pilot or the full population,
reporting recovered and regressed questions against v12.

