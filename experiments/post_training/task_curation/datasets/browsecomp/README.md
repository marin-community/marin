# BrowseComp-Plus offline BM25 intake

This source is evaluation-only. It preserves all 830 pinned questions, the
original search-only query template, the full offline BM25 index and its
Qwen3 tokenizer. `search` returns five results with snippets capped at 512
tokens. The actor receives neither relevance labels nor reference answers.
The source answer-equivalence prompt and parser run in a separate verifier.

`taskcompendium.browsecomp` declares the versioned tool interface. The
RolloutEngine provider invokes the retriever inside the task's declared
image; Harbor registers that same provider for production TaskSpec trials.
The task advertises only its source interaction tools.

## Prepare the full population

Build the image from this directory's Dockerfile. Its context contains the
`browsecomp` package, the frozen index at `browsecomp/index/bm25`, tokenizer
files at `browsecomp/tokenizer` and the `STATE` manifest from `tasks.py` at
`browsecomp/state.json`. These assets belong in the image; `--input-root`
contains only the pinned question dataset snapshot. Pass the built image's
digest to preparation. Use a new
INTERNAL GHCR package for this source, as required by the current
[admission policy](../../../../../docs/references/task-curation.md#current-admission-policy).
The image contains no question-specific reference answers.

After [stage 0](../../../../../docs/references/task-curation.md#stage-0-binding-license-gate)
has passed for the pinned question, corpus, index and code:

```bash
uv run python -m experiments.post_training.task_curation.datasets.browsecomp.prepare \
  --input-root PINNED_QUERY_SNAPSHOT \
  --actor-image ghcr.io/marin-community/rl-data-browsecomp-plus@sha256:IMAGE_DIGEST \
  --license-receipt REVIEWED_QUERY_LICENSE.json \
  --output-root NEW_CONVERSION_ROOT
```

Preparation uses the dataset-owned artifact graph. It writes TaskCompendium
task directories, normalized rows, exact TaskSpec hashes and a conversion
manifest. It rejects an incomplete population or repeated query identity.
This step supplies conversion evidence; it does not claim admission.

Continue with the [source procedure](../../../../../docs/references/task-curation.md#source-procedure):
complete-population census and exact
deduplication, lexical measurement against the approved retained inventory and
shared evaluation Bloom, private publication and readback, final-pin grading
and derived-contract census, static retention and SAMPLE. Evaluation overlap
is measured without deleting benchmark tasks. Register the source only when
its exact final pin has all required receipts.

The upstream release has reference answers but no executable source oracle.
Use the documented [no-oracle retention policy](../../../../../docs/references/task-curation.md#sources-without-oracle-solutions).
Do not turn copied answers into purported original oracle solutions. SAMPLE
quality review remains required.
