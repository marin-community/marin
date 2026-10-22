# Reproduce the pretraining source pool

Marin publishes the code needed to retrieve and normalize its pretraining
sources. The canonical catalog is
[`lib/marin/src/marin/datakit/sources.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/sources.py).
Each entry records a stable Marin source name, an ordered source-processing
recipe, and an approximate token count used for initial mixture weighting.

The catalog describes the active source pool at a particular Marin commit. To
reconstruct the inputs to a specific training run, check out the Marin commit
recorded by that run and use its copy of `sources.py`. The model's training
configuration determines which catalog entries were sampled and at what
weights; the catalog alone does not specify an exact model mixture or token
stream.

## Why Marin publishes source recipes

Marin does not publish the pool as a single mirrored Hugging Face dataset.
Researchers retrieve each dataset from its original provider. This keeps the
provider's attribution, license, and access controls in the retrieval path. It
also means that a provider can change or withdraw the source used by future
reconstructions without relying on Marin to remove a second public copy.

Some providers require an account or acceptance of dataset-specific terms.
Reconstruction does not bypass those requirements. Review each provider's
license and obtain access before running its download step.

For Hugging Face sources, Marin records the dataset repository ID and pins a
specific repository revision. The pin is intended to prevent changes on a
provider's moving branch from changing the reconstructed files. This guarantee
depends on Hugging Face continuing to serve that revision. Sources fetched
from the live web, APIs, or other mutable upstreams do not always provide an
equivalent immutable revision. For those sources, a Marin commit pins the
retrieval and transformation code, but a later run may retrieve different
content.

## Trace a source to its provider

Every catalog entry is a `DatakitSource`. Its `normalize_steps` field contains
an ordered `StepSpec` chain ending in the normalized Parquet artifact consumed
by downstream Datakit stages.

To inspect a source:

1. Find its Marin name in `all_sources()` in
   [`sources.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/sources.py).
2. Follow the imported `*_normalize_steps` factory to its module under
   [`marin/datakit/download/`](https://github.com/marin-community/marin/tree/main/lib/marin/src/marin/datakit/download).
3. Read the factory and the dependencies of its first returned step. These
   lead to the raw download and any source-specific transforms.
4. Record the provider dataset ID, pinned revision, selected files, schema,
   and normalization parameters from that module.

For example, the `coderforge` row points to
`coderforge_normalize_steps`. That factory calls `download_hf_step` with the
provider ID `togethercomputer/CoderForge-Preview`, revision `060fca9`, and the
selected Parquet splits. It then renders each trajectory as text and runs the
shared normalization step. The resulting chain preserves both the provider
identity and the exact Marin transform.

Many simple Hugging Face sources use `hf_normalize_steps`, which makes the same
provider ID, revision, file selection, and schema parameters visible in one
call. Family modules such as `nemotron_v2.py` define one shared provider
download and a separate normalization step for each catalog subset.

## Materialize the catalog

Clone Marin and check out the commit associated with the training run:

```bash
git clone https://github.com/marin-community/marin.git
cd marin
git checkout <marin-commit>
uv sync --all-packages --extra=cpu
```

Choose a durable destination for the raw and normalized artifacts. Marin
resolves relative step paths under `MARIN_PREFIX`; the destination can be a
local directory or an object-store URI supported by `fsspec`.

```bash
export MARIN_PREFIX=/path/to/pretraining-pool
export HF_TOKEN=<token-for-sources-you-can-access>
```

List the source names at the checked-out revision:

```bash
uv run python - <<'PY'
from marin.datakit.sources import all_sources

for name in all_sources():
    print(name)
PY
```

Start with one source and materialize its complete chain:

```bash
uv run python -m experiments.datakit.scripts.trigger_sources \
  --sources coderforge
```

Use a comma-separated list to select several sources. Omitting `--sources`
materializes every entry in the catalog:

```bash
uv run python -m experiments.datakit.scripts.trigger_sources
```

This is a large distributed data-processing workload. `trigger_sources` hands
each terminal normalization step to Marin's `StepRunner`, which follows
transitive dependencies, shares common family downloads, and skips artifacts
that already completed under the same prefix. A local run can download simple
sources, but large normalization steps require an Iris/Fray environment or an
equivalent deployment of Marin's execution stack.

Pass `--downloads-only` to stop at each chain's first step. This is useful for
auditing provider retrievals, but a first step may include a source-specific
transform and is not guaranteed to be the untouched provider files. Trace its
dependencies when the raw download itself is required.

## Sources with separate reconstruction steps

Three catalog entries currently begin from pre-staged Parquet because their raw
provider data requires an additional hydration or extraction process. Their
download modules pin the code and configuration that produced those shards:

| Catalog entry | Reconstruction pointer |
| --- | --- |
| `common-crawl-focus-2026-22` | [`common_crawl_focus.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/common_crawl_focus.py) links the exact Focus Crawl extractor and launch configuration. |
| `nemotron_code_v1/content` | [`nemotron_code_v1_content.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/nemotron_code_v1_content.py) links the pinned resolver, Software Heritage graph configuration, ID mapping, and object downloader. |
| `nemotron_code_v2/content` | [`nemotron_code_v2_content.py`](https://github.com/marin-community/marin/blob/main/lib/marin/src/marin/datakit/download/nemotron_code_v2_content.py) links the same pinned hydration pipeline for the v2 metadata. |

Run the linked reconstruction process first, place its output at the relative
path named by the module under `MARIN_PREFIX`, and then run `trigger_sources`
for that catalog entry. Each pre-staged step checks that input shards are
present before normalization. Use the shard counts recorded in the module to
verify that staging completed.

Provider availability can change after a Marin revision is published. A
provider removal, license change, or expired credential may prevent a later
byte-for-byte reconstruction even though the original recipe remains pinned.

## Proposal: a metadata-only Hero dataset

We should publish an index of the documents in Hero's pretraining dataset.
Someone should be able to download the original sources, run our text
transforms and normalization, and use this index to recover the selected
corpus. They would avoid rerunning deduplication, decontamination, classifiers,
and clustering.

The first release would cover `store_4d2e363d`, the 292-source store used by
`experiments/grug/moe_hero_ep/harrier_mix_2026_08_18.py`. It would contain
metadata for the documents available to training. Reproducing which documents
training sampled, how often, and in what order also requires the run's mixture
schedule and training configuration.

### What we would publish

The dataset would be Parquet, with one row per document occurrence in the
selected corpus and no text or tokens. An occurrence here is a retained
document produced by the source transform, independent of training sampling
and token chunking:

| Field | Contents |
| --- | --- |
| `marin_source` | The source name in Marin's catalog. |
| `id` | Our existing hash of the normalized text. |
| `source_id` | The upstream ID retained by normalization, when available. Stored as a JSON-encoded string because its type varies across sources. |
| `source_rows` | A nonempty, ordered list of upstream row locators: `source_ref`, `row_offset`, `file`, and `file_row_offset`. These identify the source-manifest entry, zero-based split row, provider file, and zero-based row within that file. None may be null. |
| `cluster_id` | The document's topic cluster. |
| `quality_score` | Its stored quality classifier score. |
| `content_type` | The classifier label used to calibrate that score. |
| `quality_bucket` | The quality bucket used by the final store. |

The canonical `id` is the 32-character hexadecimal `xxh3_128` hash of the
normalized UTF-8 text, after whitespace compaction. We do not need a second
text hash.

`source_id` is useful for finding a document, but it is optional and its meaning
varies. Stack v3 retains a repository ID; some SWE transforms put a text hash
in this field. FineTranslations and Institutional Books discard upstream
fields while constructing the text. Every released document must therefore
have upstream row coordinates even when `source_id` is absent. The text hash
checks that reconstruction produced the expected document.

Some sources retain repeated documents. The index must preserve those
occurrences, and the reconstruction tool must respect their counts. Treating
the released IDs as a set would lose this information.

Files would follow the final store's `cluster=<id>/quality=<bucket>/` layout
and its 384 task partitions. A separate export ledger would retain normalized
shard paths, row offsets, and task IDs for validation. These internal
coordinates would stay out of the public document rows.

A source manifest would accompany the rows. It would record provider dataset
names, revisions, file selections, and the code and parameters needed to
construct normalized text. A release manifest would identify the exact input
artifacts, tokenizer, classifier versions, quality calibration cutpoints,
fuzzy-dedup exemptions, and shard assignment. We should include other existing
classifier scores where available, with their model provenance; the prototype
currently exports the quality score, content-type label, and topic assignment.

### How reconstruction would work

For each released row, the user would retrieve the upstream rows identified
by `source_rows`, apply our source-specific text transform, and run
normalization. They would check the resulting hash against `id`, then emit
the document in its assigned bucket. Repeated index rows produce repeated
occurrences; duplicate rows elsewhere in the provider dataset do not add
copies. Tokenization can happen afterward.

Each locator's source-manifest entry identifies the provider dataset,
revision, config, and split. Its `row_offset` is the document row number
across that entire split, before our filtering, sorting, or normalization.
The manifest must name the coordinate system. `hf_viewer_parquet` means the
offset follows the Dataset Viewer's converted Parquet order and can be used
with its [`/rows` endpoint](https://huggingface.co/docs/dataset-viewer/en/rows),
with `length=1`. `source_manifest` means the offset follows our recorded
provider-file order and must be resolved through that manifest. The manifest
must list every file in split order and its row count so that a global offset
can be resolved to a file and row. These orders can differ; sorting downloaded
filenames does not establish Viewer offsets.
The file path and `file_row_offset` also allow direct reads from pinned
Parquet files. Non-Hugging-Face sources must define an equivalent stable
row order in their manifest.

The Viewer has no historical-revision parameter and may serve a different
revision from the one we used. Eight current sources also originate outside
Hugging Face's row datasets. Every source can have stable file and record
coordinates, but universal access through `/rows` would require additional
hosting. The reconstruction tool must verify Viewer revision and content,
and use pinned provider files when necessary. A successful lookup at the
same numeric offset is not sufficient evidence of a matching revision.

The transforms need to be documented precisely. For FineTranslations, for
example, a document contains both the original text and its English
translation. The upstream ID determines their concatenation order. A user
must run that folding step before normalization; hashing either side on its
own will not find the released document.

Other sources have their own assembly rules. Institutional Books chooses
between corrected and source page lists, then joins the pages. SWE-zero
renders messages, while SWE ConTree already contains rendered text. Stack v3
serializes repositories. Biodiversity joins pages across raw records and may
need records from several files to reconstruct one book.

Offsets must be captured before transformation and carried through to the
selected document. A one-row transform such as Institutional Books retains
one locator. Biodiversity needs an ordered locator for every page used to
assemble the book, including pages from other files. A normalized artifact
offset or a row number in an intermediate transformed dataset cannot stand
in for these upstream coordinates.

### How we would build it

Ideally, we could just strip the text from a canonicalized dataset. However,
we don't have the dataset in that form. The final selection
happens in the store builder, which joins tokenized documents with the filter
and classifier outputs before writing bucketed token caches. Those final
caches no longer retain document IDs.

The exporter would repeat that selection using the existing normalized and
attribute shards. It would reuse the stored deduplication and contamination
decisions and classifier outputs. It must verify document IDs and order when
joining the shards, then apply the same exemptions and bucket assignment as
the original build.

For this store, quality buckets are calculated from the score using
content-type-specific cutpoints: score thresholds chosen separately for each
content-type label. The document's label selects the threshold set, and its
quality score determines which bucket it falls into within that set. Thus,
the same score can map to different buckets for different content types. The
exporter must reuse the frozen store's thresholds rather than estimate new
ones from the exported documents. Unknown content types use the stored
`default` cutpoints. Copying a bucket from an older scoring format would give
the wrong result.

The production job should stream batches, write each task's metadata under
its bucket directories, and mark a task complete only after all its files
are written. Before publication, we would reconcile document and token counts
per source, bucket, and task against the existing store. The original ordered
shard list and task assignment are needed to reproduce its partitioning.
The export must fail if a selected occurrence has no complete upstream row
mapping. We must backfill missing coordinates by replaying the source
transforms and joining their outputs to the selected documents, preserving
the input rows used to construct each output.

### What the trial established

The bounded trial passed across all 292 sources. We exported 17,507 selected
document occurrences, checked the sample file counts and schemas, and
confirmed that the files contain no text or token columns. Separately, we
replayed 1,167 real staged input rows through normalization and matched the
resulting nonempty documents to their production normalized IDs.

The trial used the frozen store's selection attributes and quality calibration.
Some older staged inputs were still in R2, so those sources were replayed
from their original storage locations. The local tests cover repeated IDs, token chunks, filtering, calibration,
reconstruction counts, and rejection of incomplete upstream coordinates.

A later provider-row trial reconstructed four documents each from
FineTranslations, Stack v3, SWE ConTree, and SWE-zero and matched their
production normalized IDs. FineTranslations also passed retrieval through
Hugging Face `/rows`: four rows from its `abk_Cyrl/train` config, served at
the pinned source revision, reproduced those same IDs. That config has one
file, and the original and Viewer files have the same LFS content digest,
which establishes their row-order equivalence. These are sampled normalization
checks; final-selection membership is a separate check.
Institutional Books also matched four production normalized IDs after a
follow-up with a 96 MiB read cap. For Biodiversity, a bounded scan of 2,048
raw page rows reconstructed four candidate books. All four matched both
the production normalized hash and the retained book identity. This proves
reconstruction of those books' normalized text; it does not certify a
complete list of their historical input rows, since empty pages and whitespace
compaction can leave the hash unchanged.

The provenance audit recovered provider revisions and selected file lists
for all 284 Hugging Face sources. Eight sources use other download routes.
Complete split-wide file ordering and row counts have not yet been verified
across the catalog, so the audit does not establish global offsets for every
source.

Selection and normalization used separate samples. Both operations passed
sampled checks across all 292 sources. They do not yet show a complete provider-download-to-selected-text
round trip: some replay inputs had already undergone source-specific
transforms. The trial recorded staged-input and normalized-row coordinates;
it did not establish complete upstream split offsets for the selected rows.
That remains a release requirement. We also have not checked complete-corpus
counts or final token order.

### Work before release

The next step is to backfill upstream row locators and add a reconstruction
tool with a worked example for each source family, including the assembly
rules above. The release gate requires complete locators on every exported
row and sampled lookups through Hugging Face `/rows` where the coordinates
use `hf_viewer_parquet` and the Viewer serves the pinned revision. Otherwise,
sampled reads from pinned provider files must pass. Both routes must reproduce
the normalized hash after transformation.
Multirow sources must pass with all their contributing rows.
We need to recover and document
transform provenance where artifacts came from dirty working trees or older
metadata records; the store's commit alone does not pin those transforms.

We would then run the full metadata export, reconcile its counts with the
store, and measure its compressed size before uploading it. Publication should
include the manifests, a dataset card, and instructions for reconstructing one
source or one bucket. The locators should let users fetch the contributing
rows directly, subject to the provider's file format and access methods.
The trial tools live under
`experiments/datakit/scripts/metadata_*`; they provide the export and replay
checks, but are not yet the public reconstruction interface.
