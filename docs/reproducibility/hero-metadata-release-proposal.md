# Proposal: a metadata-only Hero dataset

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

## What we would publish

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
| `harrier_embedding` | The stored 1,024-dimensional Harrier vector as `int8`, or null when no stored vector is available. |
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

We should include the existing Harrier embeddings so users can cluster or
analyze the selected documents without running the embedding model again.
The vectors retain the stored int8 quantization.
The release manifest must pin the embedding artifact, model repository and
revision, dimension, quantization scale and range, and the input preprocessing.
Multiplying the stored `int8` values by the recorded scale gives approximate
floating-point vectors. We should preserve the stored bytes.

Harrier embedded one representative per candidate duplicate cluster in an
earlier deduplication pass. The final store uses verified duplicate decisions
and exempts some sources, so documents omitted by the embedding pass can still
be selected. Its embedding shards can therefore have fewer rows than the
normalized shards. Embeddings must be joined by document ID, with the same
vector carried to repeated selected occurrences. Conflicting stored vectors
for the same ID must be rejected. We must report coverage
and null counts per source; a missing vector must not silently become zeros or
cause a selected document to disappear. Filling gaps would be a separate
embedding job. At 1,024 bytes per vector, the vectors add about 1.024 TB per
billion populated rows before Parquet overhead and compression. The trial
should measure their compressed size and coverage before the full export.

A source manifest would accompany the rows. It would record provider dataset
names, revisions, file selections, and the code and parameters needed to
construct normalized text. A release manifest would identify the exact input
artifacts, tokenizer, classifier versions, quality calibration cutpoints,
fuzzy-dedup exemptions, and shard assignment. We should include other existing
classifier scores where available, with their model provenance; the prototype
exports the quality score, content-type label, and topic assignment, with an
optional Harrier shard input for testing the embedding join. Existing trial
results below predate that embedding input and do not establish its coverage.

## How reconstruction would work

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
hosting. In strict reconstruction mode, the tool must verify Viewer revision
and content, and use pinned provider files when necessary. A successful lookup
at the same numeric offset is not sufficient evidence of a matching revision.

The reconstruction client should also offer explicit content-verified recovery
for `hf_viewer_parquet` locators when the current Viewer revision differs.
Removing an earlier provider row can shift the offsets of surviving rows.
Inspect the recorded offset and up to 100 rows on either side, clipped to the
split bounds. This configurable search radius gives at most 201 candidate rows.
Even if the recorded offset matches, inspect the window for ambiguous matches.
Fetch slices in bounded requests and keep every response in an attempt on the
same observed revision. Reject partial or truncated values. Do not expand to
an unbounded scan. `source_manifest` offsets must use their pinned-file path;
they cannot be substituted for Viewer offsets.

Every candidate must pass the documented source transform and normalization
before its hash can match the released `id`. Use stable provider IDs as
additional identity checks where available. If several rows match and their
original occurrence cannot be distinguished, report ambiguous provenance.
Emit one recovered occurrence for each release-index occurrence; finding
duplicate provider rows must not add copies. Record the original locator,
observed revision, recovered coordinates, and matching hash in a recovery
report, leaving the release manifest intact. A content match establishes the
selected text; it does not establish the original row's provenance at a changed
revision, even if the bounded window contains only one match.

This fallback can recover small positional shifts while source order and text
remain stable. It cannot recover a removed target or guarantee recovery after
reordering, large deletions, or text edits. The initial recovery client should
support transforms that reconstruct a document from one provider row. Assembled
documents still require all contributing rows in the documented order; finding
one nearby page is insufficient. Keep those documents on the pinned-file path
until complete input assembly can be verified. If a bounded search fails or
leaves ambiguous provenance, use pinned files when available or report the
document as unresolved. The current draft implements strict coordinate and
Viewer-response validation; window search belongs in the reconstruction client
when that client is built.

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

## How we would build it

Ideally, we could just strip the text from a canonicalized dataset. However,
we don't have the dataset in that form. The final selection
happens in the store builder, which joins tokenized documents with the filter
and classifier outputs before writing bucketed token caches. Those final
caches no longer retain document IDs.

The exporter would repeat that selection using the existing normalized and
attribute shards. It would reuse the stored deduplication and contamination
decisions and classifier outputs. It must verify document IDs and order when
joining the shards, then apply the same exemptions and bucket assignment as
the original build. Harrier embeddings are a separate sparse ID join and must
not participate in the dense row-count or positional-alignment checks.

For this store, quality buckets are calculated from the score using
content-type-specific cutpoints: score thresholds chosen separately for each
content-type label. The document's label selects the threshold set, and its
quality score determines which bucket it falls into within that set. Thus,
the same score can map to different buckets for different content types. The
exporter must reuse the frozen store's thresholds rather than estimate new
ones from the exported documents. Unknown content types use the stored
`default` cutpoints.

The production job should stream batches, write each task's metadata under
its bucket directories, and mark a task complete only after all its files
are written. Before publication, we would reconcile document and token counts
per source, bucket, and task against the existing store. The original ordered
shard list and task assignment are needed to reproduce its partitioning.
The export must fail if a selected occurrence has no complete upstream row
mapping. We must backfill missing coordinates by replaying the source
transforms and joining their outputs to the selected documents, preserving
the input rows used to construct each output.

## What the trial established

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

An embedding metadata audit recovered the pinned Harrier model, revision,
dimension, and quantization parameters for 291 sources. The remaining source,
`common-crawl-focus-2026-22`, points to a repacked embedding artifact whose
metadata omits those fields; its recorded parent artifacts were unavailable.
That provenance gap remains open. This audit read artifact metadata and did
not measure selected-document embedding coverage.

Selection and normalization used separate samples. Both operations passed
sampled checks across all 292 sources. They do not yet show a complete provider-download-to-selected-text
round trip: some replay inputs had already undergone source-specific
transforms. The trial recorded staged-input and normalized-row coordinates;
it did not establish complete upstream split offsets for the selected rows.
That remains a release requirement. We also have not checked complete-corpus
counts or final token order.

## Work before release

The next step is to backfill upstream row locators and add a reconstruction
tool with a worked example for each source family, including the assembly
rules above. The release gate requires complete locators on every exported
row and sampled lookups through Hugging Face `/rows` where the coordinates
use `hf_viewer_parquet` and the Viewer serves the pinned revision. Otherwise,
sampled reads from pinned provider files must pass. Both routes must reproduce
the normalized hash after transformation.
Multirow sources must pass with all their contributing rows. Harrier vectors
must match the stored embedding rows, carry their frozen model and quantization
provenance, and have measured coverage for every source. The release must
distinguish confirmed missing vectors from a trial scan that stopped at its
read budget. A public export must stop if requested embedding lookups remain
unresolved.
We need to recover and document
transform provenance where artifacts came from dirty working trees or older
metadata records; the store's commit alone does not pin those transforms.

We would then run the full metadata export, reconcile its counts with the
store, and measure its compressed size before uploading it. Publication should
include the manifests, a dataset card, and instructions for reconstructing one
source or one bucket. The locators should let users fetch the contributing
rows directly, subject to the provider's file format and access methods.
The draft exporters and locator validation live under
`experiments/datakit/store/`. The exploratory inventory and replay tools are
preserved in a complete Marin checkout in the
[research archive](https://github.com/marin-community/marin-logbooks/blob/5460b8a8c94a344600c58163ac040813f2a43f86/logbooks/10527-hero-metadata/notebook.md).
