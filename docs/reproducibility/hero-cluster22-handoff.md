# Hero cluster 22 reconstruction handoff

Cluster 22 has been reconstructed from original provider inputs: **150 sources,
151,474,843 selected document occurrences**. Source reconciliation and the final
cluster aggregate passed on October 31, 2026. The output is private, normalized
text with canonical IDs. It is ready for inspection by someone with access to
Marin's CoreWeave storage.

The first milestone is complete. The broader
[metadata release proposal](hero-metadata-release-proposal.md) still has release
work remaining, especially upstream occurrence locators and embedding coverage.
No public dataset was published. The reconstruction monitor was deleted after
completion; there is no live watcher to take over.

## Code and artifacts

The code checkpoint and this handoff are on
`codex/hero-cluster22-reconstruction` in `marin-community/marin`. This branch
starts at `3bf94b358dcb29d422e47f37b98d5030c52406c8`, the metadata proposal's
merged-main checkpoint. It has not been refreshed against later main changes.
The original proposal branch is `codex/hero-metadata-release`, associated with
[PR #10527](https://github.com/marin-community/marin/pull/10527). Publishing this
checkpoint does not assert that either branch is merged or ready for release.

All completed outputs use this base:

```text
s3://marin-us-east-02a/marin/user/dlwh/hero-cluster-reconstruction-20261029-02a
```

| Artifact | Path relative to that base |
| --- | --- |
| Current reconstruction plan | `reconstruction-plan-v11.json` |
| Final cluster completion marker | `reconstruction-v1/_complete.json` |
| Per-source inputs, index, candidates, outputs and audit bindings | The plan's explicit `source_roots` mapping |

Use v11. Earlier plans contain superseded provider recipes or incomplete
inventories; some older jobs used those plans. V11 changes only Software
Heritage v1 relative to v10 and preserves the other 149 source versions and
output roots. Never infer a source's current root from a previous plan or a
job's name.

The frozen selection comes from
`s3://marin-us-east-02a/marin/datakit/store_4d2e363d`. The v11 plan pins each
source's provider paths, normalization/renderer hashes, selected ledger
artifacts, selected counts and output root. Some selection ledger paths remain
in the original RNO bucket. They were read as pinned metadata; the raw corpus
was not bulk-copied between regions.

The final marker was independently read back at `2026-10-31T18:34:51Z`:

```json
{
  "complete_cluster": true,
  "plan_sha256": "8088ac6f435b4bf916677f111fba06831e945c9bf51f1c6862fd2139fdc0f924",
  "reconstructed_occurrences": 151474843,
  "sources": 150,
  "target_cluster": 22,
  "validation": "original-provider text hashes and selected occurrence counts; historical occurrence provenance not established"
}
```

The marker's exact stored bytes have SHA256
`f69470165709a8f6b61e14bf13bf731be315d8a98a6bd6b06e2a206935b31334`.
`plan_sha256` is the plan's semantic manifest digest, computed by the plan
verifier; it is not a checksum of the JSON file's formatting.

## Inspect the result

Use the reconstruction branch and Marin's normal `uv` environment. CoreWeave
access needs `CW_KEY_ID` and `CW_KEY_SECRET`. The original operator used
`~/.creds/cw-storage.env`; a coworker should use their own credentials. Do not
copy credentials into the repository or a handoff message.

These commands only read existing artifacts:

```bash
uv run --no-sync fsutil cat \
  s3://marin-us-east-02a/marin/user/dlwh/hero-cluster-reconstruction-20261029-02a/reconstruction-v1/_complete.json

uv run --no-sync fsutil cp \
  s3://marin-us-east-02a/marin/user/dlwh/hero-cluster-reconstruction-20261029-02a/reconstruction-plan-v11.json \
  /tmp/hero-cluster22-plan-v11.json
```

The plan is about 98 MB of metadata. To find one source's output files, read its
completion marker. Its `outputs` entries contain Parquet paths and occurrence
counts:

```python
import json

from rigging.filesystem.buckets import filesystem_for

with open("/tmp/hero-cluster22-plan-v11.json") as stream:
    plan = json.load(stream)

source_name = "hplt_v3"
marker_uri = plan["source_roots"][source_name] + "/reconstructed/_complete.json"
fs, key = filesystem_for(marker_uri)
with fs.open(key, "r") as stream:
    source = json.load(stream)

assert source["complete"]
assert source["missing_ids"] == source["missing_occurrences"] == 0
for output in source["outputs"]:
    print(output["path"], output["occurrences"])
```

Final text Parquet columns are `marin_source`, `id` and `text`. `id` is the
canonical normalized-text `xxh3_128` hexadecimal ID. Repeated output rows carry
the frozen selected multiplicity. Do not deduplicate these files by ID or add
copies just because multiple provider rows have the same content. The final
text files do not contain the proposed public metadata schema, upstream row
locators, embeddings or token order. Quality-bucket membership is in each source entry's `ledgers` list in the plan.
For the `source_name` above, locate nonempty ledger Parquets with:

```python
entry = next(item for item in plan["sources"] if item["marin_source"] == source_name)
for ledger in entry["ledgers"]:
    if ledger["selected_occurrences"]:
        print(ledger["output"], ledger["selected_occurrences"])
```

Those metadata rows contain `marin_source`, `id`, `cluster_id` and
`quality_bucket`, alongside the retained metadata fields. Within a source,
count ledger rows by `(id, quality_bucket)` to recover the selected bucket
multiplicities. Repeated IDs can occur in several buckets; an ID-to-one-bucket
lookup loses information. This does not recover training token order.

## What passed, and what remains uncertain

Each source reducer read all of its pinned provider-file checkpoints, checked
candidate text against canonical IDs and content digests, rejected conflicting
text, required every selected ID and expanded the exact selected multiplicity.
The cluster aggregate then required every current source/index binding, zero
missing occurrences, globally unique output paths, the final Parquet schema and
footer counts, and the exact total across all 150 sources. CTX and Software
Heritage audit validation also ran during aggregation.

Independent output readbacks checked actual SQLite index hashes, file bindings
and output schemas/counts. Small sources received full final-text hash and
multiset checks; larger sources received bounded final-text samples in addition
to the source reducers' complete candidate validation. The aggregate is a
metadata/footer closure check. It does not independently reread every final
text row. Do not describe the result as a second, full-text rehash of all
151 million output occurrences.

Two audited exclusions need to remain in any subsequent report:

- **DaVinci CTX:** the reader reproduced the archived 16-row malformed-UTF8 batch
  exclusion policy. All 1,086 file audits are retained, including 256 excluded
  raw rows across 16 batches. All 16,977 selected occurrences passed full final
  canonical-hash and multiplicity readback.
- **Software Heritage v1:** every nonempty payload had to pass both advertised
  plain SHA1 and Git blob SHA1 before decoding. Three `present=True` empty
  payloads were explicitly audited; two failed their advertised identities.
  There were also 816 absent/empty rows and one hash-verified blank row. The
  820 total filtered rows and 513,109,847 eligible rows match the frozen counters.
  All 436,573 selected occurrences were reconstructed. The empty-row replay is
  historically compatible; exact dirty-tree execution and repaired upstream
  identities are not established. Advertised-identity membership for excluded
  faults remains unknown. V11 preserves these disclosures in the source audit.

Matching text and selected counts does not establish which historical provider
occurrence supplied a duplicate document. Historical occurrence provenance
remains unestablished. The milestone also does not establish full 292-source
release coverage, complete public locators, selected Harrier vector coverage,
training sampling frequency or final token order.

## Implementation map

Paths below are relative to the repository root. Within a table cell, bare
filenames share the directory of the preceding full path:

| Area | Entry points |
| --- | --- |
| Frozen source census and selected ledgers | `experiments/datakit/scripts/metadata_cluster_census.py`, `metadata_cluster_selection.py`, `metadata_cluster_verification.py` |
| Provider-file scans, selected SQLite indices, source reducers and aggregate | `experiments/datakit/scripts/metadata_cluster_reconstruction.py` |
| Text renderers and normalization pins | `experiments/datakit/store/provider_reconstruction.py`, `reconstruct_selected_source.py`, `structured_provider_rows.py`, `davinci_provider_rows.py`, `external_provider_rows.py` |
| SWH empty-payload policy and audit validators | `experiments/datakit/store/software_heritage_empty_rows.py` |
| Common Corpus original-provider reconstruction | `experiments/datakit/scripts/metadata_cluster_external_reconstruction.py`, `metadata_cluster_external_completion.py` |
| Common Crawl WARC replay | `experiments/datakit/scripts/metadata_focus_provider.py`, `metadata_focus_reconstruction.py`, `metadata_focus_completion.py`; `experiments/datakit/store/focus_provider_replay.py` |
| RNO-to-02a checkpoint migration | `experiments/datakit/scripts/metadata_reconstruction_migration.py` |
| Metadata export and duplicate selection | `experiments/datakit/store/metadata.py` |

The generic driver has `prepare`, `index`, `scan`, `reconcile` and `aggregate`
modes. Scans are partitioned by `--task-count`, which must equal the number of
Iris replicas. A file marker certifies only that file's candidates; source
completion requires all pinned files and selected IDs. A source completion
marker cannot substitute for a whole-cluster aggregate.

For later recipe or inventory fixes, create a new source version and fresh root,
preserving unchanged source roots in a new combined plan. Do not relabel old
markers under a new policy. Index-only reuse needs unchanged ledger descriptors
and actual SQLite SHA verification. Incomplete attempts are not checkpoints.
Runtime timeout recovery reused only completed, source/index-bound file markers.

The pushed checkpoint includes four typing annotations added after execution.
The Common Corpus scanner hashes its own source file, so even that annotation
changes its current adapter digest. Existing completed output remains valid
under its recorded pins. A future rerun must check the pinned code and plan
against the intended implementation; do not bypass a digest mismatch or assume
this branch can resume every historical checkpoint unchanged.

Checkpoint validation passed 83 focused tests, scoped typing and the required
lint entry point. To rerun the focused suite:

```bash
uv run --no-sync pytest experiments/datakit/scripts/tests experiments/datakit/store/tests
```

## Coworker takeover

1. Check out the reconstruction branch and inspect the final marker and v11
   plan. No reconstruction job needs restarting for this completed milestone.
2. Decide whether the next deliverable is a reviewable reconstruction tool or
   the broader metadata release. The current branch is an execution checkpoint;
   integration against current main and PR review are separate work.
3. For a metadata release, resolve complete upstream locators and duplicate
   occurrence ambiguity, measure selected embedding coverage, and validate the
   release schema across all intended sources. Keep missing provenance explicit.
4. If another cluster is chosen, freeze its selection and provider plan first.
   Use fresh output roots, a bounded CPU allocation and heartbeat monitoring.
   Never restart a cluster, copy a raw corpus across regions, or publish a
   dataset as an incidental part of reconstruction.

The original local workspace keeps detailed job commands and per-source
readbacks under ignored `scratch/hero-cluster*.json`. These local files are not
part of the branch and are not required to locate the completed outputs. The
remote v11 plan and source markers are the shared artifact inventory. The
terminal aggregate job was
`/dlwh/hero-cluster22-final-aggregate-v11-20261031`; its successful result and
marker hash above are the completion record.
