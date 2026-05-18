# Clustering pipeline — full-corpus run journal

End-to-end log of producing K=5000 / K=1000 / K=40 Luxical-One clusters
across every Datakit source, plus the quality analysis. Each section is
filled in as the work happens; intentionally rough notes, not a polished
report.

## Goal

Produce three artifacts under `gs://marin-eu-west4/datakit/cluster/`:

- `cluster/sample_centroids_<hash>/` — ~10M-row stratified embedding sample
  (per-source parquet shards, int8)
- `cluster/train_centroids_<hash>/` — K=5000 FAISS centroids + lookups to
  K=1000 and K=40
- `cluster/assign/<source>_<hash>/` × 113 — per-source `AssignmentAttrData`
  (cluster id at each K, plus dist to k_train centroid)
- `cluster/summarize_k{40,1000,5000}_<hash>/cluster_stats_{K}.json` —
  per-cluster top c-TF-IDF terms + representative excerpts

…then evaluate cluster quality (precision/recall-style) and write it up.

## Architecture

DAG (`experiments/datakit/cluster/exp_full_clusters.py`):

```
for src in all_sources():
    embed_<src>     EmbeddingAttrData (int8, 192-dim, co-partitioned)
        │
        └─► sample_centroids (per-source Zephyr parquet shards)
                │
                └─► train_centroids (FAISS K=5000 + agglom-merge → K=1000/40)
                        │
        assign_<src>  ◄─┘   AssignmentAttrData (co-partitioned)
            │
            └─► summarize_k{40,1000,5000}  cluster_stats_{K}.json
```

Embedding model: `DatologyAI/luxical-one` (192-dim, int8-quantized at
scale 0.6/127). Trained with spherical K-means; coarser K views derived
from agglomerative-merge (`method="average"` on cosine distance) of the
K=5000 centroids — keeps the per-doc assignment a single FAISS search.

## Run history

Each rerun preserves prior outputs via `StepRunner` cache hits (same
`hash_attrs` + same upstream dep hashes → same `output_path`).

| Job | Started (UTC) | Outcome | Notes |
| --- | --- | --- | --- |
| `embed-clusters-full-20260515-150702` | 2026-05-15 15:07 | 10 embeds failed (OOM / worker-ping) | Pre-rename baseline; surfaced the high-RAM sources |
| `embed-clusters-full-20260516-002008` | 2026-05-16 00:20 | killed (priority issue, see feedback memory) | first rerun, used `--priority production` mistakenly |
| `embed-clusters-full-20260516-002552` | 2026-05-16 00:25 | killed | 7/8 high-RAM embeds succeeded at 32g; finepdfs still OOM at 32g |
| `embed-clusters-full-20260516-010941` | 2026-05-16 01:09 | failed | finepdfs OK at 64g/window=1024; transient GCS 416 on climblab-ja status file |
| `embed-clusters-full-20260516-031124` | 2026-05-16 03:11 | killed (manually) | ClimbLab-Ja download (~400 GB, ~3.5h); single-process sample stuck → preempted → restarted |
| `embed-clusters-full-20260516-152209` | 2026-05-16 15:22 | killed (bug) | First Zephyr-sample attempt; `string indices must be integers` from missing `.window()` |
| `embed-clusters-full-20260516-154942` | 2026-05-16 15:49 | killed | Zephyr coord OOM tracking 100K tasks (one big context) |
| `embed-clusters-full-20260516-160321` | 2026-05-16 16:03 | killed | Per-source contexts but serial — ~1.7 min/source projected ~3h |
| `embed-clusters-full-20260516-172653` | 2026-05-16 17:26 | killed (after train) | parallel=8 contexts; sample done in 49 min; train load 100K shards serially → slow |
| **`embed-clusters-full-20260516-192054`** | 2026-05-16 19:20 | **in flight** | train load parallelized to 64 threads; all stages succeeded up to summarize |

## Knobs that matter (final)

- `EMBED_WINDOW = 4096`, `HIGH_RAM_EMBED_SOURCES` get `ram=32g`,
  `finepdfs` gets `ram=64g, window=1024` (its row groups have very long
  PDF docs)
- `N_PER_SOURCE_FOR_SAMPLE = 100_000` → stratified sample ≈ 10.5M rows
  (sources smaller than the cap contribute everything they have)
- `K_TRAIN = 5000`, `K_VIEWS = (40, 1000)`
- Sample: 8 parallel `ZephyrContext`s, each with up to 128 workers
- Train load: 64-thread parallel parquet read (was the dominant bottleneck
  before the fix — 100K small files × GCS round-trip)

## Decision log (key gotchas)

1. **Per-source coord vs global coord.** A single ZephyrContext spanning
   all (source, shard) pairs OOM'd its coordinator at ~100K tasks. Split
   into one context per source.
2. **`.window()` is required.** Without it, `flat_map(load_file)` produces
   individual records and `map_shard` doesn't get the batched
   `Iterator[list[dict]]` shape its body assumed.
3. **Parquet output > npz.** Single `sample.npz` made sample non-resumable
   and forced the upstream concat to happen at sample time. Per-shard
   parquet survives preemption, defers concat to train.
4. **64-thread parquet load.** 100K small files × ~100ms GCS round-trip =
   hours serially. ThreadPoolExecutor flips the bottleneck to bandwidth
   instead of latency.

## Results (post-run)

_To be filled in once `cluster/summarize_k{40,1000,5000}` succeed._

### Cluster size distribution
_TBD_

### Source × cluster co-occurrence
_TBD_

### Top terms (eyeball)
_TBD_

## Quality analysis

_To be filled in._

### Approach

Precision/recall on unsupervised clusters needs a stand-in for "ground
truth". Two complementary measures:

- **Intra-cluster precision (Claude oracle).** Sample N docs from each
  of M clusters; ask Claude to judge whether they share a coherent
  topic/style. Score = fraction of clusters judged coherent. Run at
  K=40 (cheap, eyeballable) and K=1000 (the main downstream consumer).
- **Source-purity recall.** Sources with natural sublabels — `starcoder2/*`
  (per-language programming corpora), `finepdfs/*` (per-language web
  PDFs), `safety_pt/*` (per-policy) — should partition cleanly across
  clusters. Compute the % of mass that each source's dominant cluster
  captures; high = good recall of the "this-source-belongs-together" signal.

Anthropic API key is read from `.marin.yaml` (`ANTHROPIC_API_KEY` env
var); never printed in artifacts or logs.

### Results

_TBD_

## Status

Updated as the in-flight run progresses.
