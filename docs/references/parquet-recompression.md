# Recompressing Parquet datasets

`scripts/ops/storage/recompress_parquet.py` migrates Parquet objects to zstd
level 3 with page indexes. It assigns one object to each Zephyr task and uses 16
workers by default.

Run the job on a cluster local to the bucket. Start with one dataset prefix;
bucket-wide globs can contain millions of objects and produce a correspondingly
large Zephyr plan. Omitting `--apply-to-quiescent-prefix` reads Parquet footers
and reports candidate objects, bytes, and rows without writing data.

```bash
uv run iris --cluster=marin job run \
  --cpu 1 --memory 2GB --disk 8GB --priority batch \
  --target-cluster cw-us-east-02a -- \
  python scripts/ops/storage/recompress_parquet.py \
  's3://marin-us-east-02a/marin/normalized/example/**/*.parquet' \
  --workers 16
```

Before applying the migration, stop every producer that can write to the
selected prefix. S3-compatible object stores do not provide a portable
compare-and-swap operation for replacing an object. The migration checks the
source fingerprint immediately before replacement, which detects changes made
while the shard was read, but a writer racing with the final copy could still
be overwritten. Keep the prefix quiescent until the Iris job succeeds.

Repeat the command with `--apply-to-quiescent-prefix` after confirming the
dry-run counters and producer shutdown. Each task streams the source in bounded
batches, buffers practical output row groups, and writes a `.tmp.<uuid>`
sibling. It validates the schema, row count, zstd codec, and page indexes before
replacing the source. Before writing, a retry removes abandoned temporary
siblings for that source. A running 16-worker pool creates at most 16 new staged
objects. After a whole-job interruption, rerun the same glob to reclaim earlier
siblings as their sources are retried. Object-store copy and incomplete-
multipart overhead are separate from staged objects.

Reruns skip zstd objects that already have page indexes. A non-zstd rewrite is
discarded when it is not smaller. Existing zstd objects without page indexes
retain the rewritten output even if the indexes add a small amount of metadata.

The command rejects `tmp/ttl=Nd` inputs because replacement would restart the
object lifecycle clock.

## Page-index calibration

Page-index overhead depends on the number of data pages and indexed columns.
On one 409,438,075-byte Snappy shard from
`marin/datakit/tokenize/svg_a50a1068`, the migration produced nine row groups.
With PyArrow's 20,000-row page cap, the zstd output was 270,937,075 bytes
without page indexes and 270,942,306 bytes with them. The page indexes added
5,231 bytes, or 0.00193%.

At the selected 256-row cap, the output was 270,963,059 bytes without indexes
and 270,973,452 bytes with them. The indexes added 10,393 bytes, or 0.00384%.

The same local shard showed a compression knee when varying the maximum rows
per page. These outputs all include page indexes:

| Maximum rows per page | Output bytes | Change from 1,024 |
|---:|---:|---:|
| 1,024 | 270,935,826 | baseline |
| 512 | 270,959,443 | +0.0087% |
| 256 | 270,973,452 | +0.0139% |
| 128 | 275,114,056 | +1.54% |
| 64 | 281,425,100 | +3.87% |

Zephyr uses 256 rows per page. This gives four times finer page-level pruning
than 1,024 rows while retaining the same compression on this shard. Smaller
pages need evidence from representative query workloads to justify their
storage cost.

Calibrate row-group sizes separately against query pruning and CPU. Compare 64,
128, and 256 MiB uncompressed row-group targets on representative predicate
queries, recording output bytes, bytes read, elapsed time, and CPU. PyArrow
targets 1 MiB data pages by default and does not currently use page indexes to
reduce its own reads, so include the engines that motivated the index request
in the query benchmark.
