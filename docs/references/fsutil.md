# fsutil: browsing Marin's buckets

`fsutil` lists, reads, sizes, copies, and removes objects across every bucket declared in
`config/*.yaml` — GCS, CoreWeave AI Object Storage, and R2 — from one command, and opens
an interactive browser over all of them.

```bash
uv run fsutil                        # interactive browser, starting at the bucket list
uv run fsutil gs://marin-us-central2/scratch   # bare URL: browser on a directory, viewer on a file
uv run fsutil buckets                # what is declared, and whether credentials are set
uv run fsutil ls -l gs://marin-us-central2/scratch
uv run fsutil ls -R gs://marin-us-central2/scratch/checkpoints
uv run fsutil ls 's3://marin-us-east-02a/*/config.json'
uv run fsutil cat s3://marin-us-east-02a/iris/my-job/config.json
uv run fsutil show gs://marin-us-central2/documents/shard-00000.parquet
uv run fsutil du s3://marin-us-east-02a/iris/my-job
uv run fsutil usage s3://marin-us-east-02a -o usage-report.md
uv run fsutil cp -r s3://marin-us-east-02a/iris/my-job/logs /tmp/logs
uv run fsutil mv /tmp/run.json gs://marin-us-central2/archive/
uv run fsutil rsync --dry-run --delete /tmp/checkpoints gs://marin-us-central2/checkpoints
uv run fsutil verified-copy gs://marin-us-central2/exports/model s3://marin-na/marin/exports/model
uv run fsutil hash gs://marin-us-central2/archive/run.json
uv run fsutil rm -R s3://marin-us-east-02a/tmp/expired-prefix s3://marin-us-east-02a/tmp/old-prefix
```

Paths are always full URLs — `gs://bucket/key`, `s3://bucket/key`, or a local path.
There is no current bucket to keep track of, and a copy may name a different backend on
each side.

## Credentials

GCS uses application default credentials:

```bash
gcloud auth application-default login
```

CoreWeave object storage uses an access key pair from the
[CoreWeave console](https://console.coreweave.com/object-storage/access-keys):

```bash
export CW_KEY_ID=<key-id>
export CW_KEY_SECRET=<key-secret>
```

R2 buckets use `R2_KEY_ID` / `R2_KEY_SECRET` the same way. Either backend also accepts
the generic `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` pair, which is enough when only
one of them is in play.

`fsutil buckets` reports which backends are reachable, so an unconfigured one shows up
before a command fails against it. A backend with no credentials only breaks commands
that touch its buckets; the rest keep working.

## Commands

| Command | What it does |
|---------|--------------|
| `ls [URL ...] [-lR]` | Immediate children or glob matches; with no URL, the declared buckets. `-l` adds size and modification time; `-R` lists every descendant beneath a literal URL |
| `cat URL [--raw]` | Print a file. Tabular JSON, JSONL, and parquet render as a table, and `--raw` writes stored bytes to stdout |
| `head URL [-n N]` | Print the first N lines, or the first N rows of a parquet file |
| `stat URL` | The object's metadata as the backend reports it |
| `du URL` | Total bytes and object count beneath a prefix |
| `usage URL [--workers N] [-o REPORT.md]` | Metadata-only prefix breakdown ranked by size and time since the newest write |
| `find PATTERN` | Paths matching a glob, e.g. `gs://marin-us-central2/x/**/*.json` |
| `cp SRC ... DST [-r] [-n]` | Copy one or more sources between any backends. `-r` includes directories; `-n` preserves existing destination files |
| `mv SRC ... DST [-r]` | Move or rename one or more sources. Sources are removed after every copy succeeds |
| `rsync SRC DST [--delete] [--dry-run] [--checksum]` | Synchronize the files beneath two directories or prefixes |
| `verified-copy SRC DST [--workers N] [--status-prefix URL]` | Resume and verify a cross-backend prefix copy, then publish a completion manifest |
| `hash URL ... [--hex]` | Stream complete files and print MD5 digests in base64 or hexadecimal |
| `rm URL ... [-r] [--workers N]` | Remove one or more objects. `-r` or `-R` recursively removes every prefix; remote prefixes delete while they list and show progress |
| `browse [URL]` | The interactive browser |
| `show URL` | The interactive viewer on one file; page-down scans through a parquet's rows |

`cp` and `mv` append each source basename when the destination is an existing directory,
ends in `/`, receives multiple sources, or receives a glob expression. A single literal source
copied to any other destination uses that destination as the exact output name. `-r` is required
for a directory. Transfers stream through the machine running `fsutil`, including copies
between two object stores. Quoted glob sources are expanded by the source filesystem. `mv`
finishes all copies before removing any source. Local-to-local `cp -r` and `mv -r` preserve
empty directories.

`rsync` compares size and modification time first. For equal-sized local files with different
times it compares MD5; object stores use provider MD5 metadata when both sides supply it. When
neither side supplies comparable timestamps or digests, equal sizes are treated as unchanged.
`--checksum` reads both equal-sized files and compares their MD5 digests regardless of metadata.
Extra destination files remain by default. `--delete` removes them after all copies succeed.
`--dry-run` prints the same copy and delete plan without changing the destination. Source and
destination directories may not overlap.

`verified-copy` hashes each source object while uploading it. For S3-compatible destinations, it
uses fixed-size multipart parts, calculates the corresponding content-derived ETag, and compares
that value with destination metadata after the upload. Other destinations are read back and checked
against the source SHA-256. The command records the SHA-256 and source and destination identities
under a sibling `<DST>.verified-copy-status` prefix. A retry skips both object reads when those
identities, the destination path and size, and the verified record still match. Sources without
stable generation, version, checksum, ETag, or modification metadata are read again. The command
writes `<DST>/.verified-copy-manifest.json` after every object verifies; consumers should treat that
manifest as the prefix's readiness marker. An existing completion manifest makes the destination
immutable: the command returns immediately when source paths and sizes still match and fails if they
changed. Use a fresh destination prefix for a changed export.

`hash` reads each complete object. Its columns are `url` and `md5`; digests use base64
by default. `--hex` selects hexadecimal output.

`du` scans prefixes with up to 128 concurrent metadata-bearing listings. A prefix that
exceeds one listing page is split at the next `/` through three directory levels, then
paginated flat. Both object stores drive their own paging call, so a prefix with millions
of objects directly below it arrives one page at a time.

Recursive `rm` on a remote prefix runs on that same parallel page scanner and deletes
each page as it lands, rather than scanning the whole prefix first. Deletes start
immediately, and the objects held in memory are bounded by the requests in flight
instead of by the size of the prefix. One batch is one request, at each backend's
documented maximum: 1,000 keys for S3 `DeleteObjects`, 100 sub-requests for the GCS
batch endpoint. `--workers` sets the requests in flight, defaulting to 16 and accepting
up to 256. Failed S3 batches are retried with backoff on transient errors.

`--workers` is worth raising on S3, which serves a much higher write rate, and worth
lowering to delete politely beside a running job. On GCS the default is already at the
bucket's ceiling: it admits roughly 1,000 writes per second before it returns 429, and
deletes count as writes. 60M objects therefore need about 17 hours whatever the client
does. At that scale an object lifecycle rule costs nothing and needs no listing. It is
the better tool, and it is why throwaway data belongs under a `ttl=` prefix that a rule
already covers.

`usage` uses the same parallel metadata-page scanner as `du`, defaults to 128 workers,
accepts up to 1,024 workers, and writes a Markdown report. Starting at the bucket root,
it descends into every prefix at or above the 1 TiB threshold and shows the resulting
exact prefixes in descending size order.
`--prefix-threshold` changes that threshold and accepts values such as `1TB`, `1TiB`,
or `512GiB`. `--prefix-depth` controls how many path components the in-memory scan
retains, with a default of three.

An interactive terminal shows active listing threads, a cropped prefix currently
returning pages, remaining open prefixes, listing pages, objects, logical bytes
cataloged, and object and byte rates. Captured logs get the same counters every ten
seconds without terminal control characters.

Deletion candidates are ranked by stale TiB-years: prefix size in TiB multiplied by the
years since its newest object write. This favors prefixes that are both large and
inactive. `[objects]` labels cover objects written directly under the preceding prefix,
and `[root objects]` covers the bucket root. All other rows are concrete prefixes.

`cat`, `head`, and the browser decompress `.gz`, `.bz2`, `.xz`, and `.lzma` files by
suffix. A `data.json.gz` preview uses the JSON table view. `cat --raw` writes the compressed
bytes.

Formatted file previews are capped at 10 MB after decompression. `cat --raw` is capped
at 10 MB of stored bytes. Use `cp` to fetch a whole object.

A `.parquet` file is read through its footer rather than from the head of the object,
so `cat`, `head`, `show`, and the browser show its schema, its row count, and its rows.
`head -n` bounds the rows, not the printed lines, and `cat` and `head` read from the
first row group alone. `show` and the browser's viewer instead page through the whole
file: page-down decodes the next batch of rows, row group by row group. Parquet's
smallest readable unit is a whole column chunk, so a row group that decodes to more
than 10 MB is reported and skipped instead of read — copy that file to read it. The
footer itself is read whatever its size, so the 10 MB cap above bounds the column data
rather than the whole preview.

Parquet previews need pyarrow in the environment. It is not a `marin-rigging`
dependency, because rigging sits under every other package, but the marin workspace
installs it.

## The browser

`fsutil browse` starts at the bucket list, so descending into GCS or CoreWeave is the
same keystroke. A bare `fsutil URL` skips the command name: a directory opens in the
browser and anything else in the viewer.

| Key | Action |
|-----|--------|
| `j` / `k`, arrows | Move; `g` / `G` jump to top and bottom |
| `enter` / `l` | Open a prefix, or preview a file |
| `backspace` / `h` | Up one level |
| `/` | Filter the listing by name |
| `s` | Cycle sort: name, size, modified |
| `d` | Total the highlighted prefix |
| `y` | Show the highlighted entry's full URL |
| `r` | Re-list |
| `q` | Quit, or close the file viewer |

Opening a file — from the browser or with `fsutil show URL` — pages with `j` / `k`,
page-up / page-down, and `g` / `G`. Paging down a parquet file keeps decoding rows
until the file ends; other formats show the bounded preview.

## Where the bucket list comes from

Buckets and backends are declared in `config/*.yaml` under `data.region_buckets`, and
each S3-compatible backend's endpoint and credential variables under `data.stores`.
`fsutil` builds one filesystem per backend from that config, so adding a bucket is a
config change rather than a code change. The same routing is available to library code
as `rigging.filesystem.buckets.filesystem_for(url)`, which is the way to reach two S3 backends
from one process — the process-wide `AWS_*` / `FSSPEC_S3` variables can only describe
one at a time.
