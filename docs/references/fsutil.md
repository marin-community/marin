# fsutil: browsing Marin's buckets

`fsutil` lists, reads, sizes, copies, and removes objects across every bucket declared in
`config/*.yaml` — GCS, CoreWeave AI Object Storage, and R2 — from one command, and opens
an interactive browser over all of them.

```bash
uv run fsutil                        # interactive browser, starting at the bucket list
uv run fsutil buckets                # what is declared, and whether credentials are set
uv run fsutil ls -l gs://marin-us-central2/scratch
uv run fsutil ls 's3://marin-us-east-02a/*/config.json'
uv run fsutil cat s3://marin-us-east-02a/iris/my-job/config.json
uv run fsutil du s3://marin-us-east-02a/iris/my-job
uv run fsutil usage s3://marin-us-east-02a -o usage-report.md
uv run fsutil cp -r s3://marin-us-east-02a/iris/my-job/logs /tmp/logs
uv run fsutil rm -R s3://marin-us-east-02a/tmp/expired-prefix
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
| `ls [URL] [-l]` | Immediate children or glob matches; with no URL, the declared buckets. `-l` adds size and modification time |
| `cat URL [--raw]` | Print a file. Tabular JSON, JSONL, and parquet render as a table, and `--raw` writes stored bytes to stdout |
| `head URL [-n N]` | Print the first N lines, or the first N rows of a parquet file |
| `stat URL` | The object's metadata as the backend reports it |
| `du URL` | Total bytes and object count beneath a prefix |
| `usage URL [-o REPORT.md]` | Metadata-only prefix breakdown ranked by size and time since the newest write |
| `find PATTERN` | Paths matching a glob, e.g. `gs://marin-us-central2/x/**/*.json` |
| `cp SRC DST [-r]` | Copy between any two locations, including across backends |
| `rm URL [-r]` | Remove an object. `-r` or `-R` recursively removes a prefix; remote prefixes show object-count progress |
| `browse [URL]` | The interactive browser |

`du` scans prefixes with up to 32 concurrent metadata-bearing listings. Recursive `rm`
uses S3's 1,000-key bulk delete and GCS's 20-key batch delete, with up to eight batches
in flight.

`usage` uses the same parallel, bucket-routed scanner as `du`, then writes a Markdown
report. Prefixes are split as deeply as possible while keeping each displayed group at
least 1 TiB by default. Smaller siblings are shown as a lexical range, while every
top-level prefix remains visible even when it is smaller. `--min-prefix-size` changes
the floor and accepts values such as `1TB`, `1TiB`, or `512GiB`.

Deletion candidates are ranked by stale TiB-years: group size in TiB multiplied by the
years since the newest object write in that group. This favors prefixes that are both
large and inactive. A ranged row such as `users/iris/a/ … d/` is a display rollup.
Select concrete prefixes from the range before using `rm`.

`cat`, `head`, and the browser decompress `.gz`, `.bz2`, `.xz`, and `.lzma` files by
suffix. A `data.json.gz` preview uses the JSON table view. `cat --raw` writes the compressed
bytes.

Formatted file previews are capped at 10 MB after decompression. `cat --raw` is capped
at 10 MB of stored bytes. Use `cp` to fetch a whole object.

A `.parquet` file is read through its footer rather than from the head of the object,
so `cat`, `head`, and the browser show its schema, its row count, and its first rows.
`head -n` bounds the rows, not the printed lines. Parquet's smallest readable unit is a
whole column chunk, so the rows come out of the first row group alone, and a first row
group that decodes to more than 10 MB is reported instead of read — copy that file to
read it. The footer itself is read whatever its size, so the 10 MB cap above bounds the
column data rather than the whole preview.

Parquet previews need pyarrow in the environment. It is not a `marin-rigging`
dependency, because rigging sits under every other package, but the marin workspace
installs it.

## The browser

`fsutil browse` starts at the bucket list, so descending into GCS or CoreWeave is the
same keystroke.

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

## Where the bucket list comes from

Buckets and backends are declared in `config/*.yaml` under `data.region_buckets`, and
each S3-compatible backend's endpoint and credential variables under `data.stores`.
`fsutil` builds one filesystem per backend from that config, so adding a bucket is a
config change rather than a code change. The same routing is available to library code
as `rigging.filesystem.filesystem_for(url)`, which is the way to reach two S3 backends
from one process — the process-wide `AWS_*` / `FSSPEC_S3` variables can only describe
one at a time.
