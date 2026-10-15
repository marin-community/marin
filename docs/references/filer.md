# Filer

Filer is a private Marina applet for browsing object storage and inspecting data.
Open [Filer](https://applets.marina.oa.dev/a/7f536ce3-0562-40f1-b209-e6110d8ab4e8/),
choose a configured bucket, or paste an `s3://bucket/key` or `gs://bucket/key` URL.
Bucket routing comes from the same cluster configuration as `fsutil`. Access uses
Marina's service credentials; buckets outside that account's permissions return an
access error. Local filesystem paths and HTTP URLs are rejected.

Folder listings load one backend page at one directory level. **Load next listing
page** appends another page. Name filtering and sorting cover the loaded entries.
Breadcrumbs and **Parent** navigate upward. **Copy link** preserves the location in
the URL; recent locations are stored in the browser.

| File | Views |
| --- | --- |
| Parquet | Paged table, schema, JSON, column selection, record inspector |
| CSV, TSV, JSON arrays, JSONL, NDJSON | Table, JSON, source text, record inspector |
| JSON objects, YAML | Formatted JSON and source text |
| Markdown | Sanitized document and source text |
| Text, logs, code, XML, HTML, SVG | Source text; HTML and SVG are displayed without execution |
| PNG, JPEG, GIF, WebP, AVIF | Image |
| PDF | Rendered pages with previous/next controls |
| MP3, WAV, Ogg, FLAC, MP4, WebM | Browser audio/video controls; codec support depends on the browser |
| ZIP, TAR, compressed TAR | Member table; contents are never extracted |
| Other binary files | Hex preview when the initial bytes contain a NUL |

Click a table row to inspect its fields. Nested objects, arrays, and JSON stored in
strings expand into a tree. Table sorting and search cover the current page only.
**Export page** downloads the current page as JSON. Parquet column selection changes
which columns the backend reads. Text table column selection projects the response after parsing the bounded
preview. It does not reduce object-store reads.

## Read limits

- Parquet: footer at most 4 MiB; selected column chunks at most 32 MiB uncompressed
  per row group; at most 200 returned rows. Oversized groups first return their
  schema, an empty table, and a notice directing the user to select fewer columns. Paging uses footer row counts
  to skip preceding groups; a large offset within a group still decodes preceding
  batches in that group.
- Text: at most 10 MiB after decompression, with at most 2 MiB of source text sent
  to the browser. Text tables retain at most 10,000 rows. Truncation is disclosed.
  `.gz`, `.bz2`, `.xz`, and `.lzma` text previews are decompressed by fsutil.
- Media and archives: objects larger than 10 MiB are rejected. For compressed
  archives this limit applies to compressed size. Archive listings retain at most
  5,000 members. Media is fetched as a bounded whole object, without a streaming range API.
- JSON API responses: at most 4 MiB. A page that exceeds the limit returns an error;
  reduce its row count or, for Parquet, select fewer columns.

Filer does not query entire datasets or modify storage. A displayed total for a
truncated text table counts the retained preview rows. Parquet totals come from
its footer. Reads may cross regions, but opening a file does not copy a dataset.

## Preview, release and operate

Sources live in `infra/marina/applets/filer/`. The applet UUID is
`7f536ce3-0562-40f1-b209-e6110d8ab4e8`. Run these commands from the repository root.
Preview branch work, including uncommitted edits, with Docker running:

```bash
uv run marina release infra/marina/applets/filer --local
```

This builds the package, starts disposable Postgres and Marina, and prints a local
revision URL. It does not require main or a clean checkout and does not update the
live applet. Stop it with Ctrl-C. Use `--dry-run` instead for build/import validation
without starting the local stack.

The local backend uses your shell's storage credentials, not Marina's deployed
service account. Configure GCS ADC or the relevant S3 credentials as described in
[fsutil](fsutil.md). Filer reads real objects with those permissions; it exposes
no storage writes.

For a production release, validate and inspect the current revision first:

```bash
uv run marina validate infra/marina/applets/filer
uv run marina release-status infra/marina/applets/filer --json
uv run marina applets versions 7f536ce3-0562-40f1-b209-e6110d8ab4e8 --json
uv run marina release infra/marina/applets/filer --base-version <current>
```

Replace `<current>` with the current version reported by `marina applets versions`.
Production release defaults to a clean checkout at the current remote main.
`release.toml` binds this directory to its UUID; a default release records the source commit and
subtree hash. `release-status` compares that tree with the local checkout.
Use `--allow-dirty` to release branch or uncommitted source without verified Git
provenance; `release-status` then reports `unrecorded` from a clean checkout.

Keep the applet private: its backend uses Marina's storage credentials. Backend
code runs as a trusted plugin in Marina's process. It needs no database tables or
migrations. Frontend dependencies are bundled; PDF rendering uses a packaged worker.

`infra/marina/Pulumi.marin-marina.yaml` declares `filer.marina.oa.dev` in
`marin-marina:applet_domains`. The Marina stack manages its DNS and domain mapping.
The hostname-to-applet binding lives in Postgres's `marina.applet_hosts` table
and is resolved on each request, independently of the serving Cloud Run revision.
The deployment migration preserves the existing Filer binding. Applet operators
can change it with `uv run marina applets hosts bind <hostname> <applet-id>`.
The named host redirects its root to the current applet revision at `/v/<revision>/`.
Release selects a new current revision for new visits; existing revision URLs
continue to serve that revision while it is retained.
It uses the existing Marina IAP authentication. Production rollouts load the DNS
token from Secret Manager through `marin-deploy marina rollout`.
