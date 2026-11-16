# Large Data In Git

The large-data merge guard keeps data dumps out of the repository. Committed
data files are limited to 100 KB each, and one change may add at most 250 KB of
committed data across all its files, so a dataset cannot be sharded into small
pieces to slip through. Data is anything with a data-file suffix (JSON, CSV,
Parquet, NumPy, archives) or a NUL byte in its first 8 KB. The guard runs in
pre-commit and CI lint.

## Publish Data To Hugging Face

Move the data to a private `open-athena` dataset repository and commit a
reference to it instead of the bytes:

```bash
# Copy the file into the dataset repository and capture its revision.
uv run fsutil cp --remote s3://marin-us-east-02a/experiments/my-run/results.json \
  "hf://datasets/open-athena/marin-experiment-data/my-run/results.json"
```

Commit a small JSON reference beside the code that reads the data:

```json
{
  "repo": "open-athena/marin-experiment-data",
  "revision": "<commit sha or tag>",
  "path": "my-run/results.json",
  "sha256": "<sha256 of the file>"
}
```

Readers resolve the reference with the pinned `hf://datasets/<repo>@<revision>/<path>`
URL through `fsutil` or fsspec — see [fsutil](../references/fsutil.md) for the URL
forms and credentials — and verify the downloaded bytes against `sha256`.

## The Allowlist

`infra/large-data-allowlist.json` records the data files that predate the guard
and must stay in the repository. Once `origin/main` carries the allowlist, an
entry may only shrink (never grow), must record exactly the file's committed
size, and must name a file that still exists; new entries are rejected, so a
dataset that was not adopted must be published as above. To move an allowlisted
file out, publish it as above and delete its entry.
