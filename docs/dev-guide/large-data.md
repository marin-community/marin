# Large Data In Git

Do not commit datasets, run outputs, or result dumps. They bloat every clone
and cannot be reviewed like code. The advisory lint rule `ml-committed-data`
flags them in review — see `infra/lint/rules/cruft/ml-committed-data.yaml`.

Publish the data instead and commit a reference to it:

- Object storage, such as `s3://marin-us-east-02a/experiments/my-run/`.
- A Hugging Face dataset, such as `hf://datasets/open-athena/marin-experiment-data/my-run/`.

`fsutil` owns the copy command — see [fsutil](../references/fsutil.md) for the
URL forms and credentials:

```bash
uv run fsutil cp --remote s3://marin-us-east-02a/experiments/my-run/results.json \
  hf://datasets/open-athena/marin-experiment-data/my-run/results.json
```

Readers resolve the reference with the pinned `hf://datasets/<repo>@<revision>/<path>`
URL through `fsutil` or fsspec.

Small test fixtures are fine. The 500 KB large-file check in
`infra/pre-commit.py` remains the hard backstop for individual files.
