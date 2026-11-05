# BioTasks verifiable-task source library

This campaign starts from the source-project inventory in
[BioTasks issue #5](https://github.com/Open-Athena/biotasks/issues/5) and the
document inventory in [issue #16](https://github.com/Open-Athena/biotasks/issues/16).
The initial cohort is deliberately limited to the 57 counted, content-confirmed
documents with individual source review at BioTasks revision
`a342f41d9d0674cbe280a33a2f537285547552d3`.

`import_reviewed.py` downloads those exact source bytes, verifies every recorded
SHA-256, and stores each unique document once under the private content-addressed
prefix:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/objects/sha256/...
```

It then writes an immutable campaign catalog and publishes `manifest.json` last:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/
  campaigns/biotasks/reviewed57-a342f41d-v1/
    source-catalog.jsonl
    manifest.json
```

Run from the Marin repository root after loading the CoreWeave object-storage
credentials into `CW_KEY_ID` and `CW_KEY_SECRET`:

```bash
uv run python \
  experiments/post_training/capability_driven_envs/biotasks_library/import_reviewed.py
```

The importer executes no notebook code. Its output remains
`internal_unreviewed_do_not_publish`: individual source review in BioTasks is
useful selection evidence, not a redistribution-rights audit, environment audit,
or finding that a document can support a verifiable task. The catalog is an
input to later checklist, curator-packet, proposal, and Taskforge campaigns.
Source project identity must remain the split boundary when tasks are derived.

## Notebook objective-task campaign

`run_notebooks_v1_zephyr.py` processes the complete pinned cohort with one outer
Zephyr shard per source document. It strips Jupyter outputs, asks the author model
for zero to four concrete objective ideas without imposing a quota, runs each
hydrated idea through Taskforge, and obtains a separate blinded usefulness review.
`finalize_notebooks_v1.py` materializes compact campaign views and screening bands.

The complete recovered snapshot is stored at:

```text
s3://marin-us-east-02a/marin4sc/taskforge/task-library/
  campaigns/biotasks/notebooks/v1/snapshots/complete-1/
```

It contains 57 source dispositions, 87 non-null ideas from 49 productive sources,
and eight explicit null-source records. Taskforge structurally accepted all 87
non-null TaskSpecs; this only establishes buildability and valid reference/negative
controls. The blinded GLM screen recommended 40 accept, 17 revise, and 27 reject,
with three review calls retained as errors. These screening bands are not scientific
review, rights approval, solver calibration, or release approval. All artifacts
remain `internal_unreviewed_do_not_publish`.
