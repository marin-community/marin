# Vendored Marin Agent Guidelines

!!! note
In this file, we use leading `/` to refer to paths relative to the repository root.

The `/lib/marin` directory contains a vendored copy of Marin. Read the shared instructions in `/AGENTS.md` first; the notes below call out directory-specific expectations.

## Workflow Notes


* Build on the shared practices in `/AGENTS.md`; capture only `/lib/marin`-specific expectations here.
* Keep this guide updated whenever you learn a `/lib/marin`-specific best practice.
* Capture packaging or release checklists in `/docs/recipes/` so other agents can repeat them.

## Packaging and Code Layout

* The packaging metadata lives in `/lib/marin/pyproject.toml`. Update version pins, extras, or dependency groups together with the corresponding upstream change.
* When you add new modules under `/lib/marin/src/`, follow the shared docstring and import conventions from the root guide and ensure the files remain automation-friendly.

## Data Access

* Do not special-case Google Cloud Storage unless absolutely necessary. Use `fsspec.open` and other fsspec helpers so code stays filesystem-agnostic.
* You generally should not copy data artifacts (for example `.json` or `.parquet` files) to the local filesystem when writing Python code—stream them through fsspec instead.
* Avoid hard-coding GCS paths such as `gs://marin-us-central2/foo/bar`. Prefer referencing pipeline steps; if you must inject a literal path, wrap it with `InputName.hard_coded` and call out the follow-up risk. This is **critical** for large files or directory trees.
* Again, NEVER EVER EVER load GCS files from across region if they are more than a few MB.

## Testing

* Re-run the relevant tests with `uv run --package marin pytest` targeting any suites that exercise the vendored package (for example a future `lib/marin/tests` directory) before submitting changes.
* Do not relax tolerances or add hacks to silence failures—fix the underlying issue or coordinate on the shared guide.

Add any additional directory-specific conventions here as they emerge.
