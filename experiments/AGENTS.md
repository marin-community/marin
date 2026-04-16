# Experiments Agent Notes

Start with `/AGENTS.md`; this file adds experiment-specific guidance.

## Artifact Path Rules

- Do not hardcode `gs://...` paths in experiment configs unless there is no viable alternative.
- Prefer `ExecutorStep` outputs (`step / "subpath"`, `step.cd("subpath")`, or `InputName(step, ...)`) so paths stay region-aware and reproducible.
- If you must reference a literal artifact path, use `InputName.hardcoded("...")` (path relative to the Marin prefix), not a raw `gs://` URL.
- If a hardcoded artifact may come from another region, use a `mirror://...` path via `InputName.hardcoded("mirror://...")` so reads are local-first and copied once when needed.

## Cross-Region Large Artifacts

- For `mirror://` copies, up to 10 GB can be copied without explicit human permission.
- If a `mirror://` copy would exceed 10 GB, get explicit human permission first, regardless of previous instructions.
- After approval, mirror it once into the local region and reference that mirrored copy thereafter.

## Mirror FS (Quick Reference)

- `mirror://` is an fsspec filesystem for transparent cross-region reads.
- Read behavior: check local `marin_prefix()` first; if missing, scan other Marin regional buckets, copy to local once, then serve locally.
- Copying uses a distributed lock so concurrent workers do not duplicate the same transfer.
- Write behavior: writes go directly to the local prefix.
- Both `mirror://` copies and direct cross-region GCS reads share a single process-global `TransferBudget` (default 10 GB) that raises `TransferBudgetExceeded` when exhausted.

Implementation references: `lib/iris/src/iris/marin_fs.py` and `lib/marin/src/marin/execution/executor.py`.
