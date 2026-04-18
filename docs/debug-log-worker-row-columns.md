# Debugging Log for worker-row-columns

Investigate `sqlite3.OperationalError: no such column: total_cpu_millicores` from the Iris controller scheduling loop.

## Initial status

The controller started selecting `WorkerRow` columns:

- `total_cpu_millicores`
- `total_memory_bytes`
- `total_gpu_count`
- `total_tpu_count`
- `device_type`
- `device_variant`

but the `workers` table schema created at startup did not include them. The failure surfaced in the scheduling loop as an unhandled thread exception.

## Hypothesis 1

The denormalized worker scheduling fields were added to controller code paths, but no migration added them to the `workers` table.

## Changes to make

- Add an idempotent migration that adds the worker scheduling columns.
- Backfill existing rows from `metadata_proto`.
- Add a deterministic test that fresh `ControllerDB` startup exposes the new columns.

## Results

Confirmed by inspecting fresh `ControllerDB` startup: `PRAGMA table_info(workers)` did not contain the new columns before this fix.

## Future Work

- [ ] Add coverage for other denormalized row-model columns so missing migrations fail earlier.
