# H100 evidence Nsight Compute units row

## Failure boundary

The fifteenth reviewed H100 launch reached the first Nsight Compute worker
after source authentication, generated-backend compilation, numerical gates,
Nsight Systems timing, and cache-protocol validation. Nsight Compute produced
its report, raw CSV, and public SASS export. The runner then interpreted the
first raw-page CSV record as a kernel and rejected its empty `Kernel Name`.

The negative launch is sealed separately under
`h100_contract_map_evidence_fifteenth_launch_failure_84123e_v0`. The complete
task log is 13,220 bytes with SHA-256
`170ca215b9964429802731e0d8825eacbc14c0d800495e0da7f9b74b69dd9a5c`.
No timing report or 24-record evidence bundle was accepted, and the job was not
retried.

## Format audit

Pinned Nsight Compute raw-page CSV is wide rather than one metric per row. Its
first record is a units row: all 11 identity columns are empty, while each
requested metric column contains its unit. The retained failure exposed a
274-column header whose ordered-name SHA-256 is
`5b728a44b7b41580760ba03a3fcd2f1b1a203703a404261969ca45868f91a298`.
Only the closed identity and requested-metric projection is retained; unrelated
profiler columns and the 11,992-byte raw diagnostic are not checked in.

The requested metric units are an empty dimensionless block-size unit,
`register/thread`, `byte/block`, `block`, and `%`, according to metric role.

## Repair

The parser now requires the 11 standard identity columns and all requested
metric columns, one exact units row in first position, and at least one later
kernel row. It rejects missing or duplicate columns, missing, reordered,
repeated, or mutated units rows, empty later identity rows, duplicate wide
kernel rows, and invalid metric values. Extra profiler columns remain opaque;
the evidence contract neither exposes nor depends on the other 254 columns.
Exact kernel identity and metric coverage checks remain unchanged downstream.
The CSV must also be a regular, NUL-free UTF-8 file within a 1 MiB bound, and
malformed-row errors identify only the required field rather than serializing
the profiler row.

A bounded real-format fixture covers the observed wide shape. Mutation tests
exercise every requested unit, missing and duplicate units rows, reordered
rows, blank kernel data, and unit strings used as data. A production-boundary
test executes `_run_ncu_profile` with only the external profiler process and
SASS-export call replaced, so the real parser, report hashing, SASS validation,
and evidence construction are observed together.

## Validation boundary

This is a source-only repair. It does not build an image, query a GPU, relaunch
the H100 job, or claim that later profiler and final-bundle gates pass.
