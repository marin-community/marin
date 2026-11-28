# Ingest entrypoint retirement

The maintained entrypoints are `atlas-ingest` and `atlas-repair`. Historical
launchers under `new-rl-data-ingest/agent-*` are retired as ingestion entrypoints.
They remain evidence; the command never imports or executes them. Dataset-specific
conversion belongs in the checked-in dataset modules selected by reviewed config.

The frozen full inventory records 135,686 paths: 131,808 copied scripts, 2,691
original scripts, 1,165 wrappers mentioning maintained helpers, three friction
records and 19 maintained helpers. The external task record contains
`inventory.tsv` (SHA-256
`260e8e8f433b7562f9ac525fe598136ba8c5e529ac2e2ee77ef62d0bd5a853f1`) and the
per-path `retirement.tsv` (SHA-256
`0397525ec0ab3db3faba97f4340267bc1279b4938760235bb7975574c5d3a5ed`). These large
historical ledgers are not committed as result dumps. Retirement means a script
is no longer an authorized command dependency; it does not assert that every
historical converter or experiment was ported.

| Historical responsibility | Maintained destination |
| --- | --- |
| Queue selection, concurrent pickup | `queue.py`, `requester.py`, `host_inputs.py` |
| Candidate license/provenance, original receipt inheritance | `licensing.py`; existing `license_check`, `license_resolution`, `license_publication` |
| Pinned regional staging and raw snapshot readback | `upstream.py`, `raw_archive.py`; existing HF downloader and `raw_source_publish` |
| Format adaptation and TaskSpec production | `conversion.py`, `converter_binding.py`; existing QUICK `process_rows` |
| Dedup, complete actor/integrity census | `normalized_population.py`; existing `normalize_quick`, `published_census`, `census_binding` |
| Fingerprints, locators and overlap CI | `population.py`, `overlap.py`, `registry_update.py`; existing fingerprint, population-cache and duplicate-scan modules |
| Backend survey and reference/noop proof | `backend.py`; existing `backend_capabilities` and taskcompendium controls |
| Lexical scan and retained selection | `lexical.py`, `publication.py`; existing `lexical_scan`, `within_source_census`, `remediation_publish` |
| SAMPLE panel, launcher and immutable observations | `qc.py`, `submission.py`; existing `admission_inventory`, `launch_sample_qc`, `sample_qc_receipts` |
| Private retained pin, remeasurement and source binding | `publication.py`, `ingest_registration.py`; existing retained census/grading/binding modules |
| Sealed retries, exact worker bootstrap, regional submit | `graph.py`, `receipts.py`, `worker.py`, `preflight.py`, `submission.py` |
| Content-addressed index update and serialized landing | `landing.py`, `qualification_run.py`, `registry_update.py`; existing `atlas-push` |

The friction records supply failure examples, not policy authority. The graph
addresses their repeated classes: exact worker parsing instead of handwritten
bootstraps; explicit source/region/output identity; full raw snapshots instead of
data-only archives; actual population counts before shard reads; original license
provenance after repinning; reviewer-specific observation validation; and index
compare-and-swap instead of choosing one side of a merge. Runtime-specific
repairs, unsupported task contracts and source-specific ruling decisions still
require maintained converters and actual qualification evidence. A generic
placeholder cannot assert them.

`inventory.py` can regenerate the discovery ledger without importing or executing
historical scripts:

```bash
uv run --no-sync python -m experiments.post_training.task_curation.ingest_stages.inventory \
  --root <historical-ingest-directory> --output <external-inventory-directory>
```

Its discovery classifications deliberately remain `*-review`: matching a module
name in text is not semantic proof of adopted behavior. The frozen retirement
ledger records the explicit entrypoint retirement policy above.
