# TaskTrove TaskCompendium ingestion

`inventory_tasktrove.py` counts metadata from the pinned Clean release without reading archive bytes. `ingest_tasktrove.py` then streams `tasks/` and `sft/` in Parquet row-group order, opens archive columns only for row groups containing supported answer modes, and writes a private catalog and a disposition ledger. The metadata inventory and conversion totals are separate: a metadata-eligible row is not counted as imported until the archive parser accepts it.

The current importer allowlist is `mcq`, `math`, and `numeric`. MCQA accepts the reviewed `nemotron_mcqa` converter and template `c814af4f124d`; the math importer accepts its own supported source templates. Numeric mode uses the generic `NumericSpec` contract, but the pinned Clean release inventory currently contains no numeric rows. Other modes receive an `out-of-scope` ledger entry with the observed mode. A malformed supported archive receives a `rejected` entry and reason. Duplicate source subset/path pairs are recorded once as imported and then as duplicates; conflicting archive bytes for a repeated identity are rejected.

Run the conversion inside the data-region Iris cluster so source Parquet and private outputs stay in-region. The destination must be a unique durable S3 prefix, and the job environment must inherit the cluster's configured storage credentials:

```bash
TASKTROVE_RELEASE_URI=s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.18.3
TASKTROVE_OUTPUT_URI=s3://marin-us-east-02a/marin/taskcompendium/tasktrove/2026.09.18.3/<unique-run-id>
```

Pass those two variables to the job and run:

```bash
python -m experiments.post_training.taskcompendium.ingest_tasktrove
```

`IRIS_OUTPUT_DIR` receives only a compact run summary. The durable S3 prefix contains:

- `private-catalog.parquet`: serialized private TaskSpecs, original ordered tags, per-archive SHA256, source subset/path, route, and exact source metadata fields when present;
- `ingestion-ledger.parquet`: one row for each processed input row, including split, Parquet object pin, disposition, and rejection/out-of-scope reason;
- `public-candidates.jsonl`: the PublicTask-v1 field allowlist for the packaging-approved MCQA and math source cohorts in the `tasks/` split only. These candidate rows remain private pending release packaging;
- `ingestion-manifest.json`: release manifest digest, input Parquet object identities, counts by disposition and accepted split/mode/family/converter, rejection reasons, payload byte measures, source metadata fields, and artifact URIs/sizes/object IDs.

The current public candidate cohorts are `laion__nemotron-gym-knowledge-mcqa-v2`, `laion__nemotron-gym-math-openmathreasoning-v2`, and `laion__nemo-prism-math-v3`. All-puzzles and other deferred sources remain private. The `sft/` split is excluded from the candidate projection even when its source name matches an eligible cohort. Candidate rows omit verifier specifications and gold answers. Route and converter provenance remain in the private catalog and should be joined by task ID when packaging RL and SFT cohorts. The candidate file is not a public release; packaging owns attribution, lineage caveats, and any eventual upload.

The default internal runtime cap is 840 seconds (`TASKTROVE_INGEST_RUNTIME_SECONDS`). If the scan reaches that cap or hits an unexpected input/storage error, the runner closes its Parquet writers, writes an `ingestion-manifest.json` with `status: partial` and completed-row counts, then exits with an error. The ledger count is the completed prefix; a partial run never implies full-release counts.
