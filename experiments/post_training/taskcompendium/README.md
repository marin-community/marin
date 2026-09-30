# TaskTrove TaskCompendium ingestion

`inventory_tasktrove.py` counts metadata from the pinned Clean release without reading archive bytes. `ingest_tasktrove.py` then streams `tasks/` and `sft/` in Parquet row-group order, opens archive columns only for row groups containing supported answer modes, and writes a private catalog and a disposition ledger. The metadata inventory and conversion totals are separate: a metadata-eligible row is not counted as imported until the archive parser accepts it.

The current importer allowlist is `mcq`, `math`, and `numeric`. MCQA accepts the reviewed `nemotron_mcqa` converter and template `c814af4f124d`; the math importer accepts its own supported source templates. Numeric mode uses the generic `NumericSpec` contract, but the pinned Clean release inventory currently contains no numeric rows. Other modes receive an `out-of-scope` ledger entry with the observed mode. A malformed supported archive receives a `rejected` entry and reason. Duplicate source subset/path pairs are recorded once as imported and then as duplicates; conflicting archive bytes for a repeated identity are rejected.

Run the conversion inside the data-region Iris cluster so source Parquet and private outputs stay in-region. The destination must be a unique durable S3 prefix, and the job environment must inherit the cluster's configured storage credentials:

The runner package is a root UV workspace member. For a scoped Iris environment, install it with `--sync-package taskcompendium --extra ingest --extra math`; the `ingest` extra supplies PyArrow and Marin storage access, and `math` supplies the math verifier dependency.

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
- `candidate-proof.jsonl`: one private per-candidate join row with the source Parquet object pin and per-archive SHA256, for packaging validation. It contains no TaskSpec or verifier data;
- `ingestion-manifest.json`: release manifest digest, input Parquet object identities, counts by disposition and accepted split/mode/family/converter, rejection reasons, payload byte measures, source metadata fields, and artifact URIs/sizes/object IDs.

For a focused conversion pass, set `TASKTROVE_INGEST_MODES` to a comma-separated subset of `mcq`, `math`, and `numeric`. The runner still records one ledger disposition for every input row in both splits, but reads task archives only from row groups that contain a selected mode and imports only selected-mode rows. Nonselected rows receive an `out-of-scope` disposition with an explicit mode-selection reason. The manifest records the selected modes; materialized archive bytes can include nonselected rows in a selected Parquet row group, while parsed bytes count only archives sent to an importer.

The current public candidate cohorts are `laion__nemotron-gym-knowledge-mcqa-v2`, `laion__nemotron-gym-math-openmathreasoning-v2`, and `laion__nemo-prism-math-v3`. All-puzzles and other deferred sources remain private. The `sft/` split is excluded from the candidate projection even when its source name matches an eligible cohort. Candidate rows omit verifier specifications and gold answers. Route and converter provenance remain in the private catalog and should be joined by task ID when packaging RL and SFT cohorts. The candidate file is not a public release; packaging owns attribution, lineage caveats, and any eventual upload.

After an ingestion run has status `complete`, run `audit_tasktrove_ingest.py` in the same region with `TASKTROVE_OUTPUT_URI` pointing at its durable prefix. It reads only the manifest, ledger, candidate JSONL, and proof JSONL; it does not read private catalog specifications or source archive bytes. It writes `ingestion-audit.json` beside the run and a small Iris output summary. The report counts each ledger disposition by split, source, mode, family, converter, and original tag; tag counts increment once per distinct tag on a row. It also checks that every eligible imported task has exactly one allowlisted PublicTask-v1 candidate and one matching source-proof row, including the archive SHA256, and verifies artifact hashes against the ingestion manifest. A failed audit blocks the candidate handoff.

```bash
python -m experiments.post_training.taskcompendium.audit_tasktrove_ingest
```

For a fresh candidate source-rights metadata inventory, set `TASKTROVE_OUTPUT_URI` to the completed ingestion prefix, `TASKTROVE_RIGHTS_AUDIT_OUTPUT_URI` to a new private S3 prefix, and `TASKTROVE_AUDIT_REVISION` to the full commit running the audit. Then run `audit_tasktrove_rights_metadata.py`. It validates the candidate/ledger/proof/tag joins, hashes the ledger and catalog, and inventories the recorded license and attribution fields without loading TaskSpecs or source archives. It applies no clearance: every candidate remains held until the exact source cohort is reviewed.

```bash
python -m experiments.post_training.taskcompendium.audit_tasktrove_rights_metadata
```

After the audit passes, `export_tasktrove_accepted.py` verifies the exact audit-manifest digest and joins the ledger, candidates, proof rows, and private catalog rights metadata. It recomputes artifact hashes, checks unique row-level joins, preserves ordered tags, and records source terms and disposition counts by source, split, family, converter, template, and tag. It reads no archive payloads or private TaskSpecs. Export clearance is limited to two exact `tasks` tuples: MCQA (`laion__nemotron-gym-knowledge-mcqa-v2`, `qa-short-answer`, `nemotron_mcqa`, template `c814af4f124d`) and Prism math (`laion__nemo-prism-math-v3`, `math-answer`, `nemotron_math`, template `5ee94cf985a9`). Both use their pinned NVIDIA CC-BY-4.0 source-card revisions and require all inspected archive rights fields to remain absent. Other sources, splits, families, converters, or templates stay held. It maps `tasks` to public `train` and excludes `sft/`.

Set `TASKTROVE_OUTPUT_URI` to the completed ingestion prefix, `TASKTROVE_RIGHTS_AUDIT_MANIFEST_URI` to the audited `manifest.json`, `TASKTROVE_ACCEPTED_OUTPUT_URI` to a new private S3 prefix, and `TASKTROVE_PROJECTION_REVISION` to the full exporter commit. The script validates the audit manifest against the exact ingestion artifacts before writing `AcceptedPublicRecord-v1` groups under `tasktrove_clean/train/<source>/<family>.jsonl`:

```bash
python -m experiments.post_training.taskcompendium.export_tasktrove_accepted
```

The resulting prefix is a private packaging input, not a public release. Each wrapper contains only the PublicTask-v1 task and the exact SourceProof fields. Packaging must validate the typed schema and run correct/incorrect bulk-output Harbor trials for each exported group before any Hub upload. The projection manifest records per-group source assets, counts, digests, rights-card revisions, and the audit-manifest digest. The first reviewed projection is bound to audit manifest SHA256 `ab71556290d1ce54e596588b549de95ddd68366cd8fd3bb18b77ed9e99f0eed1` and canonical normalized rights-term inventory SHA256 `165124b2f8fa95b3c2d013136eb5be0638cf0a170142cc0b4bfebd5297de0389`. The inventory digest hashes its JSON array with UTF-8 encoding, sorted keys, compact separators, and no trailing newline.

`trial_tasktrove_projection.py` selects one exact accepted row from each reviewed cohort, resolves its private Parquet row through a unique ledger join, verifies its archive SHA256 against `source_proof`, re-imports it, and checks that the typed PublicTask record matches the exported row. It then runs fixed correct and incorrect responses through the Harbor trial path without calling a model endpoint. The private Iris output contains Harbor traces and a bounded outcome summary. Run it with the same `TASKTROVE_OUTPUT_URI` and `TASKTROVE_ACCEPTED_OUTPUT_URI`, using the `harbor`, `ingest`, and `math` extras.

The default internal runtime cap is 840 seconds (`TASKTROVE_INGEST_RUNTIME_SECONDS`). If the scan reaches that cap or hits an unexpected input/storage error, the runner closes its Parquet writers, writes an `ingestion-manifest.json` with `status: partial` and completed-row counts, then exits with an error. The ledger count is the completed prefix; a partial run never implies full-release counts.
