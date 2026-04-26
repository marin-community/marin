# Public Diagnostic Logs Training Sources (#5094)

Last verified: April 26, 2026.

This inventory scopes training-data sourcing only. Eval slicing is tracked separately in issue #5093.

| Source | License / Rights | Size (compressed) | Format | Status | Contamination Risk |
| --- | --- | ---: | --- | --- | --- |
| GHALogs (Zenodo 14796970) | Creative Commons Attribution Share Alike 4.0 International (`cc-by-sa-4.0`) | 143,425,404,506 bytes | `runs.json.gz`, `repositories.json.gz`, `github_run_logs.zip` | Training-ready by license; still requires sanitization | High (CI logs can include tokens, internal paths) |
| LogChunks (Zenodo 3632351) | Creative Commons Attribution 4.0 International (`cc-by-4.0`) | 24,108,826 bytes | `LogChunks.zip` (XML) | Eval-only | Medium (failure chunks may include identifiers) |
| LogHub (`logpai/loghub`) | Custom research/academic-only license text | 7,513,088 bytes (repo metadata) | mixed plaintext logs | Eval-only | Medium |
| GitHub fixture/golden/stack traces from accepted source corpora | Inherits accepted source-corpus licensing and provenance | Upstream corpus size-dependent (StarCoderData reference: 310,802,033,041 bytes) | parquet -> sanitized jsonl | Training-ready (sample-only in this issue) | Medium |
| Marin-owned CI/Iris/Zephyr logs | Internal | n/a | internal logs | Eval-only | High |
| #5093 heldout eval slices | Eval holdout policy | n/a | eval slices | Eval-only | High |

Source metadata came from the official APIs on April 26, 2026:
- `https://zenodo.org/api/records/14796970`
- `https://zenodo.org/api/records/3632351`
- `https://api.github.com/repos/logpai/loghub`

## Split Policy

Use deterministic hash partitioning on stable source keys (repo + path):

- `issue_5093_holdout`: 1% reserved and never trainable
- `dev`: 1%
- `test`: 1%
- `train`: 97%

This policy is codified in `marin.datakit.download.diagnostic_logs.assign_partition`.

## Sanitization Rules

Apply sanitization before partition materialization and tokenization:

- redact GitHub tokens (`ghp_`, `github_pat_`, etc.)
- redact AWS access keys (`AKIA...`)
- redact key/value secrets (`token=...`, `password: ...`, `api_key=...`)
- replace email addresses with per-document pseudonyms (`<USER_0_EMAIL>`)
- replace user home paths (`/Users/<name>`, `/home/<name>`, `C:\Users\<name>`) with per-document pseudonyms
- replace other same-document instances of usernames discovered from email addresses or home paths when the candidate is specific enough to avoid common log words
- redact internal Marin GCS paths (`gs://marin-*`)

Rules are codified in `marin.datakit.download.diagnostic_logs.sanitize_diagnostic_log_text`.

## Sample-Only Ingest Plan (This Issue)

This issue intentionally avoids full-corpus pull. Executable wiring is capped:

- max parquet files: 8
- max rows scanned: 200,000
- output: partitioned sanitized jsonl + metadata counters

The experiment entry point is `experiments/exp5094_public_diagnostic_logs.py` with required `--source_path` to an already staged corpus path.

## Initial Tranche Estimate

Given the sample cap and observed diagnostic-log path sparsity in code corpora, the first training tranche should be treated as a pilot:

- expected retained sample bytes: ~50MB to ~500MB
- expected retained sample tokens: ~0.01B to ~0.15B
- promotion to full ingest is deferred until GHALogs sanitization/governance is reviewed and eval-only packaging is defined for LogChunks/LogHub.
