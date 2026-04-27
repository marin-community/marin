# Public Diagnostic Logs Training Sources (#5094)

Last verified: April 26, 2026.

This inventory scopes training-data sourcing only. Eval slicing is tracked separately in issue #5093.

| Source | License / Rights | Size (compressed) | Format | Status | Contamination Risk |
| --- | --- | ---: | --- | --- | --- |
| GHALogs (Zenodo 14796970) | Creative Commons Attribution Share Alike 4.0 International (`cc-by-sa-4.0`) | 143,425,404,506 bytes | `runs.json.gz`, `repositories.json.gz`, `github_run_logs.zip` | Training-ready by license; still requires sanitization | High (CI logs can include tokens, internal paths) |
| LogChunks (Zenodo 3632351) | Creative Commons Attribution 4.0 International (`cc-by-4.0`) | 24,108,826 bytes | `LogChunks.zip` (XML) | Eval-only | Medium (failure chunks may include identifiers) |
| LogHub (`logpai/loghub`) | Custom research/academic-only license text | 7,513,088 bytes (repo metadata) | mixed plaintext logs | Eval-only | Medium |
| Marin-owned CI/Iris/Zephyr logs | Internal | n/a | internal logs | Eval-only | High |
| #5093 heldout eval slices | Eval holdout policy | n/a | eval slices | Eval-only | High |

Source metadata came from the official APIs on April 26, 2026:
- `https://zenodo.org/api/records/14796970`
- `https://zenodo.org/api/records/3632351`
- `https://api.github.com/repos/logpai/loghub`

## Sanitization Rules

Apply sanitization before any training or eval materialization:

- redact GitHub tokens (`ghp_`, `github_pat_`, etc.)
- redact AWS access keys (`AKIA...`)
- redact key/value secrets (`token=...`, `password: ...`, `api_key=...`)
- replace email addresses with per-document pseudonyms (`<USER_0_EMAIL>`)
- replace user home paths (`/Users/<name>`, `/home/<name>`, `C:\Users\<name>`) with per-document pseudonyms
- replace other same-document instances of usernames discovered from email addresses or home paths when the candidate is specific enough to avoid common log words
- redact internal Marin GCS paths (`gs://marin-*`)

Rules are codified in `marin.datakit.download.diagnostic_logs.sanitize_diagnostic_log_text`.

## Ingest Scope

This PR only records source gating and reusable sanitization. It does not add extraction or tokenization wiring. Promotion to full ingest is deferred until GHALogs sanitization/governance is reviewed and eval-only packaging is defined for LogChunks/LogHub.
