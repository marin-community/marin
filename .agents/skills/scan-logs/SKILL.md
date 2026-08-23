---
name: scan-logs
description: Use scripts/logscan.py only when explicitly requested to scan an oversized log or to perform Gemini-backed log analysis.
---

# Scan large logs

`grep` finds matching original lines; `summarize` produces a Markdown report.
They can run independently or as a pipeline.

## Prerequisites

`GEMINI_API_KEY` must be set. The log is sent to an external Gemini API and
incurs model usage; do not send secrets or sensitive logs without approval.

## Modes

### grep — find matching lines

Returns line-numbered original lines matching a natural-language query.

```bash
uv run scripts/logscan.py grep <logfile> "<query>"
```

### summarize — produce a markdown report

Summarizes the log around the query, hierarchically reducing large inputs.

```bash
uv run scripts/logscan.py summarize <logfile> "<query>"
```

Output is a markdown report on stdout.

### Piping modes together

Narrow first, then summarize through `--stdin`:

```bash
uv run scripts/logscan.py grep log.txt "errors" \
  | uv run scripts/logscan.py summarize --stdin "summarize these errors"
```

Behavior-changing options are `--chunk-tokens`, `--concurrency`, `--model`,
`--stdin`, and `--verbose`; inspect `--help` for current defaults.

## Output

- **grep**: Line-numbered matching lines to stdout. Progress to stderr.
- **summarize**: Markdown report to stdout. Progress and token usage to stderr.

Both modes report progress and token usage to stderr. Read the matched source
lines before accepting a generated diagnosis.
