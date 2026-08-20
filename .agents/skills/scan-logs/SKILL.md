---
name: scan-logs
description: Scan logs too large to read directly, using Gemini (scripts/logscan.py).
---

# Scan logs

Use when a log is too large for context (usually >1,000 lines) or when triaging
errors in Iris, Zephyr, or training logs. `GEMINI_API_KEY` is required.

`grep` returns line-numbered original matches; `summarize` returns a markdown
report. Use them independently or pipe narrowed matches into a summary:

```bash
uv run scripts/logscan.py grep <logfile> "<query>"
uv run scripts/logscan.py summarize <logfile> "<query>"
uv run scripts/logscan.py grep log.txt "errors" \
  | uv run scripts/logscan.py summarize --stdin "summarize these errors"
```

`--stdin` replaces the logfile. Relevant options are `--chunk-tokens N`
(defaults 5000 for grep, 50000 for summarize), `--concurrency N` (16),
`--model NAME` (`gemini-2.5-flash-lite`), and `-v/--verbose` (per-chunk output
to stderr). Results go to stdout; progress and token usage go to stderr.

Use `grep` to locate a failure region before reading exact lines for `debug`, and
use `summarize` for broad canary triage or failed-job analysis.
