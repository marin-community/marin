---
name: logscan
description: Scan large log files using Gemini to find errors, patterns, and anomalies. Use when confronted with large log files that are too big to read directly.
---

# Skill: Logscan

Use `scripts/logscan.py` to analyze large log files. Two composable modes —
`grep` (find matching lines) and `summarize` (produce a markdown report) — can
be used independently or piped together.

## When to Use

- Log files too large to read in context (>1000 lines)
- Searching for errors, anomalies, or patterns in job/worker/controller logs
- Triaging failures from Iris, Ray, Zephyr, or training jobs
- Any time you need to understand what happened in a big log file

## Prerequisites

The `GEMINI_API_KEY` environment variable must be set.

## Modes

### grep — find matching lines

Returns the original log lines (with line numbers) that match a natural-language
query. Uses small chunks (~5k tokens) for precision.

```bash
uv run scripts/logscan.py grep <logfile> "<query>"
```

Output goes to stdout as `<line_number>: <line>`, one per match.

### summarize — produce a markdown report

Summarizes the log into a coherent narrative focused on the query. Uses larger
chunks (~50k tokens) and hierarchically reduces per-chunk summaries into a
final report.

```bash
uv run scripts/logscan.py summarize <logfile> "<query>"
```

Output is a markdown report on stdout.

### Piping modes together

grep's stdout feeds directly into summarize via `--stdin`:

```bash
uv run scripts/logscan.py grep log.txt "errors" \
  | uv run scripts/logscan.py summarize --stdin "summarize these errors"
```

This is useful to first narrow down to relevant lines, then get a summary.

## Arguments

| Argument | Description |
|---|---|
| `mode` | `grep` or `summarize` |
| `logfile` | Path to the log file (optional if `--stdin`) |
| `query` | Natural language description of what to look for |
| `--chunk-tokens N` | Tokens per chunk (default: 5000 for grep, 50000 for summarize) |
| `--concurrency N` | Max parallel requests (default: 16) |
| `--model NAME` | Gemini model (default: `gemini-2.5-flash-lite`) |
| `-v, --verbose` | Print per-chunk results to stderr |
| `--stdin` | Read input from stdin instead of a file |

## Examples

```bash
# Find OOM errors in a training log
uv run scripts/logscan.py grep /tmp/train.log "out of memory errors or OOM kills"

# Summarize TPU failures
uv run scripts/logscan.py summarize /tmp/worker.log "TPU errors, device failures, or FAILED_PRECONDITION"

# grep then summarize for focused analysis
uv run scripts/logscan.py grep /tmp/controller.log "timeout" \
  | uv run scripts/logscan.py summarize --stdin "what caused the timeouts?"

# Use a more capable model for complex analysis
uv run scripts/logscan.py summarize /tmp/big.log "race conditions or deadlocks" --model gemini-2.5-flash
```

## Output

- **grep**: Line-numbered matching lines to stdout. Progress to stderr.
- **summarize**: Markdown report to stdout. Progress and token usage to stderr.

Both modes print token usage stats to stderr when complete.

## Integration with Other Skills

- **babysit-\***: Use logscan to analyze logs from failed jobs before deciding on recovery
- **debug-\***: Use `grep` to find the failure region, then `Read` specific line ranges
- **canary-triage**: Use `summarize` to scan canary ferry logs for the root cause

## Tips

- Use `grep` first to narrow down, then `summarize` the filtered output via `--stdin`
- For very large files (>100k lines), `summarize` handles hierarchical reduction automatically
- `grep` uses small chunks (5k tokens) for precision; `summarize` uses large chunks (50k) for context
- Add `-v` to see per-chunk results as they complete
