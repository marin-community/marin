---
name: logscan
description: Scan large log files using Gemini to find errors, patterns, and anomalies. Use when confronted with large log files that are too big to read directly.
---

# Skill: Logscan

Use `scripts/logscan.py` to analyze large log files. This tool chunks the file,
sends each chunk to Gemini in parallel, and produces a merged summary with
highlighted lines.

## When to Use

- Log files too large to read in context (>1000 lines)
- Searching for errors, anomalies, or patterns in job/worker/controller logs
- Triaging failures from Iris, Ray, Zephyr, or training jobs
- Any time you need to understand what happened in a big log file

## Prerequisites

The `GEMINI_API_KEY` environment variable must be set.

## Usage

```bash
uv run scripts/logscan.py <logfile> "<query>" [options]
```

### Arguments

| Argument | Description |
|---|---|
| `logfile` | Path to the log file |
| `query` | Natural language description of what to look for |
| `--chunk-tokens N` | Tokens per chunk (default: 10000) |
| `--overlap F` | Overlap fraction between chunks (default: 0.2) |
| `--concurrency N` | Max parallel requests (default: 16) |
| `--model NAME` | Gemini model (default: `gemini-2.5-flash-lite`) |

### Examples

```bash
# Find OOM errors in a training log
uv run scripts/logscan.py /tmp/train.log "out of memory errors or OOM kills"

# Look for TPU failures
uv run scripts/logscan.py /tmp/worker.log "TPU errors, device failures, or FAILED_PRECONDITION"

# Investigate slow startup
uv run scripts/logscan.py /tmp/controller.log "worker startup delays or timeouts"

# Use a more capable model for complex analysis
uv run scripts/logscan.py /tmp/big.log "race conditions or deadlocks" --model gemini-2.5-flash
```

## Output

The tool prints:
1. Per-chunk summaries (streamed as chunks complete) to stdout
2. A final merged summary with the ~20 most important highlighted lines

Stderr shows progress (`[3/12] chunk 5 done`).

## Integration with Other Skills

- **babysit-\***: Use logscan to analyze logs from failed jobs before deciding on recovery
- **debug-\***: Use logscan as a first pass to identify the failure region, then read specific line ranges
- **canary-triage**: Use logscan to scan canary ferry logs for the root cause

## Tips

- Start with a broad query, then narrow down based on the summary
- For very large files (>100k lines), increase `--chunk-tokens` to reduce the number of API calls
- The overlap ensures errors near chunk boundaries aren't missed
