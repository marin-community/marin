# Hero checkpoint completions

[Latest report](https://storage.googleapis.com/marin-public/rav/hero-completions/latest/index.html)
· [Actions workflow](../../../../../.github/workflows/marin-hero-completions.yaml)

The report compares completions across retained permanent hero checkpoints.
Token colors show model probabilities. Hover, focus, or tap a token for its
probability and the five most likely alternatives.
Each prompt in [prompts.json](prompts.json) includes an expected answer, which
is scored separately. Open-ended expected answers are examples, and probability
does not measure answer correctness. Probabilities use the model distribution
before temperature scaling.

## Run

Submit all unfinished checkpoints at production priority:

```bash
gh workflow run marin-hero-completions.yaml --ref main -f submission=all -f priority=production
```

Use the Actions summary for Iris job links, status, and errors.
Each checkpoint job requires **64 GPUs**. The backfill skips completed results,
active jobs, and requests that exhausted their retries. The checkpoint selection
comes from [`CHECKPOINT_RUNS`](config.py).

The workflow runs hourly. With its default, `submission=next`, it submits no new
request while jobs are active. Otherwise, it submits at most one unfinished
request. Priorities are `batch`, `interactive`, or `production`. A manual
invocation applies its selected priority to all discovered requests, including
new checkpoints, and their retries. Future checkpoints use batch priority
unless a manual invocation sets a different priority.

Each workflow invocation publishes the results already available. It does not
wait for GPU jobs to finish. New completions appear after a later invocation
updates the report. A successful workflow does not mean that sampling completed.

## Recovery

Each request permits three attempts. Active jobs and retries retain their
original source commit, so merging a fix does not change them.
To rerun checkpoints with corrected sampler code, change `release` in
[config.py](config.py) through a PR, then invoke the workflow. A new release
creates new requests for all retained checkpoints, including prior successes.
