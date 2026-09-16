# Hero checkpoint completions

[Latest report](https://storage.googleapis.com/marin-public/rav/hero-completions/latest/index.html)
· [Actions workflow](../../../../../.github/workflows/marin-hero-completions.yaml)

The report compares three completions per prompt across retained permanent hero
checkpoints. The three samples use different seeds, fixed across checkpoints.
Expected-answer token probabilities are computed once per prompt and checkpoint.
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
Each checkpoint job requests **32 GB200 GPUs** across eight nodes and uses a batch
size of 32 across the job. With `submission=all`, two checkpoint jobs can run concurrently when
64 GPUs and the corresponding node resources are available. The backfill skips
completed results, active jobs, and requests that exhausted their retries. The
checkpoint selection comes from [`CHECKPOINT_RUNS`](config.py).

The workflow runs hourly. With its default, `submission=next`, it submits no new
request while jobs for the current sampling specification are active. Otherwise,
it submits at most one unfinished request. Priorities are `batch`, `interactive`,
or `production`. A manual invocation saves its selected priority for subsequent
attempts of all discovered requests. Active jobs retain their assigned priority.
Future checkpoints use batch priority unless a manual invocation sets a different
priority.

Each request identifies a checkpoint and a sampling specification: the prompt bank,
sampler release, completion count, and other generation settings in
[config.py](config.py). Each workflow invocation publishes available results for
the current specification. Older results stay stored and do not block publication
or enter new retries. The workflow does not wait for GPU jobs to finish. New
completions appear after a later invocation updates the report. A successful
workflow does not mean that sampling completed.

## Recovery

Each request permits three attempts. The workflow records its source commit when
it first saves a request. All attempts use that commit for the sampler code, so
merging a fix does not change existing requests.
To rerun checkpoints with corrected sampler code, change `release` in
[config.py](config.py) through a PR, then invoke the workflow. A new release
creates new requests for all retained checkpoints, including prior successes.
