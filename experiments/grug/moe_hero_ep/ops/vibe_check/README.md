# Hero checkpoint completions

[Latest report](https://storage.googleapis.com/marin-public/rav/hero-completions/latest/index.html)
· [Actions workflow](../../../../../.github/workflows/marin-hero-completions.yaml)

The report compares three completions per prompt across retained permanent hero
checkpoints. The three samples use different seeds, fixed across checkpoints.
Expected-answer token probabilities are computed once per prompt and checkpoint.
They include a final end-of-sequence (EOS) token, which measures the probability
of stopping after the reference. The report shows EOS and includes it in the totals.
Token colors show model probabilities. Hover, focus, or tap a token for its
probability and the five most likely alternatives.
Each prompt in [prompts.json](prompts.json) includes an expected answer, which
is scored separately. Open-ended expected answers are examples, and probability
does not measure answer correctness. Probabilities use the model distribution
before temperature scaling.
The sampler encodes each prompt and reference together. It rejects references
that change the prompt tokens. Put fixed formatting, such as a function-header
newline, in the prompt.

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
checkpoint selection comes from [`hero_checkpoint_paths()`](../../checkpoints.py),
which reads the current hero run and its ancestors.

The workflow runs hourly. With its default, `submission=next`, it submits no new
request while jobs for the current sampling specification are active. Otherwise,
it submits at most one unfinished request. Priorities are `batch`, `interactive`,
or `production`. A manual invocation saves its selected priority for subsequent
attempts of all discovered requests. Active jobs retain their assigned priority.
Future checkpoints use batch priority unless a manual invocation sets a different
priority.

Each request identifies a checkpoint and a sampling specification: the prompt bank,
sampler release, completion count, and other generation settings in
[config.py](config.py). The report retains completed results across sampling
versions. It updates only when new usable results arrive. Missing historical
scores appear as unavailable. Unusable files produce a warning and do not block
other results. The report labels each sampling version in the checkpoint selector.
Scheduling uses only the current specification. The workflow does not wait for
GPU jobs to finish, so workflow success does not mean that sampling completed.

## Recovery

Each request permits three Iris job submissions. Each job permits 1,000 preemption retries.
The workflow records its source commit when it first saves a request.
All attempts use that commit for the sampler code, so
merging a fix does not change existing requests.
To rerun checkpoints with corrected sampler code, change `release` in
[config.py](config.py) through a PR, then invoke the workflow. A new release
creates new requests for all retained checkpoints, including prior successes.
