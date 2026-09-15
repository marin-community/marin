# Hero checkpoint completions

[Latest report](https://storage.googleapis.com/marin-public/rav/hero-completions/latest/index.html)
· [Actions workflow](../../../../../.github/workflows/marin-hero-completions.yaml)

## Operation

Run discovery, report publication, and the status summary:

```bash
gh workflow run marin-hero-completions.yaml --ref main
```

Read the Actions summary for Iris job links, queue state, and errors.
A successful submission does not mean that sampling started or completed.

To submit all unfinished checkpoints at production priority in one invocation:

```bash
gh workflow run marin-hero-completions.yaml --ref main -f submission=all -f priority=production
```

`submission=all` submits one job for each unfinished request in the current
discovery set. Discovery uses the run list and step limits in
[`CHECKPOINT_RUNS`](config.py) and includes only retained permanent checkpoints.
It skips completed results, active jobs, and exhausted retries.
It uses the current prompt bank and excludes requests for previous prompt banks.
Each job needs 64 GPUs. Iris starts jobs as resources become available.

`submission=next` is the default for manual and scheduled invocations. It waits
until no sampling jobs are active, then submits at most one request.
The workflow saves the selected priority for the discovered requests and their
retries. Hourly invocations continue retries at the saved priority.
Future checkpoints use batch priority unless an invocation sets their priority.
An active job keeps its original priority. The next attempt uses the saved priority.
The three-attempt limit still applies.

The CLI also accepts `reconcile --submission all --priority production` from a clean checkout.
Use Actions for the shared sample store because its concurrency group
serializes submissions. The priority choices are `batch`, `interactive`, and
`production`. Manual workflow runs default to `batch`. Scheduled runs preserve
saved priorities. Iris checks the caller's permission for the selected priority.
Omit the CLI's `--priority` option to preserve saved priorities.

The submission invocation publishes results already available at that time.
Later workflow invocations publish results from the new jobs to the latest
report. Submission does not wait for all GPU jobs to finish.

For a status query without job submission, run from the repository root with
CoreWeave storage credentials and Iris authentication:

```bash
uv run --no-sync python -m experiments.grug.moe_hero_ep.ops.vibe_check status
```

## Recovery

Correct the sampler or access error before another attempt.
Requests retain their attempt counts after a fix. Retries use the original source commit.

For exhausted requests, change `release` in [config.py](config.py) through a PR.
This creates new requests for all retained checkpoints, including prior successes.
