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

To produce completions for all currently retained checkpoints at production
priority, start the workflow once:

```bash
gh workflow run marin-hero-completions.yaml --ref main -f priority=production
```

The workflow saves this priority for the discovered requests and their retries.
Hourly invocations continue those requests at the saved priority, with one
64-GPU sampling job at a time. Completed sample sets do not run again.
Future checkpoints use batch priority unless an invocation sets their priority.
An active job keeps its original priority. The next attempt uses the saved priority.
The three-attempt limit still applies.

The CLI also accepts `reconcile --priority production` from a clean checkout.
Use Actions for the shared sample store because its concurrency group
serializes submissions. The priority choices are `batch`, `interactive`, and
`production`. Manual workflow runs default to `batch`. Scheduled runs preserve
saved priorities. Iris checks the caller's permission for the selected priority.
Omit the CLI's `--priority` option to preserve saved priorities.

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
