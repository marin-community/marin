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
