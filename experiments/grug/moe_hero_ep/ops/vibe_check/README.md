# Hero checkpoint completions

This workflow samples every retained permanent checkpoint in the production hero
lineage. It checks the queue hourly and publishes a daily comparison report after
08:00 UTC. It updates one comment on [issue 8827](https://github.com/marin-community/marin/issues/8827)
with the report link.

The [latest report](https://storage.googleapis.com/marin-public/hero/completions/latest/index.html)
lets readers select two checkpoints and a prompt. It links to raw result JSON
and shows pending and failed sample sets. The first successful publication creates this URL.

## Inputs and sampling

- [prompts.json](prompts.json) holds 34 fixed prompts with IDs, seeds, and source links.
  Edit it through a PR. Prompt text and results are public; do not add private data.
- `CHECKPOINT_RUNS` in [config.py](config.py) selects runs for sampling.
  Update it when the production run changes. Discovery includes permanent checkpoints
  up to each run's step limit. Deleted checkpoints are unavailable.
- Each job uses 64 GB200 GPUs at batch priority. The scheduled queue runs one job at a time;
  newer steps run first. Sampling restores checkpoint weights and the saved router bias,
  without optimizer state. It uses master weights when present. Missing tensors fail the job.
- Generation uses a 4,096-token context, temperature 0.2, at most 200 new tokens,
  and a pinned tokenizer. These are base-model continuations with fixed per-prompt seeds.
  The sampler has no KV cache. Live sampling has not verified restore support
  for every retained checkpoint in the lineage.
- Prompt or generation-setting changes create new sample sets for all retained
  checkpoints. Previous results remain available. An unrelated code commit does not repeat sampling.

## Storage and access

| Data | Location |
| --- | --- |
| Queue and results | `s3://marin-us-east-02a/marin/hero-completions/v1/` |
| Public result JSON | `gs://marin-public/hero/completions/results/` |
| Daily HTML reports | `gs://marin-public/hero/completions/YYYY.MM.DD/` |
| Latest report redirect | `gs://marin-public/hero/completions/latest/index.html` |

The HTML page loads result JSON from public GCS. Checkpoint tensors stay in CoreWeave storage.

The [Actions workflow](../../../../../.github/workflows/hero-completions.yaml) runs from `main`.
It needs repository secrets `CW_ACCESS_KEY_ID`, `CW_SECRET_ACCESS_KEY`, and
`IRIS_CI_GCP_SA_KEY`, plus issue-write permission for its GitHub token.
Before deployment, approve the account in `IRIS_CI_GCP_SA_KEY` for Iris IAP,
federation, writes to `marin-public`, and submission to `TARGET_CLUSTER` in [config.py](config.py).
Access changes require separate approval.
The workflow does not change the hero run or cluster.

## Operation and recovery

For a read-only inventory, run from the repository root with CoreWeave credentials:

```bash
uv run --no-sync python -m experiments.grug.moe_hero_ep.ops.vibe_check inventory
```

To run the scheduled workflow on demand, use `gh workflow run hero-completions.yaml --ref main`.
For one request, use the same sampler inside a 64-GPU job at the request's source commit:

```bash
uv run --no-sync python -m experiments.grug.moe_hero_ep.ops.vibe_check.sample \
  --request completion-request.json --store-root s3://your-bucket/vibe-checks
```

The request JSON follows `SampleRequest` in [completions.py](completions.py).
It pins the checkpoint, prompt bank, generation settings, and source commit.

Actions runs `reconcile` hourly and `report` daily. Keep all invocations in its
shared concurrency group. Report retries reuse the day's saved snapshot and do not start GPU jobs.
The page warns when checkpoint inventory is more than 36 hours old. Check Actions for workflow failures.

Scheduling timeouts defer a request for six hours. Three sampling failures exhaust
its retry budget. A missing Iris job becomes a failure after 48 hours; service errors
do not prove that a job stopped. Requests keep their original source commit for retries.

After correcting a failed sampler or an access problem, change `release` in
[config.py](config.py) to retry exhausted requests. This creates new requests for
all retained checkpoints, including prior successes. Change the release when
restore or decoding behavior changes.
