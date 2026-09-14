# Hero checkpoint completions

This workflow samples retained permanent checkpoints in the configured hero runs.
It discovers checkpoints hourly and publishes a daily comparison report after
08:00 UTC. It updates one comment on [issue 8827](https://github.com/marin-community/marin/issues/8827)
with the report link.

The [latest report](https://storage.googleapis.com/marin-public/rav/hero-completions/latest/index.html)
lets readers select two checkpoints and a prompt. It links to raw result JSON
for completed sample sets. The first successful publication creates this URL.

## Inputs and sampling

- [prompts.json](prompts.json) holds 34 fixed prompts with IDs, seeds, and source links.
  Edit it through a PR. Prompt text and results are public; do not add private data.
- `CHECKPOINT_RUNS` in [config.py](config.py) selects runs for sampling.
  Update it when the production run changes. Discovery includes permanent checkpoints
  up to each run's step limit. Deleted checkpoints are unavailable.
- Each job uses 64 GB200 GPUs at batch priority. The workflow runs one job at a time.
  Newer steps run first. Sampling restores checkpoint weights and the saved router bias,
  without optimizer state. It uses master weights when present. Missing tensors fail the job.
- Generation uses a 4,096-token context, temperature 0.2, at most 200 new tokens,
  and a pinned tokenizer. These are base-model continuations with fixed per-prompt seeds.
  The sampler has no KV cache. Live sampling has not verified restore support
  for every retained checkpoint in the lineage.
- All 34 prompts use one 64-row batch. Larger banks use successive batches in the same job.
  The worker writes one result file after all prompts succeed. A retry reruns the full bank.
- Prompt or generation-setting changes create new sample sets for all retained
  checkpoints. Previous results remain available. An unrelated code commit does not repeat sampling.

## Storage and access

| Data | Location |
| --- | --- |
| Requests, attempts, results, and failure markers | `s3://marin-us-east-02a/marin/users/rav/hero-completions/` |
| Public result JSON | `gs://marin-public/rav/hero-completions/results/` |
| Daily HTML reports | `gs://marin-public/rav/hero-completions/YYYY.MM.DD/` |
| Latest report redirect | `gs://marin-public/rav/hero-completions/latest/index.html` |

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
This runs discovery and publication with the same daily publication limit.
For one request, use the same sampler inside a 64-GPU job at the request's source commit:

```bash
uv run --no-sync python -m experiments.grug.moe_hero_ep.ops.vibe_check.sample \
  --request completion-request.json --store-root s3://your-bucket/vibe-checks
```

Copy a saved `requests/<sample-id>.json` from the S3 root for `completion-request.json`.
For a new request, use `SampleRequest` in [completions.py](completions.py).
It pins the checkpoint, prompt bank, generation settings, and source commit.

Actions runs `reconcile` hourly and `report` daily. Keep these invocations in its
`hero-checkpoint-completions` concurrency group. Direct sampler jobs do not use this limit.
Report retries reuse the day's saved snapshot and do not start GPU jobs.
The report reads completed result files, including results from previous prompt banks.
Check Actions and Iris for workflow failures and active jobs.

Each sample set gets three total job attempts. Failures, preemptions, missing jobs,
and scheduling timeouts use this budget. Requests keep their original source commit for retries.
Service errors stop the controller. An active job blocks new submissions, even after its result arrives.
After three attempts without a result, `failures/<sample-id>.txt` stops retries.
The controller saves `attempts/<job-name>.txt` before submission. These markers keep
the retry limit after Iris deletes job history. A lost submission also consumes an attempt.

After correcting a failed sampler or an access problem, change `release` in
[config.py](config.py) to retry exhausted requests. This creates new requests for
all retained checkpoints, including prior successes. Change the release when
restore or decoding behavior changes.
