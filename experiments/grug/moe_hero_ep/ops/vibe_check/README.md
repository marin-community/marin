# Hero checkpoint completions

This workflow samples retained permanent checkpoints in the configured hero runs.
It discovers checkpoints and updates the current comparison report hourly.
It also saves one nonempty daily snapshot. The report day starts at 08:00 UTC.
It updates one comment on [issue 8827](https://github.com/marin-community/marin/issues/8827)
with the report link.

The [latest report](https://storage.googleapis.com/marin-public/rav/hero-completions/latest/index.html)
lets readers select two checkpoints and a prompt. It links to raw result JSON
for completed sample sets. The first successful publication creates this URL. It shows an empty report
until the first result arrives. Later publications replace it with the available results.

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
| Current report | `gs://marin-public/rav/hero-completions/latest/index.html` |

The HTML page loads result JSON from public GCS. Checkpoint tensors stay in CoreWeave storage.

The [Actions workflow](../../../../../.github/workflows/marin-hero-completions.yaml) runs from `main`.
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

To run the scheduled workflow on demand, use `gh workflow run marin-hero-completions.yaml --ref main`.
This runs discovery, report publication, and the status summary.
For one request, use the same sampler inside a 64-GPU job at the request's source commit:

```bash
uv run --no-sync python -m experiments.grug.moe_hero_ep.ops.vibe_check.sample \
  --request completion-request.json --store-root s3://your-bucket/vibe-checks
```

Copy a saved `requests/<sample-id>.json` from the S3 root for `completion-request.json`.
For a new request, use `SampleRequest` in [completions.py](completions.py).
It pins the checkpoint, prompt bank, generation settings, and source commit.

Actions runs `reconcile`, `report`, and `status` hourly. Keep these invocations in its
`hero-checkpoint-completions` concurrency group. Direct sampler jobs do not use this limit.
The current report reads completed results from all prompt banks on each invocation.
An empty result set does not create a daily snapshot. After the first result arrives,
publication saves the day's snapshot before upload. Retries reuse that snapshot,
but the current report can include newer results. Previously published daily pages stay fixed.
Report publication does not start GPU jobs.

The Actions summary shows the completed count, checkpoint steps, Iris job links,
job states, and queue or error details. The summary step also runs after submission
or publication errors if authentication succeeds. A successful submission does not
mean that sampling started or completed. The sampler can wait for 64 free GB200 GPUs.
For a read-only status summary, run:

```bash
uv run --no-sync python -m experiments.grug.moe_hero_ep.ops.vibe_check status
```

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
