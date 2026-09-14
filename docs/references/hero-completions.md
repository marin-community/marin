# Hero checkpoint completions

The completion workflow samples every retained permanent checkpoint in the production hero
lineage. It checks the queue each hour and updates a public report each day after 08:00 UTC.
It also updates one GitHub Actions comment on [issue 8827](https://github.com/marin-community/marin/issues/8827).
For the access requirements, see [Schedule and recovery](#schedule-and-recovery).

The report's stable address is
[`hero/completions/latest/index.html`](https://storage.googleapis.com/marin-public/hero/completions/latest/index.html).
The first successful scheduled publication creates this address.
Select two checkpoints and a prompt to compare their completions. The inventory includes
pending and failed sample sets. Each completed set links to its raw JSON and generation settings.

## Inputs and lineage

`experiments/grug/moe_hero_ep/completion_prompts.json` holds the prompt bank. Each entry has a
stable ID, exact prompt text, seed, and source link. Changes go through a normal pull request.
Issue text is a source reference, not an executable or automatically changing input.

The bank contains 34 prompts. The first ten retain the
[original manual prompts and seeds](https://github.com/marin-community/marin/issues/8827#issuecomment-5489155657).
Another 22 prompts provide tests of code, arithmetic, context tracking, logic, missing information,
scientific explanations, patterns, Spanish translation, and narrative continuity.
Two prompts add English-to-Polish and Polish-to-English translation. The English-to-Polish
prompt contains a haiku about Rafal and a compiler. The 24 added prompts are specific to
this bank, and their source links point to the JSON file. The examples use the supplied
first names in fictional scenarios. Each added prompt has a distinct fixed seed.

`experiments/grug/moe_hero_ep/production_run.json` is shared with `trigger_hero.sh`. It names
the active production run, its version, cluster, and accepted ancestors with step boundaries.
Update this declaration as part of a production handoff. Do not add trial runs. The sampler
does not infer production status from job names or recent GitHub comments.

Discovery reads checkpoint directory listings and `metadata.json`. It selects only checkpoints
whose metadata says `is_temporary: false`. It does not read tensors during discovery. A
checkpoint needs committed metadata before it enters the queue. Checkpoints beyond an
ancestor's handoff boundary are excluded. Deleted checkpoints cannot be reconstructed.

One sample set contains all prompts for one checkpoint and one sampling configuration. Its
request records those inputs and a source commit. An attempt is one Iris job for that request.
The queue keeps all discovered requests. Newer steps run first, followed by retained history.
A missed schedule does not drop checkpoints. A prompt-bank or generation-setting change
creates a new sample set for every retained checkpoint. Earlier sets remain in the history.

## Sampling and storage

Each attempt uses 16 GB200 nodes with four GPUs per node. Only one sample job can be active.
The job runs at batch priority. Its scheduling limit is 24 hours and its task limit is four
hours. Iris retries preemption within the same job, up to its standard 1,000-retry limit.
Other task retries are disabled. A scheduling timeout defers the request for six hours without
using its failure budget. After three sampling failures, the report keeps the failure visible.
An absent attempt becomes
a visible failure 48 hours after submission was first planned. "Absent" means Iris returned
job-not-found for that attempt's name. A service error does not prove that a job stopped.

The sampler restores weights directly from the exact checkpoint. It reads authoritative
master weights when present. These hold the full-precision model values when the training
state also stores a lower-precision copy. It applies the router bias that training saved for
the next forward pass. It creates no optimizer
state. Missing tensors fail the attempt. There is no random-weight or older-checkpoint fallback.

Generation uses the native dropless evaluation backend, a 4,096-token context, temperature
0.2, and at most 200 new tokens. Each prompt has its own random stream. The tokenizer is pinned
to a Hugging Face commit. The sampler uses base-model continuations with special tokens and
no chat template. Results record prompt and generated token IDs, decoded text, EOS or limit
stop reasons, checkpoint metadata identity, model configuration, and source commit.

The data-sharded mesh requires a 64-row batch. Unused rows contain filler prompts. The sampler
has no KV cache. Full-rack throughput and restore compatibility across the entire
historical checkpoint set require live validation. Fixed-seed samples are qualitative evidence,
not quality scores. They need not reproduce the old manual sampler's global random stream.

| Data | Location |
| --- | --- |
| Queue and immutable results | `s3://marin-us-east-02a/marin/hero-completions/v1/` |
| Public result JSON, copied once | `gs://marin-public/hero/completions/results/` |
| Dated public reports | `gs://marin-public/hero/completions/YYYY.MM.DD/` |
| Stable report redirect | `gs://marin-public/hero/completions/latest/index.html` |

Checkpoint tensors stay in CoreWeave storage. Only small result objects go to GCS. Prompt
text, generated text, checkpoint paths, and generation settings are public. Do not add private
prompts to this bank. The browser treats generated text as plain text, never HTML.

## Schedule and recovery

`.github/workflows/hero-completions.yaml` runs from `main` at minute 17 each hour. It needs
the existing `CW_ACCESS_KEY_ID`, `CW_SECRET_ACCESS_KEY`, and `IRIS_CI_GCP_SA_KEY` repository
secrets. The service account needs Iris IAP and federation access plus write access to
`marin-public`. The workflow token needs issue-write permission. No daily operator step is needed.
The workflow does not start, stop, or change the hero or its cluster.

Before deployment, confirm that the workflow's service account is an approved submitter in
the target cluster's allowlist. Access changes require separate approval. The workflow does
not change cluster access or restart a controller.

The queue records an attempt before submission. A lost submit response is recovered through
the same deterministic Iris job name, with replacement disabled. Retries bundle the request's
original Git commit. The queue checks for a committed result before it retries a terminal job.
It waits for the active job to stop before it starts another allocation.

The workflow's queue controller discovers checkpoints and checks job status each hour. Report publication is
a separate workflow step. It can read the saved queue even if the controller step fails.
Its first attempt saves the queue's requests and statuses as that day's snapshot.
A publication retry reuses that snapshot and existing result objects. It never
starts a GPU job. The stable link changes only after the dated page is uploaded. Keep all
controller and report invocations in the workflow's shared concurrency group.

The report shows its inventory timestamp and a warning when it is more than 36 hours old.
GitHub reports workflow failures through its normal notification settings. Check the Actions
run for storage, authentication, or service errors. The browser cannot issue an alert when
GitHub stops invoking the schedule.

For a read-only inventory with configured CoreWeave credentials:

```bash
uv run --no-sync python -m scripts.ops.hero_completions inventory
```

Failed requests remain visible after their failure budget is exhausted. Correct the cause
before another attempt. To retry after a sampler, access, or service correction, change `release="hero-native-v1"`
in `experiments/grug/moe_hero_ep/completion_config.py` to a new release. This creates new
requests for every retained checkpoint, including checkpoints that succeeded before.
Change the release when restore or decoding behavior changes. Requests keep the source
commit that first created them. An unrelated change to `main` does not repeat sampling.
