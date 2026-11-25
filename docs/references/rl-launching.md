# Launching RL experiments

Marin RL experiments use one artifact entry point for planning and execution. The entry point
constructs the complete graph locally, which makes deterministic configuration errors fail before
Iris accepts a coordinator job or allocates GPUs.

## Experiment entry point

An RL launch module defines a `main` that returns the terminal artifact handles and uses
`rl_build_options`:

```python
import click

from marin.execution.lazy import ArtifactStep
from marin.rl.cli import rl_build_options


@click.command()
@click.option("--scale", type=click.Choice(["smoke", "full"]), required=True)
@rl_build_options
def main(scale: str) -> ArtifactStep:
    return build_workflow(scale).evaluation


if __name__ == "__main__":
    main()
```

Run the module without `--run` to build, validate, and print the graph:

```bash
uv run python -m experiments.post_training.example --version 2026.09.21 --scale smoke
```

Add `--run` to submit it:

```bash
uv run python -m experiments.post_training.example --version 2026.09.21 --scale smoke --run
```

Outside Iris, `--run` submits a 4-CPU coordinator with 16 GB RAM and 64 GB disk. Each
`IrisSkyRLExecution` sets the coordinator deadline for its workload. CoreWeave coordinators route
through the Marin hub to their target cluster. The coordinator runs the same module and arguments
inside Iris, where `--run` executes the artifact graph. This preserves one construction and
validation path. The submitter forwards `DAYTONA_API_KEY`, `DAYTONA_RL_API_KEY`,
`DAYTONA_EVAL_API_KEY`, and `TOGETHER_API_KEY` when present in its environment.
The RL and eval keys let rollout and evaluation use separate Daytona credentials.
Each execution declares its required names in `required_env_vars`. The launcher
checks the union across the graph before submission and reports all absent or
empty names in one error, without their values. Long-loop RL recipes require
`DAYTONA_RL_API_KEY`, `DAYTONA_EVAL_API_KEY`, `DAYTONA_TARGET`, `MARIN_PREFIX`,
`MARIN_TEMP_PREFIX`, `HF_TOKEN`, and their declared judge key, such as `TOGETHER_API_KEY`.
The US Daytona recipe uses `DAYTONA_TARGET=us`; resolve its separate RL and EVAL
keys through the [Daytona credential procedure](https://github.com/marin-community/marin/blob/main/.agents/ops/daytona.md).
Iris forwards available HF/W&B/CoreWeave credentials and the current application-secret
bundle as read-only files; ordinary child jobs inherit that bundle by default.
See [secret delivery](https://github.com/marin-community/marin/blob/main/lib/iris/docs/secrets.md).

Use `--sft PRESET` on the RL command to train a starting model and use its export for RL.
Both stages use the RL layout's storage prefix.

```bash
MARIN_PREFIX=s3://marin-us-east-02a/marin uv run python -m experiments.post_training.snowball_rl \
  --preset train-long --sft mixture-262k-32node --version 2026.11.19.1 --plan
```

The Mochi RL presets default to the completed production Mochi SFT export. The export
supplies its own tokenizer. Use `--init NAME@VERSION` to start from another published model,
or select an SFT preset with `--sft`. Mochi writes artifacts under the cluster's
`tmp/ttl=3d/` prefix. Set `MARIN_PREFIX` to use a durable prefix.

```bash
uv run python -m experiments.post_training.mochi_rl \
  --preset train-medium --sft mixture-262k-176node --version 2026.11.19.1 --plan
```

Use Mochi's `train-medium-agentic` for two updates on a task mix. Pass a versioned selection
containing both task families with `--tasks NAME@VERSION`. Enable recovery with
`--checkpoints latest`. Add `--export latest` to export the final policy.

The Snowball and Mochi preset commands use `--resume NAME@VERSION` to start from
the newest retained recovery checkpoint of a finished run. Use a checkpointing
preset and an immutable version. If the source checkpoint changes between
submission and coordinator startup, the command refuses the launch and shows
both step numbers. The source checkpoint stays in temporary storage for the
retention period recorded by its run.

Use the run's resolved launch document to export its best committed native
checkpoint, including after an early stop.

```bash
uv run --project vendor/skyrl --extra cpu python -m cloud.iris.export_hf_checkpoint \
  --launch-config ./training-launch.yaml --recover-best
```

## One Hydra launch document

Marin and MarinSkyRL share one `SkyRLLaunchConfig` YAML document. There is no parallel request
envelope and no internal command-line rendering layer.

The nested runtime uses the checkout's RolloutEngine for task execution and model requests.
Harbor tasks use Shellbox. This runtime does not support Harbor agent harnesses such as OpenCode and Terminus.
See [task rollouts](task-rollouts.md) for task execution and backend limits.

The launcher runs SkyRL's frozen CPU environment in a child process through
the vendored uv version. Both uv selections prefer a compatible system interpreter
over cached managed builds, matching Iris task setup. SkyRL has its own lock and
is excluded from Marin's workspace. The child discards the parent's Python
environment selection. The Linux worker-construction preflight opens and closes
a pidfd before building Shellbox machines and records its interpreter and base
prefix in the validation receipt. SkyRL installs Marin libraries and Harbor from
editable paths in the checkout.
Marin calls SkyRL through the executable.

Iris submits the Marin repository through its standard workspace bundle.
Workers run SkyRL from `/app/vendor/skyrl` and install the project in editable
mode with the frozen Megatron/vLLM profile. Marin artifacts and SkyRL run
records use `rigging.provenance` for checkout and launch context alongside
the resolved configuration.
Outside a Marin checkout, the launcher raises `RuntimeError` before submission.

```text
Marin experiment
  SkyRLSpec + IrisSkyRLExecution + resolved artifacts
                         |
                         v
  source launch.yaml: run, runtime, iris, ray,
                      artifacts, inputs, skyrl recipe
                         |
                         v
MarinSkyRL launch host
  compose skyrl recipe against ppo_base_config once
  inject canonical model/data/output values
  validate topology, strategy, entrypoint, TP, and batches
                         |
                         v
  resolved launch.yaml ---------> Iris allocation and submission
           |                               |
           +-------------------------------+
                         |
                         v
Iris task
  stage immutable inputs -> patch task-local paths -> run(cfg.skyrl)
```

Before submitting the GPU job, the CPU coordinator mirrors each Hub policy,
`local_inference` teacher, and Hub draft model into the regional bucket. It reuses
the existing cache and its retention period without refreshing object ages.
Resolved mirror locators are recorded in `inputs.hugging_face_models`.

GPU nodes restore policy and teacher snapshots from the mirror into their local
HF cache before Ray starts. A missing mirror marker or file falls back to the
existing direct-Hub download. Draft locators are rewritten to the mirror before
submission; warm drafts read it without a cache lock. If any listed draft file or
the completion marker is missing, node preparation repairs the mirror through
the existing Hub cache path. Local and object-store inputs keep their loading paths.

`prepare` only validates the config. Submitting Hub models requires Iris job
context; run the [experiment entry point](#experiment-entry-point) with `--run`
to start its CPU coordinator. A local launcher fails before downloading weights.

The top-level sections name what they configure: `iris` owns allocation and routing, `ray` owns
cluster bootstrap, `inputs` owns immutable model and data locators, `artifacts` owns durable and
temporary outputs, and `skyrl` is the trainer's Hydra subtree. `launch_config` means the complete
declarative input to a launch. `role_plan` means a derived resource calculation. Neither is called a
protocol because no independently versioned message exchange exists.

Experiments render their final source recipe directly into the `skyrl` mapping. There is no
`SkyRLSpec.overrides` escape hatch and MarinSkyRL never translates fields into dotted command-line
arguments. The launch host resolves Hydra defaults once; the resolved document is the input to
allocation, task bootstrap, training, checkpoint export, and the persisted run manifest.

```text
                    +-> derive Iris resources
resolved launch.yaml +-> configure Ray and staging
                    +-> call the registered run(cfg.skyrl)
                    +-> derive checkpoint-export launch.yaml (only when requested)
```

This shape keeps validation mechanical. Adding a launch setting means adding one structured field and
reading it where the behavior lives; it does not require adding matching CLI flags, argv builders, and
forwarding tests at every process boundary.

## Publish a checkpoint export

The checkpoint-export GPU job stops after writing and verifying the object-store
export. When its request includes `hf_hub_repo_id`, the launcher waits for that
job to exit, then submits a CPU Iris job to upload through the existing Hub
publisher. The CPU job targets the export's CoreWeave cluster or pins the GCS
bucket's region on GCP. It reserves staging disk from the export manifest's file
sizes and downloads up to four files at a time into Iris's disk-backed `/cache`
mount. This bounds concurrent shard buffers and avoids Docker's memory-backed
`/tmp`.
Hub credentials and object-store credentials use Iris's inherited secret files.
`hf_hub_private`, `hf_hub_revision`, and both `hf_upload_mode` values are preserved.

A failed upload leaves `hf_export_request.json` pending. Rerun the same request:
a verified object-store export skips GPU conversion and submits only CPU
publication. A missing or corrupt export still requires conversion. For example,
with the original checkpoint directory, resolved training configuration, and
export placement in these variables:

```bash
uv run --directory vendor/skyrl --frozen --extra cpu --no-group dev \
  python -m cloud.iris.export_hf_checkpoint \
  --request "$CHECKPOINT_PATH" --launch-config "$LAUNCH_CONFIG" \
  --cluster "$CLUSTER" --gpu-variant "$GPU_VARIANT"
```

Requests become complete only after all requested publication succeeds. Manual
`--no-wait` exports remain available without a Hub destination; Hub publication
requires waiting so the CPU job starts after the GPU allocation is released.

## SkyRL role plan

`SkyRLRolePlan` is the source of truth for policy, rollout, and batch geometry. Every parallelism
field is required:

- policy node and per-node GPU counts;
- inference engine count;
- tensor, pipeline, data, and expert parallel sizes;
- train, policy mini-batch, and per-GPU micro-batch sizes;
- samples per prompt and whether roles are colocated.

Render these values into `config_yaml` from the role plan. Do not maintain parallel literals or
dotted Hydra overrides for the same values. `SkyRLSpec` compares the rendered YAML to the role plan
when the experiment constructs the artifact, then writes that recipe into the launch document.

For one inference engine:

```text
engine slice = tensor parallel * pipeline parallel
engine ranks = engine slice * data parallel
```

Disaggregated vLLM engines can span hosts when TP=PP=1 and EP=DP exceeds the GPUs per host.
The runtime assigns one GPU to each worker. It checks that each worker runs on its assigned
host, uses a distinct physical GPU, and reports the parallel ranks required by the layout.
The rollout pool uses `ceil(engine count * DP / GPUs per host)` hosts. For example, 48 policy
GPUs plus one DP=EP=12 rollout engine require 15 four-GPU hosts. The planner also accepts
other configured counts with valid Megatron geometry, with no rack-size cap. Runtime evidence
for this cross-host path covers the 48/12 layout within one NVLink domain. Other layouts,
including engines that share or partly fill a host, need separate runtime qualification.

Resource-count validation alone does not keep an expert-parallel group within one
NVLink domain. Without a locality policy, Iris places the combined policy and rollout
allocation, and Ray assigns workers within it. An expert group may communicate across
the interconnect between hosts or racks. To keep the trainer and each engine local,
enable [NVLink role placement](../dev-guide/skyrl.md#keep-each-model-group-within-one-nvlink-domain).
Check placement and communication-backend support before running a new layout. Accepting a
72-GPU policy in the planner does not establish a full-rack NVL72 trainer with inference elsewhere.

Engines that fit on one host retain whole-engine packing: each host holds
`floor(GPUs per host / engine ranks)` engines. Other disaggregated engines must still fit
on one host. For colocated roles, each TP-by-PP slice must fit within and divide a policy node;
DP replicas occupy separate slices. Expert parallel size must divide `tensor parallel * data parallel`;
pipeline stages form separate expert groups. Total planned GPUs are:

```text
policy GPUs = policy nodes * policy GPUs per node
rollout GPUs = engine count * engine ranks

separate roles: policy GPUs + rollout GPUs
colocated roles: policy GPUs, which must equal rollout GPUs
```

The allocation must provide exactly the planned hosts, including whole-host rounding and
host-local packing gaps. For example, four 8-GPU policy nodes plus four node-sized rollout
engines require eight hosts. Leaving rollout DP at one needs only five hosts; an eight-host
allocation then disagrees with the derived role plan and is rejected before submission.

The preflight requires positive role dimensions, checks batch divisibility, and requires an explicit
forward microbatch size to equal the training microbatch size. It checks local engine placement,
rejects unsupported transport settings and requires an explicit KL-loss
setting. For every launcher, preflight checks policy and reference Megatron parallelism against
the GPUs assigned to each role. Megatron builds separate dense and expert grids. The role GPU count
must be divisible by both `TP * PP * CP` and `ETP * EP * PP`, where ETP is expert tensor
parallelism. An omitted or null ETP inherits the role's TP before these checks.
Context-parallel ranks participate in the expert grid, so EP need not divide dense
data parallelism.

Adopted long-loop RL rounds also stage model metadata and the pinned tokenizer
during their CPU no-run preflight, then run the training driver's vocabulary
check. A model-only export uses its explicitly declared policy tokenizer in a
separate local metadata view; the binding records the model identity, tokenizer
revision and file hashes. Embedded checkpoint tokenizers remain independent
and must match the selected tokenizer's token-to-ID mapping. This check reads
metadata without downloading weight shards or modifying the published export.
Its result is recorded as `vocabulary_check` in the native preflight receipt.

Set `trainer.rollout_buffer.groups_per_family` for fixed numbers of prompt groups per task family in
each training batch. Each Parquet training row supplies its `family`. The counts must sum to
`trainer.train_batch_size`. Use `batch_policy: full_batch` and leave `data.sampling.kind` unset.
Each family needs its group count times `max_staleness_steps + 1` usable rows so that every batch in
the rollout window can use distinct prompts. A rejected group releases a slot in its own family.
Each family draws prompts from its own seeded passes. Checkpoints retain the position in each
family’s order. Omit the family counts to use the configured prompt order, including curriculum sampling.

## Artifact and runtime boundaries

Identity-bearing inputs belong in `SkyRLSpec`: model and tokenizer,
data artifacts, a starting native checkpoint, topology, seed, retention, and the rendered config. Cluster routing,
host resources, priority, and retry policy belong in `IrisSkyRLExecution`; changing placement must
not fork artifact identity. A versioned task mix (`RlCurriculum`) also belongs in `SkyRLSpec`; it
renders the trainer's sampling settings and is named in the run record. See
[RL curricula](rl-curricula.md).

The `records` tracker requires a fresh attempt ID and `iris.max_retries=0`.
Its Iris jobs also use zero preemption retries. After a preemption, start a fresh
attempt with `trainer.resume_mode=from_path` and `trainer.resume_path` pointing
to its saved `global_step_N` checkpoint. Known checkpoint ancestors contribute
metrics through the resumed boundary; a checkpoint without recorded ancestry
contributes only the new attempt's metrics. Other tracker backends retain Iris's
ordinary preemption retry budget.

`IrisSkyRLExecution.container_profile` accepts an `iris.rpc.job_pb2.ContainerProfile` enum for the
coordinator and trainer jobs. Enum names are serialized at the launch boundary. It defaults to `None`,
which leaves the Iris default in effect. In-pod sandbox graders need namespace capabilities: use
`job_pb2.CONTAINER_PROFILE_PRIVILEGED` and give CPU preflight children the same profile. Iris requires
admin authorization for this elevated profile. Grader commands still run inside their own sandbox.

Use artifact dependencies for model and data handoffs. A raw checkpoint path hides provenance and
can point at an incomplete export. Keep resume behavior and terminal export explicit. Use Marin's
temporary-storage helper for checkpoints and trajectories so paths have a bounded lifetime and do
not repeat a source bucket prefix.

Shared W&B trackers bound final SDK uploads to 30 seconds per process. The
training entrypoint first waits up to 30 seconds for remote node loggers, then
finishes its own uploads using the same SDK deadline. These waits can take up to
60 seconds, plus other cleanup. Upload timeouts produce warnings and preserve
the workload's success or original exception. W&B may show the run as Crashed
or Failed after a primary upload timeout even when training succeeds. Some W&B
records may remain unuploaded. Other cleanup can still exhaust the existing
120-second teardown watchdog.

Marin sets the run's export root (`trainer.export_path`) to `<run output>/exports`.
Set `trainer.dump_data_batch: true` to save diagnostic `TrainingInputBatch`
pickle files under `<trainer.export_path>/dumped_data/global_step_<step>_training_input.pkl`.
Local and cloud export paths use the same format. Dumps contain the batch after
importance-sampling corrections and advantage calculation. Group UIDs and raw
rewards are removed before dumping.

SkyRL keeps the two most recent recovery checkpoints by default. For an RSI loop
that needs only the latest recovery point, set
`retention=SkyRLRetentionPolicy(resume_checkpoint_count=1)` in `SkyRLSpec`.
Marin translates this policy to `trainer.max_ckpts_to_keep`; standalone SkyRL
recipes can set that field directly. Cleanup runs after the replacement checkpoint
has uploaded and its latest-checkpoint marker has been published. Pending HF
exports and the selected best native checkpoint remain protected beyond this
rolling window. On success, waiting preset launches apply `checkpoints` and `keep_every` after
writing the terminal receipt. Detached launches keep the trainer's rolling window. HF export is
opt-in. `skyrl_step(..., export_hf=True)` requests a durable terminal model and a separate export
job. With `run.export_hf: false`, `SkyRLRun.hf_model_uri` is null. Successful runs still write
evaluation, resolved configuration, and terminal result records. Pipelines that serve the
resulting model for downstream evaluation must request export.

Every HF export writes `generation_config.json` through SkyRL's Hugging Face
model lifecycle utilities. It keeps the explicit fields of the source model's
generation config and fills in the rest:
`bos_token_id`, `eos_token_id` as the union of the tokenizer EOS id and the
end-of-turn tokens the chat template emits (128001 and 128009 for the Llama 3
header template), and `do_sample: true`, `temperature: 0.6`, `top_p: 0.95`.
vLLM applies these sampling values to requests that omit them. SkyRL rollouts
send explicit sampling parameters, and behavior-logprob training sets
`generation_config: vllm` so the checkpoint's defaults never reach rollouts.

With `SkyRLSpec`, configuration rendering binds the `best_checkpoint` callback's
`initial_model_uri` to the resolved model input before artifact fingerprinting.
Standalone recipes must set this URI explicitly.

The `best_checkpoint` callback accepts a `selection_metric` whose values must be
finite normalized scores in [0, 1], with larger values preferred. Set
`save_after_fraction` to delay trained-checkpoint selection until that fraction
of the total steps has elapsed. With the default one-observation window,
step-zero evaluation records the initial model as a candidate without saving
weights; if it remains best, load the
`model_uri` recorded in `best_checkpoint.json`. Optional `preservation_max_drop`
entries map metric names to allowed absolute drops from their step-zero values;
all gates must pass before a candidate can replace the best checkpoint. The
candidate must attempt `expected_tasks`, score at least `minimum_scored_tasks`,
and pass `eval/selection_eligible`. With preservation gates, an ineligible
step-zero evaluation fails immediately. The
baseline and selection policy are recorded in `best_checkpoint.json` and the
baseline is restored on resume. Keep metric definitions and the selection policy
unchanged on resume.

For selection over several evaluations, set `ema_window` and
`evaluation_interval` on `best_checkpoint`. A window of five with interval two
combines `selection_metric` at the five scheduled optimizer steps ending at the
current step. The newest observation has weight one; each older observation's
weight is multiplied by `1 - 2 / (5 + 1)`. Divide the weighted sum by the sum of
the weights. A missing observation leaves the window ineligible. Preservation
gates still use the current evaluation. The callback records observation history
and selection settings; changing the window or interval on resume fails.

For asynchronous mixtures, `trainer.rollout_buffer.consumed_route_quotas` sets
the number of prompt groups from each route in every consumed batch.
`source_routes` maps each input's `data_source` to a route. Quotas require
`batch_policy: full_batch` and must sum to the training batch size in prompt
groups. Rejected groups are replaced within their route. Generation slots
rotate between routes so one repeatedly failing route cannot occupy every slot.

`trainer.rollout_buffer.full_batch_fill` is an opt-in bound for `full_batch`
with dynamic sampling disabled. For example:

```yaml
full_batch_fill:
  straggler_timeout: 600
  timeout: 900
```

Both clocks start when the learner begins admitting a batch, after any
evaluation/publish fence. Rejected groups retain their original source,
route and family slot; a replacement cannot silently draw from another source
in the same route. At `straggler_timeout`, an already valid lookahead group
can exchange slots with a pending group of exactly the same source, route and
family. The pending prompt keeps generating for the donor's later batch;
neither prompt is dropped or scored again. Both directions of the exchange
must fit the policy-staleness window. The source allocation drawn for each
batch and its route/family counts therefore remain unchanged, although the
within-source prompt order and staleness can change. Existing admission
rejection still excludes unusable groups; this does not eliminate that
within-source selection bias.

At `timeout`, admission fails with `BatchFillDeadlineExceeded` if a full valid
batch is unavailable. The absolute clock covers admission callbacks and payload
fetches and does not reset on partial progress, rejection or replacement.
Failure never trains an underfilled batch or relaxes quotas. A deadline bounds
waiting, not the existence of usable data: without a matching ready reserve,
it can stop the run instead of producing an update. These settings do not
shorten trajectory budgets, change timeout verifier scoring, add generation
slots or change any existing recipe when left null. Pending source slots
survive checkpoint/restart; a new process starts a new admission clock.

The startup concurrency event records both limits. Native
`rollout_batch_straggler_exchange` events record the batch pair, promoted UID,
pending lease and stratum. `rollout_batch_fill_deadline` records an exhausted
batch and its pending strata. Require source provenance on every input, use a
fresh artifact version, and review these events and actual fill times before
claiming a performance improvement.

The `wall_time_budget` callback stops after a completed update when `seconds`
have elapsed since this process's first optimizer completion. Startup is
excluded; subsequent evaluation and checkpoint work counts toward the budget.
Set `groups_per_step` to the prompt-group batch size. The callback writes
`training-budget.json` under the export root. Its `drain_margin` records the
planned allowance in seconds for finalization; it does not enforce a deadline.
Normal trainer finalization drains work after the stop. A fresh
process starts a fresh clock, including when it restores a checkpoint.

Native checkpoints default to lifecycle-managed TTL storage derived from the RL artifact name and version.
Set `SkyRLRetentionPolicy.native_checkpoint_storage` to
`SkyRLCheckpointStorage.PERMANENT` to write native checkpoints under
`<artifact_output>/checkpoints`, as the SFT checkpointer does. Attempts, traces and
Ray logs keep their temporary storage lifetime. The launch document and terminal
run record use the same checkpoint root, including for native recovery and draft
checkpoints.
After runtime changes, use a fresh RL artifact version. Reusing the old
version with `resume_mode=latest` can load private Torch or distributed state serialized by the old
runtime. Reuse a version across runtime changes only after establishing checkpoint compatibility or
selecting a fresh checkpoint root.

### Native checkpoint inputs

Set `SkyRLSpec.starting_checkpoint` to an artifact handle for a completed
`global_step_N` directory. Give the checkpoint artifact an immutable name and
version. Give the continuation a new RL artifact version when changing that
input; reusing an existing run version with a different starting checkpoint fails.
The launch document records the starting checkpoint's URI and artifact identity
(`name@version:fingerprint`) under `inputs.starting_checkpoint`; the trainer reads native state directly.
The input requires `resume_mode=latest`; omit any recipe `trainer.resume_path`.
Completed immutable SkyRL artifact versions also reject configuration drift instead of
returning a cached result.

Automatic `resume_mode=latest` first uses the current run's latest checkpoint.
If that run has no checkpoint marker, it uses the supplied starting checkpoint,
or starts fresh if none was supplied. Repeat the same continuation command to
recover its progress. A corrupt marker, missing checkpoint or load error still
fails; it never silently falls back to older state. A continuation writes to
its own output root, leaving the starting run's checkpoints unchanged. Without
`starting_checkpoint`, the trainer's explicit `from_path` and `none` modes remain available.

The initial import starts the configured dataset from the beginning. Leave
`restore_dataloader_state=true` to recover this run's data position and queued
rollouts after interruption. Optional `reset_global_step_on_resume` and
`reset_distillation_token_count_on_resume` apply only to the initial import;
recovery preserves the counters saved by this run.

Training limits use the absolute trainer step. To apply two more updates from
`global_step_500`, set `max_steps=502` and enough `epochs` to reach that target.
A checkpoint already at or beyond the target completes without another update.
Set `reset_global_step_on_resume=true` to count a new stage from zero instead;
this leaves the restored optimizer and scheduler state intact.

### Frozen Micro210 evaluation

`python -m experiments.evaluation.micro210` builds the canonical Micro210
evaluation graph. Supply `--model-uri`, the SHA256 of its
`.marinskyrl-model-manifest.json` through `--model-manifest-sha256`, a tokenizer
repository and full commit through `--tokenizer-uri` and `--tokenizer-revision`,
and the run identity through `--name` and `--version`. The command prints the
graph by default; `--run` submits it through Iris.

The protocol lives in durable object storage. The code pins its URI, revision,
SHA256 and byte size, and verifies those bytes before reading the policy.
The model binding cannot set sampling, sandbox resources, harness deadlines or
the judge. Arbitrary task-subset overrides remain forbidden. The graph checks the prepared task bytes and model
manifest before GPU allocation, then writes `canonical-evaluation.json` with the
protocol hash beside the run outputs. Sandbox resources come from each task.
The bound tokenizer must map the frozen stop IDs to `<|end_of_text|>` and
`<|eot_id|>`; an incompatible tokenizer fails before allocation. Retained
trajectories use the supplied run identity.

Use repeated `--benchmark <name>` options for a diagnostic or correction subset.
The standard graph resolves this allow-list against the content-pinned prepared
rows and keeps their order, repetitions, prompts and task settings unchanged.
Unknown benchmark names fail before GPU allocation. `canonical-evaluation.json`
records the original row indices and both input hashes beside the unchanged
policy hash. Omitting the option evaluates the full suite. A subset result is
not a full-suite result; label its coverage when reporting it.

Mochi exports use the adopted rounds in
`experiments/post_training/long_loop/rounds/mochi-{base,sft}-eval.yaml` through
`experiments.post_training.long_loop.round_runner`. Its serving section declares
InfiniBand transport, `NCCL_IB_HCA=ibp`, and a 4-MiB NCCL buffer. The historical
256-KiB qualification predates the NVLSTree corruption finding in #11649.

Preflight the exact command from a clean checkout before adding `--run`:

```bash
uv run python -m experiments.post_training.long_loop.round_runner \
  --round experiments/post_training/long_loop/rounds/mochi-sft-eval.yaml \
  --through eval --version <fresh-version> \
  --source-revision "$(git rev-parse HEAD)" --validation-output /tmp/mochi-eval
```

The policy and prepared-task references use CoreWeave S3 so regional tasks can
read the pinned bytes. Verify admission from an Iris CPU task in the evaluation
region as well as locally; Mac GCS credentials do not establish remote access.

Inspect `launch.yaml`, `compiled.yaml`, and the required/allocated GPU counts in
`graph.json`. Preflight rejects local model rank counts larger than the physical
allocation. Runtime verifies the API judge's authenticated model catalog before
allocating rollout GPUs. Historical gate-2 evaluations using the earlier frozen
MiniMax policy are preserved at `benfeuer/mochi-micro210-gate2-frozen`;
its commit `da0663fbea` includes the self-hosted judge launch and receipt checks.

## Pre-launch review

Before submitting a large run:

1. Print the plan with the intended immutable version and selection options.
2. Check the resolved role plan and topology, especially engine count and DP/TP/PP/EP.
   Check that explicit forward and training microbatch sizes agree.
   For `fully_async`, confirm the train batch and policy mini-batch are equal.
3. Confirm the vendored runtime supports the chosen strategy, CUDA stack, model, and vLLM
   geometry. After runtime changes, use a fresh RL artifact version unless its checkpoints are
   explicitly compatible.
4. Confirm request and response token budgets match the dataset and chat template.
5. Check coordinator and worker CPU, host memory, disk, concurrency, timeout, and credentials.
6. Check W&B project/entity, Hub upload behavior, checkpoint retention, resume mode, and terminal
   export.
7. Run a smoke that follows the same runtime, artifact handoffs, and role geometry as the full run.

Past delayed failures have included missing coordinator dependencies, invalid nested job names,
wrong W&B entities, undersized coordinator or GPU-host memory, untyped checkpoint handoffs, backend
and CUDA capability mismatches, malformed Iris endpoints, tokenizer or chat-template drift, staging
races, rendezvous port collisions, and invalid draft artifacts. Add a construction-time check when a
failure is deterministic from the artifact inputs. Keep service outages, hardware faults, and
data-dependent numerical failures in runtime diagnosis rather than guessing at them in the launcher.

For live submission, monitoring, or recovery, follow the `launch-rl` and `use-iris` skills and
`lib/iris/OPS.md`. Do not restart a cluster or resubmit a failed run without explicit authority.

If a long-loop coordinator fails after its RL and evaluation children succeed,
recover the reporting tail with `experiments.post_training.long_loop.postprocess`.
Its hash-pinned plan records the fresh `version`, `postprocessing_revision`,
original `resolved_spec` URI and SHA-256, completed `artifacts` by stage,
`artifact_record_sha256` pins, `additional_evidence`, `sft_geometry`, and any
explicit `accepted_unscored_attempts`. Omit `recovery` when the native evaluation
is already complete. The graph adopts the producers and runs reporting,
SFT accounting and archiving on CPUs:

```bash
uv run python -m experiments.post_training.long_loop.postprocess \
  --plan-uri <immutable-plan-uri> --plan-sha256 <sha256> --version <fresh-version>
# After inspecting the plan, repeat with --run to submit.
```

Canonical reporting verifies the prepared Parquet hash, derives cohort identities
from its rows, and reads only the completed evaluation attempt. It does not rely
on the older `selection.json` preparation file or its historical attempt count.
An explicitly recorded unscored attempt remains unscored and blocks native
checkpoint cleanup; preserving it does not certify full evaluation coverage.
To continue after a successful report, include its `report` and `eval_inputs`
artifacts in the plan. Accounting replays only the consumed data and trainer
configuration, without loading model or optimizer plugins. Supply its `pack`,
`measured_step_seconds`, and `timing_source` through `accounting_settings` when
the original archive settings omit them; pin the native timing evidence in
`additional_evidence`.
The recovery plan may override `task_timeouts.accounting` and
`task_timeouts.archive` in seconds without changing the adopted producers.
Accounting saves an immutable sample prefix and `.pin.json` every 4,096 samples
under `progress/`. After an interruption, supply the latest complete pin's JSON
contents as `accounting_resume`; replay verifies both the consumed configuration and native
checkpoint before continuing after that prefix.
After accounting succeeds, adopt its `sft_data_accounting` artifact to retry only
the archive. When population reports were published separately from accounting,
set `archive.sft_data_accounting.population_evidence` in the resolved recovery
specification to filename/source/SHA-256 pins. The archive verifies these pins,
requires the same population/padding/decision evidence, and records the supplement
without changing the accounting producer.

For the current curation, training and evaluation recipe, see the
[Snowball RL experiments](../recipes/rl-experiments.md) and the
[selection handoff](task-curation.md#train-on-a-snowball-selection).

### Cluster network selection

On `cw-us-east-02a`, the SkyRL launcher defaults `NCCL_IB_HCA` to `ibp` so RDMA
uses the InfiniBand ports instead of the hosts' additional `mlx5_*` Ethernet
ports. An explicit runtime setting takes precedence. The interconnect gate uses
the same environment as the launched workers and must pass before they start.

### NCCL SIMPLE buffers

Policy and reference roles resolve `nccl_buffer_size_bytes` to 4,194,304 bytes
(4 MiB) by default. The role runtime environment sets `NCCL_BUFFSIZE` explicitly,
including inference engines that receive the policy's weight-sync setting. This
prevents an inherited small-buffer value from replacing the resolved default.
A `null` value deliberately delegates sizing to the process environment and NCCL
config files.

NCCL 2.30.7 can return incorrect NVLSTree reductions when its tuned staging payload
exceeds a SIMPLE FIFO slot. A 256-KiB buffer reproduced this failure through both
the native C API and ProcessGroupNCCL. A 1-MiB buffer also failed at 512-MiB
messages; 4-MiB controls passed the tested 16–512-MiB range. See
[the investigation and source-library A/B](https://github.com/marin-community/marin/issues/11649).
This mitigation uses the existing library and preserves byte verification. It
does not establish correctness for every message size or topology. Check resolved
role settings and the effective `NCCL_BUFFSIZE` in worker INIT logs before scaling
a new configuration. Reference workers now retain larger buffers, so include that
memory in colocated-role capacity estimates.
