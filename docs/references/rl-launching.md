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
Iris forwards available HF/W&B/CoreWeave credentials and the current application-secret
bundle as read-only files; ordinary child jobs inherit that bundle by default.
See [secret delivery](https://github.com/marin-community/marin/blob/main/lib/iris/docs/secrets.md).

Use `--sft PRESET` on the RL command to train a starting model and use its export for RL.
Both stages use the RL layout's storage prefix.

```bash
MARIN_PREFIX=s3://marin-us-east-02a/marin uv run python -m experiments.post_training.snowball_rl \
  --preset train-long --sft mixture-262k-32node --version 2026.11.19.1 --plan
```

The Mochi RL presets default to the verified CoreWeave copy of the production Mochi SFT
step-400407 export. Its SkyRL model manifest binds the weights to the SFT run's tokenizer.
Use the `train-long` preset to plan the standard one-rack loop with the default initializer.

```bash
uv run python -m experiments.post_training.mochi_rl \
  --preset train-long --version YYYY.MM.DD --plan
```

The one-rack presets use the pinned two-layer DSpark draft with four proposals when
the resolved initializer matches step 400407. Selecting a different task mix with
`--tasks` keeps that default. Three-rack presets use plain decoding pending
measurements with their 262k context and FlashInfer serving configuration.

Use `--init NAME@VERSION`
to start from another published model or select an SFT preset with `--sft`.
Mochi writes artifacts under the cluster's
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
uv run --isolated --frozen --project vendor/skyrl --extra cpu python -m cloud.iris.export_hf_checkpoint \
  --launch-config ./training-launch.yaml --recover-best
```

The `cpu` extra runs on macOS. The `vllm` and `megatron` extras and GPU torch
builds do not run on macOS; export and training jobs run on Iris.

## One Hydra launch document

Marin and MarinSkyRL share one `SkyRLLaunchConfig` YAML document. There is no parallel request
envelope and no internal command-line rendering layer.

SkyRL runs prepared TaskSpecs through RolloutEngine and Shellbox. It supplies the serving engine's
sampled token IDs and log probabilities to the same session for shell, Pi, and Terminus-2 tasks.
The launch recipe binds agent settings, machine resources, and separately scoped training and
checkpoint evaluation Daytona credentials. See [task rollouts](task-rollouts.md) for execution
and backend limits.

The launcher runs the vendored uv version through `uv tool run`, then starts
SkyRL's frozen CPU environment with `uv run --project vendor/skyrl`. Both uv
selections prefer a compatible system interpreter over cached managed builds,
matching Iris task setup. SkyRL has its own lock and is excluded from Marin's
workspace. The child discards the parent's Python environment selection. SkyRL
installs the Marin libraries from editable paths in the checkout.
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

Coordinated vLLM data-parallel startup uses `trainer.model_load_retry` for both
staged local weights and S3 streaming. An actor startup failure recreates the
whole engine gang with fresh rendezvous ports, within the existing
`generator.engine_init_timeout_seconds` deadline.

DP rendezvous ports are bound and held on the engine's target node, excluding
that node's `/proc/sys/net/ipv4/ip_local_port_range`, kernel-reserved service
ports, and Iris's default named-port range. Kernel reservations also cover
deployment-specific named-port ranges published by Iris's worker bootstrap.
The same reserved RPC and process-group port vector is passed explicitly to
every rank. Rank zero releases the sockets immediately before vLLM binds them;
excluding the kernel's ephemeral range prevents outbound connections from
taking those ports during the handoff. The former fixed 20000–29999 pool
overlapped Iris's widened ephemeral range.

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
uv run --isolated --frozen --project vendor/skyrl --extra cpu --no-group dev \
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

A rollout window holds `max_staleness_steps + 1` batches. The training pool needs a distinct
prompt for each group slot in the window and one spare after dataset preparation.

Set `trainer.rollout_buffer.groups_per_family` for fixed numbers of prompt groups per task family in
each training batch. Each Parquet training row supplies its `family`. The counts must sum to
`trainer.train_batch_size`. Use `batch_policy: full_batch`.
Each family needs one distinct usable prompt per open group slot in that family and one spare.
A rejected group releases a slot in its own family.
Leave `data.sampling.kind` unset for seeded passes within each family. The
`prompt-thompson-source`, `prompt-thompson-contextual`, `prompt-thompson-cost-aware`, and
`prompt-thompson-within-source-cost-aware` orders rank eligible prompts within the leased family
and learn from completed optimizer steps. The within-source variant allocates each family's draw
window using the reference row weights in that family, capped by available rows per source.
Checkpoints retain each family's draw position.
Omit the family counts to use other prompt orders, including curriculum sampling.

`prompt-thompson-within-source-pass-cost-aware` draws from the least-visited eligible
prompts in the scheduled source. A visit counts when the loader offers the prompt, so
stale or discarded work consumes a visit. A quarantined or live prompt catches up when
it becomes eligible again. Each source window keeps its selected prompts distinct.
Set `data.sampling.epsilon=1` for uniform draws within the same pass and source window.
This order uses a flat catalog and checkpoints visit counts, the pending source window,
its Thompson draw, and the loader's quarantine.

The Qwen3-0.6B sampler experiment provides `smoke-{source-only,contextual,cost-aware}`
and `pilot-{source-only,contextual,cost-aware}` presets in `experiments.post_training.qwen_sampler`.
The `-seed29` pilot variants provide a second paired seed. All three adaptive arms share
a discounted regularized logistic Thompson draw, distinct prompt windows, and 20%
exploration from the reference mixture. The source-only arm uses source indicators.
The contextual arms add public prompt length, text shape, and hashed word sequences
centered within each training source. The cost-aware arm also fits generated token
counts and applies a bounded inverse-cost weight.

A useful dispatch has a fully verified group with normalized score spread of at least
0.05 and contributes nonzero advantage tokens to an optimizer update. Feedback comes
after the optimizer, with generated tokens charged for discarded, unverified, and
untrained groups. Checkpoints retain the acquisition window, model, and random state.
SkyRL checks the prepared held-out prompts for overlap before training and collects
sampler feedback from training rollouts only. Use `data.sampling.decay` to discount older
observations. The bin curriculum's `reversion_mass` must be zero for prompt Thompson
and adaptive source orders.

Set `data.sampling.kind=adaptive-source` to favor sources whose answers receive different
verifier scores within a prompt group. `source_column` selects the prepared row label
(`data_source` by default, `family` for labeled task sets, or `teacher_route` for stamped
Parquet). `domain_weights` sets the reference shares. An empty map uses the filtered row
counts. At least two answers per prompt are required.

When selection fills a batch and before its optimizer update, the sampler discounts the
counts from previous batches by `decay`. It adds one observation per fully verified
group judged for the batch, including groups discarded by dynamic sampling. It weights each
source's reference share by its estimated mixed-group rate under a Beta prior
(`prior_alpha`, `prior_beta`), then reserves `epsilon` of the total mass for the reference
mix. Set `epsilon=1` for a fixed-share control with the same draw and checkpoint behavior.
For the Thompson comparison, set `kind=adaptive-source-thompson`. It uses the same
verified groups and draws one rate per source from its Beta posterior at initialization
and after each feedback batch. It multiplies each draw by its reference share, normalizes
the results, floors each source at
`epsilon / number_of_sources`, and renormalizes. All subsequent draw windows use these
weights until the next feedback batch. Checkpoints preserve the current weights and the
posterior random-number state.

Source draws are stochastic, and rows are unique within a draw window. In-flight prompts,
source sizes and admission can change the realized mix. The `curriculum/<source>/` metrics
use these denominators:

- `proposed_rows` counts sampler draws since the last feedback update, including rows
  skipped because their UID is live or quarantined. `assigned_groups` counts one
  dispatch for each group sent to a worker, including a group found by scanning eligible
  rows or sent again for regeneration.
- `rejected/<reason>` counts completed groups rejected before feedback, attributed to the
  batch that inspected them. Reasons include fully masked, too few eligible answers, stale
  and duplicate UIDs. Absent rejection keys count as zero.
- `groups` counts groups judged for the batch, including dynamic-filter discards.
  `graded_groups` counts those with complete verifier scores. `informative_groups` counts
  fully graded groups with at least two different scores. `weight` is the next target share.

With asynchronous prefetch, assignments and judgments in one update can refer to different
groups. With `trainer.dump_grader_report=true`, the training section of
`grader_report.json` reports per-subset verifier and rollout failures. Join its
`training_tasks` counts to the selection manifest by task ID to compute per-source
verified coverage across completed answers, including groups rejected by admission.

## Artifact and runtime boundaries

Identity-bearing inputs belong in `SkyRLSpec`: model and tokenizer,
data artifacts, a starting native checkpoint, topology, seed, retention, and the rendered config. Cluster routing,
host resources, priority, and retry policy belong in `IrisSkyRLExecution`; changing placement must
not fork artifact identity.

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

For a single immutable Parquet task selection, the SkyRL driver caches validated rows and prompt
lengths for 14 days in the checkpoint storage backend's temporary bucket. The key includes the
selection's Parquet and manifest bytes, the loaded fast tokenizer and template, task runtime settings,
and the installed preparation code. Each launch applies its own prompt limit to the cached lengths.
A cache miss validates the selection, writes batches of prepared rows and lengths to driver-local disk,
and publishes the Parquet object when complete. A hit decodes the object in batches to driver-local
disk. The driver memory-maps the prepared rows during training, so its local disk needs space for
the uncompressed dataset as well as the staged Parquet file.

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

The `best_checkpoint` callback accepts a `selection_metric` whose values must be
finite normalized scores in [0, 1], with larger values preferred. Set
`save_after_fraction` to delay trained-checkpoint selection until that fraction
of the total steps has elapsed. Step-zero evaluation still records the initial
model as a candidate without saving weights; if it remains best, load the
`model_uri` recorded in `best_checkpoint.json`. Optional `preservation_max_drop`
entries map metric names to allowed absolute drops from their step-zero values;
all gates must pass before a candidate can replace the best checkpoint. The
step-zero evaluation must meet the configured coverage and eligibility checks
when preservation gates are set; otherwise the callback fails immediately. The
baseline and selection policy are recorded in `best_checkpoint.json` and the
baseline is restored on resume. Keep metric definitions and the selection policy
unchanged on resume.

The temporary checkpoint root is derived from the RL artifact name and version.
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

For the current curation, training and evaluation recipe, see the
[Snowball RL experiments](../recipes/rl-experiments.md) and the
[selection handoff](task-curation.md#train-on-a-snowball-selection).

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
