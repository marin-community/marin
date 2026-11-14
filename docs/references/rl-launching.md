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

## One Hydra launch document

Marin and MarinSkyRL share one `SkyRLLaunchConfig` YAML document. There is no parallel request
envelope and no internal command-line rendering layer.

The nested runtime uses the checkout's RolloutEngine for task execution and model requests.
Harbor tasks use Shellbox. This runtime does not support Harbor agent harnesses such as OpenCode and Terminus.
See [task rollouts](task-rollouts.md) for task execution and backend limits.

The launcher runs `uv run --isolated --project vendor/skyrl --frozen --no-group dev
marinskyrl` in a child process. SkyRL has its own lock and is excluded from Marin's
workspace. The child discards the parent's Python environment selection. SkyRL
installs Marin libraries and Harbor from editable paths in the checkout.
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

Resource-count validation does not ensure that an expert-parallel group stays within one
NVLink domain. Iris places the combined policy and rollout allocation, and Ray assigns workers
within it. An expert group may communicate across the interconnect between hosts or racks.
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
rejects unsupported transport settings and separate critic roles, and requires an explicit KL-loss
setting. For every launcher, preflight checks policy and reference Megatron parallelism against
the GPUs assigned to each role. Megatron builds separate dense and expert grids. The role GPU count
must be divisible by both `TP * PP * CP` and `ETP * EP * PP`, where ETP is expert tensor
parallelism. An omitted or null ETP inherits the role's TP before these checks.
Context-parallel ranks participate in the expert grid, so EP need not divide dense
data parallelism.

## Artifact and runtime boundaries

Identity-bearing inputs belong in `SkyRLSpec`: model and tokenizer,
data artifacts, a starting native checkpoint, topology, seed, retention, and the rendered config. Cluster routing,
host resources, priority, and retry policy belong in `IrisSkyRLExecution`; changing placement must
not fork artifact identity.

`IrisSkyRLExecution.container_profile` accepts an `iris.rpc.job_pb2.ContainerProfile` enum for the
coordinator and trainer jobs. Enum names are serialized at the launch boundary. It defaults to `None`,
which leaves the Iris default in effect. In-pod sandbox graders need namespace capabilities: use
`job_pb2.CONTAINER_PROFILE_PRIVILEGED` and give CPU preflight children the same profile. Iris requires
admin authorization for this elevated profile. Grader commands still run inside their own sandbox.

Use artifact dependencies for model and data handoffs. A raw checkpoint path hides provenance and
can point at an incomplete export. Keep resume behavior and terminal export explicit. Use Marin's
temporary-storage helper for checkpoints and trajectories so paths have a bounded lifetime and do
not repeat a source bucket prefix.

SkyRL keeps the two most recent recovery checkpoints by default. For an RSI loop
that needs only the latest recovery point, set
`retention=SkyRLRetentionPolicy(resume_checkpoint_count=1)` in `SkyRLSpec`.
Marin translates this policy to `trainer.max_ckpts_to_keep`; standalone SkyRL
recipes can set that field directly. Cleanup runs after the replacement checkpoint
has uploaded and its latest-checkpoint marker has been published. Pending HF
exports and the selected best native checkpoint remain protected beyond this
rolling window. HF export is opt-in: `skyrl_step(..., export_hf=True)` requests a durable terminal
model and a separate export job. The structured launch config defaults `run.export_hf`
to false as well. Without export, successful runs retain native recovery checkpoints,
in-training evaluation, resolved configuration and terminal result records; `SkyRLRun.hf_model_uri`
is null. Pipelines that serve the resulting model for downstream evaluation must request export.

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
`global_step_N` directory. Use an immutable adoption name and version for each
starting checkpoint. Use a new RL artifact version for each starting checkpoint;
automatic recovery with the same version prefers that version's existing checkpoints.
The launch document records its URI and identity under
`inputs.starting_checkpoint`; the trainer reads native state directly.

Automatic `resume_mode=latest` first uses the current run's latest checkpoint.
If that run has no checkpoint marker, it uses the supplied starting checkpoint,
or starts fresh if none was supplied. Repeat the same continuation command to
recover its progress. A corrupt marker, missing checkpoint or load error still
fails; it never silently falls back to older state. Explicit `from_path` always
loads the supplied path, and `none` disables loading. A continuation writes to
its own output root, leaving the starting run's checkpoints unchanged.

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
[quick RL loop](../recipes/quick-rl-loop.md).
