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
validation path. The submitter forwards `DAYTONA_API_KEY`, `HF_TOKEN`, and `WANDB_API_KEY` when they
are present in its environment.

## One Hydra launch document

Marin and MarinSkyRL share one `SkyRLLaunchConfig` YAML document. There is no parallel request
envelope and no internal command-line rendering layer.

```text
Marin experiment
  SkyRLSpec + IrisSkyRLExecution + resolved artifacts
                         |
                         v
  source launch.yaml: run, runtime, iris, ingress, ray,
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
                    +-> derive checkpoint-export launch.yaml
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

For disaggregated roles, one engine's ranks must fit within one node. For colocated roles, each
TP-by-PP slice must fit within and divide a policy node; DP replicas occupy separate slices. Expert
parallel size must divide `tensor parallel * data parallel`; pipeline stages form separate expert
groups. Total planned GPUs are:

```text
policy GPUs = policy nodes * policy GPUs per node
rollout GPUs = engine count * engine ranks

separate roles: policy GPUs + rollout GPUs
colocated roles: policy GPUs, which must equal rollout GPUs
```

That total must equal `num_nodes * gpus_per_node`. For example, four 8-GPU policy nodes plus four
node-sized rollout engines require 64 GPUs. Leaving rollout DP at one describes only 36 GPUs and is
rejected during graph construction.

The preflight also requires positive dimensions, checks batch divisibility, and rejects a runtime
profile whose installed backend contradicts `trainer.strategy`. The `fully_async` entry point has a
stricter batch contract: `train_batch_size` must equal `policy_mini_batch_size`.

## Artifact and runtime boundaries

Identity-bearing inputs belong in `SkyRLSpec`: the pinned MarinSkyRL runtime, model and tokenizer,
data artifacts, topology, seed, retention, and the rendered config. Cluster routing,
host resources, priority, and retry policy belong in `IrisSkyRLExecution`; changing placement must
not fork artifact identity.

Use artifact dependencies for model and data handoffs. A raw checkpoint path hides provenance and
can point at an incomplete export. Keep resume behavior and terminal export explicit. Use Marin's
temporary-storage helper for checkpoints and trajectories so paths have a bounded lifetime and do
not repeat a source bucket prefix.

The temporary checkpoint root is derived from the RL artifact name and version, not only from the
runtime commit. When `SkyRLRuntime.commit` changes, use a fresh RL artifact version. Reusing the old
version with `resume_mode=latest` can load private Torch or distributed state serialized by the old
runtime. Reuse a version across a runtime repin only after establishing checkpoint compatibility or
selecting a fresh checkpoint root.

## Pre-launch review

Before submitting a large run:

1. Print the plan with the intended immutable version and selection options.
2. Check the resolved role plan and topology, especially engine count and DP/TP/PP/EP.
   For `fully_async`, confirm the train batch and policy mini-batch are equal.
3. Confirm the pinned MarinSkyRL commit supports the chosen strategy, CUDA stack, model, and vLLM
   geometry. If the commit changed, use a fresh RL artifact version unless its checkpoints are
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
