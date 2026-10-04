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
  compose skyrl recipe against ppo_base_config
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
arguments. The launch host resolves Hydra defaults; the resolved document is the input to
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

## Typed recipes

`marin.skyrl_recipe` contains Schema classes generated from MarinSkyRL's YAML and the immutable,
sparse Recipe API. Only explicitly authored fields appear in `recipe.to_skyrl()`. Hydra supplies
runtime defaults on the launch host.

Construct a `SkyRLRecipe` from typed sections and pass it as `SkyRLSpec(recipe=..., hardware=...)`.
`SkyRLHardware` declares the GPU variant and GPUs per node. The launcher derives physical node
count, runtime profile and ingress mode from the composed recipe.

```python
from marin.skyrl_recipe import Algorithm, ContextBudget, Generator, Placement, SkyRLRecipe, Trainer
from marin.rl.skyrl import SkyRLHardware

recipe = SkyRLRecipe(
    context_budget=ContextBudget(request_window_tokens=2048, max_new_tokens_per_turn=1024, max_turns=1),
    trainer=Trainer(
        strategy="megatron",
        placement=Placement(colocate_all=False, colocate_policy_ref=True,
                            policy_num_nodes=1, policy_num_gpus_per_node=8,
                            ref_num_nodes=1, ref_num_gpus_per_node=8),
        train_batch_size=64, policy_mini_batch_size=32, micro_train_batch_size_per_gpu=4,
        algorithm=Algorithm(use_kl_loss=True),
    ),
    generator=Generator(backend="vllm", run_engines_locally=True, num_inference_engines=8,
                        inference_engine_tensor_parallel_size=1,
                        inference_engine_pipeline_parallel_size=1,
                        inference_engine_data_parallel_size=1,
                        inference_engine_expert_parallel_size=1, n_samples_per_prompt=4),
)
hardware = SkyRLHardware(gpu_variant="H100", gpus_per_node=8)
```

Marin requires explicit placement, inference geometry, batch sizes, samples per prompt,
`trainer.algorithm.use_kl_loss`, strategy, backend and local-engine mode. It checks `model_fields_set`
for these declarations and requires policy GPU width to match the hardware. Read authored policy
through typed attributes, such as `recipe.trainer.placement.policy_num_nodes`; effective values come
from the launcher's prepare mode. MarinSkyRL owns resource arithmetic and trainer batch validation.

Use `SkyRLRecipe.combine(base=BASE, preset=PRESET, policy=POLICY, arm=ARM)` for independent named
parts. Different values for one leaf, or a parent and its descendant, raise with both part names.
Lists are whole values. Use `recipe.merge(PATCH)` for an intentional override, such as optimizer
or model parallelism tuning. Pass computed checkpoint schedules as another named part so a setting
that contradicts the schedule raises.

`recipe.with_settings(["context_budget.max_turns=4"])` parses values using schema types, merges them
onto the complete document, and validates it. Scalar settings use typed strings; arrays and mappings
use JSON. Unknown fields, wrong types and owned fields identify their paths during construction.
A derived token limit names `context_budget` as its owner. Run seed, retention, model paths and
other launch-owned values belong in the launch envelope. Harbor and engine kwargs remain intentional
open mappings with the shared ownership checks.

## Pin and copied schema

`config/update-external.py` copies the flat schema at the commit in the MarinSkyRL external lock
into `marin.skyrl_recipe`. Its adjacent provenance records that commit, per-file hashes and one
content hash. Resolve the external lock before running the updater. `--check` compares local Git
objects at the pinned commit with the exact copied inventory and content, without fetching or
writing. Use `--schema-source /path/to/MarinSkyRL` when the uv Git cache has no such checkout.

Marin format, license and type-file checks exclude copied source; callers remain type checked.
The commit-and-hash check owns the copy. Bot updates admit only the copied inventory and provenance
alongside the project lock and generated pins. The required unit-test launch-document check builds
every supported producer configuration and loads each document with the installed pin. A breaking
pin migration is a human PR that changes the pin, copy and producers together.

## Artifact and runtime boundaries

Identity-bearing inputs belong in `SkyRLSpec`: the pinned MarinSkyRL runtime, model and tokenizer,
data artifacts, hardware, seed, retention, and the typed recipe. Cluster routing,
host resources, priority, and retry policy belong in `IrisSkyRLExecution`; changing placement must
not fork artifact identity.

Use artifact dependencies for model and data handoffs. A raw checkpoint path hides provenance and
can point at an incomplete export. Keep resume behavior and terminal export explicit. Use Marin's
temporary-storage helper for checkpoints and trajectories so paths have a bounded lifetime and do
not repeat a source bucket prefix.

The temporary checkpoint root is derived from the RL artifact name and version, not only from the
runtime commit. When `MARIN_SKYRL.commit` changes, use a fresh RL artifact version. Reusing the old
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
