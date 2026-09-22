---
name: launch-rl
description: Define, validate, submit, or restart a Marin SkyRL experiment through its artifact main. Use for RL launch files, SkyRLRolePlan or SkyRLTopology changes, MarinSkyRL configuration, RL smoke tests, and requests to launch or relaunch RL on Iris.
---

# Launch RL

Make the experiment's artifact graph the only launch interface. Constructing the graph must validate
deterministic SkyRL launch geometry and config before Iris allocates GPUs.

## Define the experiment

- Define a Click `main` that constructs and returns the requested terminal `ArtifactStep` handles.
- Decorate it with `@rl_build_options`. Do not add a second hand-written `iris job run` or invoke
  `marinskyrl iris launch` from an experiment.
- Put policy and rollout identity in `SkyRLSpec`, including a fully explicit `SkyRLRolePlan`.
  Placement-only details belong in `IrisSkyRLExecution`.
- Render every role-plan setting that has a YAML counterpart from the plan. Do not repeat DP, TP,
  PP, EP, engine count, placement, or batch sizes as unrelated literals or Hydra overrides.
- Express models and data as artifact dependencies. Pin tokenizer and source revisions; do not depend
  on an ambient checkpoint path or whatever happens to be on a worker.

The rollout arithmetic is:

```text
engine slice = TP * PP
engine GPUs = num engines * engine slice * DP
separate roles = policy GPUs + engine GPUs
colocated roles require policy GPUs == engine GPUs
```

For separate roles, each engine's `TP * PP * DP` must fit on one node. For colocated roles, the
`TP * PP` slice must divide a policy node and DP replicas occupy separate slices. EP must divide
`TP * DP`. The result must equal `num_nodes * gpus_per_node`; unused requested GPUs usually mean DP
or the engine count is wrong.

## Validate before launch

Run the main without `--run` first, using an immutable calendar version for a real experiment:

```bash
uv run python -m experiments.<module> --version YYYY.MM.DD <selection-options>
```

Graph construction checks the topology arithmetic, batch divisibility, runtime-profile strategy, and
agreement between the rendered config and role plan. It also requires `train_batch_size ==
policy_mini_batch_size` for the `fully_async` entry point. Treat a preflight failure as a recipe
error; fix the single source of truth instead of weakening the validator.

Before a large launch, also check:

- the runtime commit contains the required backend and CUDA behavior;
- resume mode, checkpoint retention, terminal export, and Hub upload behavior are explicit;
- context and generation budgets match the chat template and task;
- task CPU, memory, disk, concurrency, and required credentials are sufficient;
- W&B entity/project and model/tokenizer identities are valid;
- a smoke uses the same role geometry and runtime path as the intended full run.

## Submit and observe

Launch through the same main:

```bash
uv run python -m experiments.<module> --version YYYY.MM.DD <selection-options> --run
```

Choose the terminal stage explicitly when a main offers one. For example, use `--stage rl` when the
request is only to train; a default evaluation stage may add downstream GPU work.

Outside Iris, `--run` submits a CPU coordinator after validation. Inside Iris, it executes the
artifact graph. CoreWeave coordinators must submit through the Marin hub with an explicit target
cluster and a deadline sized for queue wait plus runtime. Set required `DAYTONA_API_KEY`, `HF_TOKEN`,
and `WANDB_API_KEY` values on the submit host; the coordinator forwards values that are present.

For a live submission or inspection, also use `use-iris` and read `lib/iris/OPS.md`. Record the
coordinator job ID, child SkyRL job ID, artifact version, runtime commit, and resolved topology in
the task's existing durable surface. Start with read-only inspection. Never restart an Iris cluster,
cancel a run, or resubmit a failed run without the authority required by the selected operations
workflow.

## Diagnose preflight gaps

If a failure occurs only after allocation, decide whether a deterministic input could have exposed
it. Add that check at the artifact generator boundary with a behavior-focused regression test. Good
preflight candidates include contradictory topology/config values, missing pinned dependencies,
invalid job names, impossible worker resources, missing identity fields, and unsupported
runtime/strategy combinations. Runtime service outages, hardware faults, and data-dependent model
failures belong in operational diagnosis, not speculative launch validation.
