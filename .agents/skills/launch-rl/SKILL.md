---
name: launch-rl
description: Define, validate, submit, or restart a Marin SkyRL experiment through its artifact main. Use for RL launch files, SkyRLRolePlan or SkyRLTopology changes, MarinSkyRL configuration, RL smoke tests, and requests to launch or relaunch RL on Iris.
---

# Launch RL

Use the experiment's artifact graph as the launch interface. For the configuration model, role
arithmetic, and ownership boundaries, read [RL launching](../../../docs/references/rl-launching.md).

## Define the experiment

- Define a Click `main` that constructs and returns the requested terminal `ArtifactStep` handles.
- Decorate it with `@rl_build_options`. Do not add another submission path to an experiment.
- Put artifact identity and a fully explicit `SkyRLRolePlan` in `SkyRLSpec`. Put cluster placement
  in `IrisSkyRLExecution`.
- Render role geometry from the plan. Do not repeat DP, TP, PP, EP, engine count, placement, or
  batch sizes as independent literals or Hydra overrides.
- Express models and data as artifact dependencies. Pin tokenizer and source revisions; do not depend
  on an ambient checkpoint path or whatever happens to be on a worker.
- Keep the generated `SkyRLLaunchConfig` YAML as the only Marin-to-MarinSkyRL boundary. Extend that
  config instead of adding request envelopes or internal per-field command-line flags.

## Validate before launch

Run the main without `--run` first, using an immutable calendar version for a real experiment:

```bash
uv run python -m experiments.<module> --version YYYY.MM.DD <selection-options>
```

Treat a preflight failure as a recipe error. Fix the source config or role plan instead of weakening
the validator. Check the printed runtime commit, physical allocation, role geometry, entrypoint,
batch sizes, artifact identities, output paths, and terminal stage.

Use a fresh RL artifact version whenever `SkyRLRuntime.commit` changes. The temporary checkpoint
root follows the artifact name and version, so reusing a version can make `resume_mode=latest` load
private Torch or distributed state written by the old runtime. Reuse an RL version across a repin
only when checkpoint compatibility has been established explicitly or resume points at a fresh
checkpoint root.

Before a large launch, run a smoke through the same strategy, artifact handoffs, and role geometry.
Confirm the runtime contains the required backend fixes and that host resources, credentials,
context budget, checkpoint/export policy, and W&B identity are explicit.

## Submit and observe

Launch through the same main:

```bash
uv run python -m experiments.<module> --version YYYY.MM.DD <selection-options> --run
```

Choose the terminal stage explicitly when a main offers one. For example, use `--stage rl` when the
request is only to train; a default evaluation stage may add downstream GPU work.

For a live submission or inspection, use `use-iris`. Record the coordinator ID, child job ID,
artifact version, runtime commit, and topology in the task's durable surface. Verify all expected
tasks join, the configured model-loading path is active, optimizer steps advance, and terminal
checkpoint/export metadata exists before calling the smoke successful.

## Diagnose preflight gaps

If a failure was deterministic from the artifact inputs, add a construction-time check with a
behavior-focused regression test. Diagnose service outages, hardware faults, and data-dependent
failures operationally. Cancel or resubmit only with current-thread authorization; a retry uses a
fresh artifact version when the runtime commit changed.
