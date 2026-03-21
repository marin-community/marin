# TUNIX_RL_MASTER.md

This document is the main orientation guide for anyone trying to understand the
RL stack in this checkout of `tunix`, especially:

- how the Tunix RL codebase is structured,
- how meshes and role assignment work,
- what "collocated" and "disaggregated" mean in practice,
- how vLLM rollout is integrated,
- what we tried for multi-host async RL,
- why those attempts failed on multi-host TPU,
- what is safe to do now,
- and what realistic next steps exist.

This document is intentionally detailed. It is meant to save the next agent or
new contributor from having to rediscover the same architecture and failure
modes from scratch.

## 1. Scope and Snapshot

This note describes the local repo at:

- Repo root: `/Users/ahmed/code/tunix`
- Base upstream snapshot from local notes: commit `b3de6ec3fdb9831cb7e5f8f1f6ff7f6da87dc819`
- Current branch in local notes: `main`

Important context for this workspace:

- There are local uncommitted docs and scripts related to async RL experiments.
- Some of those notes describe code changes that are not present in the current
  checked-in Python implementation.
- This document calls that out explicitly where it matters.

## 2. Recommended Reading Order

If you are new to this repo, read in this order:

1. `README.md`
2. `CODEX_README.md`
3. `tunix.md`
4. `docs/performance.md`
5. `docs/rollout.md`
6. `tunix/cli/grpo_main.py`
7. `tunix/rl/rl_cluster.py`
8. `tunix/rl/rl_learner.py`
9. `tunix/rl/grpo/grpo_learner.py`
10. `tunix/generate/vllm_sampler.py`
11. `tunix/rl/reshard.py`
12. `TUNIX_ASYNC_RL.md`
13. `pathways_async_rl.md`
14. `disaggregate_claude_v3.md`

The public docs are good for concepts, but the code is the source of truth.

## 3. What Tunix Is

Tunix is a JAX/Flax NNX post-training library for LLMs. In this repo, the RL
center of gravity is:

- `tunix/rl/rl_cluster.py`
- `tunix/rl/rl_learner.py`
- `tunix/rl/grpo/grpo_learner.py`
- `tunix/rl/ppo/ppo_learner.py`
- `tunix/rl/rollout/*`
- `tunix/generate/*`

The mental model is:

- Tunix is a library-first training system.
- The CLI is a thin wrapper that builds config, model, mesh, and learner
  objects.
- Distribution is expressed with JAX meshes and shardings.
- RL is composed from reusable roles rather than one monolithic trainer object.

## 4. Repo Areas That Matter for RL

### 4.1 Main entrypoints

- `tunix/cli/grpo_main.py`
- `tunix/cli/peft_main.py`

For RL in this repo, `python -m tunix.cli.grpo_main ...` is the main CLI path.

### 4.2 Config system

- `tunix/cli/base_config.yaml`
- `tunix/cli/config.py`

The CLI merges YAML + command-line overrides into a structured config, then
constructs:

- model config,
- tokenizer config,
- mesh config,
- training config,
- rollout config,
- algorithm config.

### 4.3 RL control plane

- `tunix/rl/rl_cluster.py`
- `tunix/rl/rl_learner.py`
- `tunix/rl/grpo/grpo_learner.py`

These files decide:

- which models exist,
- which mesh each role uses,
- whether weights are shared or copied,
- whether async rollout is enabled,
- when rollout, logprob computation, training, and weight sync happen.

### 4.4 Rollout engines

- `tunix/rl/rollout/base_rollout.py`
- `tunix/rl/rollout/vanilla_rollout.py`
- `tunix/rl/rollout/vllm_rollout.py`
- `tunix/rl/rollout/sglang_jax_rollout.py`
- `tunix/generate/vllm_sampler.py`

The rollout engine is pluggable. For the work discussed here, `vllm` is the
important one.

### 4.5 Weight movement and resharding

- `tunix/generate/utils.py`
- `tunix/rl/reshard.py`

This is the critical area for disaggregated multi-host behavior.

### 4.6 Inference-side auxiliary models

- `tunix/rl/inference/inference_worker.py`

The inference worker hosts non-actor models such as:

- reference,
- critic,
- reward.

## 5. RL Stack: High-Level Architecture

In a standard GRPO setup, Tunix has these conceptual components:

1. Actor trainer
2. Rollout engine
3. Reference model
4. Critic model, optionally
5. Reward function or reward model
6. Weight sync path between actor and rollout

The rough loop is:

1. Generate responses from prompts using the current rollout policy.
2. Score the responses.
3. Compute advantages.
4. Compute old policy logprobs / reference logprobs if needed.
5. Train the actor.
6. Sync actor weights back into the rollout engine.
7. Repeat.

### 5.1 Where this loop lives

- The CLI builds an `RLCluster` in `tunix/cli/grpo_main.py`.
- The GRPO learner logic lives in `tunix/rl/grpo/grpo_learner.py`.
- The generic RL control flow lives in `tunix/rl/rl_learner.py`.

### 5.2 GRPO specifics

`GRPOConfig` in `tunix/rl/grpo/grpo_learner.py` defines the main algorithm
knobs, including:

- `num_generations`
- `num_iterations`
- `beta`
- `epsilon`
- `epsilon_high`
- `advantage_estimator`
- `loss_algo`

This is standard GRPO-style group sampling with multiple completions per prompt.

## 6. How the CLI Builds an RL Run

The main object in `tunix/cli/grpo_main.py` is `GrpoPipeline`.

Important methods:

- `create_rollout_config()`
- `create_role_to_mesh()`
- `create_cluster_config()`
- `create_rl_cluster()`

### 6.1 Mesh creation

The CLI constructs meshes through `create_mesh()` in `tunix/cli/config.py`.

Today, in this checkout, the relevant behavior is:

- it reads `mesh.shape`,
- it reads `mesh.axis_names`,
- it validates them,
- then it calls `jax.make_mesh(...)`.

The checked-in CLI does **not** currently parse custom `device_ids` or
`device_start_idx` / `device_end_idx` fields from mesh config.

That point matters later, because some local async launcher scripts pass custom
device assignments that the current CLI parser does not actually consume.

### 6.2 Default mesh config example

From `tunix/cli/base_config.yaml`:

```yaml
model_config:
  mesh:
    shape: "(2,2)"
    axis_names: "('fsdp','tp')"
```

For RL, the CLI can assign different meshes to:

- actor
- reference
- rollout

through:

- `model_config`
- `actor_model_config`
- `reference_model_config`
- `rollout_model_config`

### 6.3 Role-to-mesh mapping

In `tunix/cli/grpo_main.py`, `create_role_to_mesh()` returns:

```python
{
    Role.ACTOR: actor_mesh,
    Role.REFERENCE: reference_mesh,
    Role.ROLLOUT: rollout_mesh,
}
```

That map is the core switch that determines whether the run behaves like
collocated or disaggregated execution.

### 6.4 End-to-end CLI construction flow

When you run:

```bash
python -m tunix.cli.grpo_main tunix/cli/base_config.yaml ...
```

the high-level object construction flow is:

1. `HyperParameters` loads YAML and command-line overrides.
2. `GrpoPipeline.create_role_to_mesh()` constructs meshes.
3. `GrpoPipeline.create_rollout_config()` builds a `RolloutConfig`.
4. `GrpoPipeline.create_cluster_config()` packages all cluster-level settings.
5. `GrpoPipeline.create_rl_cluster()` builds an `RLCluster`.
6. The pipeline creates the GRPO learner and dataset.
7. Training begins.

This is important because many async failures happen during step 5, before the
actual training loop has even started.

## 7. RLCluster: The Real Center of Gravity

`tunix/rl/rl_cluster.py` is the most important RL file in the repo.

It is responsible for:

- loading actor / rollout / reference / critic / reward,
- establishing which roles can share model state,
- constructing the rollout engine,
- constructing the inference worker,
- constructing the trainer(s),
- providing APIs to rollout, score, train, and sync.

### 7.1 Backbone sharing

The first major decision in `RLCluster.__init__` is backbone sharing.

`_init_backbone_sharing_map()` checks whether roles share the same mesh. In the
most important case:

- if `Role.ACTOR` mesh equals `Role.ROLLOUT` mesh,
- then actor and rollout are marked as sharing backbone.

That is the implementation-level basis for collocated mode.

### 7.2 Model loading behavior

`_load_model()` can reshard an existing NNX model to a target mesh by:

1. splitting the model state,
2. building target shardings,
3. calling `reshard.reshard_pytree(...)`,
4. merging the new state back into the graph.

This is one place where cross-mesh movement can happen.

### 7.3 Rollout initialization behavior

In `_init_cluster()`:

- `vanilla` rollout loads a separate rollout model if needed.
- `vllm` rollout does **not** load a fully independent rollout model from disk.
- for non-vanilla rollout engines, Tunix initially points `self.rollout_actor`
  at `self.train_actor` unless meshes are shared.

That means the first vLLM initialization path often starts from actor state and
then transfers that state into the vLLM-side model representation.

This is one of the most important implementation details for understanding the
multi-host failure.

### 7.4 What `_init_cluster()` actually constructs

`_init_cluster()` has three major phases:

1. Initialize rollout.
2. Initialize inference worker.
3. Initialize trainer(s).

This order matters:

- rollout is constructed before the main actor training loop runs,
- so initial actor-to-rollout sync happens early,
- and if rollout initialization itself requires cross-mesh movement, the run can
  fail before the first training step.

That is exactly what happened in the multi-host async attempt with vLLM.

## 8. Inference Worker

The inference worker is intentionally simple.

`tunix/rl/inference/inference_worker.py` hosts:

- reference model
- critic model
- reward model

It provides:

- `get_rewards(...)`
- `get_ref_per_token_logps(...)`
- `get_values(...)`

So the actor trainer is not doing every forward pass itself. Tunix decomposes
the RL stack into:

- rollout,
- training,
- inference-side computations.

## 9. Rollout Engines

Tunix has three rollout modes relevant here:

- `vanilla`
- `vllm`
- `sglang_jax`

### 9.1 `vanilla`

This is the simplest Tunix-native generation path. It is conceptually easiest
for understanding control flow.

### 9.2 `vllm`

This is the important one for async RL work in this repo.

The vLLM integration is built from:

- `tunix/rl/rollout/vllm_rollout.py`
- `tunix/generate/vllm_sampler.py`

High-level flow:

1. Build a `VllmSampler`.
2. Initialize the vLLM engine.
3. Take the actor model state.
4. Transfer / map / reshard that state into the vLLM-owned model state.

### 9.3 vLLM config surface

`RolloutConfig` in `tunix/rl/rollout/base_rollout.py` contains many vLLM knobs,
including:

- `rollout_vllm_model_version`
- `rollout_vllm_hbm_utilization`
- `rollout_vllm_init_with_random_weights`
- `rollout_vllm_tpu_backend_type`
- `rollout_vllm_server_mode`
- `rollout_vllm_async_scheduling`
- `rollout_vllm_additional_config`
- `rollout_mapping_config`
- `tensor_parallel_size`
- `data_parallel_size`

### 9.4 Important operational note

The docs in `docs/rollout.md` explicitly say:

- Tunix + vLLM is WIP
- CLI support for vLLM rollout is WIP

That does not mean it never works. It means advanced configurations should be
treated as experimental unless the exact path has been validated.

## 10. The Weight Sync Path

This is the most important technical section for understanding the async
experiments.

### 10.1 What weight sync means here

In RL, rollout needs current policy weights.

If rollout and actor share the same model object on the same mesh, sync is
effectively free.

If rollout is a separate engine or separate mesh, Tunix must explicitly move
weights from actor state into rollout state.

### 10.2 The actor-to-rollout sync API

`RLCluster.sync_weights()`:

1. selects actor params,
2. chooses LoRA-only or full-param sync depending on model setup,
3. calls `self.rollout.update_params(...)`,
4. tracks sync metrics.

### 10.3 vLLM update path

For vLLM rollout, this goes through:

1. `VllmRollout.update_params(...)`
2. `VllmSampler.update_params(...)`
3. `transfer_state_with_mappings(...)`
4. `reshard.reshard_pytree(...)`

This is the concrete path where mesh-to-mesh movement happens.

### 10.4 The exact critical chain

The concrete initialization/sync path is:

1. `tunix/rl/rollout/vllm_rollout.py`
2. `self._sampler.load_checkpoint(state)`
3. `tunix/generate/vllm_sampler.py:load_checkpoint`
4. `tunix/generate/vllm_sampler.py:update_params`
5. `tunix/generate/utils.py:transfer_state_with_mappings`
6. `tunix/rl/reshard.py:reshard_pytree`
7. reshard implementation
8. usually `jax.device_put(...)` in OSS mode

### 10.5 Why mappings exist

vLLM parameter names and structure do not necessarily match Tunix trainer
parameter naming exactly.

So Tunix:

- maps keys,
- optionally transposes tensors,
- aligns shapes,
- casts dtypes,
- then reshares to target shardings.

That extra mapping layer is important, because failures can happen even before
or after the sharding step depending on backend/model combination.

### 10.6 Weight sync metrics

`RLCluster.sync_weights()` tracks:

- total transfers,
- successful transfers,
- failed transfers.

This is useful for future debugging. If a run starts but periodically dies
during training rather than at initialization, these counters are the first
place to check to determine whether the problem is:

- rollout generation,
- trainer update,
- or actor-to-rollout sync.

## 11. Resharding: The Crucial Implementation Detail

`tunix/rl/reshard.py` is where Tunix chooses the primitive used to move arrays
between shardings / meshes.

### 11.1 Two paths in the code

`reshard_pytree(...)` tries reshard backends in order:

1. Pathways utils experimental reshard
2. fallback to `jax.device_put`

In OSS mode, unless Pathways proxy support is really available, this usually
means the code falls back to `jax.device_put(...)`.

### 11.2 Why that matters

On a single host, `jax.device_put(...)` across local-device shardings can be
fine.

On multi-host, using Python-side `device_put` to move a large pytree between
disjoint meshes on different physical hosts is exactly where the local async
experiments broke.

## 12. Collocated vs Disaggregated: The Intended Design

The public performance docs describe two execution patterns:

- collocated
- disaggregated

The examples in `docs/performance.md` are accurate at the conceptual level.

### 12.1 Collocated mode

All roles share the same mesh.

Example:

```python
devices = jax.devices()
devices_mesh = np.array(devices).reshape(len(devices), 1)
mesh = Mesh(devices_mesh, axis_names=("fsdp", "tp"))

role_to_mesh = {
    Role.ACTOR: mesh,
    Role.REFERENCE: mesh,
    Role.ROLLOUT: mesh,
}
```

Implementation consequences:

- actor and rollout can share the same backbone object,
- no cross-mesh rollout weight sync is required,
- the workflow runs sequentially over shared hardware.

Operationally, this is the safest option on multi-host TPU in this repo.

### 12.2 Disaggregated mode

Different roles get different meshes.

Example from docs:

```python
devices = jax.devices()
split = int(len(devices) / 2)
rollout_devices = np.array(devices[:split]).reshape(split, 1)
train_devices = np.array(devices[split:]).reshape(split, 1)
rollout_mesh = Mesh(rollout_devices, axis_names=("fsdp", "tp"))
train_mesh = Mesh(train_devices, axis_names=("fsdp", "tp"))

role_to_mesh = {
    Role.ACTOR: train_mesh,
    Role.REFERENCE: train_mesh,
    Role.ROLLOUT: rollout_mesh,
}
```

Implementation consequences:

- actor and rollout no longer share backbone,
- weight sync becomes explicit,
- async rollout becomes eligible.

### 12.3 How Tunix decides async rollout

In `tunix/rl/rl_learner.py`, async rollout is enabled when:

```python
actor_mesh != rollout_mesh
```

That is intentionally simple. But it also means:

- Tunix does not distinguish "safe disaggregated" from "unsafe disaggregated"
  by topology.
- It does not validate whether the mesh split is actually valid on a multi-host
  TPU topology.

This simplicity is fine on single-host setups. It becomes risky on multi-host.

## 13. What Async Actually Means in This Codebase

In `tunix/rl/rl_learner.py`, async rollout is not "independent distributed
programs per role." It is closer to:

- a single overall JAX-driven program,
- with overlapping data preparation / rollout / training behavior,
- under the assumption that different role meshes can be driven safely in the
  same runtime context.

`_run_all_micro_batch_steps(...)` uses a thread pool and queueing to overlap
stages. This is useful, but it is not the same thing as Pathways-style central
dispatch across independently owned device groups.

That distinction is the heart of the multi-host problem.

## 14. TPU Topology and Why It Matters

For the async experiments, physical topology is not a side issue. It is the
core issue.

### 14.1 v5p examples

Useful mental table:

| TPU type | Chips | Hosts | Chips per host |
| --- | ---: | ---: | ---: |
| v5p-8 | 4 | 1 | 4 |
| v5p-16 | 8 | 2 | 4 |
| v5p-32 | 16 | 4 | 4 |

### 14.2 JAX multi-host model

In normal JAX multi-controller SPMD:

- each host runs the same Python program,
- `jax.devices()` returns global devices,
- each host only directly owns its local subset,
- collectives across hosts are planned by XLA/PJRT.

The important constraint is:

- one host does not get to opt out of a `jit` just because its local role is
  logically "inactive."

That is where people often over-import an MPI or Ray mental model that does not
apply here.

## 15. Why Multi-Host Full-Mesh Training Works

A normal working multi-host training job looks like this:

```text
Full mesh over all devices
Host 0 contributes its local chips
Host 1 contributes its local chips
Both hosts enter the same compiled program
XLA inserts collectives over ICI
```

This is the standard, supported, well-tested path.

So the problem is **not**:

- "multi-host TPU is broken"

The problem is:

- "trying to run disjoint role meshes inside one JAX SPMD job is not the same
  thing as a normal full-mesh multi-host job"

## 16. Why Multi-Host Disaggregated Broke for Us

This section summarizes the local design docs:

- `TUNIX_ASYNC_RL.md`
- `pathways_async_rl.md`
- `disaggregate_claude_v3.md`

### 16.1 The goal

We wanted a true async RL setup for MATH500 on `v5p-16`:

- rollout on one set of chips,
- actor training on another set of chips,
- overlap generation with training,
- sync weights between them.

### 16.2 The first obvious split: host-separated

Natural first split on `v5p-16`:

- host 0 devices for rollout
- host 1 devices for actor

Example:

```text
Rollout mesh: [0,1,2,3]
Actor mesh:   [4,5,6,7]
```

This is conceptually clean, but it collides with JAX SPMD reality.

### 16.3 The host-separated problem

This design assumes:

- actor training can "really run on host 1 only"
- rollout can "really run on host 0 only"

But under normal JAX multi-controller SPMD:

- both hosts still run the same compiled programs,
- both hosts still need a valid participation story,
- and cross-mesh transfers are not magically upgraded to runtime-managed ICI
  operations just because the two meshes are part of one Python process tree.

### 16.4 The concrete failure we saw

The local notes captured a crash during initial vLLM weight sync:

```text
jax._src.dispatch._device_put_sharding_impl
jax._src.api.device_put
tunix/rl/reshard.py:reshard_pytree
tunix/generate/utils.py:transfer_state_with_mappings
tunix/generate/vllm_sampler.py:update_params
tunix/generate/vllm_sampler.py:load_checkpoint
tunix/rl/rollout/vllm_rollout.py:__init__
tunix/rl/rl_cluster.py:_init_cluster
```

This is exactly consistent with the code.

The critical event is:

- actor state exists on the actor mesh,
- vLLM target state exists on the rollout mesh,
- Tunix attempts to map and reshard the actor state into the rollout state,
- the OSS reshard path uses `jax.device_put`,
- that cross-mesh, cross-host movement crashes.

### 16.5 Why this is not just "a vLLM bug"

vLLM is involved because it is the rollout backend. But the deeper problem is
more general:

- the current RL stack assumes one JAX/SPMD execution context can safely manage
  independently assigned role meshes,
- and the fallback reshard primitive is not a robust solution for full-model
  cross-host transfer between disjoint meshes.

## 17. The Interleaved-Mesh Attempt

The next idea was to keep every host "participating" by giving each role some
devices from each host.

Example:

```text
Actor mesh:   [0,1,4,5]
Rollout mesh: [2,3,6,7]
```

The local async launcher script follows this style. It sets:

- `ROLLOUT_DEVICE_IDS=[0,1,4,5]`
- `ACTOR_DEVICE_IDS=[2,3,6,7]`

in `scripts/launch_rloo_llama3_1_8b_math500_async_tpu.sh`.

### 17.1 Why this was attractive

This tries to avoid the "one host has no local devices in a role mesh" problem.

### 17.2 Why it is still bad

This kind of split is not a normal contiguous physical TPU rectangle.

The local notes correctly identify the likely result:

- XLA / runtime errors,
- invalid communication planning,
- internal TPU runtime failures,
- or hangs / crashes around collectives and resharding.

Even when the program gets farther than the host-separated case, it is still
working against the physical topology rather than with it.

## 18. Why the Transfer-Mesh Proposal Does Not Solve the Core Problem

`disaggregate_claude_v3.md` proposes:

- actor mesh,
- rollout mesh,
- plus a third full transfer mesh used only for sync.

This is a clever idea, and it is worth understanding. But for this repo and
this execution model, it does not fix the root issue.

### 18.1 What the proposal tries to fix

It tries to replace:

- raw cross-host `device_put` between disjoint meshes

with:

- a compiled full-mesh collective-based transfer.

That could help with weight movement itself.

### 18.2 Why it is still insufficient

The fundamental issue is not only "how to copy weights."

It is also:

- who is driving which mesh,
- under what distributed runtime contract,
- while training and rollout are supposedly overlapping.

Normal JAX SPMD still expects hosts to participate together in compiled calls.
So even a better copy primitive does not create a true Pathways-style dispatch
model.

### 18.3 Bottom line

Transfer mesh might be an interesting research experiment. It is not a credible
near-term fix for this repo’s current multi-host async design.

## 19. Pathways in This Repo: Real, Partial, and Misleading Signals

One thing that can confuse a newcomer is that "Pathways" shows up in several
places in the repo.

### 19.1 Where Pathways appears

- `README.md` mentions multi-host distributed training with Pathways.
- `tunix/cli/grpo_main.py` has a `--pathways_bns` flag.
- `tunix/cli/grpo_main.py` can set JAX backend to `"pathways"`.
- `tunix/rl/reshard.py` has a Pathways utils reshard path.
- some rollout code comments mention single-controller behavior.

### 19.2 What this does **not** mean

It does **not** mean the ordinary OSS async RL path is already running on a
Pathways-like centralized runtime.

Without an actual Pathways backend / proxy environment, the normal behavior in
this checkout is still the ordinary JAX path, and resharding falls back to
`jax.device_put`.

### 19.3 Why this matters

A newcomer could easily think:

- "README says Pathways"
- therefore
- "disaggregated multi-host async should work out of the box"

That inference would be wrong for this local workflow.

## 20. Current Local Experiment Artifacts vs Checked-In Behavior

This is a critical section for future agents.

### 20.1 Local async docs exist

This checkout contains local design docs:

- `TUNIX_ASYNC_RL.md`
- `pathways_async_rl.md`
- `disaggregate_claude_v3.md`

These are useful and mostly technically sound on the main questions.

### 20.2 Local async launch script exists

This checkout also contains:

- `scripts/launch_rloo_llama3_1_8b_math500_async_tpu.sh`

This script passes custom mesh device assignments such as:

- `model_config.mesh.device_ids=...`
- `actor_model_config.mesh.device_ids=...`
- `rollout_model_config.mesh.device_ids=...`

### 20.3 But the checked-in CLI does not parse those fields

The current `tunix/cli/config.py` in this checkout still constructs meshes from:

- `mesh.shape`
- `mesh.axis_names`

using:

- `jax.make_mesh(...)`

It does **not** currently consume `mesh.device_ids`.

So the async launcher script reflects a local experimental intent, not a fully
wired and validated CLI feature in the current Python implementation.

This discrepancy should be assumed until proven otherwise.

## 21. Concrete Failure Analysis: Where Mesh Sync Failed

This is the most concrete answer to "what exactly broke?"

### 21.1 The source object

In non-vanilla rollout mode, `RLCluster` initially sets:

```python
self.rollout_actor = self.train_actor
```

That means the initial rollout state source is actor state.

### 21.2 The sink object

`VllmRollout.__init__` constructs a vLLM sampler and immediately does:

```python
state = nnx.state(model)
self._sampler.load_checkpoint(state)
```

So the first sync happens during rollout construction, not later after a clean
steady-state loop has already been established.

### 21.3 The transformation layer

`VllmSampler.update_params(...)` calls:

- key mapping,
- transpose hooks,
- shape alignment,
- dtype alignment,
- then resharding into the vLLM-side destination state.

### 21.4 The failing primitive

`transfer_state_with_mappings(...)` eventually calls:

```python
reshard.reshard_pytree(tgt_flat_dict, sharding_dict)
```

and in OSS mode that usually means:

```python
jax.device_put(...)
```

### 21.5 Why this is fragile on multi-host disjoint meshes

At that point Tunix is trying to do all of the following at once:

- transfer a large model state,
- between distinct meshes,
- that may live on different hosts,
- from within a normal JAX SPMD program,
- without a role-isolated runtime,
- and without a dedicated robust cross-mesh transfer protocol.

That is exactly the wrong place to rely on a raw fallback `device_put`.

### 21.6 Practical interpretation

The crash is not surprising. It is the expected result of the current design
meeting a multi-host disaggregated topology it was not truly built to support.

## 22. What Works Reliably Today

### 22.1 Collocated full-mesh RL

On multi-host TPU, the safest recommendation is:

- use the full mesh,
- keep actor / reference / rollout colocated,
- rely on sequential execution across the shared hardware.

For `v5p-16`, that means a mesh like `(2,4)` across all devices.

This is also the mode that best matches:

- the public docs,
- normal JAX multi-host expectations,
- and the actor/rollout sharing logic in `RLCluster`.

### 22.2 Single-host disaggregated experiments

Disaggregated mode is much more plausible on single-host hardware, because:

- there is no cross-host addressability problem,
- there is no inter-host collective planning issue,
- and `device_put`-style movement is staying within one physical host context.

That does not make every configuration automatically safe, but it removes the
main failure mode described in this document.

### 22.3 vLLM with conservative settings

For colocated mode:

- use moderate `rollout_vllm_hbm_utilization`
- coordinate rollout HBM with trainer/inference needs

For disaggregated mode, docs say you can push HBM utilization higher because
rollout owns dedicated hardware. That is true in principle, but only after the
topology/runtime story itself is valid.

## 23. What Does Not Work in This Checkout

These should be treated as unsupported or effectively broken until proven
otherwise.

### 23.1 Multi-host disaggregated RL on one ordinary JAX SPMD job

Specifically:

- actor on one host subset
- rollout on another host subset
- in the same distributed JAX process group
- with in-memory weight sync between them

This is the exact failure mode we have already hit.

### 23.2 Assuming local async launcher script equals supported CLI feature

It does not.

The launcher script and the Python config implementation are currently out of
sync on device-id mesh assignment behavior.

### 23.3 Assuming Pathways is implicitly active

It is not, unless the run is actually configured to use the Pathways backend
and environment.

## 24. Practical Recommendations for the Next Agent

If your goal is to get MATH500 or similar RL training running, do this:

1. Start from the sync collocated script:
   `scripts/launch_rloo_llama3_1_8b_math500_tpu.sh`
2. Use full-mesh collocated execution on the TPU slice.
3. Validate the run end-to-end before changing topology.
4. Treat async vLLM CLI configurations as experimental unless they are backed by
   a concrete successful run in the same environment.

Useful local scripts to know:

- `scripts/launch_rloo_llama3_1_8b_math500_tpu.sh`
  - safest starting point for the local MATH500 work
- `scripts/launch_rloo_llama3_1_8b_math500_async_tpu.sh`
  - experimental async launcher, not a supported baseline
- `scripts/grpo_demo_llama3_qwen2.py`
  - useful reference for Python-level experimentation and custom mesh work
- `scripts/load_marin_env.sh`
  - loads local secrets from `.marin.yaml`

If your goal is to continue the async/disaggregated investigation, do this:

1. Do not start from the assumption that mesh reshaping alone will fix it.
2. Separate "copy primitive problem" from "runtime execution model problem."
3. Be explicit about whether the experiment is:
   - single-host,
   - multi-host JAX SPMD,
   - or multi-process / multi-slice orchestration.

## 25. Recommended Next Steps

These are listed in priority order.

### 25.1 Short-term, production-minded path

Use collocated mode on multi-host TPU.

Reason:

- it matches the actual working JAX distributed model,
- it avoids cross-mesh actor-to-rollout sync,
- it avoids topology hacks,
- it is the least surprising configuration for this repo.

### 25.2 Short-term, investigative path

Validate disaggregated behavior only on single-host TPU first.

Reason:

- this isolates the async/control-flow design from the multi-host topology
  problem,
- and gives a cleaner signal about whether vLLM + RL sync is otherwise sound.

### 25.3 Medium-term, architecture path

If true async multi-host RL is the actual requirement, move to a multi-process
architecture:

- separate actor process,
- separate rollout process,
- external coordinator,
- explicit weight sync transport.

This is the main conclusion of `pathways_async_rl.md`, and it is the most
credible open-source approximation of the desired Pathways-style behavior.

Possible sync transports:

- Orbax/GCS checkpoints
- shared filesystem / SSD
- custom gRPC tensor streaming

### 25.4 Low-confidence research path

Prototype a transfer-mesh experiment only if the goal is research, not
delivery.

Do not confuse this with a likely near-term fix.

## 26. Questions a Newcomer Should Ask Before Touching This Code

1. Is this run supposed to be collocated or disaggregated?
2. Is the hardware single-host or multi-host?
3. Is the rollout engine `vanilla`, `vllm`, or `sglang_jax`?
4. Is Pathways actually enabled, or is this ordinary JAX?
5. Are the mesh assignment fields I am passing actually parsed by the current
   CLI implementation?
6. Does this run require in-memory weight sync across different meshes?
7. Am I trying to solve a topology problem with a config change that only
   changes logical mesh labels?

If those questions are not answered first, it is very easy to waste time.

## 27. Minimal Examples

### 27.1 Safe collocated mental model

```python
devices = jax.devices()
mesh = Mesh(np.array(devices).reshape(2, 4), ("fsdp", "tp"))

cluster_config = ClusterConfig(
    role_to_mesh={
        Role.ACTOR: mesh,
        Role.REFERENCE: mesh,
        Role.ROLLOUT: mesh,
    },
    rollout_engine="vllm",
    training_config=training_config,
    rollout_config=rollout_config,
)
```

What this buys you:

- one real distributed program,
- one full mesh,
- no cross-mesh actor-to-rollout reshard at steady state.

This is the best model to keep in your head when debugging a normal multi-host
Tunix RL run.

### 27.2 Conceptual disaggregated example

```python
devices = jax.devices()
split = len(devices) // 2
rollout_mesh = Mesh(np.array(devices[:split]).reshape(split, 1), ("fsdp", "tp"))
actor_mesh = Mesh(np.array(devices[split:]).reshape(split, 1), ("fsdp", "tp"))
```

This matches the docs conceptually, but on multi-host hardware it should not be
read as "safe by default."

### 27.3 Local async launcher caveat

The local async launcher script passes device IDs like:

```bash
rollout_model_config.mesh.device_ids="[0,1,4,5]"
actor_model_config.mesh.device_ids="[2,3,6,7]"
```

But the current checked-in CLI parser does not implement those fields. So do
not assume this command line alone changes actual mesh device assignment unless
you have separately verified the config implementation in the current checkout.

## 28. Additional Operational Notes

### 28.1 LoRA changes the sync cost model

Tunix is capable of syncing LoRA-only parameters rather than all parameters.
This matters because:

- LoRA reduces HBM footprint,
- LoRA reduces sync payload size,
- and some collocated vs disaggregated tradeoffs look different when only LoRA
  deltas need to move.

That said, LoRA does **not** solve the multi-host runtime-model problem. It
only reduces the amount of data that needs to move once the topology/runtime is
already valid.

### 28.2 CPU offload is helpful but orthogonal

`offload_to_cpu` can reduce HBM pressure in collocated mode by moving inactive
models off device. This is useful for memory management, but it does not solve:

- cross-host disjoint mesh execution,
- or cross-mesh vLLM sync correctness.

Treat it as a memory knob, not an async-topology fix.

### 28.3 MATH500-specific caution

The local MATH500 launchers use long sequence settings, relatively high
generation count, and vLLM rollout. That combination is useful for stressing
the async design, but it also means failures can come from:

- topology/runtime problems,
- HBM budgeting mistakes,
- sequence-length pressure,
- or rollout backend instability.

When debugging, separate those concerns instead of changing all of them at once.

## 28. Final Summary

The simplest accurate summary is:

- Tunix RL is a JAX mesh-driven RL framework with pluggable rollout engines and
  explicit role-to-mesh assignment.
- Collocated mode is the safe and natural mode for multi-host TPU in this repo.
- Disaggregated mode is conceptually supported, but the current async design is
  much more reliable on single-host than on multi-host.
- Our multi-host async attempt failed concretely during actor-to-vLLM weight
  sync, on the `load_checkpoint -> update_params -> transfer_state_with_mappings
  -> reshard_pytree -> jax.device_put` path.
- The deeper issue is not just one bad copy primitive. It is that normal JAX
  SPMD is not a Pathways-style central dispatcher for independently scheduled
  role meshes.
- The most realistic route to true multi-host async RL is separate processes or
  slices per role with explicit orchestration and explicit weight transport.

If you only remember one thing from this document, remember this:

- full-mesh multi-host JAX works,
- disjoint multi-host role meshes inside one ordinary JAX async RL job are a
  different problem,
- and this repo, as currently wired in this checkout, does not solve that
  problem for us.
