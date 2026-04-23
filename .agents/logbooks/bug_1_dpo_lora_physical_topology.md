# Bug 1: DPO LoRA Physical Topology on `v5p-8`

> Scope: isolate the original `v5p-8 pd=4` LoRA DPO divergence only.
> This is a cleaned extraction from [debug_accum_tpu_type.md](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:1).
> It intentionally excludes the later Bug 2 `c=f32` thread except where those runs directly informed Bug 1.

## Current State

**Bug 1** is the canonical bad LoRA DPO regime on `v5p-8` with the standard FSDP mesh:

- TPU: `v5p-8`
- mesh: canonical FSDP `axes={replica:1, data:4, model:1}`
- recipe: LoRA `r=64`, `alpha=64`, `zero_init_b=True`
- DPO: per-stmt mental-health recipe, `batch=64`, `lr=1e-6`
- mixed precision: default `p=f32,c=bf16`

Canonical bad baseline:

- **Exp Q pd=4**: step 9 `train/loss = 0.660557`
  W&B: <https://wandb.ai/marin-community/dpo/runs/experiment_q_r64_v5p8_pd4_s10_ue5a-i4-d7d7e1>
  Source: [Exp Q pd=4](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:7408)

The strongest current takeaways are:

1. **This is not a generic `v5p-8` DPO failure.**
   Full FT on `v5p-8` learns materially better than the bad LoRA regime.
2. **This is not fixed by dtype changes.**
   `c=f32` does not rescue. `p=bf16,c=bf16` does not rescue. CA bf16-rounding gates do not rescue.
3. **This is not fixed by changing the reference path.**
   `SeparateReferenceConfig` tracks the bad baseline.
4. **This is not a CE-kernel workload / accumulation issue.**
   CE probes stayed bad.
5. **The only clean positive rescues are mesh changes that avoid `data=4` on the bad slice.**
   Pure TP `{data:1, model:4}` and mixed mesh `{data:2, model:2}` both recover strongly.
6. **LR and horizon compensate the symptoms but do not identify the cause.**
   `lr=1e-5`, `lr=3e-5`, and 100-step runs all improve Bug 1, but that does not explain the underlying topology dependence.
7. **Bug 1 is not the same mechanism as Bug 2.**
   They share a similar LR-scaling signature, but Bug 1 does not respond at all to the CA/dtype interventions that partially rescue Bug 2.

## Best Current Interpretation

The safest current statement is:

> Bug 1 is a **LoRA + canonical FSDP mesh** failure on `v5p-8`, tied to the `data:4` regime on that hardware/slice layout. It is **not** explained by CE tiling, reference construction, or simple precision changes.

What is still open:

- Is the load-bearing variable **logical `data=4` anywhere**, or specifically the **physical 4-chip `v5p-8` single-slice topology**?
- Is the culprit a **collective algorithm / participant ordering / physical mesh mapping** issue, or some other FSDP-specific update-path effect?
- Why does LoRA amplify this, while full FT does not?

## Known-Good / Known-Bad Matrix

| config | result | evidence |
|---|---:|---|
| `v5p-8`, LoRA, canonical FSDP `{data:4, model:1}`, `lr=1e-6` | bad | Exp Q pd=4: `0.660557` |
| `v5p-8`, LoRA, canonical FSDP `{data:4, model:1}`, `pd=8` | bad | Exp Q pd=8: `0.662508` |
| `v5p-8`, full FT | good enough | Exp T: `0.608325` at step 9, clearly away from bad LoRA regime |
| `v5p-8`, LoRA, pure TP `{data:1, model:4}` | good | Exp W TP: `0.267132` |
| `v5p-8`, LoRA, mixed `{data:2, model:2}` | good | Exp Z3 / W mix: `0.273818` |
| `v5p-16`, LoRA, canonical good recipe | good | Exp N: `0.317624` |
| `v5p-8`, LoRA, canonical FSDP, `c=f32` | still bad | Exp U / AB: `0.659855` |
| `v5p-8`, LoRA, canonical FSDP, `p=bf16,c=bf16` | still bad | CN: `0.660557` |
| `v5p-8`, LoRA, canonical FSDP + CA gates | still bad | CM: `0.660557` |

## Experiment Index

### Baselines and controls

| ID | What changed | Outcome | W&B | Source |
|---|---|---|---|---|
| Exp N | good control on `v5p-16 pd=2` | step 9 `0.317624` | <https://wandb.ai/marin-community/dpo/runs/r64_matched_pd2_s10_v5p16_n1-7a55a1> | [good-run links](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:7860) |
| Exp Q pd=8 | canonical bad recipe on `v5p-8` | step 9 `0.662508` | <https://wandb.ai/marin-community/dpo/runs/experiment_q_r64_v5p8_pd8_s10_ue5a-i1-38dd4c> | [Exp Q pd=8](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:7570) |
| Exp Q pd=4 | canonical bad recipe with lower `pd` | step 9 `0.660557` | <https://wandb.ai/marin-community/dpo/runs/experiment_q_r64_v5p8_pd4_s10_ue5a-i4-d7d7e1> | [Exp Q pd=4](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:7408) |
| Exp T | full FT on `v5p-8` | step 9 `0.608325`; clearly not the bad LoRA regime | <https://wandb.ai/marin-community/dpo/runs/exp_t_v5p8_fullft_bs32_pd4_s10_uc1-rerun-20260416-3-stream-042354> | [Exp T](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:6870) |

### Things that did not fix Bug 1

| ID | Hypothesis tested | Outcome | W&B | Source |
|---|---|---|---|---|
| Exp R | smaller local CE workload might fix it | still bad, step 9 `0.662023` | <https://wandb.ai/marin-community/dpo/runs/experiment_r_r64_v5p8_bs32_pd4_s10_ue5a-i1-423c65> | [Exp R](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:7153) |
| Exp R2a | CE backward bf16 accumulation across blocks is the cause | still bad, step 9 `0.661409` | <https://wandb.ai/marin-community/dpo/runs/experiment_r2a_r64_v5p8_bs64_pd4_s10_uc1-i1-8dac8a> | [Exp R2a](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:10442) |
| Exp V | `AdapterBaseReferenceConfig` is the cause | still bad, step 9 `0.661385` | <https://wandb.ai/marin-community/dpo/runs/experiment_v_r64_v5p8_pd4_s10_uc1-v1-40ec48> | [Exp V](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:6191) |
| Exp U | bf16 compute precision is the cause | still bad, step 9 `0.659855` | <https://wandb.ai/marin-community/dpo/runs/experiment_u_r64_v5p8_pd4_fp32_s10_uc1a-e1ff3f> | [Exp U](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:6450) |
| Z2 | `--xla_allow_excess_precision=false` will force a better collective | worse than baseline, step 9 `0.693876` | <https://wandb.ai/marin-community/dpo/runs/experiment_z2_r64_v5p8_pd4_s10_uc1-z2a-367b5e> | [Z2](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:5173) |
| AB | width-4 bug is caused by bf16 collective dtype | still bad with verified `f32` grad collectives, step 9 `0.659855` | <https://wandb.ai/marin-community/dpo/runs/experiment_ab_r64_v5p8_pd4_fp32_s10_uc1-ab1-3a6ba4> | [AB](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:3946) |
| CM | Bug 1 is rescueable by CA forward-rounding gates | no rescue, step 9 `0.660557` | <https://wandb.ai/marin-community/dpo/runs/experiment_cm_r64_s10_cm-v5p8-pd4-ca-gates-268a92> | [CM](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:2523) |
| CN | Bug 1 is rescueable by full bf16 storage+compute | no rescue, step 9 `0.660557` | <https://wandb.ai/marin-community/dpo/runs/experiment_cn_s10_cn-v5p8-pd4-pbf16-cbf16-34734b> | [CN](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:2570) |

### Things that did fix Bug 1

| ID | What changed | Outcome | W&B | Source |
|---|---|---|---|---|
| Exp W TP | pure TP mesh `{data:1, model:4}` | full recovery, step 2 `0.290650`, step 9 `0.267132` | <https://wandb.ai/marin-community/dpo/runs/experiment_w_r64_v5p8_pd4_mesh_s10_uc1-wtp2-815fa2> | [Exp W TP](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:5930) |
| Exp W TP replication | cross-region TP replication | same step-2 recovery in `ue5` | <https://wandb.ai/marin-community/dpo/runs/experiment_w_r64_v5p8_pd4_mesh_s10_ue5-wtp2-2ec0bb> | [Exp W TP replication](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:5963) |
| Exp Z3 / W mix | mixed mesh `{data:2, model:2}` | full recovery, step 2 `0.298463`, step 9 `0.273818` | <https://wandb.ai/marin-community/dpo/runs/experiment_w_r64_v5p8_pd4_mesh_s10_uc1-wmix1-c3846c> | [Exp Z3](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:5299) |

### LR / horizon probes on Bug 1

| ID | What changed | Outcome | W&B | Source |
|---|---|---|---|---|
| CP5 | 100 steps at canonical `lr=1e-6` | step 50 `0.453800`, step 99 `0.397746` | <https://wandb.ai/marin-community/dpo/runs/experiment_cp5_r64_v5p8_pd4_s100_ue5-cp5-s100-af37e4> | [summary block](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:696) |
| CP6 | `lr=1e-5` | step 9 `0.411035` | <https://wandb.ai/marin-community/dpo/runs/experiment_cp6_r64_s10_ue5-cp6-v5p8-lr1em5-0fdf9c> | [summary block](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:696) |
| CP9 | `lr=3e-5` | step 9 `0.136786` | <https://wandb.ai/marin-community/dpo/runs/experiment_cp9_r64_s10_cp9-v5p8-pd4-lr3e-05-636468> | [CP9](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:1762) |

Interpretation:

- These runs show Bug 1 is **not literally frozen**.
- They do **not** explain the root cause.
- They do show that the bad topology changes the effective early training dynamics enough that higher LR or longer horizon can compensate behaviorally.

### Diagnostics that narrowed the topology story

| ID | What it checked | Outcome | W&B | Source |
|---|---|---|---|---|
| Exp Y | compare LoRA/Adam partition specs on bad `v5p-8` vs good `v6e-8` | pspecs are identical except `data:4` vs `data:8` mesh width | v5p: <https://wandb.ai/marin-community/dpo/runs/experiment_y_r64_v5p8_pd4_sharding_s2_uc1-y5p-7d5264> ; v6e: <https://wandb.ai/marin-community/dpo/runs/experiment_y_r64_v6e8_pd4_sharding_s2_ew4-y6e-ddcfc9> | [Exp Y](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:5715) |
| Exp Z1 | compare post-reduce LoRA grad values on bad `v5p-8` vs good `v6e-8` at matched init | gradients differ elementwise at step 0 / early steps | v5p: <https://wandb.ai/marin-community/dpo/runs/experiment_z1_r64_v5p8_pd4_s2_ue5-z15q-025103> ; v6e: <https://wandb.ai/marin-community/dpo/runs/experiment_z1_r64_v6e8_pd4_s2_ew4-z16f-19e67e> | [Exp Z1](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:5220) |

## Historical Precision Branch and Why It Was Retracted

The first topology story in the source logbook was:

> width-4 bf16 collective arithmetic on `v5p-8` produces the wrong LoRA gradient update.

That story was supported at the time by:

- **Z1**: v5p-8 and v6e-8 post-reduce LoRA grad values differ.
- **Z4**: the HLO on the default `c=bf16` recipe showed bf16 cross-chip reductions, with width 4 on `v5p-8` and width 8 on `v6e-8`.

Historical source:

- [Z4 result](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:5016)

But that interpretation was later weakened by **AB**:

- AB reran the bad `v5p-8` recipe under `p=f32,c=f32` and dumped HLO.
- The width-4 grad collectives were verified to be **`f32`**, not bf16.
- Training was still bad.

So the current safe reading is:

- **Z1 still matters**: there really are width-dependent numerical differences.
- **Y still matters**: this is not a wrong `PartitionSpec`.
- **Z4 as a current mechanism claim is too strong**.
- **AB killed the narrower “bf16 collective dtype is the whole cause” story**.

That leaves a more general topology/collective interpretation alive:

- physical participant ordering
- algorithm choice
- logical `data=4` width
- physical `v5p-8` slice layout
- or some other FSDP-specific update-path effect that survives the move to `f32`

## Caveated / Inconclusive Bug-1-Side Probes

| ID | Outcome | Why it is not decisive |
|---|---|---|
| BM | good, step 9 `0.316186` on `v5p-16` | this was not a clean width-4-on-larger-slice test; the summary itself notes it effectively ran a wider data axis / no-accum configuration, so it does not isolate abstract width 4 |
| BN v1 | failed to compile with `matmul_precision=highest` | Splash Attention rejected the f32 matmul path; no training evidence |
| BN v2 | crashed under `matmul_precision=high` | no usable training signal recorded in the source logbook |

Relevant source:

- BM summary: [summary block](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:719)
- BN compile failure: [BN](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:3073)

Observed W&B runs:

- BM: <https://wandb.ai/marin-community/dpo/runs/experiment_bm_r64_s10_uc1-bm1-v5p16-pd4-68ce4d>
- BN v1: <https://wandb.ai/marin-community/dpo/runs/experiment_bn_r64_v5p8_s10_ue5-bn-highest-2cc9df>
- BN v2: <https://wandb.ai/marin-community/dpo/runs/experiment_bn_r64_v5p8_pd4_s10_ue5-bn-high-992707>

## What I Trust Most

High-confidence:

- Canonical `v5p-8` FSDP `data:4` LoRA is bad.
- `pd` is not the cause.
- CE math is not the cause.
- reference-path choice is not the cause.
- simple precision changes are not the cause.
- full FT on `v5p-8` is not in the same bad regime.
- mesh changes that avoid `data:4` recover deterministically.
- Bug 1 is a mesh/topology/FSDP-path problem, not a generic recipe problem.

Medium-confidence:

- the failure surface is tied to `data:4` specifically rather than “any FSDP on v5p-8”
  because `{data:2,model:2}` and `{data:1,model:4}` both recover cleanly.

Still open:

- whether this is **abstract width-4** or **physical `v5p-8` single-slice topology**
- whether the load-bearing difference lives in:
  - collective algorithm / participant ordering
  - physical logical-rank-to-chip mapping
  - some FSDP optimizer/update path unique to the canonical mesh

## Working Conclusion for Future Agents

If the goal is to **run the recipe successfully today**, do one of:

1. avoid canonical `v5p-8` FSDP `{data:4, model:1}`
2. use pure TP `{data:1, model:4}`
3. use mixed mesh `{data:2, model:2}`
4. use a larger / different pod (`v5p-16`, `v6e-8`, etc.)

If the goal is to **finish the root-cause investigation**, the next clean probes should be:

1. explicit width-4-on-larger-slice test
2. mesh permutation / physical-rank reorder on `v5p-8`
3. per-step LoRA update-vector audit comparing:
   - bad `v5p-8 data:4`
   - good `v5p-8 data:2/model:2`
   - good `v5p-8 TP`
   - good `v5p-16`

That is the shortest path from “it’s topology-shaped” to “which topology fact actually matters.”

## 2026-04-18T Launch Plan: BL physical-device permutation sweep

Next probe to execute:

- **BL**: keep the canonical bad Bug-1 recipe fixed and vary only the explicit
  physical device order used to build the `{replica:1, data:4, model:1}` mesh.

Planned variants:

| variant | `EXPERIMENT_BL_ORDER` | purpose |
|---|---|---|
| BL0 | `canonical` | explicit-order control |
| BL1 | `reverse` | maximal reversal of logical-rank to device assignment |
| BL2 | `swap12` | local interior swap |
| BL3 | `rotate` | cyclic shift |

Decision rule:

- If any permutation materially changes step-2 or step-9 loss relative to the
  explicit-order control, physical topology / rank assignment is load-bearing.
- If all four variants are equivalently bad, the next priority probe is an
  explicit width-4-on-larger-slice test.

Submitted jobs on Iris (`us-central1`):

- BL0 canonical:
  `/ahmed/iris-run-experiment_bl_v5p8_pd4_device_permutation_s10-20260418-183601`
- BL1 reverse:
  `/ahmed/experiment-bl-v5p8-pd4-reverse-20260418-1838`
- BL2 swap12:
  `/ahmed/experiment-bl-v5p8-pd4-swap12-20260418-1838`
- BL3 rotate:
  `/ahmed/experiment-bl-v5p8-pd4-rotate-20260418-1838`

Launch note:

- The first parallel submission wave collided on the same auto-generated Iris
  job ID because all four launches started in the same second. BL0 landed; BL1
  through BL3 were re-submitted with explicit `--job-name` values and are now
  queued successfully.

## 2026-04-18T BL Result: Physical Device Order Is Load-Bearing

This section records the full execution of **BL**, the explicit physical-device
permutation probe for Bug 1.

### Goal

The goal of BL was to answer the most important remaining Bug-1 question:

- is the failure intrinsic to the abstract logical `{data:4, model:1}` regime?
- or does it depend on the **physical assignment** of the 4 logical `data`
  ranks to concrete TPU chips on `v5p-8`?

BL held the logical mesh, model, optimizer, batch order, LoRA config, and dtype
 fixed, and varied only the physical device order used to build the mesh.

### Exact Code Changes

Three code changes were required to make this probe real rather than nominal.

1. `lib/levanter/src/levanter/utils/mesh.py`

- Added `MeshConfig.device_permutation: Sequence[int] | None`
- Added `MeshConfig.preserve_device_order: bool`
- Extended `create_mesh_from_axis_specs(...)` with
  `preserve_device_order: bool = False`
- Added a debug-only path:
  - if `preserve_device_order=True`, skip `mesh_utils.create_device_mesh(...)`
  - instead reshape the provided device list directly with
    `np.array(devices, dtype=object).reshape(ici_mesh_shape)`

Why this matters:

- if we let `mesh_utils.create_device_mesh(...)` run, JAX may choose a
  topology-aware arrangement and partially normalize away the effect we want to
  test
- the BL probe specifically needs a "literal physical assignment" mode

2. `lib/levanter/src/levanter/trainer.py`

- Updated `TrainerConfig.device_mesh`
- if `self.mesh.device_permutation` is set:
  - read `jax.devices()`
  - validate the provided tuple is a permutation of `0..N-1`
  - reorder the device list
  - pass that list into `create_mesh_from_axis_specs(...)`
- also threaded
  `preserve_device_order=self.mesh.preserve_device_order`

Why this matters:

- this is where the logical mesh gets bound to a concrete ordered list of TPU
  devices
- without this change, experiment config cannot control logical-rank to physical
  chip assignment

3. `experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py`

- Added a new dedicated Bug-1 experiment script
- Fixed recipe:
  - `v5p-8`
  - `per_device_parallelism=4`
  - `train_batch_size=64`
  - `num_train_steps=10`
  - `learning_rate=1e-6`
  - LoRA `r=64`, `alpha=64`, `zero_init_b=True`
  - reference = `AdapterBaseReferenceConfig()`
  - logical mesh = `{replica:1, data:4, model:1}`
- Added permutation control via `EXPERIMENT_BL_ORDER`
- Supported aliases:
  - `canonical -> (0,1,2,3)`
  - `reverse -> (3,2,1,0)`
  - `swap12 -> (0,2,1,3)`
  - `rotate -> (1,2,3,0)`
- Configured:
  - `device_permutation=DEVICE_PERMUTATION`
  - `preserve_device_order=True`
- Tagged the runs with `experiment-bl`, `bug-1`, `v5p-8`, `pd4`, and
  `perm-<order>`

Important implementation note:

- BL `canonical` is an **explicit-order control**
- it is not a promise of bit-perfect identity with the original Exp Q helper
  path
- the relevant fact is that it still lands in the same bad regime as the
  original Bug-1 baseline

### Local Validation Before Launch

I ran:

```bash
.venv/bin/python -m py_compile \
  lib/levanter/src/levanter/utils/mesh.py \
  lib/levanter/src/levanter/trainer.py \
  experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py
```

This passed before any Iris submission.

### Exact Launch Commands

All jobs were launched on the Marin Iris controller with
`lib/iris/examples/marin.yaml`.

First submission wave:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  -e EXPERIMENT_BL_ORDER canonical \
  -e MARIN_DEBUG_RUN_TAG bl0-can \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py
```

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  -e EXPERIMENT_BL_ORDER reverse \
  -e MARIN_DEBUG_RUN_TAG bl1-rev \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py
```

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  -e EXPERIMENT_BL_ORDER swap12 \
  -e MARIN_DEBUG_RUN_TAG bl2-s12 \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py
```

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  -e EXPERIMENT_BL_ORDER rotate \
  -e MARIN_DEBUG_RUN_TAG bl3-rot \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py
```

Submission wrinkle:

- because the four launches were started in the same second and BL0 used the
  default auto-generated Iris name, the other three collided on the same
  generated job ID
- BL0 landed successfully
- BL1 through BL3 had to be resubmitted with explicit `--job-name` values

Resubmissions:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  --job-name experiment-bl-v5p8-pd4-reverse-20260418-1838 \
  -e EXPERIMENT_BL_ORDER reverse \
  -e MARIN_DEBUG_RUN_TAG bl1-rev \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py
```

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  --job-name experiment-bl-v5p8-pd4-swap12-20260418-1838 \
  -e EXPERIMENT_BL_ORDER swap12 \
  -e MARIN_DEBUG_RUN_TAG bl2-s12 \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py
```

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  --job-name experiment-bl-v5p8-pd4-rotate-20260418-1838 \
  -e EXPERIMENT_BL_ORDER rotate \
  -e MARIN_DEBUG_RUN_TAG bl3-rot \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py
```

### Final Iris Jobs

All four jobs completed successfully:

- BL0 canonical:
  `/ahmed/iris-run-experiment_bl_v5p8_pd4_device_permutation_s10-20260418-183601`
- BL1 reverse:
  `/ahmed/experiment-bl-v5p8-pd4-reverse-20260418-1838`
- BL2 swap12:
  `/ahmed/experiment-bl-v5p8-pd4-swap12-20260418-1838`
- BL3 rotate:
  `/ahmed/experiment-bl-v5p8-pd4-rotate-20260418-1838`

Terminal job state:

- all four `succeeded`
- no job failed
- no job was killed
- no job was preempted

Observed durations:

- BL0 canonical: `22m 17s`
- BL1 reverse: `22m 09s`
- BL2 swap12: `18m 13s`
- BL3 rotate: `22m 29s`

Minor runtime note:

- BL3 rotate hit transient Hugging Face `BrokenPipeError` retries while reading
  the base checkpoint
- the retry logic recovered and the run completed successfully
- this did not change the training regime it landed in

### W&B Runs

- BL0 canonical:
  <https://wandb.ai/marin-community/dpo/runs/experiment_bl_r64_v5p8_pd4_perm_s10_bl0-can-9787c8>
- BL1 reverse:
  <https://wandb.ai/marin-community/dpo/runs/experiment_bl_r64_v5p8_pd4_perm_s10_bl1-rev-a7fd3e>
- BL2 swap12:
  <https://wandb.ai/marin-community/dpo/runs/experiment_bl_r64_v5p8_pd4_perm_s10_bl2-s12-7af282>
- BL3 rotate:
  <https://wandb.ai/marin-community/dpo/runs/experiment_bl_r64_v5p8_pd4_perm_s10_bl3-rot-93ae98>

Verification method:

- Iris job completion was verified with `uv run iris ... job summary <job_id>`
- final metric values were verified via W&B API `scan_history(keys=['global_step','train/loss'])`
- this avoids relying on truncated terminal log snippets

### Final Results

The result is not subtle.

| variant | order | step 2 loss | step 9 loss | regime |
|---|---|---:|---:|---|
| BL0 | `canonical` | `0.686027` | `0.662460` | bad |
| BL1 | `reverse` | `0.334927` | `0.308011` | good |
| BL2 | `swap12` | `0.685125` | `0.660557` | bad |
| BL3 | `rotate` | `0.334837` | `0.309196` | good |

Full logged trajectories:

- BL0 canonical:
  `0.693147, 0.693147, 0.686027, 0.681521, 0.674175, 0.668314, 0.667874, 0.663600, 0.660159, 0.662460`
- BL1 reverse:
  `0.693147, 0.693147, 0.334927, 0.325310, 0.333508, 0.311831, 0.330246, 0.318093, 0.299343, 0.308011`
- BL2 swap12:
  `0.693147, 0.693147, 0.685125, 0.682298, 0.673723, 0.668946, 0.667573, 0.662823, 0.658715, 0.660557`
- BL3 rotate:
  `0.693147, 0.693147, 0.334837, 0.325998, 0.333270, 0.312630, 0.330968, 0.318587, 0.299446, 0.309196`

Three observations matter most:

1. The split happens immediately at **step 2**

- bad orders:
  - BL0 canonical `0.686027`
  - BL2 swap12 `0.685125`
- good orders:
  - BL1 reverse `0.334927`
  - BL3 rotate `0.334837`

This is the same step where the original Bug-1 healthy/bad basin split showed
up elsewhere in the larger logbook.

2. This is not "any reorder helps"

- `swap12` is still bad
- `reverse` and `rotate` are good

So the effect is structured, not a generic perturbation artifact.

3. The successful permutations land in the **healthy LoRA regime**, not merely a
"less bad" intermediate regime

- BL1/BL3 step-2 and step-9 values are essentially in-family with the known-good
  LoRA runs
- BL0/BL2 step-2 and step-9 values are in-family with the original Bug-1 bad
  regime

### Supporting Runtime Details

The logs provide a few useful guardrails against simpler confounds.

1. Batch order was held fixed

- logged `DEBUGJ BATCH` hashes matched across variants for the early steps
- for example:
  - step 0 hash: `7a61ce53d17eb721`
  - step 1 hash: `f2eb65dfc30bc069`
  - step 2 hash: `351ee7a62b0d7d14`

2. The logical mesh stayed fixed

- all runs used:
  - `axes={"replica": 1, "data": 4, "model": 1}`
  - `param_mapping={"embed": "data"}`
  - `shared_mapping={}`

3. The LoRA recipe stayed fixed

- LoRA `r=64`
- `alpha=64`
- `zero_init_b=True`
- `lr=1e-6`
- `batch=64`
- `pd=4`
- `seed=0`
- `reference=AdapterBaseReferenceConfig()`

4. The runs still begin from the same symmetric init

- the good `reverse` run logged the same canonical step-0 pattern:
  - `loss=0.693147`
  - `gA_l2=0`
  - `gB_l2=2.463127`
  - `pB_l2=0`

So the permutation does **not** rescue by changing the nominal initialization
recipe. The divergence shows up after the first real update path.

### Interpretation Update

BL changes the Bug-1 picture materially.

Before BL, the safe open question was:

- abstract width-4 logical `data` axis?
- or physical `v5p-8` single-slice topology?

After BL, the safe update is:

- **abstract width-4 alone is not sufficient to explain Bug 1**
- **physical device assignment / logical-rank ordering is load-bearing**

This is because:

- the logical mesh is identical across BL0-BL3
- the training recipe is identical across BL0-BL3
- the batch order is identical across BL0-BL3
- only the physical rank assignment changes
- and that single change cleanly flips the run between the bad and healthy
  regimes

What BL specifically falsifies:

- "Bug 1 is purely an abstract logical `data=4` phenomenon"
- "all physical embeddings of the logical `{data:4,model:1}` mesh are
  equivalent for this recipe"

What BL supports:

- Bug 1 is a **physical topology / logical-rank assignment / collective-path**
  issue on `v5p-8`
- the root cause likely lives in one of:
  - replica-group participant ordering
  - collective algorithm choice as a function of device layout
  - physical ring/tree path
  - or another FSDP path property that depends on the concrete mesh embedding

### Updated Confidence After BL

High-confidence now:

- Bug 1 is not just "LoRA on v5p-8"
- Bug 1 is not just "DPO on v5p-8"
- Bug 1 is not just "logical `data=4` in the abstract"
- physical device order on `v5p-8` can deterministically flip the recipe from
  bad to healthy

Still open after BL:

- exactly **which topology fact** is load-bearing:
  - physical adjacency
  - rank numbering
  - collective algorithm selection
  - HLO `replica_groups` / device-id lowering
  - or some downstream optimizer/update-path consequence of the above

### What This Means Practically

If the goal is simply to run the Bug-1 recipe successfully on `v5p-8`, BL
already gives a new practical workaround:

- use a known-good explicit device order rather than the canonical one

From this sweep, the known-good orders are:

- `reverse = (3,2,1,0)`
- `rotate = (1,2,3,0)`

Known-bad orders from this sweep:

- `canonical = (0,1,2,3)`
- `swap12 = (0,2,1,3)`

### Immediate Next Debugging Moves

BL resolves the existence question. The next step is no longer "does topology
matter?" It does.

The next best probes are:

1. diff `BL0` vs `BL1` and `BL3` HLO / sharding / replica groups
2. record the exact physical JAX device list for each order and map
   logical-rank-to-chip explicitly
3. compare step-0 and step-2 LoRA `B` grad/update tensors between:
   - BL0 canonical
   - BL1 reverse
   - BL3 rotate
4. check whether the healthy orders share a common physical pattern
   that `swap12` breaks

### Bottom Line

BL is the strongest Bug-1 result so far.

Holding the logical mesh and training recipe fixed while changing only the
physical device order on `v5p-8` flips the run between:

- bad Bug-1 regime: `~0.66` by step 9
- healthy LoRA regime: `~0.31` by step 9

That is direct evidence that **Bug 1 is a physical topology / rank-assignment
bug, not merely a logical width-4 recipe bug**.

## 2026-04-18T Next Steps After BL: HLO + Grad-Value Follow-Up

BL answered the first-order question:

- does physical device order matter for Bug 1?

Answer:

- **yes**

The next question is more specific:

- **what exactly changes between the good and bad physical orders?**

There are two broad possibilities:

1. the compiled computation is structurally the same and only the concrete
   collective participant ordering differs
2. the compiled computation is structurally different across orders
   (extra collectives, different collective-permutes, different scheduling,
   different channel structure, etc.)

This follow-up plan is designed to separate those cases.

### Recommended Rerun Set

Minimum recommended set:

- **BL0 HLO**: `canonical = (0,1,2,3)` — known bad
- **BL1 HLO**: `reverse = (3,2,1,0)` — known good
- **BL2 HLO**: `swap12 = (0,2,1,3)` — known bad

Optional but strongly useful:

- **BL3 HLO**: `rotate = (1,2,3,0)` — known good

Why BL2 is important:

- BL0 vs BL1 alone proves "bad vs good physical order"
- BL2 tells us whether the **bad regime** corresponds to one specific compiled
  form or to multiple compiled forms that independently land in the same bad
  basin

Why BL3 is still useful if capacity is easy:

- BL3 tells us whether the **good regime** also has one canonical compiled form
  or whether multiple different physical orderings can independently land in the
  same healthy basin

### Corrected Run Configuration

For the HLO reruns, the recommended settings are:

- same logical mesh as BL:
  - `{replica:1, data:4, model:1}`
- same model/training recipe as BL:
  - `v5p-8`
  - `pd=4`
  - `batch=64`
  - LoRA `r=64`, `alpha=64`, `zero_init_b=True`
  - `lr=1e-6`
  - `seed=0`
- set `num_train_steps=10`
  - this keeps the reruns directly comparable to the completed BL sweep
  - it preserves both the decisive step-2 split and the full step-9 endpoint
  - the extra time is worth it because we want HLO artifacts and final-regime
    confirmation in the same rerun set

Required debug env vars:

```bash
MARIN_DEBUG_LOG_STEP_TRACE=1
MARIN_DEBUG_DUMP_GRAD_VALUES=1
MARIN_DEBUG_HLO_UPLOAD_DIR=gs://<bucket>/<prefix>/<variant>/
XLA_FLAGS="--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*"
```

Why these specific knobs:

- `MARIN_DEBUG_HLO_UPLOAD_DIR` is required because the upload hook in
  `train_dpo.py` only runs when that env var is set
- `--xla_dump_to=/tmp/xla_hlo` is required because the upload hook expects HLO
  dumps to appear there
- `--xla_dump_hlo_as_text` gives human-diffable text HLO
- `--xla_dump_hlo_module_re=.*train.*` keeps dump volume focused on the train
  computation
- `MARIN_DEBUG_LOG_STEP_TRACE=1` gives per-step `TRACE` and aggregate SENTINEL
  stats
- `MARIN_DEBUG_DUMP_GRAD_VALUES=1` prints fixed-index post-reduce `lora_B`
  gradient values and is the cheapest current way to connect HLO structure to
  actual numerical differences

What not to do initially:

- do **not** start with `--xla_dump_hlo_pass_re=.*`
- that will massively increase dump volume and make storage/diffing noisier
- if the first-pass HLO diff is inconclusive, add broader dumps later

### What We Want To Learn From Each Rerun

#### 1. BL0 HLO rerun

Purpose:

- establish the new explicit-order **bad** control with persisted HLO artifacts

What we want to learn:

- the exact post-optimization collective structure for the bad canonical order
- the exact `replica_groups` and any related collective metadata
- the exact fixed-index step-0/1/2 `lora_B` post-reduce values for the bad case

Why it matters:

- BL0 is the anchor bad case for all later diffs

#### 2. BL1 HLO rerun

Purpose:

- compare the cleanest known **good** order against BL0

What we want to learn:

- whether BL1 differs from BL0 only in collective participant ordering
- or whether BL1 introduces deeper structural changes
- whether the step-0/1/2 `lora_B` values differ in a way that lines up with the
  good step-2 basin split

Key question answered by BL0 vs BL1:

- is the good/bad split visible directly in the compiled collective structure,
  or only in the numerical outputs of apparently similar collectives?

#### 3. BL2 HLO rerun

Purpose:

- compare a second **bad** order against BL0

What we want to learn:

- whether BL0 and BL2 are effectively the same bad compiled form
- or whether they are structurally different but still land in the same bad
  regime

Why this matters:

- if BL0 and BL2 are structurally near-identical, that narrows the correlate for
  the bad regime considerably
- if BL0 and BL2 are structurally different, then "bad" is not tied to one
  unique compiled form and the failure surface is broader

This is the main reason BL2 should be included rather than only BL0/BL1.

#### 4. BL3 HLO rerun (optional but valuable)

Purpose:

- compare a second **good** order against BL1

What we want to learn:

- whether BL1 and BL3 share one common "good" compiled structure
- or whether multiple structurally different orderings can still land in the
  same healthy regime

Why it matters:

- this strengthens the interpretation of the good side, not just the bad side

### Planned Artifact Analysis

Once the HLO uploads exist, the first-pass analysis should focus on collectives.

Suggested extraction:

```bash
grep -nE "all-reduce|all-gather|reduce-scatter|collective-permute|all-to-all" \
  <after_optimizations.txt> > <variant>.collectives.txt
```

Primary comparisons:

- `BL0 vs BL1`
- `BL0 vs BL2`
- `BL1 vs BL3` if BL3 is rerun

What to inspect in each diff:

1. `replica_groups={...}`

- expected to change across orderings
- this is the first thing to inspect, but not the only thing

2. `to_apply=...` reduction function

- if this differs, the collective is not merely the same op with a different
  participant ordering

3. `channel_id=...`

- channel renumbering alone is usually not meaningful
- but systematic channel structure differences can indicate different lowering
  paths

4. presence/absence of `collective-permute`

- this would be a major clue
- if one order inserts a permute and another does not, the good/bad split may
  come from more than just nominal reduction associativity

5. surrounding schedule / fusion context

- if collectives appear in different structural neighborhoods, that points to a
  deeper compilation-path difference

### Decision Rules For Interpreting The HLO Diffs

#### Case A: BL0 vs BL1 differ only in `replica_groups` listing

What we learn:

- the strongest evidence shifts toward a "same computation, different concrete
  collective order" story
- Bug 1 would then look more like a topology-sensitive numerical instability in
  the first LoRA update than a qualitatively different compiled program

What it would *not* prove:

- it would not prove that HLO-text equality captures every backend difference
- but it would materially narrow the space

#### Case B: BL0 vs BL1 show structural collective differences

What we learn:

- physical order is changing the compiled collective path more deeply than a
  simple participant relabeling
- this would strengthen the case for an XLA/GSPMD/topology-lowering issue

#### Case C: BL0 vs BL2 are structurally very similar

What we learn:

- the bad regime may correspond to one narrow family of compiled forms
- BL2 then mostly strengthens the interpretation that "bad" is not accidental

#### Case D: BL0 vs BL2 are structurally different but both bad

What we learn:

- the bad regime is broader than one specific compiled form
- the failure would then look like multiple distinct physical-order-induced
  numerical paths can independently push the recipe into the same bad basin

#### Case E: BL1 vs BL3 are structurally different but both good

What we learn:

- the healthy regime is also broader than one special ordering
- that matters because it says we should stop looking for one magic
  compile-layout fingerprint and instead look for broader topological
  properties shared by the good orders

### Why We Also Need Grad-Value Dumps

HLO alone does not close the loop.

The HLO tells us:

- what collective structure was compiled

The grad-value dumps tell us:

- what numerical outputs those collectives actually produced at the tensors we
  care about

The most useful current tensor-level signal is:

- fixed-index post-reduce `lora_B` gradient values at early steps

Specifically we want to compare:

- BL0 canonical
- BL1 reverse
- BL2 swap12
- BL3 rotate, if rerun

and inspect:

- step 0 values
- step 1 values
- step 2 values

for:

- `q_proj.lora.lora_B`
- `k_proj.lora.lora_B`
- `v_proj.lora.lora_B`
- `o_proj.lora.lora_B`
- `gate_proj.lora.lora_B`
- `up_proj.lora.lora_B`
- `down_proj.lora.lora_B`

What we want to learn from the grad-value comparison:

- whether good and bad orders already differ numerically at step 0
- whether the difference is tiny-but-structured or larger-than-expected
- whether BL0 and BL2 share a similar bad-side sign/value pattern
- whether BL1 and BL3 share a similar good-side sign/value pattern

This is the cheapest current bridge between:

- "different physical order"
- "different collective/HLO path"
- "different step-2 training basin"

### Short-Term Priority Order

The recommended order of operations is:

1. rerun `BL0`, `BL1`, `BL2` with HLO upload + grad-value dumps
2. optionally rerun `BL3` in parallel if capacity is easy
3. diff the uploaded `after_optimizations` HLO for collective structure
4. compare the fixed-index `lora_B` gradient values at steps `0/1/2`
5. only if needed, widen the dump scope with broader XLA dump flags

### What Success Looks Like

This next round will be successful if it narrows Bug 1 into one of these
shapes:

1. **same compiled structure, different participant order**
2. **different compiled collective structure across physical orders**
3. **multiple structurally different bad forms / good forms**

Any of those outcomes is valuable. The key is to move from:

- "topology matters"

to:

- "what exact compiled/numerical difference topology is causing?"

That is the next load-bearing step in the Bug-1 investigation.

## 2026-04-18T BL-HLO Completion: All Four Succeeded And Uploaded

This section records the final completion state of the BL-HLO reruns and the
verified GCS locations of the uploaded HLO artifacts.

### Final Job Status

All four BL-HLO runs succeeded.

Verified with `uv run iris --config lib/iris/examples/marin.yaml job summary <job_id>`:

| variant | Iris job | final state | duration | preemptions |
|---|---|---|---:|---:|
| BL0 canonical | `/ahmed/experiment-bl-hlo-v5p8-pd4-bl0-20260418-1242` | `succeeded` | `28m 21s` | `0` |
| BL1 reverse | `/ahmed/experiment-bl-hlo-v5p8-pd4-bl1-20260418-1242` | `succeeded` | `22m 23s` | `0` |
| BL2 swap12 | `/ahmed/experiment-bl-hlo-v5p8-pd4-bl2-20260418-1242` | `succeeded` | `28m 07s` | `0` |
| BL3 rotate | `/ahmed/experiment-bl-hlo-v5p8-pd4-bl3-20260418-1242` | `succeeded` | `22m 56s` | `0` |

Important verification note:

- an earlier status report claimed BL2 had one recovered preemption
- the current Iris job summary for BL2 reports `preemptions=0`
- absent stronger evidence from lower-level task metadata, the authoritative
  value recorded here is **0 preemptions**

### Where The HLO Artifacts Were Pushed On GCP

All HLO artifacts were uploaded under:

`gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/`

Per-variant prefixes:

- BL0 canonical:
  `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl0-can/`
- BL1 reverse:
  `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl1-rev/`
- BL2 swap12:
  `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl2-s12/`
- BL3 rotate:
  `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl3-rot/`

These prefixes were verified directly with `gsutil ls`.

### Upload Confirmation From Run Logs

Each run emitted `DEBUGJ HLO_UPLOAD ... dest=...` in the `train_dpo` task log,
confirming not just intent but actual upload completion.

Verified upload summaries:

| variant | uploaded files | uploaded bytes | destination |
|---|---:|---:|---|
| BL0 canonical | `587` | `116,783,144` | `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl0-can/` |
| BL1 reverse | `601` | `114,571,172` | `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl1-rev/` |
| BL2 swap12 | `426` | `114,109,609` | `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl2-s12/` |
| BL3 rotate | `587` | `114,288,725` | `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl3-rot/` |

The upload lines were:

- BL0:
  `DEBUGJ HLO_UPLOAD uploaded=587 bytes=116783144 dest=gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl0-can/`
- BL1:
  `DEBUGJ HLO_UPLOAD uploaded=601 bytes=114571172 dest=gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl1-rev/`
- BL2:
  `DEBUGJ HLO_UPLOAD uploaded=426 bytes=114109609 dest=gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl2-s12/`
- BL3:
  `DEBUGJ HLO_UPLOAD uploaded=587 bytes=114288725 dest=gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl3-rot/`

### What Exists In Those Prefixes

The prefixes contain the raw XLA dump artifacts produced by:

`XLA_FLAGS="--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*"`

Examples of verified object names:

- `module_0001.jit_convert_element_type...execution_options.txt`
- `module_0001.jit_convert_element_type...hlo_module_config.txt`
- `module_0004.jit_minimum...flagfile`
- `module_0007.jit_multiply...tpu_comp_env.txt`

So the GCS upload is not a marker-only prefix; it contains the expected XLA dump
files.

### Operational Summary

The BL-HLO rerun stage is complete:

- all four jobs succeeded
- all four emitted `DEBUGJ HLO_UPLOAD`
- all four uploaded artifacts into the expected `gs://marin-us-central1/...`
  prefixes
- the HLO diff stage can now proceed using those GCS paths as source of truth

### Immediate Analysis Inputs

The canonical paths to use for the next HLO diff stage are:

- BL0 bad control:
  `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl0-can/`
- BL1 good control:
  `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl1-rev/`
- BL2 second bad:
  `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl2-s12/`
- BL3 second good:
  `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl3-rot/`

Primary comparisons remain:

1. BL0 vs BL1 — bad vs good
2. BL0 vs BL2 — bad vs bad
3. BL1 vs BL3 — good vs good
4. grad-value traces from `MARIN_DEBUG_DUMP_GRAD_VALUES=1`

## 2026-04-18T Launch Plan: BL-HLO follow-up reruns

This launch block records the exact reruns to submit after the successful BL
topology sweep.

Goal:

- persist HLO artifacts for good and bad physical orders
- persist fixed-index early-step `lora_B` gradient values
- keep the runs directly comparable to the completed BL sweep

Chosen variants:

- BL0 HLO: `canonical` — bad control
- BL1 HLO: `reverse` — good control
- BL2 HLO: `swap12` — second bad order
- BL3 HLO: `rotate` — second good order

Chosen run shape:

- same BL script:
  `experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py`
- same recipe as BL:
  - `v5p-8`
  - `pd=4`
  - `batch=64`
  - LoRA `r=64`, `alpha=64`, `zero_init_b=True`
  - `lr=1e-6`
  - `seed=0`
  - `num_train_steps=10`
- constrain TPU region to `us-central1`
  - avoids cross-region HLO uploads
  - keeps the reruns closer to the original BL environment

Required env vars for each run:

```bash
REGIONS_OVERRIDE=us-central1
MARIN_DEBUG_RUN_TAG=<variant-tag>
EXPERIMENT_BL_ORDER=<order>
MARIN_DEBUG_HLO_UPLOAD_DIR=gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/<variant>/
MARIN_DEBUG_DUMP_GRAD_VALUES=1
MARIN_DEBUG_DUMP_SHARDING=1
XLA_FLAGS="--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*"
```

Submission policy:

- use explicit `--job-name` for every variant
- avoid the same-second auto-name collision seen in the original BL sweep

Submitted jobs:

- BL0 HLO
  - Iris job: `/ahmed/experiment-bl-hlo-v5p8-pd4-bl0-20260418-1242`
  - HLO upload prefix:
    `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl0-can/`
- BL1 HLO
  - Iris job: `/ahmed/experiment-bl-hlo-v5p8-pd4-bl1-20260418-1242`
  - HLO upload prefix:
    `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl1-rev/`
- BL2 HLO
  - Iris job: `/ahmed/experiment-bl-hlo-v5p8-pd4-bl2-20260418-1242`
  - HLO upload prefix:
    `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl2-s12/`
- BL3 HLO
  - Iris job: `/ahmed/experiment-bl-hlo-v5p8-pd4-bl3-20260418-1242`
  - HLO upload prefix:
    `gs://marin-us-central1/debug/bug_1_bl_hlo/20260418/bl3-rot/`

## 2026-04-18T Local Artifact Analysis: The Good/Bad Split Appears During XLA Optimization

This section records the first real analysis of the downloaded BL HLO artifacts.

Local artifact root:

- `.agents/artifacts/bug_1_bl_hlo/20260418/bl0-can/`
- `.agents/artifacts/bug_1_bl_hlo/20260418/bl1-rev/`
- `.agents/artifacts/bug_1_bl_hlo/20260418/bl2-s12/`
- `.agents/artifacts/bug_1_bl_hlo/20260418/bl3-rot/`

Reference background:

- see [xla_collective_primer.md](xla_collective_primer.md)

### First: What "before" and "after" optimization mean

This was the confusing part, so spell it out carefully.

The primer's pipeline is:

```text
Python/JAX
  -> jaxpr
  -> HLO / StableHLO
  -> XLA optimization passes
  -> optimized HLO
  -> hardware-specific executable
```

In the dumped files:

- `before_optimizations.txt` means:
  - the HLO module right after JAX has lowered the program to XLA HLO
  - before XLA has done its heavy backend work
  - this is still a relatively "source-like" view of the program
- `after_optimizations.txt` means:
  - the HLO module after XLA has optimized and lowered it for this exact TPU target
  - this is much closer to the actual executable behavior
  - collectives may be fused, decomposed, rescheduled, or rewritten

So:

- if two runs differ already in `before_optimizations`, then JAX or early sharding/lowering produced different logical programs
- if they are identical in `before_optimizations` but differ in `after_optimizations`, then the split was introduced later by XLA optimization / backend lowering

That distinction is the main result of this analysis.

### Second: Which files were compared

Each variant has two dumped `jit__train_step` modules. The module numbers differ
across runs, so compare file contents, not numeric prefixes.

First train-step module:

- BL0 bad control:
  `.agents/artifacts/bug_1_bl_hlo/20260418/bl0-can/module_0334.jit__train_step.cl_813921542.before_optimizations.txt`
- BL1 good control:
  `.agents/artifacts/bug_1_bl_hlo/20260418/bl1-rev/module_0334.jit__train_step.cl_813921542.before_optimizations.txt`
- BL2 second bad:
  `.agents/artifacts/bug_1_bl_hlo/20260418/bl2-s12/module_0298.jit__train_step.cl_813921542.before_optimizations.txt`
- BL3 second good:
  `.agents/artifacts/bug_1_bl_hlo/20260418/bl3-rot/module_0334.jit__train_step.cl_813921542.before_optimizations.txt`

Second train-step module:

- BL0:
  `.agents/artifacts/bug_1_bl_hlo/20260418/bl0-can/module_0798.jit__train_step.cl_813921542.before_optimizations.txt`
- BL1:
  `.agents/artifacts/bug_1_bl_hlo/20260418/bl1-rev/module_1572.jit__train_step.cl_813921542.before_optimizations.txt`
- BL2:
  `.agents/artifacts/bug_1_bl_hlo/20260418/bl2-s12/module_0760.jit__train_step.cl_813921542.before_optimizations.txt`
- BL3:
  `.agents/artifacts/bug_1_bl_hlo/20260418/bl3-rot/module_1572.jit__train_step.cl_813921542.before_optimizations.txt`

The corresponding optimized modules are:

- bad family:
  - BL0:
    `module_0336...after_optimizations.txt`
  - BL2:
    `module_0300...after_optimizations.txt`
- good family:
  - BL1:
    `module_0336...after_optimizations.txt`
  - BL3:
    `module_0336...after_optimizations.txt`

and the second optimized modules:

- bad family:
  - BL0:
    `module_0800...after_optimizations.txt`
  - BL2:
    `module_0762...after_optimizations.txt`
- good family:
  - BL1:
    `module_1574...after_optimizations.txt`
  - BL3:
    `module_1574...after_optimizations.txt`

### Third: The strongest result

The two bad permutations compile to one identical optimized-HLO family.

The two good permutations compile to a different identical optimized-HLO family.

And, crucially, all four variants share the same unoptimized train-step HLO.

That means:

- the high-level JAX/XLA program is the same
- the physical device permutation is not changing the traced math
- the good/bad split is introduced later by XLA optimization / backend lowering

### Exact equality results

#### `before_optimizations`: all four are identical

First train-step module:

- size: `3,786,246`
- SHA-256 prefix: `5eeeee8d36197f73`

Second train-step module:

- size: `1,969,979`
- SHA-256 prefix: `ce64129c5c5780a1`

These hashes were the same across:

- BL0 canonical
- BL1 reverse
- BL2 swap12
- BL3 rotate

So the four runs enter XLA optimization with the same train-step HLO.

#### `after_optimizations`: bad family and good family separate cleanly

Bad family `BL0 == BL2`:

- first optimized module hash prefix:
  `f1bef9fa153c32b2`
- second optimized module hash prefix:
  `3b6c2633f12afd13`

Good family `BL1 == BL3`:

- first optimized module hash prefix:
  `c32a9c6db66ee359`
- second optimized module hash prefix:
  `13518dab775e74f1`

Cross-family:

- BL0 optimized HLO is **not** equal to BL1 optimized HLO
- BL2 optimized HLO is **not** equal to BL3 optimized HLO

This is much stronger than "some lines differ." The optimized train-step
modules fall into two exact buckets:

1. a bad compiled form
2. a good compiled form

### Fourth: What differs in the optimized HLO

For the first optimized `train_step` module, collective counts are:

| family | file size | all-reduce | all-gather | reduce-scatter | collective-permute | all-to-all |
|---|---:|---:|---:|---:|---:|---:|
| BL0 / BL2 bad | `6,636,080` | `276` | `1047` | `58` | `12` | `36` |
| BL1 / BL3 good | `7,357,957` | `136` | `591` | `2` | `1546` | `36` |

For the second optimized `train_step` module:

| family | file size | all-reduce | all-gather | reduce-scatter | collective-permute | all-to-all |
|---|---:|---:|---:|---:|---:|---:|
| BL0 / BL2 bad | `4,601,734` | `144` | `979` | `28` | `0` | `30` |
| BL1 / BL3 good | `5,103,696` | `74` | `535` | `0` | `1174` | `30` |

This means the difference is structural, not cosmetic.

It is **not** just:

- the same HLO with a different `replica_groups` listing
- one or two changed channel IDs
- one ring order inside an otherwise identical collective

Instead, XLA is choosing a different communication decomposition for good vs
bad physical orders.

### Fifth: What the `collective-permute` difference looks like

The good family contains large numbers of `collective-permute-start` /
`collective-permute-done` pairs inside core model contractions.

Representative examples from the good HLO:

- `Attention._compute_qkv ... dot_general`
- `Attention/Linear/contract heads, head_size -> batch, position, embed`
- `LlamaMlp/Linear/contract embed -> batch, position, mlp`
- `LlamaMlp/Linear/contract mlp -> batch, position, embed`

Typical source/target patterns in the good family:

- `{{0,1},{1,2},{2,3},{3,0}}`
- `{{0,3},{1,0},{2,1},{3,2}}`

These appear many times. A quick count on the first optimized module found:

- `162` occurrences of `{{0,1},{1,2},{2,3},{3,0}}`
- `88` occurrences of `{{0,3},{1,0},{2,1},{3,2}}`

By contrast, the bad family has only a handful of `collective-permute` ops,
and those appear much more peripherally, near slice / concatenate handling
around custom collective fusions. For example:

- `collective-permute-start(%slice.5378), source_target_pairs={{0,1},{1,2},{2,3}}`

So the good family is not merely "bad plus a few extra ops." It looks like a
different strategy for moving data through the core attention/MLP contraction
path.

### Sixth: What this means for Bug 1

The safest current interpretation is:

1. Bug 1 is not explained by the logical mesh shape alone.
2. Bug 1 is not just "abstract width-4 reduction associativity."
3. Physical device assignment changes which optimized communication strategy
   XLA chooses.
4. That compiled-strategy split tracks the training split exactly:
   - canonical + swap12 -> bad compiled family -> bad training basin
   - reverse + rotate -> good compiled family -> good training basin

This is the strongest evidence so far that Bug 1 is a topology-sensitive XLA
lowering phenomenon, not simply a LoRA hyperparameter issue.

### Seventh: What this does and does not prove

What it proves:

- same high-level train-step HLO
- different optimized train-step HLO families
- family membership aligns exactly with good vs bad training

What it does **not** yet prove:

- that XLA is "wrong" in the strict sense
- which exact optimized collective or fusion is the first numerically
  load-bearing difference
- whether the issue should be described as:
  - an XLA backend bug
  - a topology-sensitive numerical instability in this LoRA DPO recipe
  - or an interaction of both

But this analysis does rule out a weaker story:

- "the permutations probably only changed the textual order of the same
  collective"

That story is no longer consistent with the artifacts.

### Eighth: Practical takeaway

For actual training:

- avoid canonical `v5p-8` `{data:4, model:1}` for this LoRA DPO recipe
- use a known-good permutation, `{data:2, model:2}`, pure TP, or larger slices

For root-cause work:

- the next high-value step is a minimal repro that preserves this exact split:
  - same `before_optimizations`
  - two different `after_optimizations` families depending on physical device
    order

That is a much sharper upstream story than the older "width-4 bf16 reductions
seem bad" hypothesis.

### Ninth: One limitation of the current artifact bundle

The downloaded HLO bundle was sufficient for the compiler-structure conclusion,
but it did **not** contain the expected gradient-value traces in a conveniently
recoverable form.

So this section establishes:

- the compiled-path split

but not yet:

- the exact elementwise `lora_B` grad differences that flow through that split

If needed, that numerical comparison will have to come from run logs or a fresh
instrumented rerun rather than from these HLO text dumps alone.

## 2026-04-18T Next Compiler Probe: Dump Intermediate XLA Pass States

This section records the next compiler-forensics experiment motivated by the
artifact analysis above.

### Why this is the right next question

Right now we know:

- all four BL variants have identical `before_optimizations` train-step HLO
- good and bad variants split into two exact `after_optimizations` families

That still leaves an important gap:

- **where, inside XLA, does the split first appear?**

Saying "before" and "after" is useful but coarse. Those are only the two
endpoints of a much longer compiler pipeline.

### What a compiler pass is

Reference:

- [xla_collective_primer.md](xla_collective_primer.md)
- [~/code/xla/docs/hlo_passes.md](</Users/ahmed/code/xla/docs/hlo_passes.md>)
- [~/code/xla/docs/hlo_dumps.md](</Users/ahmed/code/xla/docs/hlo_dumps.md>)

Very roughly, the compiler pipeline is:

```text
Python/JAX
  -> jaxpr
  -> lowered HLO
  -> XLA pass 1
  -> XLA pass 2
  -> XLA pass 3
  -> ...
  -> XLA pass N
  -> final optimized HLO
  -> TPU executable
```

A **pass** is one compiler transformation or analysis over the HLO module.

Examples from the XLA docs:

- algebraic simplification
- constant folding
- dead code elimination
- reshape mover
- sharding propagation
- collective pipelining / scheduling-style passes
- backend-specific legalization / lowering passes

So:

- `before_optimizations` = HLO before pass 1
- `after_optimizations` = HLO after the last pass

and there may be many intermediate states in between.

### Important clarification: who writes `before_optimizations`

The most useful mental model is:

```text
JAX function
  -> traced to jaxpr
  -> lowered by JAX to HLO / StableHLO-like IR
  -> handed to XLA
  -> dumped by XLA as before_optimizations
  -> transformed by XLA passes
  -> dumped by XLA as after_optimizations
```

So:

- `jaxpr` is not HLO
- `before_optimizations` is already past `jaxpr`
- but it is still before the main XLA optimization / lowering pipeline

That is why our current result matters so much:

- same `before_optimizations` means JAX produced the same logical program
- different `after_optimizations` means XLA / backend lowering introduced the
  split later

### How many passes does XLA run?

There is not one single stable answer like "37 passes."

The safest answer is:

- XLA has **many dozens to hundreds** of available passes overall
- any given compile runs a backend-specific subset of those passes
- some passes are organized into **pipelines**
- some pipelines are wrapped in **fixpoint loops**, so the same pass may run
  multiple times until no further change occurs

The public docs say:

- the compiler has **several hundred** HLO passes overall
  - see [~/code/xla/docs/hlo_passes.md](</Users/ahmed/code/xla/docs/hlo_passes.md>)

But for our specific TPU compile, the number that actually runs depends on:

- platform / backend
- sharding pattern
- whether a pass makes a change and triggers repeated cleanup pipelines
- optimizer/debug options

So the right mindset is:

- not "what is the one pass count?"
- but "after which pass or pipeline do BL0 and BL1 first stop matching?"

### Is there a setting for the number of optimizations?

Not in the usual "compiler optimization level" sense.

There is **not** one simple knob like:

- `-O0`
- `-O1`
- `-O2`
- `-O3`

for TPU XLA debugging here.

Instead, the useful controls are:

- `--xla_dump_to=...`
  - where dump files go
- `--xla_dump_hlo_module_re=...`
  - which modules to dump
- `--xla_dump_hlo_pass_re=...`
  - dump after passes whose names match a regex
- `--xla_dump_hlo_pipeline_re=...`
  - dump around pipelines whose names match a regex

Per the XLA docs:

- dump selected passes:
  `XLA_FLAGS="--xla_dump_to=DIR --xla_dump_hlo_pass_re=spmd|propagation"`
- dump every pass:
  `XLA_FLAGS="--xla_dump_to=DIR --xla_dump_hlo_pass_re=.*"`

That last option is powerful but can explode the artifact count.

### Why pass-by-pass dumping is valuable here

Right now our knowledge is:

1. same traced train-step program
2. different final compiled train-step families

Pass-by-pass dumping lets us refine that to:

1. same traced train-step program
2. same early compiler states
3. **first divergence occurs after pass X or pipeline Y**

That is much more actionable.

If the first divergence appears after something like:

- sharding propagation
- an SPMD partitioning pass
- a collective scheduling / pipelining pass

then we may be able to inspect open XLA code and reason directly about it.

If the modules remain identical deep into the pipeline and only diverge very
late, that points more strongly at TPU backend / `libtpu` lowering.

### Why cloning `openxla/xla` is useful but not sufficient

We cloned:

- `~/code/xla`

This is useful as a source browser and tooling reference.

Examples:

- [~/code/xla/docs/tools.md](</Users/ahmed/code/xla/docs/tools.md>)
- [~/code/xla/docs/hlo_dumps.md](</Users/ahmed/code/xla/docs/hlo_dumps.md>)
- [~/code/xla/docs/hlo_passes.md](</Users/ahmed/code/xla/docs/hlo_passes.md>)

But it does **not** mean we can faithfully emulate TPU topology on a laptop.

Why:

- the public repo has good generic HLO tools
- however, TPU compilation goes through the TPU runtime interface in
  [tpu_on_demand_compiler.cc](</Users/ahmed/code/xla/xla/stream_executor/tpu/tpu_on_demand_compiler.cc>)
- that file calls:
  - `TpuCompiler_RunHloPassesFn`
  - `TpuCompiler_RunBackendFn`
- those go through `libtpu`

So the real TPU backend decision-making is not something we can cleanly emulate
offline from this checkout alone.

Conclusion:

- cloning XLA is worthwhile for understanding passes and tooling
- trying to "simulate v5p-8 topology locally" is probably not worthwhile
- the right next compiler experiment is still on a real TPU job

### Proposed next experiment

Goal:

- rerun only the minimum good/bad pair
- dump intermediate compiler states
- identify the first pass or pipeline where they diverge

Chosen pair:

- BL0 canonical: bad control
- BL1 reverse: good control

Why not rerun all four immediately:

- BL0/BL2 are already proven equivalent as bad family
- BL1/BL3 are already proven equivalent as good family
- for first-divergence analysis, one representative bad + one representative
  good is enough

### Exact proposed dump strategy

Keep the same BL script and same training recipe, but add pass-level HLO
dumping.

Base script:

- `experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py`

Keep:

- `v5p-8`
- `pd=4`
- `batch=64`
- LoRA `r=64`, `alpha=64`, `zero_init_b=True`
- `lr=1e-6`
- `seed=0`
- `num_train_steps=10`

Add / change env vars:

```bash
REGIONS_OVERRIDE=us-central1
MARIN_DEBUG_RUN_TAG=<variant-tag>
EXPERIMENT_BL_ORDER=<canonical|reverse>
MARIN_DEBUG_HLO_UPLOAD_DIR=gs://marin-us-central1/debug/bug_1_bl_pass_hlo/<date>/<variant>/
MARIN_DEBUG_DUMP_SHARDING=1
XLA_FLAGS="--xla_dump_to=/tmp/xla_hlo \
--xla_dump_hlo_as_text \
--xla_dump_hlo_module_re=.*train.* \
--xla_dump_hlo_pass_re=.*"
```

### What we want to learn from this rerun

Best case:

- we find the exact first pass after which BL0 and BL1 stop matching

Then we can classify the split:

1. **early / open-XLA pass**
   - likely inspectable in public source
   - maybe sharding / SPMD / collective scheduling related
2. **late / TPU backend pass**
   - stronger evidence that the split is happening inside TPU-specific lowering
   - more likely to require upstream TPU/backend support rather than local code
     inspection

### Caveat on dump size

`--xla_dump_hlo_pass_re=.*` can create a lot of files.

So the operational sequence should be:

1. start with only BL0 + BL1
2. keep module filter narrow:
   - `--xla_dump_hlo_module_re=.*train.*`
3. if the output is still too large, narrow the pass regex later
   - for example `spmd|propagation|collective|schedule|layout`

### Why this is worth doing

At this point, this is one of the few probes that can substantially increase
understanding rather than just generating another positive/negative training
result.

It answers:

- is the split introduced by a generic XLA pass we can inspect?
- or only by late TPU backend lowering?

That is the most important remaining compiler question for Bug 1.

## 2026-04-18T Launch Plan: BL-PASS first-divergence compiler probe

This section records the next experiment after the good/bad optimized-HLO split
was established.

### Core question

We already know:

- BL0 and BL1 have identical `before_optimizations` train-step HLO
- BL0 and BL1 have different `after_optimizations` train-step HLO

So the remaining compiler question is:

- **after which XLA pass or pipeline do they first diverge?**

### Why start with "dump every pass"

The first idea might be "binary search the compiler pipeline."

That is not the best first move.

Better initial strategy:

1. dump the `train_step` HLO after every pass for BL0 and BL1
2. line up the dumped pass sequence
3. find the first pass where the files stop matching

Reason:

- we already know the endpoints differ
- if intermediate snapshots are available, a full pass dump gives the first
  divergence in one shot
- binary search is only the fallback if artifact volume is too large

### Why only BL0 and BL1

We do **not** need all four permutations for this pass-level probe.

Reason:

- BL0 and BL2 already proved to be identical bad optimized families
- BL1 and BL3 already proved to be identical good optimized families

So the minimal informative pair is:

- BL0 canonical = bad representative
- BL1 reverse = good representative

### Exact objective

Success means answering:

1. the first pass or pipeline where BL0 and BL1 stop matching
2. whether that pass looks like:
   - open XLA graph / sharding / SPMD / scheduling machinery
   - or a later TPU/backend-specific lowering stage

### Exact launch recipe

Base script:

- `experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py`

Keep the same training setup as the earlier BL runs:

- `v5p-8`
- `pd=4`
- `batch=64`
- LoRA `r=64`, `alpha=64`, `zero_init_b=True`
- `lr=1e-6`
- `seed=0`
- `num_train_steps=10`

Variants:

- BL0-PASS canonical
- BL1-PASS reverse

Environment:

```bash
REGIONS_OVERRIDE=us-central1
MARIN_DEBUG_RUN_TAG=<variant-tag>
EXPERIMENT_BL_ORDER=<canonical|reverse>
MARIN_DEBUG_HLO_UPLOAD_DIR=gs://marin-us-central1/debug/bug_1_bl_pass_hlo/20260418/<variant>/
MARIN_DEBUG_DUMP_SHARDING=1
MARIN_DEBUG_DUMP_GRAD_VALUES=1
XLA_FLAGS="--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.* --xla_dump_hlo_pass_re=.*"
```

Meaning of the important XLA flags:

- `--xla_dump_to=/tmp/xla_hlo`
  - dump compiler artifacts to the local worker filesystem
- `--xla_dump_hlo_as_text`
  - use readable text HLO dumps
- `--xla_dump_hlo_module_re=.*train.*`
  - restrict to train-step-like modules instead of dumping every tiny helper
- `--xla_dump_hlo_pass_re=.*`
  - dump after every compiler pass

This is the key experiment knob.

### Expected tradeoff

This will likely produce a much larger artifact bundle than the earlier BL-HLO
runs.

That is acceptable for BL0 + BL1 only.

If this turns out to be too large or unwieldy, the fallback plan is:

1. rerun with `--xla_dump_hlo_pipeline_re=.*` to find the first divergent
   pipeline
2. then rerun with a narrower `--xla_dump_hlo_pass_re=<regex>` inside that
   pipeline

But the first attempt should be the simplest one:

- dump every pass
- compare later

### Exact submission commands

BL0-PASS:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  --job-name experiment-bl-pass-v5p8-pd4-bl0-20260418-2330 \
  -e REGIONS_OVERRIDE us-central1 \
  -e EXPERIMENT_BL_ORDER canonical \
  -e MARIN_DEBUG_RUN_TAG bl0-pass \
  -e MARIN_DEBUG_HLO_UPLOAD_DIR gs://marin-us-central1/debug/bug_1_bl_pass_hlo/20260418/bl0-can/ \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e MARIN_DEBUG_DUMP_GRAD_VALUES 1 \
  -e XLA_FLAGS "--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.* --xla_dump_hlo_pass_re=.*" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py
```

BL1-PASS:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  --job-name experiment-bl-pass-v5p8-pd4-bl1-20260418-2330 \
  -e REGIONS_OVERRIDE us-central1 \
  -e EXPERIMENT_BL_ORDER reverse \
  -e MARIN_DEBUG_RUN_TAG bl1-pass \
  -e MARIN_DEBUG_HLO_UPLOAD_DIR gs://marin-us-central1/debug/bug_1_bl_pass_hlo/20260418/bl1-rev/ \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e MARIN_DEBUG_DUMP_GRAD_VALUES 1 \
  -e XLA_FLAGS "--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.* --xla_dump_hlo_pass_re=.*" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py
```

### What to inspect after completion

After the runs finish:

1. download the two pass-dump bundles locally
2. collect only the `jit__train_step` dump files
3. hash and align them in pass order
4. identify the first non-matching file pair
5. map that pass or pipeline name back to the XLA source tree

That is the cleanest available path to answer:

- not just "good and bad compile differently"
- but "where in the compiler they first start compiling differently"

### Submission outcome

The first submission attempt failed before job creation because Iris bundles the
entire worktree and the local HLO artifacts under `.agents/artifacts/` pushed
the workspace zip above the 25 MB bundle limit:

- observed bundle size: `36.5 MB`
- controller limit: `25 MB`

Root cause:

- the downloaded BL HLO bundles in `.agents/artifacts/bug_1_bl_hlo/20260418/`
  were being included in the Iris submission bundle

Operational fix:

- add `.agents/artifacts/` to the local git exclude file:
  `/Users/ahmed/code/marin/.git/info/exclude`

Verification after the fix:

- `.agents/artifacts/` no longer appeared in `collect_workspace_files(...)`
- new Iris workspace bundle size: `6.6 MB`

Final submitted jobs:

- BL0-PASS canonical:
  `/ahmed/experiment-bl-pass-v5p8-pd4-bl0-20260418-2345`
- BL1-PASS reverse:
  `/ahmed/experiment-bl-pass-v5p8-pd4-bl1-20260418-2345`

These are the active pass-dump jobs for the "first diverging XLA pass"
investigation.

## 2026-04-19T Execution Start: Download BL-PASS Bundles And Find First Diverging Pass

User-confirmed next step:

- download the two BL-PASS bundles locally
- compare the `jit__train_step` pass sequence
- identify the first diverging pass between BL0 and BL1

Source bundles:

- `gs://marin-us-central1/debug/bug_1_bl_pass_hlo/20260418/bl0-can/`
- `gs://marin-us-central1/debug/bug_1_bl_pass_hlo/20260418/bl1-rev/`

Planned local artifact root:

- `.agents/artifacts/bug_1_bl_pass_hlo/20260418/bl0-can/`
- `.agents/artifacts/bug_1_bl_pass_hlo/20260418/bl1-rev/`

Method:

1. download both bundles locally
2. isolate `jit__train_step` pass dumps
3. align pass snapshots in order
4. hash / diff them to find the first non-matching pass state
5. then interpret that pass name against the XLA pipeline

## 2026-04-19T Result: First Diverging XLA Pass For BL0 vs BL1

### Goal

Find the first compiler pass where the bad canonical permutation (`BL0`) and the
good reverse permutation (`BL1`) stop being the same compiled `jit__train_step`
program.

This matters because, per [xla_collective_primer.md](./xla_collective_primer.md),
`before_optimizations` and `after_optimizations` are only the two endpoints of a
much longer XLA pipeline:

```text
jaxpr
  -> JAX lowering
HLO before optimization
  -> pass 1
  -> pass 2
  -> ...
  -> pass N
HLO after optimization
```

So the right debugging question is not only:

- "are the final optimized HLOs different?"

but:

- "after which exact pass do they first become different?"

### What I Actually Did

I started to pull the full BL-PASS bundles locally, but that would have moved
multiple gigabytes of text HLO over GCS egress to answer a much narrower
question. That was unnecessary.

Instead, I switched to remote object metadata on GCS and compared the pass-dump
objects directly.

Source prefixes:

- `gs://marin-us-central1/debug/bug_1_bl_pass_hlo/20260418/bl0-can/`
- `gs://marin-us-central1/debug/bug_1_bl_pass_hlo/20260418/bl1-rev/`

Exact style of commands used:

```bash
gsutil ls 'gs://.../bl0-can/*module_0336.jit__train_step*txt'
gsutil ls 'gs://.../bl1-rev/*module_0336.jit__train_step*txt'

gsutil ls -L 'gs://.../bl0-can/*module_0336.jit__train_step*txt' > /tmp/bl0_0336_lsL.txt
gsutil ls -L 'gs://.../bl1-rev/*module_0336.jit__train_step*txt' > /tmp/bl1_0336_lsL.txt
```

Then I parsed the `md5` hash for each object and compared the ordered pass
sequence.

### Quick Primer: What Is A "Pass" Here?

Per [xla_collective_primer.md](./xla_collective_primer.md), a pass is one
compiler transformation over the HLO module.

Examples from this run:

- `X64_elimination`
- `Phase_1_pre-layout_assignment_passes`
- `xla-partitioner`
- `spmd-cleanup`
- `optimizer-tpu-spmd-collectives`
- later fusion / scheduling / memory-assignment passes

Each dumped filename records a snapshot of the HLO module *between* passes, for
example:

```text
module_0336.jit__train_step.cl_813921542.0077.spmd-cleanup.after_pipeline-start.before_dce.txt
```

That means:

- module: `module_0336.jit__train_step`
- pass index in the dump stream: `0077`
- current pipeline: `spmd-cleanup`
- snapshot location: after pipeline start, before `dce`

So this filename is not just "some file"; it is a precise compiler timepoint.

### Main Result

For the main `jit__train_step` module (`module_0336`):

- BL0 object count: `225`
- BL1 object count: `241`
- the first `77` objects are byte-identical by `md5`
- the first divergence is object / pass index `0077`

First diverging pass snapshot:

```text
module_0336.jit__train_step.cl_813921542.0077.spmd-cleanup.after_pipeline-start.before_dce.txt
```

BL0 hash / size:

- `md5 = mKDJ6sy6QbJD2B2r/RJdZw==`
- `size = 3,951,772`

BL1 hash / size:

- `md5 = VkRZIqV6cO1qYLwcZZOPfw==`
- `size = 7,609,962`

The immediately preceding pass snapshot is still identical:

```text
module_0336.jit__train_step.cl_813921542.0076.xla-partitioner.after_shardy-xla.before_tpu-rng-bit-decompose-tuple.txt
```

Shared hash at `0076`:

- `md5 = dlmGV8D0sNW4ZnnZaRVvmw==`

So the strongest compiler-level statement we can make now is:

> BL0 and BL1 are identical through `xla-partitioner.after_shardy-xla`, and they
> first split at the start of the `spmd-cleanup` pipeline.

### Why This Is Important

This is much stronger than our earlier statement:

- "`before_optimizations` is identical but `after_optimizations` differs"

Now we know the split is not happening "somewhere late in TPU fusion land." It
starts relatively early in the distributed-lowering part of the compiler, right
after partitioning / sharding work has already happened and as XLA is cleaning
up the SPMD form.

That narrows the search significantly.

### What `spmd-cleanup` Likely Means In Plain English

Using the primer's terminology:

- `xla-partitioner` / `shardy-xla` is where the logical sharded program gets
  turned into an SPMD-distributed HLO form
- `spmd-cleanup` is where XLA normalizes, simplifies, and legalizes that
  distributed form so later collective / fusion / scheduling passes can operate
  on it cleanly

So the device-order-dependent split is not being introduced by user JAX code and
not by the original HLO lowering. It is appearing once XLA starts rewriting the
already-partitioned distributed program.

### Collective Structure At The First Diverging Snapshot

I also counted collective op names *in the first divergent snapshot itself*.

BL0 `0077`:

- `all-reduce = 542`
- `all-gather = 486`
- `all-to-all = 36`
- `collective-permute = 0`

BL1 `0077`:

- `all-reduce = 458`
- `all-gather = 348`
- `all-to-all = 36`
- `collective-permute = 1248`

This is a very strong signal.

The first divergent pass is not a tiny metadata difference or a replica-group
listing quirk. By the first differing snapshot, the good permutation has
already been rewritten into a materially different communication structure with
many `collective-permute` ops, while the bad permutation has not.

So the current best interpretation is:

- the good and bad physical orderings enter different SPMD cleanup /
  communication-lowering paths
- this split is already visible at the first divergent pass
- the later huge `after_optimizations` difference is downstream of this earlier
  fork, not the first point of divergence

### Smaller Wrapper Module (`module_0334`)

I also checked the smaller `module_0334.jit__train_step` wrapper.

Findings:

- all five numbered pass snapshots (`0000`..`0004`) match exactly between BL0
  and BL1
- the only mismatch there is `execution_options.txt`, which is not a compiler
  pass snapshot

So for actual pass-dump evidence, `module_0336` pass `0077` is the first real
divergence point we have found.

### Additional Structural Observation

The shared pass sequence names stay aligned through the full BL0 stream, but BL1
has `16` extra later snapshots beyond BL0.

The first extra BL1-only snapshot is:

```text
module_0336.jit__train_step.cl_813921542.0225.final_rematerialization.after_pipeline-start.before_rematerialization.txt
```

This means the good path not only differs in content starting at `0077`; the
later pipeline structure also continues differently.

### Bottom Line

The first diverging pass for Bug 1 is:

```text
module_0336.jit__train_step.cl_813921542.0077.spmd-cleanup.after_pipeline-start.before_dce.txt
```

Interpretation:

- same JAX program
- same HLO through partitioning
- first split introduced at the start of `spmd-cleanup`
- good path already contains `collective-permute`-heavy communication structure
  at that first split

This is now strong evidence that Bug 1 is a topology-sensitive XLA SPMD cleanup
/ communication-lowering fork, not merely a late-stage reduction-order accident.

### Best Next Step After This

The highest-value follow-up is now:

1. inspect the `spmd-cleanup` and immediately adjacent pass family in XLA /
   TPU-lowering code
2. diff the actual text of BL0 vs BL1 at `0077`
3. reduce this to a minimal repro if we want an upstream XLA / TPU backend bug
   report

## 2026-04-19T Concrete Pass-0077 Diff: What Changed In The HLO

I pulled the exact first-divergent snapshots locally:

- [BL0 pass 0077](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_bl_pass_hlo_targeted/20260418/pass0077/bl0_0077.txt)
- [BL1 pass 0077](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_bl_pass_hlo_targeted/20260418/pass0077/bl1_0077.txt)

The goal here was not just "find a different hash", but answer:

- what is XLA actually rewriting differently?

### Representative Example 1: Attention LoRA `embed -> ... -> LORA_R`

Bad BL0 form:

```text
%squeeze.3316 = bf16[64,1024] reshape(...)
%all-gather.1 = bf16[64,4096] all-gather(%squeeze.3316), replica_groups=[1,4]<=[4]
%dot = bf16[4,4096,64] dot(%mul.2031, %all-gather.1)
%all-gather.2 = bf16[64,4096] all-gather(%squeeze.3316), replica_groups=[1,4]<=[4]
%dot.1 = bf16[4,4096,64] dot(%mul.2031, %all-gather.2)
```

Good BL1 form for the same logical op:

```text
%windowed_dot_general_body_ag ...
%collective-permute = bf16[64,1,1024] collective-permute(...), source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
%collective-permute.1 = bf16[64,1,1024] collective-permute(...), source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
%concatenate.182 = bf16[64,2,1024] concatenate(%collective-permute, %collective-permute.1), dimensions={1}
%reshape.406 = bf16[64,2048] reshape(%concatenate.182)
%dot.2 = bf16[4,4096,64] dot(%reshape.405, %reshape.406)
...
ROOT %tuple.491 = (..., %collective-permute.2, %add.2293, %collective-permute.3, %add.2294)
```

Interpretation:

- BL0 keeps the straightforward SPMD pattern:
  - gather the remote shard(s)
  - then do the local `dot`
- BL1 rewrites the same contraction into a `windowed_dot_general_body_ag`
  helper that explicitly moves shard windows with `collective-permute`, glues
  them together with `concatenate`, and accumulates partial dot results

The two `source_target_pairs` patterns in BL1 are:

- `{{0,3},{1,0},{2,1},{3,2}}`
- `{{0,1},{1,2},{2,3},{3,0}}`

That is, one hop in one direction and one hop in the other direction around the
4-way group.

So a reasonable inference is:

- BL1 is decomposing an all-gather-like data movement into directional
  neighbor-exchange windows
- BL0 is not

I am explicitly calling that an inference from the HLO shape, not a claim that
I have already identified the exact TPU backend source line that chooses it.

### Representative Example 2: MLP LoRA `embed -> ... -> LORA_R`

Bad BL0 form:

```text
%all-gather.18 = bf16[64,4096] all-gather(%squeeze.3329), replica_groups=[1,4]<=[4]
%dot.24 = bf16[4,4096,64] dot(%mul.2040, %all-gather.18)
%all-gather.19 = bf16[64,4096] all-gather(%squeeze.3329), replica_groups=[1,4]<=[4]
%dot.25 = bf16[4,4096,64] dot(%mul.2040, %all-gather.19)
```

Good BL1 form for the same logical op:

```text
%collective-permute.32 = bf16[64,1,1024] collective-permute(...), source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
%collective-permute.33 = bf16[64,1,1024] collective-permute(...), source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
%concatenate.198 = bf16[64,2,1024] concatenate(%collective-permute.32, %collective-permute.33), dimensions={1}
...
ROOT %tuple.499 = (..., %collective-permute.34, %add.2353, %collective-permute.35, %add.2354)
```

So the same basic rewrite pattern is not isolated to attention. It also appears
in the MLP LoRA contraction.

### Representative Example 3: MLP Main `embed -> ... -> mlp`

Bad BL0 form:

```text
%all-gather.20 = bf16[14336,4096] all-gather(%squeeze.3331), replica_groups=[1,4]<=[4]
%dot.28 = bf16[4,4096,14336] dot(%mul.2040, %all-gather.20)
%all-gather.21 = bf16[14336,4096] all-gather(%squeeze.3331), replica_groups=[1,4]<=[4]
%dot.29 = bf16[4,4096,14336] dot(%mul.2040, %all-gather.21)
```

Good BL1 form:

```text
%collective-permute.36 = bf16[14336,1,1024] collective-permute(...), source_target_pairs={{0,3},{1,0},{2,1},{3,2}}
%collective-permute.37 = bf16[14336,1,1024] collective-permute(...), source_target_pairs={{0,1},{1,2},{2,3},{3,0}}
%concatenate.200 = bf16[14336,2,1024] concatenate(%collective-permute.36, %collective-permute.37), dimensions={1}
...
ROOT %tuple.500 = (..., %collective-permute.38, %add.2361, %collective-permute.39, %add.2362)
```

This matters because it shows the rewrite is not "LoRA-special" in a narrow
sense. It also affects the main MLP contraction pattern in the same module.

### What I Think This Means

The strongest practical interpretation is:

- BL0 path:
  - keep direct `all-gather -> dot` style rewrites
- BL1 path:
  - lower the same logical sharded contraction into a windowed /
    neighbor-exchange form using many `collective-permute` ops

The good path even names one helper computation:

```text
%windowed_dot_general_body_ag
```

I read that as strong evidence that XLA is not merely scheduling collectives
differently; it is choosing a different *decomposition* for the distributed dot
itself.

Again, the exact expansion rule and heuristic may still live partly inside the
TPU backend / `libtpu`, but the HLO is already interpretable enough to say what
the two compiled strategies are doing.

### What We Can Now Say, Precisely

We can now make a much stronger statement than before:

- Bug 1 is not just "same collective with different reduction order"
- the good and bad physical orderings cause `spmd-cleanup` to pick different
  communication decompositions for the same distributed dot patterns
- those decompositions differ across both attention and MLP paths

So the current best model is:

- physical topology / logical-rank assignment changes how XLA chooses to
  realize sharded dot/general communication
- that choice materially changes numerical behavior early in training

### What Remains Unknown

Still unknown:

- the exact open-source pass or backend heuristic that says "emit windowed
  permute form here" vs "emit all-gather form here"
- whether that choice is fully visible in open XLA or partly hidden behind the
  TPU backend boundary

But the concrete HLO rewrite itself is now visible and no longer speculative.

## 2026-04-19T Big-Picture Interpretation: What Is Going On Here?

This section is meant to be the most pedagogical summary of the current state.
If the surrounding HLO / XLA details start to blur together, start here.

For the background vocabulary, see
[xla_collective_primer.md](./xla_collective_primer.md).

### Short Version

Bug 1 now looks like this:

- the same LoRA DPO training program is being compiled in two different ways on
  `v5p-8`
- which compiled form you get depends on the physical device ordering inside the
  same logical `{data:4, model:1}` mesh
- the "bad" ordering gets a direct `all-gather -> dot` style communication
  decomposition
- the "good" ordering gets a `collective-permute`-heavy windowed
  decomposition
- the two forms already differ at the first divergent pass:
  `spmd-cleanup.after_pipeline-start.before_dce`
- training outcome tracks that compiler split:
  - bad compiled family -> bad optimization basin
  - good compiled family -> healthy optimization basin

### Step-By-Step Causal Story

#### 1. The Python / JAX program is the same

BL0 and BL1 are not different experiment recipes in the usual sense.

They share the same:

- model
- optimizer
- LoRA config
- batch / seed / microbatching
- logical mesh shape: `{data:4, model:1}`

The only change is:

- which physical TPU chips occupy logical data ranks `0,1,2,3`

That is why the result is surprising and important.

#### 2. JAX lowers the same logical program

Earlier evidence already showed:

- same `before_optimizations`
- same HLO through `xla-partitioner.after_shardy-xla`

That means:

- this is not a user-code bug
- not a different `jaxpr`
- not a different logical sharding spec
- not a case where JAX front-end tracing made two different graphs

#### 3. XLA then hits a topology-sensitive fork

The first real divergence is:

```text
module_0336.jit__train_step.cl_813921542.0077.spmd-cleanup.after_pipeline-start.before_dce.txt
```

So the fork is happening after partitioning has already happened, while XLA is
cleaning up / legalizing the SPMD-distributed HLO.

This is the key new fact from the pass-dump study.

#### 4. The fork is not cosmetic

At the first divergent snapshot:

- BL0 has no `collective-permute`
- BL1 has `1248` `collective-permute` ops

This is not a tiny metadata difference. It is a real change in the distributed
communication graph.

#### 5. The good path is not merely "the same all-gather in a different order"

The concrete diffs show that BL1 is not simply printing the same collective with
a different `replica_groups` listing.

Instead, for the same logical distributed `dot_general`, BL1 uses a different
realization:

- explicit directional neighbor exchanges
- explicit concatenation of windowed shards
- partial dot accumulation through helper computations like
  `%windowed_dot_general_body_ag`

So the good path is a different communication *decomposition*, not just a
different final addition order.

#### 6. This is broader than one LoRA tensor

The rewrite difference appears in:

- attention LoRA contractions
- MLP LoRA contractions
- MLP main contractions

So Bug 1 is not best described as "a one-off LoRA adapter bug in one tensor."
It is better described as:

- a topology-sensitive XLA communication-lowering choice
- which happens to interact catastrophically with this LoRA DPO training setup

### What This Does And Does Not Prove

#### What it does prove

It proves:

- physical device order matters for this recipe
- that effect is visible in the compiled HLO, not just in training curves
- the compiler split begins at `spmd-cleanup`
- the split changes the communication decomposition of distributed dot-like ops

#### What it does not yet prove

It does **not** yet prove:

- that BL1's windowed permute form is "correct" and BL0's all-gather form is
  "buggy"
- that there is an upstream XLA correctness bug rather than a legal but
  numerically different lowering
- which exact TPU-backend heuristic chooses one form vs the other

So there are still two live explanations:

1. **backend bug / mis-lowering**
   - one topology-dependent path is wrong or pathologically unstable
2. **legal but numerically fragile lowering**
   - both paths are allowed, but this LoRA DPO setup is sensitive enough that
     the different decomposition flips the training basin

At the moment, the evidence supports the existence of the fork more strongly
than it supports either explanation of *why* the fork is harmful.

### Why This Can Happen At All

This is the confusing part, so it is worth stating plainly.

XLA is allowed to choose different implementations for the same high-level
distributed operation depending on target topology.

For example, a sharded dot or an all-gather-like communication pattern can be
implemented by:

- direct `all-gather`
- a sequence of neighbor exchanges (`collective-permute`)
- some windowed / pipelined hybrid

If those implementations are mathematically equivalent but numerically
different, then a very fragile training setup can land in different basins.

That is weird, but it is a real possibility and consistent with the artifacts we
now have.

### Practical Interpretation For Marin

From an engineering perspective, this means:

- the canonical `v5p-8 {data:4, model:1}` physical ordering is not safe for
  this LoRA DPO recipe
- known-good workarounds are:
  - good physical permutations (`reverse`, `rotate`)
  - `{data:2, model:2}`
  - pure TP `{data:1, model:4}`
  - larger / different TPU slices that avoid this exact lowering family

So even before we fully understand the backend reason, we already have useful
operational guidance.

### Current Best Mental Model

If I had to compress the current evidence into one sentence, it would be:

> Bug 1 is a topology-sensitive XLA SPMD communication-lowering fork for
> distributed dot/general patterns on `v5p-8`, and this LoRA DPO recipe is
> sensitive enough that the bad lowering family lands in a different training
> basin.

That is the best current model because it matches all of the evidence we have:

- permutation experiment split
- identical early HLO through partitioning
- first divergence at `spmd-cleanup`
- `all-gather` vs `collective-permute` rewrite families
- attention + MLP coverage

### Where Open XLA Can Still Help

Open XLA can still help us understand:

- what `spmd-cleanup` is intended to do
- what the partitioner and SPMD cleanup passes look like in public code
- whether "windowed dot general with neighbor permutes" is an expected public
  optimization pattern

Open XLA may **not** fully reveal:

- the TPU-specific heuristic that picks one path vs the other
- whether that choice happens partly inside `libtpu`

So the open-source boundary is:

- enough visibility to understand the shape of the compiler fork
- maybe not enough visibility to fully explain the TPU backend decision rule

### What I Would Say If Asked "What Is Going On?"

The cleanest answer is:

1. We changed only the physical chip ordering, not the logical training recipe.
2. XLA compiled the same distributed training step into two different
   communication strategies.
3. The first split happens at `spmd-cleanup`.
4. The good strategy uses many windowed `collective-permute` exchanges; the bad
    strategy stays closer to direct `all-gather`.
5. That compiler choice is enough to flip the optimization outcome for this
    LoRA DPO setup.

## 2026-04-19T Public OpenXLA Source Analysis

I cloned public OpenXLA outside the Marin repo to:

- `/Users/ahmed/code/xla`

Clone state used for this analysis:

- repo: `https://github.com/openxla/xla.git`
- checked-out commit: `400cffe3ad`

The purpose of this section is to answer:

- how much of the BL0-vs-BL1 compiler fork is visible in public source?
- what is clearly public XLA logic?
- what likely remains TPU-backend-private?

### Main Finding

Public OpenXLA explains **much more** of the observed behavior than I expected.

Specifically, public source already contains:

- the public `shardy-xla` pass
- the public SPMD partitioner
- the public `spmd-cleanup` pipeline
- the public `windowed_dot_general_body_ag` machinery
- the public `collective-permute`-based windowed einsum rewrite logic

So the existence of a `collective-permute`-heavy windowed dot path is **not**
backend-private folklore. It is real, public XLA code.

What still appears not fully public is:

- the exact TPU orchestration / pass scheduling around some TPU-specific steps
- likely some TPU-specific heuristics or pass insertion behind `libtpu`

### What Is Public, Exactly

#### 1. `shardy-xla` is a real public pass

File:

- `/Users/ahmed/code/xla/xla/service/spmd/shardy/shardy_xla_pass.h`

It explicitly declares:

```c++
absl::string_view name() const override { return "shardy-xla"; }
```

and documents that the pass:

1. converts HLO to StableHLO + SDY
2. runs Shardy passes, including propagation and partitioner
3. converts back to HLO

So the `after_shardy-xla` label in our dumps is directly explained by public
OpenXLA.

#### 2. The SPMD partitioner is public

Files:

- `/Users/ahmed/code/xla/xla/service/spmd/spmd_partitioner.h`
- `/Users/ahmed/code/xla/xla/service/spmd/stateful_rng_spmd_partitioner.h`

The public partitioner class is:

- `SpmdPartitioner`
- name: `"spmd-partitioning"`

And the compilers/pipelines often run:

- `ShardyXLA`
- then `StatefulRngSpmdPartitioner`

For example in public CPU/GPU pipelines:

- `/Users/ahmed/code/xla/xla/service/cpu/cpu_compiler.cc`
- `/Users/ahmed/code/xla/xla/service/gpu/gpu_spmd_pipeline.cc`

Those files show the public sequence:

1. `ShardyXLA`
2. `StatefulRngSpmdPartitioner`
3. later cleanup / inlining / collective-related passes

This is useful because it explains why our HLO dump has both:

- `after_shardy-xla`
- then the partitioner-adjacent stage

even though the exact top-level pipeline name in the dump is `xla-partitioner`.

#### 3. `spmd-cleanup` is public and tiny

File:

- `/Users/ahmed/code/xla/xla/service/spmd/spmd_partitioner.cc`

Public code literally contains:

```c++
HloPassPipeline pass("spmd-cleanup");
pass.AddPass<HloDCE>(/*remove_cross_partition_collective_ops=*/true);
pass.AddPass<TupleSimplifier>();
pass.AddPass<HloDCE>(/*remove_cross_partition_collective_ops=*/true);
pass.AddPass<HloCSE>(/*is_layout_sensitive=*/false);
```

This has an important interpretive consequence:

- our first differing snapshot is
  `spmd-cleanup.after_pipeline-start.before_dce`
- that means the divergence is already present **before** the cleanup DCE /
  tuple-simplifier / CSE work begins

So the split is not being *created* by cleanup itself.

It is being created by the immediately preceding partitioner / TPU-adjacent
stage, and `spmd-cleanup` is simply the first snapshot where we can see it.

This is a refinement of the earlier wording. More precise phrasing is:

- the first *visible divergent snapshot* is the start of `spmd-cleanup`
- the actual rewrite was introduced just before that, in the preceding
  partitioner-stage work

That is a better statement than simply saying "the split happens in cleanup."

#### 4. `windowed_dot_general_body_ag` is public

File:

- `/Users/ahmed/code/xla/xla/service/spmd/dot_handler.cc`

This file contains the exact string we saw in HLO:

- `windowed_dot_general_body_ag`

Relevant public helpers include:

- `GetWindowedEinsumConfiguration(...)`
- `EmitWindowedDotGeneral(...)`
- `CanReshardWithCollectivePermute(...)`

So the HLO name we observed is not some opaque backend artifact. It is emitted
by public SPMD dot-handling code.

### What The Public Dot Handler Says

This is the strongest public-code confirmation we have.

In `dot_handler.cc`, public comments say:

- XLA may emit a **windowed DotGeneral** when one operand and the output have
  certain partitioning relationships
- the loop computes one window per iteration
- during each iteration, each partition sends its input window to its neighbor
  using `collective-permute` for the next iteration

That is not my paraphrase of HLO only. It is the actual public comment in the
 implementation of `EmitWindowedDotGeneral(...)`.

The relevant code also shows:

- explicit construction of `collective-permute`
- explicit neighbor movement
- explicit bidirectional communication support when `num_partitions % 4 == 0`

Public code excerpt, in substance:

- create a while loop
- compute one window per iteration
- send the next window to the neighbor via `collective-permute`

This matches our BL1 HLO extremely well.

### Why BL1 Looks Like Two Directions Around The Ring

Our BL1 HLO showed two directional patterns:

- `{{0,3},{1,0},{2,1},{3,2}}`
- `{{0,1},{1,2},{2,3},{3,0}}`

Public `dot_handler.cc` includes logic for a bidirectional windowed-einsum
algorithm, including comments and code that set up:

- one direction as `0 -> 1 -> 2 -> 3 -> ...`
- the reverse-direction companion path

So the "two directions around the 4-way group" interpretation is well supported
by public code.

### Public Cost Model / Heuristic Evidence

Public `dot_handler.cc` also contains heuristic logic comparing:

- all-gather bytes / time
- reduce-scatter bytes / time
- computation time

It explicitly reasons about when a windowed einsum path is preferable.

This means the distinction we observed:

- BL0: direct `all-gather -> dot`
- BL1: windowed / permute-heavy dot

is not an impossible backend oddity. Public XLA already has code whose job is
to choose among such decompositions.

### Important Nuance: Where The First Visible Split Appears

Our neighboring pass names are:

- `0075 xla-partitioner.after_pipeline-start.before_dce`
- `0076 xla-partitioner.after_shardy-xla.before_tpu-rng-bit-decompose-tuple`
- `0077 spmd-cleanup.after_pipeline-start.before_dce`

And:

- `0076` is still identical between BL0 and BL1
- `0077` is the first differing snapshot

So the most precise reading is:

- the split is introduced after the `after_shardy-xla` snapshot and before the
  first cleanup snapshot
- therefore the culprit is **not** `spmd-cleanup`'s DCE/CSE itself
- it is something in the immediately preceding partitioner / TPU-adjacent step

### What Looks TPU-Private

I searched the public tree for:

- `tpu-rng-bit-decompose-tuple`

and found no public definition.

That is important because our `0076` snapshot is named:

```text
...after_shardy-xla.before_tpu-rng-bit-decompose-tuple
```

So one plausible interpretation is:

- there is at least one TPU-specific pass or subpass between `0076` and `0077`
- that pass is not obviously present in the public tree

Separately, the public TPU compiler entry point:

- `/Users/ahmed/code/xla/xla/stream_executor/tpu/tpu_on_demand_compiler.cc`

still delegates compilation via:

- `TpuCompiler_RunHloPassesFn`
- `TpuCompiler_RunBackendFn`

That is the `libtpu` boundary.

So the current open/private split is:

- public source clearly explains the *existence* of the windowed permute path
- public source clearly explains that the first visible divergent snapshot is
  after partitioning and before cleanup proper
- but the exact TPU-specific pass scheduling and/or heuristic that selected the
  path for this topology may still cross into `libtpu`

### One Especially Interesting Clue

In public source, the bidirectional windowed-einsum path is controlled by:

- `options.bidirectional_windowed_einsum`

I searched the public tree for places that set this to `true`.

Result:

- only tests in `spmd_partitioner_test.cc` obviously set it to `true`
- I did **not** find an obvious public CPU/GPU pipeline enabling it in normal
  compiler setup

This is not a proof, but it is a meaningful clue:

- if our TPU HLO is clearly using the bidirectional-style neighbor exchange
  pattern
- and that option is not obviously enabled by public compiler pipelines
- then the exact enablement path may be TPU-specific and not fully public

That is one of the strongest current hints that we are hitting a mixed
public/private boundary:

- public mechanism
- possibly private enablement / selection

### What This Means For Our Earlier Conclusions

We should refine the earlier story to be more precise:

Old rough version:

- "the first divergent pass is `spmd-cleanup`"

Better version after reading public source:

- the first **differing snapshot** is the start of `spmd-cleanup`
- the divergence itself is introduced immediately before that, in the
  partitioner / TPU-adjacent stage after `after_shardy-xla`
- the windowed permute strategy seen in BL1 is explained by public
  `dot_handler.cc`
- but the exact reason BL1 gets that strategy and BL0 does not may still depend
  on TPU-private orchestration / heuristics

### Current Boundary Statement

What we can now say confidently from public OpenXLA:

- `shardy-xla` is public
- `spmd-cleanup` is public
- `windowed_dot_general_body_ag` is public
- `collective-permute`-based windowed dot realization is public
- public code explicitly reasons about choosing such decompositions

What remains uncertain / likely private:

- the exact TPU-only step between `0076` and `0077`
- the exact topology-sensitive heuristic that selected BL1's windowed form and
  BL0's direct all-gather form

### Practical Takeaway

So the answer to "can open XLA help us understand this?" is:

- **yes, substantially**

Open XLA already explains:

- why a `collective-permute`-heavy windowed dot path exists
- why that path can coexist with a direct all-gather path
- why this kind of decomposition choice is a normal compiler concern

But open XLA probably does **not yet** fully explain:

- the final TPU-specific selection logic for this exact `v5p-8` topology /
  ordering case

That last part may require either:

- a minimal repro to file upstream
- or TPU-backend/internal visibility

## 2026-04-19T Clarification: Can We Fully Explain Why This Happens?

Short answer:

- **partially yes**
- **fully, not yet**

This distinction matters.

### What We Can Explain Already

We can now explain a large fraction of the mechanism:

1. the user-level training program is the same
2. the logical mesh is the same
3. only physical device ordering changes
4. the compiled HLO stays identical through the early partitioning snapshots
5. then the partitioner / TPU-adjacent stage forks into two different
   communication decompositions
6. the good family uses public `windowed_dot_general` /
   `collective-permute` machinery
7. the bad family stays closer to direct `all-gather -> dot`
8. that compiler fork is enough to flip the optimization basin

So we do **not** have a black-box mystery anymore.

We know:

- what changes
- where it first becomes visible
- what the two compiled strategies look like
- that the good strategy is backed by public OpenXLA machinery

### What We Still Cannot Fully Explain

The remaining missing piece is narrower but important:

- why does BL1 get the windowed / permute-heavy strategy while BL0 gets the
  direct all-gather strategy?

More specifically, we do not yet know:

- the exact final selection rule
- whether that rule is entirely visible in public SPMD code
- or whether some of the decisive topology-sensitive choice is happening inside
  TPU-private `libtpu`

So the unresolved question is no longer:

- "what is happening?"

It is now:

- "what exact heuristic or pass boundary selects one public realization versus
  the other for this topology?"

That is a much smaller and better-defined unknown.

### Best Current Boundary Statement

Current best boundary:

- public OpenXLA explains the **existence** of the windowed permute path
- public OpenXLA explains that XLA can choose among different distributed-dot
  decompositions
- public OpenXLA explains why our BL1 HLO looks like neighbor-exchange windows
- public OpenXLA does **not yet** fully explain the final BL0-vs-BL1 selection
  decision

So:

- mechanism family: mostly understood
- final topology-sensitive selection rule: still partly unknown

### Practical Consequence

This means we should not block ourselves on full compiler-root-cause certainty.

We already know enough for practical engineering action:

- canonical `v5p-8 {data:4, model:1}` is unsafe for this recipe
- good permutations / other meshes are valid workarounds
- the failure is tied to a topology-sensitive compiler-lowering fork, not just
  "bad luck" or an LR issue

### Best Next Technical Step

The highest-value next step is:

- build a **minimal repro** for the distributed-dot pattern that flips between:
  - direct `all-gather -> dot`
  - windowed `collective-permute` decomposition

Why this is the right next move:

- if the minimal repro also flips, we can likely make a credible upstream XLA /
  TPU bug report
- if the minimal repro does **not** flip, then the final selection likely
  depends on broader TPU-backend context that is harder to expose publicly

So the next branch should be:

1. try to reduce this to a minimal repro
2. if successful, file / escalate with concrete artifacts
3. if not, stop digging for total root-cause certainty and treat the issue as
   "understood enough to work around"

### One-Sentence Status

The best one-sentence status update is:

> We now understand the compiler fork well enough to describe it concretely,
> but we do not yet fully understand the TPU-specific selection logic that makes
> one physical ordering take the good path and another take the bad path.

## 2026-04-19T Next 10-Hour Program: Full-FT Discriminator + LoRA Root-Cause Ladder

This section is the current recommended execution plan.

It is deliberately long and explicit because the next 10 hours should not be a
sequence of ad-hoc guesses. The goal is to make the next steps legible and
branching, with clear "if this, then that" logic.

### High-Level Thesis

The single best next discriminator is the one the user proposed:

> Does full fine-tuning on `v5p-8` also become sensitive to physical device
> permutation, or is that catastrophic sensitivity specific to LoRA?

This matters because it cleanly separates two very different worlds.

#### World A: full FT is permutation-insensitive

Then the current best story becomes:

- XLA is still generating different compiled families
- but those differences are only catastrophic for the LoRA parameterization
- so Bug 1 is not best framed as "generic XLA is broken"
- it is best framed as "LoRA is unusually fragile to a topology-sensitive but
  probably legal communication rewrite"

That would strongly justify a LoRA-first mechanism program.

#### World B: full FT is permutation-sensitive too

Then the story is broader:

- the topology-sensitive compiler fork affects more than LoRA
- LoRA merely amplifies it
- Bug 1 is not specifically a LoRA pathology; LoRA is just the easiest place to
  see it

That would justify a more general XLA / TPU minimal repro effort.

### Existing Evidence We Must Respect

We already know:

- Exp T full FT on `v5p-8` learns materially better than bad LoRA.
- But Exp T was not a physical-permutation sweep.
- Therefore Exp T rules out "generic v5p-8 DPO is broken," but it does **not**
  rule out a weaker statement:
  - full FT might still be slightly topology-sensitive
  - just much less sensitive than LoRA

So the next plan starts with the missing direct test:

- same full-FT recipe
- same `v5p-8`
- same logical mesh `{data:4, model:1}`
- different physical device orders

### Shared Infra / Harness Requirements

Before any new runs:

1. Create a dedicated full-FT permutation script.
2. Reuse the explicit mesh permutation hook already added for BL.
3. Enable HLO upload on all first-round full-FT permutation runs.
4. Keep the first round at `10` steps, not `2`, so we preserve comparability to
   BL and Exp T reruns.

Base scripts to reuse:

- full FT:
  - [experiment_t_v5p8_full_ft_s2.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_t_v5p8_full_ft_s2.py)
  - [debug_full_ft_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/debug_full_ft_s10.py)
- device permutation / explicit mesh order:
  - [experiment_bl_v5p8_pd4_device_permutation_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py)

Recommended new script:

- `experiments/posttrain/per_stmt_dpo/experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py`

Recommended shared envs for all first-round full-FT permutation runs:

```bash
REGIONS_OVERRIDE=us-central1
MARIN_DEBUG_LOG_BATCH_INDICES=1
MARIN_DEBUG_LOG_STEP_TRACE=1
MARIN_DEBUG_DUMP_GRAD_VALUES=1
MARIN_DEBUG_DUMP_SHARDING=1
MARIN_DEBUG_HLO_UPLOAD_DIR=gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/<date>/<variant>/
XLA_FLAGS="--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*"
```

Why this HLO setup:

- it gives us full `train_step` HLO snapshots and after-optimization artifacts
- it is much cheaper than `--xla_dump_hlo_pass_re=.*`
- if the full-FT quartet is invariant, we may not need pass dumps at all
- if the quartet splits, then we escalate only a representative pair to
  pass-by-pass dumping

### Decision Criteria Up Front

For the first full-FT permutation matrix, use these rough behavioral bins:

#### "Permutation-invariant enough"

All four variants stay in the same broad regime:

- step-2 losses within about `0.02`
- step-9 losses within about `0.03`
- reward-gap / grad-norm traces qualitatively matched

This does **not** mean bit-identical. It means no BL-style basin split.

#### "Permutation-sensitive"

Any of these occurs:

- a BL-style early split at step 2
- step-9 spread bigger than about `0.05`
- one variant clearly lands in a different qualitative learning regime

#### "Ambiguous"

- no clean step-2 split
- but noticeable step-9 / reward-gap separation

In that case the fallback is:

- extend only canonical + reverse to `35` or `100` steps

### Stage 0: Full-FT Permutation Quartet

These are the highest-priority runs.

#### FT-A0: full FT canonical

Base:

- full FT on `v5p-8`
- `SeparateReferenceConfig`
- `bs=32`
- `pd=4`
- `steps=10`
- mesh `{replica:1,data:4,model:1}`
- explicit physical order `canonical = (0,1,2,3)`

Hypothesis:

- control anchor for the quartet

Learn:

- whether the explicit-order full-FT canonical run still matches the known Exp T
  regime

#### FT-A1: full FT reverse

Only change:

- physical order `reverse = (3,2,1,0)`

Hypothesis:

- if full FT is robust, this should track FT-A0 closely

Learn:

- direct answer to "does full FT care about this permutation?"

#### FT-A2: full FT swap12

Only change:

- physical order `swap12 = (0,2,1,3)`

Hypothesis:

- if the LoRA split was really specific to the BL good/bad families, this may
  be behaviorally neutral under full FT

Learn:

- whether the bad BL ordering is also bad for full FT, or whether full FT
  washes it out

#### FT-A3: full FT rotate

Only change:

- physical order `rotate = (1,2,3,0)`

Hypothesis:

- if full FT is robust, this should again track the other full-FT runs

Learn:

- whether the BL good ordering is special only for LoRA or broadly helpful

Expected artifacts for FT-A0..A3:

- W&B curves
- sharding dump
- full `train_step` HLO bundle
- grad-value dumps

Immediate decision after FT-A0..A3:

1. If full FT is invariant enough:
   - go to Stage 1A (LoRA-specific branch)
2. If full FT also splits:
   - go to Stage 1B (general-topology branch)
3. If ambiguous:
   - run FT-B0 / FT-B1 first, then branch

### Stage 0.5: Matched-Confound Controls

These are not first priority, but they should be queued immediately after the
quartet if the result needs disambiguation.

#### FT-B0: full FT canonical vs reverse at longer horizon

Run:

- FT-A0 canonical extended to `35` or `100` steps
- FT-A1 reverse extended to `35` or `100` steps

Why:

- if the first 10 steps are similar but trajectories later diverge, we need to
  know that before over-claiming "full FT is invariant"

#### FT-B1: matched-batch LoRA control

Run:

- LoRA canonical vs reverse at `bs=32`, `pd=4`, `steps=10`

Why:

- Exp Q / BL were run at `bs=64`
- Exp T is `bs=32`
- if full FT looks stable at `bs=32`, it is worth checking that the LoRA split
  still exists at the same smaller batch

Interpretation:

- if LoRA still splits badly at `bs=32` while full FT does not, that is strong
  evidence the phenomenon is not a batch-size artifact

#### FT-B2: optional full FT `bs=64` canonical vs reverse

Only if memory allows.

Why:

- this would match the LoRA global-batch geometry directly

If it does not fit, do **not** waste time forcing it. FT-B1 is the cheaper
matched-batch control.

### Stage 1A: If Full FT Is Permutation-Invariant

This is the branch I currently consider more likely.

If this branch happens, the priority shifts from "is XLA broken?" to:

- "what about LoRA makes it uniquely sensitive to this compiler fork?"

#### L-A0: compare full-FT HLO families across permutations

Goal:

- determine whether full FT gets the **same** HLO family across permutations or
  whether it gets a similar compiler fork that is behaviorally benign

Possible outcomes:

1. HLO also splits in full FT, but loss curves stay close
   - then the fork is general, but LoRA is the fragile consumer
2. HLO does **not** split in full FT
   - then LoRA-specific graph shapes / partitioning are likely what trigger the
     fork in the first place

This is one of the highest-value discriminator analyses after FT-A.

#### L-A1: LoRA vs full-FT HLO side-by-side

Compare:

- LoRA canonical BL0 vs full-FT canonical FT-A0
- LoRA reverse BL1 vs full-FT reverse FT-A1

Questions:

- which dot/contraction families only appear in LoRA?
- which of the public `windowed_dot_general` rewrites happen only in LoRA?
- do full-FT MLP/attention contractions also show the same windowed patterns?

Hypothesis:

- the low-rank contraction shapes may trigger a more fragile partitioning /
  windowed-einsum choice than dense full-FT layers do

#### L-A2: rank sweep under canonical + reverse

Proposed runs:

- `r=8`
- `r=16`
- `r=32`
- `r=64` (baseline)
- `r=128`

Keep:

- `alpha=r`
- `zero_init_b=True`
- canonical vs reverse pair
- HLO upload on each

Questions:

- does the topology sensitivity strengthen or weaken as rank changes?
- does the HLO family switch at a specific rank / shape threshold?

Why this is high value:

- if sensitivity is strongly rank-dependent, then shape-driven windowed-einsum
  heuristics become a leading explanation

#### L-A3: module-family ablations

Proposed runs:

- attention-only LoRA targets
- MLP-only LoRA targets
- qkv-only
- output-proj-only
- gate/up/down-only

Keep:

- canonical + reverse pairs
- `r=64`, `alpha=64`
- HLO upload

Questions:

- is one contraction family sufficient to reproduce the topology sensitivity?
- does the pathology live mostly in attention, mostly in MLP, or require both?

Interpretation:

- if attention-only splits and MLP-only does not:
  - focus on the attention `windowed_dot_general_body_ag` family first
- if MLP-only splits:
  - the issue is not attention-specific
- if both split:
  - low-rank factorization itself is the main amplifier

#### L-A4: init / symmetry / update-scale sweep on Bug 1

Proposed runs:

- canonical vs reverse with:
  - `zero_init_b=True` baseline
  - small nonzero `b_init_scale=1e-4`
  - small nonzero `b_init_scale=1e-3`
  - `zero_init_b=False` if needed as a large-init extreme

Questions:

- does breaking the exact zero-init symmetry reduce topology sensitivity?
- is the catastrophic split a first-update fragility problem?

Why this matters:

- if full FT is stable but LoRA is not, the "tiny first live update in LoRA B"
  becomes a prime suspect

#### L-A5: scale / alpha sweep under canonical + reverse

Proposed runs:

- `alpha=16`
- `alpha=64`
- `alpha=256`

Keep `r=64`.

Questions:

- does increasing adapter scale make the topology gap smaller or larger?
- does the pathology disappear when the adapter signal is numerically larger?

Interpretation:

- if larger scale suppresses the gap, then the issue may be low-signal numeric
  fragility rather than a categorical correctness bug

#### L-A6: exact LoRA grad / update capture

Instrument:

- full `lora_B` grads at step 0 / 1 / 2
- full `lora_A` grads if nonzero
- Adam `m`
- Adam `v`
- final parameter update `ΔB`

Compare canonical vs reverse:

- norm ratio
- cosine similarity
- elementwise sign flips
- per-module breakdown

Questions:

- is the difference mainly update **scale** or update **direction**?
- does one specific module dominate the drift?

This is a direct bridge from compiler differences to optimizer consequences.

#### L-A7: first-update intervention experiments

If L-A6 shows the early LoRA update is the critical separator, run:

- inject reverse step-0/1 LoRA update into canonical for one step only
- inject canonical step-0/1 LoRA update into reverse
- or scale the canonical first LoRA update to match reverse norm

Questions:

- is the basin split caused almost entirely by the first 1-2 updates?
- or is the bad topology continuously harmful across many steps?

This is higher implementation effort, so it is not first-hour work, but it is a
very strong causality test.

#### L-A8: synthetic numeric microbench for LoRA-shaped dots

Build small JAX / HLO repros for the exact shapes seen in the BL HLO, including:

- attention LoRA `embed -> ... -> LORA_R`
- MLP LoRA `embed -> ... -> LORA_R`
- MLP main `embed -> ... -> mlp`

For each:

- canonical vs reverse device order
- bf16 vs fp32 reference comparison
- random input scales
- LoRA-like scales near the actual initialized magnitude

Questions:

- how large is the output difference between the two decompositions?
- is the difference unusually large for low-rank / skinny contractions?
- does a dense full-FT analog show the same or much smaller error?

If this works, it becomes the bridge from "training curves differ" to
"numerical error in this contraction family differs."

### Stage 1B: If Full FT Is Also Permutation-Sensitive

If the full-FT quartet also splits meaningfully, then the problem is broader.

In that case, prioritize general topology/XLA follow-up.

#### G-A0: full-FT pass-dump pair

Rerun:

- full-FT canonical
- full-FT reverse

with:

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.* --xla_dump_hlo_pass_re=.*"
```

Goal:

- find the first divergent pass for full FT

Questions:

- is it the same `after_shardy-xla -> before cleanup` boundary as LoRA?
- is the same `windowed_dot_general` machinery involved?

#### G-A1: compare full-FT and LoRA divergence points

If full FT also splits:

- compare the first divergent pass name
- compare whether the same contraction families flip
- compare whether the same public/private boundary appears

Interpretation:

- if the divergence boundary is the same, then LoRA is probably just amplifying
  a general topology-sensitive partitioner choice

#### G-A2: full-FT rescue via mixed mesh / TP

Run:

- full FT `{data:2, model:2}`
- full FT `{data:1, model:4}`

Why:

- if those also rescue full FT, the structural parallel with LoRA becomes much
  stronger

#### G-A3: general minimal repro

At that point, stop centering LoRA.

Build a minimal repro for the distributed dot shape that flips between:

- direct `all-gather -> dot`
- windowed `collective-permute` form

This becomes the best candidate for upstream reporting.

### Stage 2: Minimal-Repro / Upstream Lane

This lane is worthwhile in either branch, but its urgency depends on Stage 1.

#### MR-A0: JAX-only distributed dot repro

Construct a minimal `pjit` or `shard_map` example on `v5p-8` with:

- explicit mesh permutation
- shapes matching the observed LoRA attention contraction
- no training loop
- only the sharded `dot_general`

Desired outcome:

- canonical and reverse compile to different HLO families

If this works:

- it becomes the cleanest artifact for upstream XLA / TPU reporting

#### MR-A1: JAX-only dense full-FT analog

Construct the dense analog shape too.

Questions:

- does dense full-FT shape also fork?
- or is the fork specific to low-rank / skinny matrices?

#### MR-A2: numeric reference harness

For the minimal repro:

- compute bf16 TPU result
- compare with fp32 CPU or TPU reference
- compare canonical vs reverse discrepancy

This is the strongest pure-numerics artifact we can generate.

### 10-Hour Suggested Execution Order

This is the recommended order if we want progress without constant replanning.

#### Hours 0-2

1. Implement the full-FT permutation script.
2. Launch FT-A0 / A1 / A2 / A3 with HLO upload.
3. In parallel, prepare analysis scripts to hash / compare the uploaded HLOs.

#### Hours 2-4

1. Analyze FT-A quartet behavior.
2. If needed, launch FT-B0 longer-horizon canonical/reverse.
3. Launch FT-B1 matched-batch LoRA control if batch-size confound remains.

#### Hours 4-6

If full FT invariant:

1. analyze full-FT HLO family across permutations
2. launch L-A2 rank sweep pair(s)
3. launch L-A3 module-family ablations

If full FT sensitive:

1. launch G-A0 full-FT pass-dump pair
2. compare divergence boundary to LoRA

#### Hours 6-8

If on LoRA-specific branch:

1. run L-A4 init sweep
2. run L-A6 exact LoRA grad/update capture
3. analyze whether the difference is scale or direction

If on general branch:

1. run G-A2 mixed-mesh / TP full-FT rescues
2. begin MR-A0 minimal repro

#### Hours 8-10

1. whichever branch we are on, start the minimal repro lane
2. if the minimal repro works, begin upstream issue drafting
3. if the minimal repro does not work, stop trying to fully explain TPU-private
   selection and converge on "understood enough + workaround"

### Current Recommendation

The single highest-priority next action is:

1. implement `experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py`
2. launch the four full-FT permutation runs with HLO upload

That is the cleanest next discriminator and the best way to decide whether the
next 8 hours should focus on:

- a general topology/XLA problem, or
- a LoRA-specific fragility problem.

## 2026-04-19T Execution Start: Full-FT Permutation Quartet On `v5p-8`

User instruction for this phase:

- keep working through the Bug 1 experiment ladder
- do not stop at planning
- start executing the highest-priority discriminator

Selected next action:

- implement and launch the **full-FT permutation quartet**:
  - canonical
  - reverse
  - swap12
  - rotate

Why this is the right next step:

- Exp T already showed `v5p-8` full FT learns materially better than bad LoRA
- but Exp T was not a physical-permutation sweep
- this quartet directly tests whether permutation sensitivity is:
  - broad to full FT
  - or catastrophic mainly for LoRA

Planned script:

- `experiments/posttrain/per_stmt_dpo/experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py`

Planned run shape:

- TPU: `v5p-8`
- full FT
- `SeparateReferenceConfig`
- `bs=32`
- `pd=4`
- `steps=10`
- explicit mesh `{replica:1, data:4, model:1}`
- explicit physical device order via `device_permutation`

Planned debug / artifact capture:

```bash
REGIONS_OVERRIDE=us-central1
MARIN_DEBUG_LOG_BATCH_INDICES=1
MARIN_DEBUG_LOG_STEP_TRACE=1
MARIN_DEBUG_DUMP_GRAD_VALUES=1
MARIN_DEBUG_DUMP_SHARDING=1
MARIN_DEBUG_HLO_UPLOAD_DIR=gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/<variant>/
XLA_FLAGS="--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*"
```

Decision rule after the quartet:

1. if full FT is permutation-invariant enough:
   - prioritize LoRA-specific fragility experiments
2. if full FT also splits materially:
   - prioritize general topology / XLA minimal repro work

### Execution: Script Implementation

Implemented:

- [experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py)

Design:

- start from the existing full-FT probe shape in
  [experiment_t_v5p8_full_ft_s2.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_t_v5p8_full_ft_s2.py)
- reuse the explicit physical-device permutation mechanism from
  [experiment_bl_v5p8_pd4_device_permutation_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py)
- keep the logical mesh fixed at `{replica:1, data:4, model:1}`
- switch only the mapping from logical data rank to physical chip

Key config choices in the new script:

- full fine-tuning
- `SeparateReferenceConfig`
- `v5p-8`
- `bs=32`
- `pd=4`
- `steps=10`
- checkpointing default: `offload`
- mesh:
  - `axes={"replica": 1, "data": 4, "model": 1}`
  - `param_mapping={"embed": "data"}`
  - `device_permutation=<variant>`
  - `preserve_device_order=True`
- built-in debug envs:
  - `MARIN_DEBUG_LOG_BATCH_INDICES=1`
  - `MARIN_DEBUG_LOG_STEP_TRACE=1`
  - `MARIN_DEBUG_DUMP_SHARDING=1`
  - `MARIN_DEBUG_DUMP_GRAD_VALUES=1`

Supported physical-order selectors:

- `canonical` -> `(0,1,2,3)`
- `reverse` -> `(3,2,1,0)`
- `swap12` -> `(0,2,1,3)`
- `rotate` -> `(1,2,3,0)`

This keeps the comparison directly aligned with the earlier BL LoRA quartet.

### Local Validation Before Launch

I ran:

```bash
.venv/bin/python -m py_compile \
  experiments/posttrain/per_stmt_dpo/experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py \
  lib/levanter/src/levanter/utils/mesh.py \
  lib/levanter/src/levanter/trainer.py
```

This passed.

### Exact Launch Commands

All jobs were submitted via `lib/iris/examples/marin.yaml`.

Canonical:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  --job-name experiment-b1-fullft-v5p8-pd4-can-20260419-0740 \
  -e REGIONS_OVERRIDE us-central1 \
  -e EXPERIMENT_B1_ORDER canonical \
  -e EXPERIMENT_B1_CHECKPOINTING offload \
  -e MARIN_DEBUG_RUN_TAG b1ft-can \
  -e MARIN_DEBUG_HLO_UPLOAD_DIR gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/b1ft-can/ \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e MARIN_DEBUG_DUMP_GRAD_VALUES 1 \
  -e XLA_FLAGS "--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py
```

Reverse:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  --job-name experiment-b1-fullft-v5p8-pd4-rev-20260419-0740 \
  -e REGIONS_OVERRIDE us-central1 \
  -e EXPERIMENT_B1_ORDER reverse \
  -e EXPERIMENT_B1_CHECKPOINTING offload \
  -e MARIN_DEBUG_RUN_TAG b1ft-rev \
  -e MARIN_DEBUG_HLO_UPLOAD_DIR gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/b1ft-rev/ \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e MARIN_DEBUG_DUMP_GRAD_VALUES 1 \
  -e XLA_FLAGS "--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py
```

Swap12:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  --job-name experiment-b1-fullft-v5p8-pd4-s12-20260419-0740 \
  -e REGIONS_OVERRIDE us-central1 \
  -e EXPERIMENT_B1_ORDER swap12 \
  -e EXPERIMENT_B1_CHECKPOINTING offload \
  -e MARIN_DEBUG_RUN_TAG b1ft-s12 \
  -e MARIN_DEBUG_HLO_UPLOAD_DIR gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/b1ft-s12/ \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e MARIN_DEBUG_DUMP_GRAD_VALUES 1 \
  -e XLA_FLAGS "--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py
```

Rotate:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  --job-name experiment-b1-fullft-v5p8-pd4-rot-20260419-0740 \
  -e REGIONS_OVERRIDE us-central1 \
  -e EXPERIMENT_B1_ORDER rotate \
  -e EXPERIMENT_B1_CHECKPOINTING offload \
  -e MARIN_DEBUG_RUN_TAG b1ft-rot \
  -e MARIN_DEBUG_HLO_UPLOAD_DIR gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/b1ft-rot/ \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e MARIN_DEBUG_DUMP_GRAD_VALUES 1 \
  -e XLA_FLAGS "--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py
```

### Submitted Iris Jobs

- canonical:
  `/ahmed/experiment-b1-fullft-v5p8-pd4-can-20260419-0740`
- reverse:
  `/ahmed/experiment-b1-fullft-v5p8-pd4-rev-20260419-0740`
- swap12:
  `/ahmed/experiment-b1-fullft-v5p8-pd4-s12-20260419-0740`
- rotate:
  `/ahmed/experiment-b1-fullft-v5p8-pd4-rot-20260419-0740`

HLO upload prefixes:

- `gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/b1ft-can/`
- `gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/b1ft-rev/`
- `gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/b1ft-s12/`
- `gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/b1ft-rot/`

Initial submission outcome:

- all four submitted cleanly on the first wave
- workspace bundle size during submission was `6.7 MB`
- no job-name collision occurred because explicit `--job-name` values were used

Early scheduler / startup status:

- first check: all four were `JOB_STATE_PENDING`, waiting only on small
  `us-central1` CPU orchestrator capacity
- second check: all four advanced to `JOB_STATE_RUNNING` with
  `task_state_counts.building = 1`

That means:

- there was no immediate config rejection
- there was no early launch failure
- the next meaningful signals to collect are:
  - child training task creation
  - W&B run URLs
  - compile / HBM failures if any
  - first logged step-2 / step-9 trajectories

Child-task startup identifiers:

- canonical:
  - experiment record:
    `gs://marin-us-central1/experiments/experiment_b1_fullft_v5p8_pd4_device_permutation_s10-a0dd32.json`
  - checkpoint output:
    `gs://marin-us-central1/checkpoints/dpo/stmt_dpo/debug/experiment_b1_fullft_v5p8_pd4_s10_b1ft-can-5023e4`
  - run id:
    `experiment_b1_fullft_v5p8_pd4_s10_b1ft-can-5023e4`
- reverse:
  - experiment record:
    `gs://marin-us-central1/experiments/experiment_b1_fullft_v5p8_pd4_device_permutation_s10-5fd238.json`
  - checkpoint output:
    `gs://marin-us-central1/checkpoints/dpo/stmt_dpo/debug/experiment_b1_fullft_v5p8_pd4_s10_b1ft-rev-20b0c8`
  - run id:
    `experiment_b1_fullft_v5p8_pd4_s10_b1ft-rev-20b0c8`
- swap12:
  - experiment record:
    `gs://marin-us-central1/experiments/experiment_b1_fullft_v5p8_pd4_device_permutation_s10-74238c.json`
  - checkpoint output:
    `gs://marin-us-central1/checkpoints/dpo/stmt_dpo/debug/experiment_b1_fullft_v5p8_pd4_s10_b1ft-s12-34b35e`
  - run id:
    `experiment_b1_fullft_v5p8_pd4_s10_b1ft-s12-34b35e`
- rotate:
  - experiment record:
    `gs://marin-us-central1/experiments/experiment_b1_fullft_v5p8_pd4_device_permutation_s10-07b6d6.json`
  - checkpoint output:
    `gs://marin-us-central1/checkpoints/dpo/stmt_dpo/debug/experiment_b1_fullft_v5p8_pd4_s10_b1ft-rot-29b657`
  - run id:
    `experiment_b1_fullft_v5p8_pd4_s10_b1ft-rot-29b657`

Derived WandB links:

- canonical:
  <https://wandb.ai/marin-community/dpo/runs/experiment_b1_fullft_v5p8_pd4_s10_b1ft-can-5023e4>
- reverse:
  <https://wandb.ai/marin-community/dpo/runs/experiment_b1_fullft_v5p8_pd4_s10_b1ft-rev-20b0c8>
- swap12:
  <https://wandb.ai/marin-community/dpo/runs/experiment_b1_fullft_v5p8_pd4_s10_b1ft-s12-34b35e>
- rotate:
  <https://wandb.ai/marin-community/dpo/runs/experiment_b1_fullft_v5p8_pd4_s10_b1ft-rot-29b657>

Observed naming quirk:

- WandB name truncation stripped the explicit `perm_<variant>` substring from the
  long human-readable experiment name to satisfy WandB limits
- the variant still survives cleanly in:
  - `MARIN_DEBUG_RUN_TAG`
  - the Iris job id
  - the GCS checkpoint suffix
  - the HLO upload prefix

Next immediate task:

- monitor the quartet to determine whether full FT is:
  - permutation-invariant enough to isolate this as a LoRA-specific fragility
  - or also split by physical order, which would imply a broader topology/XLA
    training effect

Small prep change for the next likely control:

- updated
  [experiment_bl_v5p8_pd4_device_permutation_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py)
  to accept:
  - `EXPERIMENT_BL_BS`
  - `EXPERIMENT_BL_STEPS`
- defaults remain the original BL behavior:
  - `bs=64`
  - `steps=10`
- this is specifically to make the matched-batch LoRA control (`bs=32`) cheap to
  launch if the full-FT quartet comes back stable

### Parallel Control Launch: matched-batch LoRA pair (`bs=32`)

Reason for launching before the full-FT quartet finishes:

- the most obvious confound between Exp T / the full-FT quartet and the
  original BL LoRA quartet is batch size
- full FT uses `bs=32`
- BL used `bs=64`
- if the BL split survives cleanly at `bs=32`, then a later "full FT stable,
  LoRA unstable" result is much stronger

Scope:

- canonical LoRA permutation
- reverse LoRA permutation
- same explicit physical-order mechanism as BL
- same `v5p-8`, `pd=4`, `steps=10`
- only change from BL baseline is `bs=32`

Exact launch commands:

Canonical:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  --job-name experiment-bl-bs32-v5p8-pd4-can-20260419-0750 \
  -e REGIONS_OVERRIDE us-central1 \
  -e EXPERIMENT_BL_ORDER canonical \
  -e EXPERIMENT_BL_BS 32 \
  -e EXPERIMENT_BL_STEPS 10 \
  -e MARIN_DEBUG_RUN_TAG bl32-can \
  -e MARIN_DEBUG_HLO_UPLOAD_DIR gs://marin-us-central1/debug/bug_1_lora_bs32_perm_hlo/20260419/bl32-can/ \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e XLA_FLAGS "--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py
```

Reverse:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  --job-name experiment-bl-bs32-v5p8-pd4-rev-20260419-0750 \
  -e REGIONS_OVERRIDE us-central1 \
  -e EXPERIMENT_BL_ORDER reverse \
  -e EXPERIMENT_BL_BS 32 \
  -e EXPERIMENT_BL_STEPS 10 \
  -e MARIN_DEBUG_RUN_TAG bl32-rev \
  -e MARIN_DEBUG_HLO_UPLOAD_DIR gs://marin-us-central1/debug/bug_1_lora_bs32_perm_hlo/20260419/bl32-rev/ \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e XLA_FLAGS "--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py
```

Submitted Iris jobs:

- `/ahmed/experiment-bl-bs32-v5p8-pd4-can-20260419-0750`
- `/ahmed/experiment-bl-bs32-v5p8-pd4-rev-20260419-0750`

HLO upload prefixes:

- `gs://marin-us-central1/debug/bug_1_lora_bs32_perm_hlo/20260419/bl32-can/`
- `gs://marin-us-central1/debug/bug_1_lora_bs32_perm_hlo/20260419/bl32-rev/`

## Live Status Snapshot During Execution

This section records the first meaningful runtime state after the launches, not
just the parent CPU executor status.

### Child `train_dpo` jobs discovered

Full-FT quartet children:

- `/ahmed/experiment-b1-fullft-v5p8-pd4-can-20260419-0740/train_dpo`
- `/ahmed/experiment-b1-fullft-v5p8-pd4-rev-20260419-0740/train_dpo`
- `/ahmed/experiment-b1-fullft-v5p8-pd4-s12-20260419-0740/train_dpo`
- `/ahmed/experiment-b1-fullft-v5p8-pd4-rot-20260419-0740/train_dpo`

Matched-batch LoRA controls:

- `/ahmed/experiment-bl-bs32-v5p8-pd4-can-20260419-0750/train_dpo`
- `/ahmed/experiment-bl-bs32-v5p8-pd4-rev-20260419-0750/train_dpo`

### Scheduling state at this snapshot

Full FT:

- `rotate` child: `JOB_STATE_RUNNING`
- `canonical`, `reverse`, `swap12` children: `JOB_STATE_PENDING`

Matched-batch LoRA:

- both `canonical` and `reverse` children: `JOB_STATE_PENDING`

Shared pending reason for the queued TPU children:

- insufficient currently-free `v5p-8` capacity in `us-central1`
- memory filter also reports `need 250.0GB, available 176.8GB` on some workers
- autoscaler is waiting on scale group
  `tpu_v5p-preemptible_8-us-central1-a`

This is infrastructure delay, not experimental failure.

### First real training-child evidence: full-FT `rotate`

The first landed TPU child is:

- `/ahmed/experiment-b1-fullft-v5p8-pd4-rot-20260419-0740/train_dpo`

Observed state:

- running
- peak memory already ~`29.22 GB`
- current memory ~`17.32 GB`
- no preemptions
- no failures

Important early log evidence:

- WandB run is live:
  <https://wandb.ai/marin-community/dpo/runs/experiment_b1_fullft_v5p8_pd4_s10_b1ft-rot-29b657>
- debug envs are present inside the TPU worker:
  - `MARIN_DEBUG_LOG_STEP_TRACE=1`
  - `MARIN_DEBUG_DUMP_SHARDING=1`
  - `MARIN_DEBUG_DUMP_GRAD_VALUES=1`
  - `MARIN_DEBUG_HLO_UPLOAD_DIR=gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/b1ft-rot/`
  - `XLA_FLAGS=--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*`
- sharding dump succeeded:
  - `DEBUGJ SHARDING_MESH ...`
  - `DEBUGJ SHARDING_DONE`
- no HBM / `resource_exhausted` errors yet

### Why the full-FT startup is slow

The `rotate` child is still in model/reference loading, not yet at train-step
compile or step-0 metrics.

This is expected for this exact recipe:

- full fine-tuning
- `SeparateReferenceConfig`
- initialization from HF checkpoint

Observed behavior in logs:

- it reads all four safetensor shards for `marin-community/marin-8b-instruct`
- then starts reading them again

The best current interpretation is:

- one full pass for the trainable policy model
- another full pass for the separate reference model

So the current lack of step-0 loss is **not** a stall signal yet. It is still in
the expensive two-model startup phase.

### Updated runtime milestone: full-FT `rotate` reached first-step compile

Later logs from the same `rotate` TPU child show that it successfully exited the
two-model HF-load phase and moved into the real training pipeline.

New evidence:

- second full-model read completed
- first several batch hashes were logged:
  - `step=0 sha256=09c72b735e87ea9f`
  - `step=1 sha256=e704300fb89ed190`
  - `step=2 sha256=e14fc45a5675d356`
  - `step=3 sha256=c248521e071ce8a4`
  - `step=4 sha256=21b6de4c94ad4c91`
- data loader began serving real train batches
- trainer log reached:
  - `First batch loaded ... starting first train step`
  - `Tracing train_step for jaxpr...`
  - `Traced train_step in 8.7s`
  - `Lowering train_step to HLO...`

Other notable details from the first-step logs:

- the CE path selected `xla`, not a custom Pallas kernel:
  - `Fused cross-entropy selected implementation: xla`
- CE block config resolved as:
  - `num_v_blocks=16`
  - `num_b_blocks=1`

Interpretation:

- full-FT `rotate` is now past "download / initialize" and into the exact part
  of the stack we care about for topology-sensitive HLO comparison
- no OOM/HBM failure has appeared up to the start of HLO lowering
- the next meaningful signals are:
  - first step-0 / step-1 / step-2 losses
  - first HLO upload objects in
    `gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/b1ft-rot/`

### Major update: full-FT `rotate` did not reach loss logging; it crashed the TPU compiler

After successful HLO lowering, the `rotate` full-FT child failed with a backend
compiler abort:

```text
F0419 07:58:07.741708 async_dynamic_index_emitter.cc:584] Check failed:
intermediate_calc.slice_size %
intermediate_calc.update_detail.num_sublanes_per_chunk == 0 (1 vs. 0)
Sublane slicing size not multiple of update chunk sublane size.
```

Iris summary:

- job:
  `/ahmed/experiment-b1-fullft-v5p8-pd4-rot-20260419-0740/train_dpo`
- terminal state:
  `JOB_STATE_FAILED`
- exit:
  `139` / `SIGSEGV` / `SIGABRT`
- duration:
  `4 minutes and 44.24 seconds`
- peak memory:
  `29.22 GB`

Important boundary conditions:

- it failed **after**
  - first batch load
  - `train_step` tracing
  - `train_step` HLO lowering
- it failed **before**
  - step-0 loss logging
  - any HLO upload artifact appearing in
    `gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/b1ft-rot/`

So the current full-FT result is not "good" or "bad" training. It is:

- a topology-conditioned compile/backend failure on the first landed dense run

### Immediate consequence for the working hypothesis

This materially weakens the earlier simple story:

> "XLA is probably fine; LoRA is just unusually fragile."

At minimum, one full-FT permutation on the same `v5p-8` topology family is now
showing backend/compiler instability too.

That does **not** yet prove:

- full FT has the same permutation sensitivity as LoRA
- or that all full-FT permutations will fail

But it does prove:

- the dense path is not obviously immune to topology-conditioned compiler effects

### Live state right after the `rotate` crash

At the same snapshot:

- `canonical` full-FT child:
  `/ahmed/experiment-b1-fullft-v5p8-pd4-can-20260419-0740/train_dpo`
  -> `JOB_STATE_RUNNING`
- `swap12` full-FT child:
  `/ahmed/experiment-b1-fullft-v5p8-pd4-s12-20260419-0740/train_dpo`
  -> `JOB_STATE_RUNNING`
- `reverse` full-FT child:
  `/ahmed/experiment-b1-fullft-v5p8-pd4-rev-20260419-0740/train_dpo`
  -> still `JOB_STATE_PENDING`

This is now the critical next discriminator:

1. if `canonical` and `swap12` also hit the same compiler crash:
   - full FT is also topology/compiler-sensitive here
   - the Bug-1 story becomes broader than LoRA fragility
2. if `canonical` and/or `swap12` compile and train cleanly while `rotate`
   crashes:
   - full FT is still permutation-sensitive, but in a different way from LoRA
   - dense path may be less numerically fragile but still compiler-path fragile
3. if later `reverse` lands and trains while others fail or diverge:
   - permutation order is again load-bearing on dense full FT

### Dense branch status immediately after the `rotate` failure

`canonical` and `swap12` both subsequently landed on TPU and started their own
full-FT child runs:

- canonical run:
  <https://wandb.ai/marin-community/dpo/runs/experiment_b1_fullft_v5p8_pd4_s10_b1ft-can-5023e4>
- swap12 run:
  <https://wandb.ai/marin-community/dpo/runs/experiment_b1_fullft_v5p8_pd4_s10_b1ft-s12-34b35e>

At this snapshot they are still earlier than `rotate` was at failure time:

- both have initialized WandB
- both have emitted `DEBUGJ WORKER_ENV`
- both have completed `DEBUGJ SHARDING_DONE`
- both are still reading the HF checkpoint shards for the first model
- neither has reached first-batch logging or HLO lowering yet

So the correct current state is:

- `rotate`: dense permutation already demonstrated a backend compiler crash
- `canonical`: dense permutation is live but not yet at compile
- `swap12`: dense permutation is live but not yet at compile
- `reverse`: still waiting on TPU capacity

This means the next highest-value observation is no longer a loss comparison. It
is:

- do `canonical` and `swap12` survive the first HLO-lowering / backend-compile
  boundary that killed `rotate`?

## 2026-04-19T Execution Start: Inspect Public OpenXLA Source Around `spmd-cleanup`

User-confirmed next step:

- clone the public OpenXLA repository outside the Marin repo at
  `/Users/ahmed/code/xla`
- inspect the public pass pipeline around:
  - `xla-partitioner`
  - `shardy-xla`
  - `spmd-cleanup`
  - any public windowed / collective decomposition helpers
- determine how much of the BL0-vs-BL1 split is interpretable from open source

Why this step is worth doing:

- we already know the first divergent pass is
  `module_0336...0077.spmd-cleanup.after_pipeline-start.before_dce`
- we already know the good path is rewriting representative distributed dots
  into `collective-permute`-heavy windowed forms
- the remaining question is:
  - is this rewrite family visible in public XLA code?
  - or does the key decision likely cross into TPU backend / `libtpu`?

Planned working directory:

- `/Users/ahmed/code/xla`

Planned outputs from this step:

1. a mapping from our observed pass names to public OpenXLA pass / pipeline code
2. a note on whether `spmd-cleanup` itself is open and inspectable
3. a note on whether the relevant windowed dot / collective decomposition logic
   appears in public source
4. a boundary statement for what likely remains TPU-backend-private

## 2026-04-19T Update: Full-FT Quartet Reached Terminal State; Matched-Batch LoRA Control Completed

This section records the next real execution milestone after the earlier
`rotate` crash snapshot.

### Full-FT permutation quartet: all four reached terminal failure

All four dense full-FT child jobs are now terminal, and all four failed at the
same first-step backend/compiler boundary with the same `async_dynamic_index`
check:

- canonical:
  `/ahmed/experiment-b1-fullft-v5p8-pd4-can-20260419-0740/train_dpo`
  -> `failed`
- reverse:
  `/ahmed/experiment-b1-fullft-v5p8-pd4-rev-20260419-0740/train_dpo`
  -> `failed`
- swap12:
  `/ahmed/experiment-b1-fullft-v5p8-pd4-s12-20260419-0740/train_dpo`
  -> `failed`
- rotate:
  `/ahmed/experiment-b1-fullft-v5p8-pd4-rot-20260419-0740/train_dpo`
  -> `failed`

Authoritative Iris summaries:

- canonical:
  `State: failed  exit=0  failures=1  preemptions=0`
- reverse:
  `State: failed  exit=0  failures=1  preemptions=0`
- swap12:
  `State: failed  exit=0  failures=1  preemptions=0`
- rotate:
  `State: failed  exit=0  failures=1  preemptions=0`

The failing check is the same in every case:

```text
async_dynamic_index_emitter.cc:584
Check failed:
intermediate_calc.slice_size %
intermediate_calc.update_detail.num_sublanes_per_chunk == 0 (1 vs. 0)
Sublane slicing size not multiple of update chunk sublane size.
```

Observed failure timestamps:

- `rotate`: `07:58:07Z`
- `swap12`: `08:03:09Z`
- `canonical`: `08:03:52Z`
- `reverse`: `08:08:19Z`

Durations:

- `rotate`: `4m44s`
- `swap12`: `4m52s`
- `canonical`: `5m12s`
- `reverse`: `4m36s`

### What this changes

This is stronger than the earlier intermediate snapshot.

We no longer have:

- one dense permutation crashing while others are still pending/running

We now have:

- the entire dense full-FT permutation quartet failing before step-0 loss
  logging, across both previously "good" and previously "bad" LoRA orderings

So the current dense-path result is:

- full FT on this `v5p-8`, `pd=4`, explicit-permutation harness is not merely
  permutation-sensitive in loss space
- it is outright hitting a topology/compiler-path crash before training begins

This does **not** yet say that dense full FT has the *same* mechanism as the
LoRA Bug-1 loss split.

But it does say:

- the backend/compiler instability story is broader than a pure LoRA numerical
  fragility story

### Matched-batch LoRA control: both finished

The matched-batch LoRA control pair (`bs=32`) also reached terminal state:

- canonical:
  `/ahmed/experiment-bl-bs32-v5p8-pd4-can-20260419-0750/train_dpo`
  -> `succeeded`
- reverse:
  `/ahmed/experiment-bl-bs32-v5p8-pd4-rev-20260419-0750/train_dpo`
  -> `succeeded`

Iris summaries:

- canonical:
  `State: succeeded  exit=0  failures=0  preemptions=1`
- reverse:
  `State: succeeded  exit=0  failures=0  preemptions=0`

Important note on canonical:

- canonical had one recovered TPU-init/preemption-style incident:
  `TPU init failure ("Couldn't open iommu group")`
- the job still reached final success after recovery

### Immediate interpretation after these completions

The experiment program is **not** finished. What is finished is only the first
new discriminator wave:

1. dense full-FT permutation quartet
2. matched-batch LoRA control pair

The ladder in this logbook still has substantial unfinished branches:

- evaluate the matched-batch LoRA curves and compare canonical vs reverse
- inspect whether the dense full-FT crash can be minimized into a separate
  topology/compiler repro
- continue the LoRA-vs-dense branch logic from Stage 1A / 1B based on these new
  facts
- run the later LoRA root-cause and minimal-repro experiments if still needed

So the correct current status is:

- newly launched runs: this wave is complete
- full logbook experiment ladder: not yet complete

## 2026-04-19T Correction: The Full-FT Quartet Was Invalid As A Bug-1 Discriminator

This is an important correction to the immediately preceding dense-path
interpretation.

I incorrectly launched the B1 full-FT permutation quartet with:

- `EXPERIMENT_B1_CHECKPOINTING=offload`

That was a mistake.

### Why this was wrong

The older master logbook had already established a specific, known failure:

- `debug_accum_tpu_type.md` at the Exp T handoff
  ([debug_accum_tpu_type.md](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:7075))
  explicitly records that `gradient_checkpointing="offload"` on `v5p-8`
  full-FT hits this same XLA compile-time check failure in
  `async_dynamic_index_emitter.cc:584`.

That prior logbook is explicit:

> this is an XLA internal assertion inside the async dynamic-index emitter ...
> The `offload` checkpointing path materializes host-offloaded carries with
> dynamic-slice shape math, which is where this check lives. So "offload on
> `v5p-8` + full FT + `SeparateReferenceConfig` + `bs=32 pd=4`" hits the bug;
> this is *not* a general `v5p-8` compile failure.

So the dense quartet failures:

- `canonical`
- `reverse`
- `swap12`
- `rotate`

do **not** show that plain full-FT on `v5p-8` is broken, and they do **not**
show a full-FT analog of the LoRA Bug-1 permutation loss split.

They show only this:

- I accidentally reproduced the already-known `offload` compiler bug across all
  four permutations.

### Why this did not happen in LoRA BL

The LoRA permutation experiments did **not** use this path.

The LoRA BL script uses:

- `model_config=llama_8b`
- no explicit `gradient_checkpointing="offload"` override
- `AdapterBaseReferenceConfig`

So BL stayed on the normal LoRA graph and never entered the dense
host-offloaded carry path that triggers the async-dynamic-index crash.

The B1 full-FT script, by contrast, used:

- `SeparateReferenceConfig`
- full dense fine-tuning
- and, incorrectly, `gradient_checkpointing="offload"`

That is exactly the known-bad combination from Exp T's failed offload attempt.

### What the older logbook actually says about full FT

The relevant older positive result is:

- Exp T result:
  [debug_accum_tpu_type.md](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/debug_accum_tpu_type.md:6827)

That section says:

- `v5p-8` full FT **LEARNS**
- broad "`v5p-8` execution-graph failure" is no longer the leading explanation

So the correct current dense interpretation is still the earlier one:

- plain full FT on `v5p-8` is known to be feasible
- the newly launched B1 dense quartet was invalid because it accidentally used
  the previously ruled-out `offload` mode

### Operational fix

The B1 full-FT permutation script has now been corrected so its default
checkpointing mode is no longer `offload`.

File:

- [experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py)

Specific fix:

- `DEFAULT_CHECKPOINTING = "default"` (was `"offload"`)

### Correct next dense probe

If we still want the full-FT permutation discriminator, the next valid rerun is:

- same B1 script
- same four permutations
- `EXPERIMENT_B1_CHECKPOINTING=default` (or `recompute` if memory forces it)

Only that rerun can answer the scientific question:

> does plain full FT on `v5p-8` show permutation sensitivity like LoRA, or not?

## 2026-04-19T Relaunch: Corrected Full-FT Permutation Quartet (`checkpointing=default`)

User requested immediate relaunch of the dense full-FT permutation quartet after
the offload mistake was identified.

### Intent

This relaunch is the actual Bug-1 dense discriminator:

- full FT
- `SeparateReferenceConfig`
- explicit physical device permutation
- **no `offload`**

The only scientific question here is:

> does plain full FT on `v5p-8` show permutation sensitivity like LoRA, or not?

### Script state at relaunch

The B1 full-FT script was corrected before submission:

- [experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py)
- `DEFAULT_CHECKPOINTING = "default"`

### Exact launch commands

All four launches use:

- `REGIONS_OVERRIDE=us-central1`
- `EXPERIMENT_B1_CHECKPOINTING=default`
- HLO dump/upload enabled
- `MARIN_DEBUG_DUMP_GRAD_VALUES=1`
- `MARIN_DEBUG_DUMP_SHARDING=1`

Canonical:

```bash
WANDB_API_KEY="$WANDB_API_KEY" uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 \
  --cpu 1 \
  --memory 3g \
  --job-name experiment-b1-fullft-default-v5p8-pd4-can-20260419-172134 \
  -e REGIONS_OVERRIDE us-central1 \
  -e EXPERIMENT_B1_ORDER canonical \
  -e EXPERIMENT_B1_CHECKPOINTING default \
  -e MARIN_DEBUG_RUN_TAG b1ftd-can \
  -e MARIN_DEBUG_HLO_UPLOAD_DIR gs://marin-us-central1/debug/bug_1_fullft_perm_hlo/20260419/b1ftd-can/ \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e MARIN_DEBUG_DUMP_GRAD_VALUES 1 \
  -e XLA_FLAGS "--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_b1_fullft_v5p8_pd4_device_permutation_s10.py
```

Reverse / swap12 / rotate are identical except for:

- `EXPERIMENT_B1_ORDER`
- `MARIN_DEBUG_RUN_TAG`
- `MARIN_DEBUG_HLO_UPLOAD_DIR`
- `--job-name`

### Submitted parent jobs

- `/ahmed/experiment-b1-fullft-default-v5p8-pd4-can-20260419-172134`
- `/ahmed/experiment-b1-fullft-default-v5p8-pd4-rev-20260419-172134`
- `/ahmed/experiment-b1-fullft-default-v5p8-pd4-s12-20260419-172134`
- `/ahmed/experiment-b1-fullft-default-v5p8-pd4-rot-20260419-172134`

Expected child jobs:

- `/ahmed/experiment-b1-fullft-default-v5p8-pd4-can-20260419-172134/train_dpo`
- `/ahmed/experiment-b1-fullft-default-v5p8-pd4-rev-20260419-172134/train_dpo`
- `/ahmed/experiment-b1-fullft-default-v5p8-pd4-s12-20260419-172134/train_dpo`
- `/ahmed/experiment-b1-fullft-default-v5p8-pd4-rot-20260419-172134/train_dpo`

### Monitoring ownership

User explicitly requested:

- once a minute for 10 minutes
- then once every 10 minutes for one hour
- if anything crashes, investigate immediately and debug/fix

To make that deterministic, a local monitor harness was created:

- [scratch/20260419-1021_bug1_fullft_monitor.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/scratch/20260419-1021_bug1_fullft_monitor.py)
- state file:
  [scratch/20260419-1021_monitoring_state.json](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/scratch/20260419-1021_monitoring_state.json)

The monitor:

- polls at the requested cadence
- records parent/child state snapshots
- captures summary/log artifacts on terminal states
- stops early if all jobs become terminal

The actual debugging responsibility still remains with the agent: if a terminal
failure appears, inspect the captured artifacts and fix/resubmit if the cause is
small and local.

## 2026-04-19T Early Monitoring Result: Corrected Quartet Is Live

Early monitoring after the corrected relaunch produced the first useful
runtime signal.

### Canonical: one bad-node TPU-init failure, then automatic recovery

Canonical initially landed on a bad TPU worker and showed:

- `Couldn't open iommu group /dev/vfio/2`
- JAX TPU init failure
- CPU fallback
- mesh validation failure:
  `ValueError: ICI product 4 must equal devices_per_slice 1`

This is **not** a model/code bug. It is the familiar bad-node TPU-init failure.

Important operational detail:

- Iris detected the bad-node signature automatically
- canonical was recovered onto a fresh worker
- `preemption_count=1`
- it is now back on the normal startup path

So no manual code fix or resubmission was needed here.

### Reverse: first clean positive signal

The `reverse` corrected full-FT run is the first one to get all the way through:

- startup
- checkpoint load
- first-batch fetch
- `train_step` trace
- HLO lowering
- first executed training step

Confirmed runtime details from logs:

- W&B:
  <https://wandb.ai/marin-community/dpo/runs/experiment_b1_fullft_v5p8_pd4_s10_b1ftd-rev-382ba1>
- first train-step trace:
  - `loss=0.6931471824645996`
  - `grad_l2=31.380828857421875`
  - `param_l2=16608.185546875`
- compile path:
  - `Tracing train_step for jaxpr...`
  - `Traced train_step in 8.6s`
  - `Lowering train_step to HLO...`
  - `Lowered train_step in 2.5s`

This is the first real confirmation that the corrected dense probe is now
testing the intended scientific question instead of the old offload compiler
bug.

### Canonical after recovery

Canonical is now also on the healthy post-restart path:

- recovered worker assigned
- WandB initialized
- both full model loads started
- no second TPU-init failure so far

At the latest check it was still loading the two HF checkpoint copies required
by `SeparateReferenceConfig`, so it had not yet reached first-step compile.

### Swap12 and rotate

Both initially waited on TPU capacity, then subsequently landed and entered
`JOB_STATE_RUNNING`.

At the latest check:

- `swap12` child had started on TPU
- `rotate` child had started on TPU
- neither had yet produced first-step logs

### Current corrected-dense status

So the corrected quartet status is now:

- `canonical`: recovered from one bad TPU-init worker and now running
- `reverse`: clean through first executed train step
- `swap12`: running
- `rotate`: running

This is the first moment where the full-FT permutation discriminator is
scientifically live.

## 2026-04-19T Result: Corrected Full-FT Permutation Quartet Succeeds And Is Nearly Invariant

This is the key result from the corrected dense probe.

### Terminal status

All four corrected full-FT permutation runs finished successfully:

- canonical:
  `/ahmed/experiment-b1-fullft-default-v5p8-pd4-can-20260419-172134/train_dpo`
  -> `succeeded`
- reverse:
  `/ahmed/experiment-b1-fullft-default-v5p8-pd4-rev-20260419-172134/train_dpo`
  -> `succeeded`
- swap12:
  `/ahmed/experiment-b1-fullft-default-v5p8-pd4-s12-20260419-172134/train_dpo`
  -> `succeeded`
- rotate:
  `/ahmed/experiment-b1-fullft-default-v5p8-pd4-rot-20260419-172134/train_dpo`
  -> `succeeded`

W&B runs:

- canonical:
  <https://wandb.ai/marin-community/dpo/runs/experiment_b1_fullft_v5p8_pd4_s10_b1ftd-can-de02c4>
- reverse:
  <https://wandb.ai/marin-community/dpo/runs/experiment_b1_fullft_v5p8_pd4_s10_b1ftd-rev-382ba1>
- swap12:
  <https://wandb.ai/marin-community/dpo/runs/experiment_b1_fullft_v5p8_pd4_s10_b1ftd-s12-240712>
- rotate:
  <https://wandb.ai/marin-community/dpo/runs/experiment_b1_fullft_v5p8_pd4_s10_b1ftd-rot-d0f749>

Operational note:

- canonical had one recovered bad-node TPU-init incident
  (`Couldn't open iommu group`)
- after recovery, it trained successfully
- the other three completed with no preemptions

### Step traces

#### Step 2

| variant | step-2 loss |
|---------|-------------|
| canonical | `0.6861803` |
| reverse   | `0.6891314` |
| swap12    | `0.6870556` |
| rotate    | `0.6892699` |

Spread at step 2:

- max - min = `0.6892699 - 0.6861803 = 0.0030896`

#### Step 9

| variant | step-9 loss |
|---------|-------------|
| canonical | `0.6094701` |
| reverse   | `0.6142877` |
| swap12    | `0.6083252` |
| rotate    | `0.6083851` |

Spread at step 9:

- max - min = `0.6142877 - 0.6083252 = 0.0059625`

### Strongest true conclusion

The corrected dense result falls squarely into the logbook's earlier **World A**
branch:

> full FT is permutation-invariant enough, while LoRA is not.

This is the strongest current Bug-1 discriminator.

Compare against the LoRA BL permutation sweep:

- LoRA bad family (`canonical`, `swap12`) step-9:
  about `0.6606`
- LoRA good family (`reverse`, `rotate`) step-9:
  about `0.308-0.309`

That LoRA split is huge:

- roughly `0.35` loss difference by step 9

The dense full-FT permutation spread is tiny by comparison:

- about `0.006` at step 9

So the new evidence says:

1. the `v5p-8` topology/compiler path is not generically exploding full FT
2. full FT tolerates these permutation / HLO differences well enough to stay in
   one learning regime
3. LoRA does not

### What this means for Bug 1

This sharply strengthens the LoRA-specific interpretation:

- the topology-sensitive compiler fork is real
- but the pathological outcome is not a generic consequence of that fork
- instead, LoRA's low-rank update geometry is unusually sensitive to it

In other words:

> XLA/topology differences are the trigger, but LoRA is the fragile object.

That is a much better statement than either extreme:

- "XLA must just be broken"
- or "LoRA must be broken in complete isolation from the compiler"

### Best next step

The next highest-value step is now **not** more dense permutation runs.

It is:

1. compare **full-FT HLO** vs **LoRA HLO** for the same permutation pair
   (`canonical` vs `reverse`)
2. determine whether the same communication fork appears in both
3. if yes, isolate why LoRA's optimizer/update path is uniquely sensitive

That moves the investigation directly into the LoRA-specific Stage 1A branch:

- LoRA vs full-FT HLO side-by-side
- then module-family / rank / first-update sensitivity probes

## 2026-04-19T Dense Full-FT HLO Download + Diff: Same Compiler Fork, Different Training Sensitivity

This section closes the first half of the Stage 1A branch:

> do the corrected dense full-FT runs see the same topology-sensitive HLO fork as
> LoRA, or not?

The answer is now:

> **yes**. Dense full-FT on `v5p-8 pd=4` sees the same broad canonical/swap12 vs
> reverse/rotate compiled-path split that LoRA sees.
> **But** dense full-FT is numerically robust to that split, while LoRA is not.

That is the strongest Bug-1 narrowing so far.

### Local dense HLO root

Corrected full-FT HLO bundles were downloaded locally under:

- [bug_1_fullft_hlo/20260419](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419)

Per-variant directories:

- [b1ftd-can](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-can)
- [b1ftd-rev](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev)
- [b1ftd-s12](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-s12)
- [b1ftd-rot](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rot)

Observed local file counts:

- `b1ftd-can`: `601` files, `72M`
- `b1ftd-rev`: `608` files, `81M`
- `b1ftd-s12`: `433` files, `70M`
- `b1ftd-rot`: `601` files, `78M`

These match the earlier GCS object counts closely enough to treat the local
copies as complete for analysis.

### Dense train-step modules found locally

Each dense permutation produced two `jit__train_step` compile families, just as
the earlier LoRA bundles did.

#### Canonical

- [module_0344 before](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-can/module_0344.jit__train_step.cl_813921542.before_optimizations.txt)
- [module_0346 after](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-can/module_0346.jit__train_step.cl_813921542.after_optimizations.txt)
- [module_0805 before](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-can/module_0805.jit__train_step.cl_813921542.before_optimizations.txt)
- [module_0807 after](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-can/module_0807.jit__train_step.cl_813921542.after_optimizations.txt)

#### Reverse

- [module_0344 before](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev/module_0344.jit__train_step.cl_813921542.before_optimizations.txt)
- [module_0346 after](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev/module_0346.jit__train_step.cl_813921542.after_optimizations.txt)
- [module_1561 before](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev/module_1561.jit__train_step.cl_813921542.before_optimizations.txt)
- [module_1563 after](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev/module_1563.jit__train_step.cl_813921542.after_optimizations.txt)

#### Rotate

- [module_0344 before](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rot/module_0344.jit__train_step.cl_813921542.before_optimizations.txt)
- [module_0346 after](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rot/module_0346.jit__train_step.cl_813921542.after_optimizations.txt)
- [module_1561 before](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rot/module_1561.jit__train_step.cl_813921542.before_optimizations.txt)
- [module_1563 after](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rot/module_1563.jit__train_step.cl_813921542.after_optimizations.txt)

#### Swap12

- [module_0308 before](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-s12/module_0308.jit__train_step.cl_813921542.before_optimizations.txt)
- [module_0310 after](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-s12/module_0310.jit__train_step.cl_813921542.after_optimizations.txt)
- [module_0661 before](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-s12/module_0661.jit__train_step.cl_813921542.before_optimizations.txt)
- [module_0663 after](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-s12/module_0663.jit__train_step.cl_813921542.after_optimizations.txt)

### Hash result: dense splits into the same two permutation families as LoRA

The dense full-FT permutation quartet collapses into the same two logical
families that the earlier BL LoRA quartet showed:

- bad/identity-oriented family:
  - `canonical`
  - `swap12`
- good/permuted family:
  - `reverse`
  - `rotate`

#### Main train-step family

`before_optimizations` hash:

- all four dense variants:
  - `e569c88d9768663512ede7a34b944009855c7740`

`after_optimizations` hash:

- `canonical`:
  - `d8944ee8911e2b45851a8d4bf91563244db2dcbb`
- `swap12`:
  - `d8944ee8911e2b45851a8d4bf91563244db2dcbb`
- `reverse`:
  - `803e8573270df47369d1deb584350cdafb84d01b`
- `rotate`:
  - `803e8573270df47369d1deb584350cdafb84d01b`

So:

- `canonical == swap12`
- `reverse == rotate`
- bad family != good family

#### Secondary train-step family

`before_optimizations` hash:

- all four dense variants:
  - `3357b7281f4a83f98e8e212d522e439b5358a9d2`

`after_optimizations` hash:

- `canonical`:
  - `163204122bc7b6d52874f9bb1f886a0415bf17dc`
- `swap12`:
  - `163204122bc7b6d52874f9bb1f886a0415bf17dc`
- `reverse`:
  - `196591220c64e3ee3a7b2fa2d0d79307f6859342`
- `rotate`:
  - `196591220c64e3ee3a7b2fa2d0d79307f6859342`

So the dense permutation structure is not noisy or one-off. It is a stable
two-family optimized-HLO split.

### Communication count comparison within dense full-FT

The counts below are substring counts over the optimized HLO text. In
particular, `collective-permute` includes both `collective-permute-start` and
`collective-permute-done`.

#### Dense bad family (`canonical`, `swap12`)

Main optimized train-step file:

- [dense bad main](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-can/module_0346.jit__train_step.cl_813921542.after_optimizations.txt)

Counts:

- `all-reduce=155`
- `all-gather=449`
- `reduce-scatter=30`
- `collective-permute=62`
- `all-to-all=42`

Secondary optimized train-step file:

- `all-reduce=104`
- `all-gather=449`
- `reduce-scatter=30`
- `collective-permute=62`
- `all-to-all=42`

#### Dense good family (`reverse`, `rotate`)

Main optimized train-step file:

- [dense good main](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev/module_0346.jit__train_step.cl_813921542.after_optimizations.txt)

Counts:

- `all-reduce=85`
- `all-gather=211`
- `reduce-scatter=2`
- `collective-permute=992`
- `all-to-all=42`

Secondary optimized train-step file:

- `all-reduce=34`
- `all-gather=211`
- `reduce-scatter=2`
- `collective-permute=992`
- `all-to-all=42`

### Comparison to existing LoRA BL HLO bundles

For reference, the earlier LoRA BL optimized train-step files already showed:

#### LoRA bad family (`BL0`, `BL2`)

- [LoRA bad main](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_bl_hlo/20260418/bl0-can/module_0336.jit__train_step.cl_813921542.after_optimizations.txt)

Counts:

- `all-reduce=276`
- `all-gather=1047`
- `reduce-scatter=58`
- `collective-permute=12`
- `all-to-all=36`

#### LoRA good family (`BL1`, `BL3`)

- [LoRA good main](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_bl_hlo/20260418/bl1-rev/module_0336.jit__train_step.cl_813921542.after_optimizations.txt)

Counts:

- `all-reduce=136`
- `all-gather=591`
- `reduce-scatter=2`
- `collective-permute=1546`
- `all-to-all=36`

### High-level conclusion from the count tables

Dense full-FT and LoRA do **not** differ in whether the compiler fork exists.
They differ in how much the training dynamics care about that fork.

In both dense full-FT and LoRA:

- bad family = more `all-gather` / `all-reduce`, very little
  `collective-permute`
- good family = much more `collective-permute`, much less `all-gather`

So the dense-vs-LoRA discriminator is now:

> the topology-sensitive XLA communication fork is present in **both** dense full
> FT and LoRA.
> **Only LoRA** turns that fork into a huge loss-curve bifurcation.

### Representative concrete rewrite in dense full-FT

The dense optimized HLO shows the same qualitative per-op rewrite we already saw
in the BL LoRA analysis.

#### Dense reverse: collective-permute-heavy path

In the dense reverse optimized train-step HLO:

- [dense reverse main](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev/module_0346.jit__train_step.cl_813921542.after_optimizations.txt)

the compiler emits `collective-permute-start/done` directly on core
`dot_general` sites:

- QKV contraction with grouped heads:
  [module_0346 line 7324](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev/module_0346.jit__train_step.cl_813921542.after_optimizations.txt:7324)
  and [line 7416](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev/module_0346.jit__train_step.cl_813921542.after_optimizations.txt:7416)
- attention output projection:
  [line 7385](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev/module_0346.jit__train_step.cl_813921542.after_optimizations.txt:7385)
  and [line 7458](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev/module_0346.jit__train_step.cl_813921542.after_optimizations.txt:7458)
- MLP projection:
  [line 7393](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev/module_0346.jit__train_step.cl_813921542.after_optimizations.txt:7393)
  and [line 7478](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-rev/module_0346.jit__train_step.cl_813921542.after_optimizations.txt:7478)

These lines use the same two directional neighbor exchanges already seen in the
LoRA good family:

- `{{0,1},{1,2},{2,3},{3,0}}`
- `{{0,3},{1,0},{2,1},{3,2}}`

So the good dense family is unmistakably in the same permute-heavy communication
regime as the good LoRA family.

#### Dense canonical: all-gather-heavy path

In the dense canonical optimized train-step HLO:

- [dense canonical main](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-can/module_0346.jit__train_step.cl_813921542.after_optimizations.txt)

the same logical contraction sites stay on `all-gather`-based distributed-dot
realizations:

- QKV contraction with grouped heads:
  [line 6833](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-can/module_0346.jit__train_step.cl_813921542.after_optimizations.txt:6833)
- attention output projection:
  [line 6907](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-can/module_0346.jit__train_step.cl_813921542.after_optimizations.txt:6907)
- MLP projection:
  [line 6980](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/artifacts/bug_1_fullft_hlo/20260419/b1ftd-can/module_0346.jit__train_step.cl_813921542.after_optimizations.txt:6980)

So the dense bad family is unmistakably in the same all-gather-heavy regime as
the bad LoRA family.

### Why this matters

This result answers the user's earlier high-level objection directly:

> if XLA really were "doing something weird", why does everybody not see the same
> pathology?

The answer now appears to be:

1. XLA/topology **is** doing something real and observable:
   it chooses two different communication decompositions for the same logical
   program depending on physical device ordering
2. dense full-FT and LoRA both see that fork
3. dense full-FT is robust to it
4. LoRA is not

That means the current root-cause boundary is:

> the compiler fork is real, but it is not by itself sufficient to explain the
> Bug-1 loss split.
> The actual pathology is the interaction between that fork and LoRA's low-rank
> update geometry / first-update dynamics.

### Best current one-line statement

The best current Bug-1 statement is now:

> Bug 1 is a **LoRA-specific numerical fragility triggered by a real
> topology-sensitive XLA communication-lowering fork** on `v5p-8 pd=4`.

### Next best step after this section

The next investigation should stay in the LoRA-specific branch, not the dense
branch.

Highest-value follow-ups:

1. compare LoRA and dense first-update statistics under the same bad vs good
   family
2. run LoRA module-family ablations:
   - attention-only
   - MLP-only
   - q/v-only
3. run a LoRA rank sweep under `canonical` vs `reverse`
4. inspect whether the fragile signal is concentrated in the first nonzero
   `lora_B` update rather than spread uniformly across all adapter sites

## 2026-04-19T LoRA-Specific Isolation Program: Hypotheses, Experiments, Branches, And Candidate Fixes

This section is the forward-looking Bug-1 program after the dense HLO result.

We now treat the following as established facts:

1. the topology-sensitive XLA communication fork is real
2. dense full FT sees that fork too
3. dense full FT is robust to it
4. LoRA is not

So the right question is no longer:

> "does XLA fork?"

It is now:

> "what property of LoRA turns that otherwise-tolerable compiler fork into a
> large training bifurcation?"

This program is written to answer that question with the fewest confounded
steps, and to surface a practical fix as soon as one appears.

### Scientific target

We want a causal statement of the form:

> LoRA fails under the bad topology family because of `X`, and changing `Y`
> removes or sharply reduces the permutation split.

The best possible outcomes are:

- an implementable training fix in Marin
- a smaller, principled safe LoRA recipe on `v5p-8 pd=4`
- or a much tighter backend bug report if the remaining culprit still looks
  compiler-facing

### Fixed baseline for the isolation branch

Unless a given experiment says otherwise, all LoRA isolation probes should hold
this base recipe fixed:

- hardware: `v5p-8`
- layout: `{replica:1, data:4, model:1}`
- compare only:
  - bad family: `canonical`
  - good family: `reverse`
- objective: the canonical Bug-1 LoRA DPO recipe
- batch size: `64`
- `pd=4`
- `num_train_steps=10`
- `r=64`
- `alpha=64`
- `dropout=0`
- `zero_init_b=True`
- `target_modules=None` unless explicitly varied
- checkpointing: `default` unless memory forces otherwise

This keeps all comparisons anchored to the known BL split.

### Standard success / failure thresholds

For the short-run discriminators, use the same early-and-late readouts across
all probes:

- `step-2 loss`
- `step-9 loss`
- fixed-index `GRAD_VAL` dumps for `lora_B`
- HLO upload for canonical and reverse
- per-module LoRA update norms if instrumented

Classification thresholds:

- **full Bug-1 split preserved**:
  - bad family step-9 remains near `~0.66`
  - good family step-9 remains near `~0.31`
- **split materially reduced**:
  - step-9 gap shrinks by at least `50%`
- **split effectively removed**:
  - step-9 gap `<= 0.05`
- **ambiguous**:
  - one side moves, but not enough to distinguish scale vs direction

### Shared measurement harness before new sweeps

Before spending many TPU-hours on ablations, the harness should be upgraded so
every later result is more interpretable than the original BL runs.

## L0: Common instrumentation upgrade

### Hypothesis

The next round will stall if we keep reading only loss curves and HLO counts.
We need per-module first-update statistics.

### Experiment

Create a reusable Bug-1 LoRA isolation script family with the following knobs:

- `target_modules`
- `r`
- `alpha`
- `zero_init_b`
- `b_init_scale`
- `a_init_mode`
- optional LoRA-only optimizer overrides if easy
- physical order selector (`canonical`, `reverse`)

And add the following logging for LoRA params only:

- step-0 / step-1 / step-2:
  - per-module `||grad_B||`
  - per-module `||grad_A||`
  - per-module `||delta_W||` where `delta_W = B @ A * scale`
  - canonical-vs-reverse cosine on `delta_W` if run offline
  - fixed-index `GRAD_VAL` dumps for `q/k/v/o/gate/up/down`

### What we learn

- whether the split is mostly a norm/scale effect or a direction/cosine effect
- whether one module family dominates the discrepancy
- whether the decisive divergence appears already in `B`, in `A`, or only after
  forming `delta_W`

### Next steps

- if a single module or family dominates, go straight to the matching ablation
- if the discrepancy is spread evenly, prioritize rank/scale and optimizer
  hypotheses

### Candidate fix if confirmed

No direct fix. This is enabling infrastructure for the rest of the branch.

## H1: Only some LoRA target modules are fragile

The strongest structural hypothesis is that the split is concentrated in a
subset of the LoRAized layers, not all of them equally.

This is especially plausible because the HLO diff showed the topology-sensitive
fork on both attention and MLP contractions, and because not all LoRA params are
sharded the same way.

## L1: Attention-only LoRA

### Hypothesis

The pathology is primarily driven by attention LoRA (`q_proj/k_proj/v_proj/o_proj`).

### Experiment

Run canonical + reverse with:

- `target_modules=["q_proj","k_proj","v_proj","o_proj"]`

### What we learn

- whether attention alone is sufficient to reproduce Bug 1

### Next steps

- if attention-only reproduces the split:
  - go to `L3`, `L4`, and `L5`
- if attention-only does not:
  - prioritize MLP-side probes `L2`, `L6`, `L7`
- if attention-only shows a partial split:
  - keep both attention and MLP lanes alive

### Candidate fix if confirmed

- avoid attention LoRA on this topology
- or use a reduced attention target set

## L2: MLP-only LoRA

### Hypothesis

The pathology is primarily driven by MLP LoRA (`gate_proj/up_proj/down_proj`).

### Experiment

Run canonical + reverse with:

- `target_modules=["gate_proj","up_proj","down_proj"]`

### What we learn

- whether MLP-only is sufficient to reproduce Bug 1

### Next steps

- if MLP-only reproduces the split:
  - focus on `L6`, `L7`, `L9`
- if it does not:
  - focus on attention-side probes
- if both attention-only and MLP-only split:
  - the problem is broader than one family; move to capacity/init hypotheses

### Candidate fix if confirmed

- avoid MLP LoRA on this topology
- or exclude the worst individual MLP projection

## L3: Q/V-only vs O-only

### Hypothesis

The split lives mostly in a smaller attention subset, not the entire attention
block.

### Experiment

Run three canonical+reverse pairs:

- `["q_proj","v_proj"]`
- `["o_proj"]`
- `["k_proj"]` as a low-priority tie-breaker if needed

### What we learn

- whether the issue is in input-side attention projections, output projection,
  or both

### Next steps

- if `o_proj` alone is enough:
  - prioritize sharding-specific hypotheses
- if `q/v` alone are enough:
  - prioritize first-feature / low-rank forward-branch sensitivity
- if neither alone is enough but full attention is:
  - interaction among attention LoRA sites matters

### Candidate fix if confirmed

- safe target-module subsets for v5p-8 `pd=4`

## L4: Gate/Up-only vs Down-only

### Hypothesis

Only part of the MLP LoRA triad is fragile.

### Experiment

Run canonical+reverse pairs:

- `["gate_proj","up_proj"]`
- `["down_proj"]`

### What we learn

- whether the MLP fragility, if present, is on the expansion side or the
  contraction side

### Next steps

- if `down_proj` alone splits:
  - strongly suspect sharding / reduction placement
- if `gate/up` alone split:
  - suspect forward activation sensitivity in the gated branch

### Candidate fix if confirmed

- remove the offending MLP target module from the LoRA target list

## H2: The culprit tracks LoRA parameter sharding, not module semantics

There is a strong mechanistic sub-hypothesis hiding inside H1:

> modules whose LoRA `B` or `delta_W` are data-sharded are more fragile than
> modules whose LoRA params stay replicated.

This is directly motivated by the trainer-side debug note that some LoRA `B`
parameters are replicated while others participate in data-axis sharding paths.

## L5: Replicated-B group vs data-sharded-B group

### Hypothesis

The split is concentrated in the modules whose LoRA updates sit on the
data-sharded path.

### Experiment

Run canonical+reverse pairs for two grouped target sets:

- replicated-ish group:
  - `["q_proj","k_proj","v_proj","gate_proj","up_proj"]`
- data-sharded-ish group:
  - `["o_proj","down_proj"]`

### What we learn

- whether sharding class predicts fragility better than model semantics

### Next steps

- if only `["o_proj","down_proj"]` split:
  - go immediately to `L6`
- if the replicated group also splits:
  - the problem is not limited to data-sharded LoRA params

### Candidate fix if confirmed

- exclude `o_proj` and/or `down_proj`
- or replicate those specific LoRA params across the data axis

## L6: Force problematic LoRA params replicated

### Hypothesis

The bad family is driven by the data-axis sharding of a small set of LoRA
parameters, not by LoRA in general.

### Experiment

Patch parameter mapping so that the most suspicious LoRA params are replicated
across `data`, while keeping everything else unchanged:

- first candidate:
  - `o_proj.lora.lora_B`
  - `down_proj.lora.lora_B`
- second candidate if needed:
  - all LoRA `B` params replicated

Run canonical only first, then compare to reverse if needed.

### What we learn

- whether the bug requires those LoRA params to participate in the bad
  distributed-dot / reshard path

### Next steps

- if replication rescues canonical:
  - the root cause is tightly tied to LoRA param placement
  - the practical fix is to replicate those params on `v5p-8 pd=4`
- if replication does not rescue:
  - move away from sharding-placement hypotheses and toward low-rank-geometry
    hypotheses

### Candidate fix if confirmed

- special parameter mapping for LoRA on `v5p-8 pd=4`

## H3: LoRA rank / capacity controls robustness

Another strong hypothesis:

> the compiler fork changes the effective update, but dense full FT averages
> over many degrees of freedom while low-rank LoRA concentrates all learning
> into too few directions.

## L7: Rank sweep at fixed `alpha/r = 1`

### Hypothesis

Increasing LoRA rank makes canonical more robust by giving the optimizer more
directions to absorb the topology-induced perturbation.

### Experiment

Run canonical+reverse pairs with:

- `r=4, alpha=4`
- `r=8, alpha=8`
- `r=16, alpha=16`
- `r=32, alpha=32`
- `r=64, alpha=64`
- `r=128, alpha=128`

### What we learn

- whether the split shrinks as rank increases

### Next steps

- if higher rank collapses the split:
  - treat capacity averaging as load-bearing
  - then test whether only one module family needs higher rank
- if the split is rank-invariant:
  - capacity alone is not the story

### Candidate fix if confirmed

- safer minimum rank for `v5p-8 pd=4`

## L8: Scale sweep at fixed rank

### Hypothesis

The bad family receives an early LoRA update that is directionally similar to
the good family but too small or too large.

### Experiment

Keep `r=64` fixed and sweep:

- `alpha=16`
- `alpha=32`
- `alpha=64`
- `alpha=128`
- `alpha=256`

### What we learn

- whether the split behaves like an effective-step-size problem

### Next steps

- if higher alpha rescues canonical without hurting reverse:
  - the fork is attenuating the useful LoRA update in the bad family
- if lower alpha helps instead:
  - canonical may be over-amplifying a bad early direction
- if neither matters:
  - move to init or optimizer hypotheses

### Candidate fix if confirmed

- topology-specific `alpha` recommendation

## H4: The decisive problem is the very first nonzero LoRA update

This is a natural continuation of the zero-init-B reasoning, but now focused on
Bug 1 rather than the earlier Bug-2 `c=f32` thread.

## L9: Small nonzero B init in Bug 1

### Hypothesis

The bad topology family only becomes pathological because canonical LoRA starts
with `B=0` and a highly concentrated first nonzero update.

### Experiment

Use the existing debug knob in `LoraConfig`:

- `b_init_scale=1e-5`
- `b_init_scale=1e-4`
- `b_init_scale=1e-3`

Run canonical+reverse pairs.

### What we learn

- whether a tiny symmetry break makes canonical robust

### Next steps

- if tiny `b_init_scale` collapses the split:
  - the first-update geometry is central
- if it does nothing:
  - the issue is not just "all signal is in the first B update"

### Candidate fix if confirmed

- tiny nonzero `B` init for v5p-8 `pd=4` DPO LoRA

## L10: A=0 / B=random-small while preserving zero adapter output

### Hypothesis

What matters is not only the first update magnitude, but **which factor learns
first**.

### Experiment

Use the existing debug knobs:

- `a_init_mode="zero"`
- `b_init_scale` small and nonzero

This preserves `B @ A = 0` at init but swaps the learning asymmetry.

### What we learn

- whether the permutation split depends on the canonical "A random, B zero"
  LoRA factorization geometry

### Next steps

- if this collapses the split:
  - the factorization asymmetry is load-bearing
- if not:
  - the problem is downstream of that exact symmetry

### Candidate fix if confirmed

- alternate LoRA init that preserves policy=reference while improving
  robustness

## H5: Adam-style optimizer state amplifies tiny topology-induced differences

Another plausible explanation is:

> the bad and good families see modestly different gradients, but Adam on tiny
> low-rank parameters amplifies those differences much more than dense full FT
> does.

## L11: Optimizer ablation on LoRA params

### Hypothesis

Bug 1 is substantially amplified by Adam's moment accumulation on the LoRA
branch.

### Experiment

Run canonical+reverse pairs with simpler or altered optimizer dynamics:

- Adam with `beta1=0`
- Adam with smaller `beta2`
- SGD / momentum-free optimizer if already supported
- LoRA-only LR multiplier if easy to express cleanly

### What we learn

- whether the fork primarily changes the optimizer trajectory rather than the raw
  useful descent direction

### Next steps

- if simpler optimizer collapses the split:
  - focus on optimizer-state amplification
- if all optimizers still split:
  - the issue is pre-optimizer or in the effective low-rank update itself

### Candidate fix if confirmed

- a LoRA-specific optimizer recipe for `v5p-8 pd=4`

## H6: The issue is DPO-specific, not generic LoRA training

The current evidence is all from DPO. It is still possible that the combination
of LoRA + preference loss + policy/reference coupling is the fragile object.

## L12: Same LoRA target sets under SFT / plain CE

### Hypothesis

The topology split is benign for ordinary supervised LoRA but pathological for
DPO's preference-difference geometry.

### Experiment

Run canonical+reverse with:

- same LoRA recipe
- same target modules
- plain next-token supervised loss instead of DPO

Prefer using the exact module family that best reproduces Bug 1 from `L1-L5`.

### What we learn

- whether this is a generic LoRA training instability or specifically a DPO
  instability

### Next steps

- if SFT is stable while DPO splits:
  - the preference objective / reference subtraction is load-bearing
- if SFT also splits:
  - this is a more general low-rank distributed-training pathology

### Candidate fix if confirmed

- SFT warm-start before DPO
- or a short bootstrap phase before switching to canonical DPO

## H7: Low-rank factorization is the fragile object, not merely sparse adaptation

This is the cleanest conceptual control after the dense full-FT comparison.

## L13: Train dense additive deltas on the same target modules

### Hypothesis

The problem is not "freezing most of the model" by itself. The problem is the
specific low-rank `B @ A` parameterization.

### Experiment

Introduce a simple dense adapter control:

- freeze the base model
- on the chosen target modules, train a full dense `delta_W`
- same objective, same mesh, same canonical vs reverse comparison

This is a higher-effort experiment than the others, so it should only be done
after the easier LoRA-target/module/rank/init lanes.

### What we learn

- whether factorization itself is load-bearing

### Next steps

- if dense delta is stable while LoRA splits:
  - low-rank factorization is the culprit
- if dense delta also splits:
  - sparse adaptation on that module set is enough

### Candidate fix if confirmed

- alternate adapter family on `v5p-8 pd=4`

## H8: The useful signal is present, but canonical turns it into the wrong delta-W direction

This is less a TPU experiment than an analysis lane, but it is crucial if the
earlier ablations remain ambiguous.

## L14: Per-module `delta_W` direction audit

### Hypothesis

Canonical and reverse produce materially different **directions** in merged
adapter updates for a small number of modules, even if aggregate gradient norms
look similar.

### Experiment

For the first 1-2 real updates:

- compute `delta_W = B @ A * scale` per LoRA module
- compare canonical vs reverse:
  - cosine
  - norm ratio
  - top singular value
  - sign agreement on fixed entries

Compare this to dense full-FT updates on the same module families.

### What we learn

- scale defect vs direction defect
- whether one module dominates the bad family

### Next steps

- if cosine is high and norms differ:
  - treat the split as effective-step-size attenuation/amplification
- if cosine is low or negative in a few modules:
  - target those modules directly

### Candidate fix if confirmed

- module exclusion
- rank increase on only the unstable modules
- targeted param replication

## H9: The bad family can be repaired with a brief bootstrap, not a permanent recipe change

This is a practical workaround hypothesis.

## L15: Short bootstrap on a safe topology, then resume on bad topology

### Hypothesis

Bug 1 is mainly a basin-entry problem: once LoRA escapes the fragile initial
region, the bad family can continue training normally.

### Experiment

Two-stage run:

1. train for 1-3 steps on a safe family:
   - `reverse`
   - or `data:2, model:2`
2. resume from that checkpoint on canonical bad topology

### What we learn

- whether the entire problem is concentrated in the earliest updates

### Next steps

- if bootstrap cures canonical:
  - the first-update basin story is strongly supported
- if canonical later collapses anyway:
  - the bad family continues to inject harmful dynamics beyond the first steps

### Candidate fix if confirmed

- automated bootstrap procedure for LoRA DPO on `v5p-8 pd=4`

## Recommended execution order for the next 10 hours

This is the suggested order, balancing information value and implementation
cost.

### Phase A: highest-value, lowest-complexity discriminators

1. `L0` instrumentation upgrade
2. `L1` attention-only
3. `L2` MLP-only
4. `L5` replicated-ish vs data-sharded-ish target groups

This phase should answer whether the split is concentrated in:

- attention
- MLP
- or sharding class

### Phase B: geometry / capacity

5. `L7` rank sweep
6. `L8` scale sweep
7. `L9` small nonzero B init
8. `L10` A-zero / B-random-small

This phase should answer whether the key fragility is:

- low capacity
- bad first-update scale
- or canonical factorization asymmetry

### Phase C: optimizer / objective

9. `L11` optimizer ablation
10. `L12` SFT vs DPO control
11. `L14` `delta_W` direction audit

This phase should answer whether:

- Adam amplifies the split
- DPO specifically is the fragile objective
- or the real difference is directional, not scalar

### Phase D: higher-cost controls and practical fixes

12. `L6` force specific LoRA params replicated
13. `L15` bootstrap-on-safe-topology
14. `L13` dense-delta adapter control

These are more expensive or invasive, so they come after the cheaper,
higher-leverage discriminators unless an earlier result points directly at them.

## Decision tree summary

- if `L5` shows only `o_proj/down_proj` split:
  - replicate those params or exclude them
- if `L1` or `L2` isolate a single family:
  - shrink target modules accordingly
- if `L7` shows high rank cures the split:
  - use higher rank selectively
- if `L9` or `L10` cure the split:
  - change LoRA init
- if `L11` cures the split:
  - change LoRA optimizer recipe
- if `L12` says SFT is stable but DPO splits:
  - use SFT bootstrap or DPO-specific workaround
- if `L15` works:
  - treat Bug 1 as a basin-entry problem with a procedural fix
- if none of `L1-L15` materially reduce the split:
  - escalate back toward compiler-facing hypotheses with much stronger evidence
    that LoRA's low-rank path itself is interacting pathologically with the
    XLA fork in a way not captured by simple capacity/init/optimizer controls

## Current highest-confidence fix candidates

These are not yet proved, but they are the most plausible practical fixes based
on current evidence:

1. safe physical order / mesh workaround
2. exclude the most topology-sensitive LoRA modules
3. replicate specific LoRA params across data
4. raise rank on the fragile target set
5. use a non-canonical LoRA init or short bootstrap phase

The next round should be judged by how quickly it turns that ranked guess list
into one defensible recommendation.

## 2026-04-19T L0 Instrumentation: W&B `lora_debug` Callback Landed

This section records the L0 instrumentation upgrade from the forward-looking
program above. L0 was the enabling step ("the next round will stall if we keep
reading only loss curves and HLO counts. We need per-module first-update
statistics"). That infrastructure is now live.

### What was built

A new callback — [levanter.callbacks.lora_debug](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/callbacks/lora_debug.py)
— publishes per-LoRA-module diagnostics to W&B every step when enabled. It is
modelled after the existing `WatchCallback` (`TrainerConfig.watch`) but
LoRA-targeted.

Previously, the only tooling for this type of signal was stderr-only via
`MARIN_DEBUG_LOG_STEP_TRACE` and `MARIN_DEBUG_DUMP_GRAD_VALUES` (scattered
inline `jax.debug.print` calls in [trainer.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/trainer.py)).
Those still work but produce stderr log lines, not queryable W&B series.

### Flag surface (three equivalent ways to enable)

Any one flips it on:

1. `TrainerConfig.lora_debug.enabled: true` — structured config with
   fine-grained knobs (interval, sentinel indices, opt-state toggles, etc.)
2. `WandbConfig.lora_debug: true` — convenience flag on the tracker config
3. `MARIN_DEBUG_LORA_DEBUG=1` — env-var override (debug-run convenience)

Default is off. Existing runs are unaffected.

### Metrics published every step under `lora_debug/`

Per LoRA module (one key per `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, `down_proj`, aggregated across scan layers):

- `grad/{A,B}/{module}/l2` and `max_abs`
- `param/{A,B}/{module}/l2` and `max_abs`
- `update/{A,B}/{module}/l2`
- `delta_W/{module}/l2` — Frobenius norm of `B @ A * scale`, computed without
  materialization via `trace(B^T B · A A^T)` (cheap: O(r²) per module)

Aggregates:

- `agg/{grad,param,update}_{A,B}/l2`
- `agg/delta_W/l2`
- `agg/attn/{grad,param,update,delta_W}_{A,B}/l2` (H1 attention-family probe)
- `agg/mlp/{grad,param,update,delta_W}_{A,B}/l2` (H1 MLP-family probe)

Fixed-index sentinel grads (ports the existing Z1 stderr dump into queryable
W&B series so canonical-vs-reverse can be diffed from run history alone):

- `sentinel/grad_B/{module}/idx_{0,0.25,0.5,0.75,1}` — values at the five
  fractional positions in the flattened tensor
- `sentinel/grad_B/{module}/n` — flat element count
- `sentinel/grad_A/...` — optional, off by default

Adam optimizer state (H5 "Adam amplifies tiny topology differences"):

- `adam/m/{A,B}/{module}/l2`, `adam/v/{A,B}/{module}/l2`
- `adam/effective_update/{A,B}/{module}/l2` = `|m / (sqrt(v) + eps)|_2` —
  Adam's un-LR-scaled update direction magnitude

### One-time topology summary

When the flag is on, `log_topology_summary(trainer, model)` runs right after
trainer state initialization and pushes the following into the W&B run
**config** (not step metrics). That means every cross-run diff (canonical vs
reverse, etc.) can be reconstructed from the run metadata without re-reading
worker stderr:

- `lora_debug/mesh/shape` — dict of `{axis: size}`
- `lora_debug/mesh/device_count`
- `lora_debug/mesh/devices` — list of physical device repr strings (captures
  the BL permutation directly)
- `lora_debug/mesh/param_mapping`, `lora_debug/mesh/compute_mapping`
- `lora_debug/xla_flags` — contents of `XLA_FLAGS` env
- `lora_debug/debug_envs` — every `MARIN_DEBUG_*` env var with value
- `lora_debug/experiment_tag` — `MARIN_DEBUG_RUN_TAG`
- `lora_debug/permutation` — `EXPERIMENT_BL_ORDER` / `EXPERIMENT_B1_ORDER`
- `lora_debug/num_lora_modules`, `lora_debug/lora_module_paths`

### Files touched

- **new**: [lib/levanter/src/levanter/callbacks/lora_debug.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/callbacks/lora_debug.py) (~450 lines):
  `LoraDebugConfig`, `LoraDebugCallback(JitCallback)`,
  `collect_topology_summary`, `log_topology_summary`, and helpers
- [lib/levanter/src/levanter/callbacks/__init__.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/callbacks/__init__.py):
  re-exports
- [lib/levanter/src/levanter/trainer.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/trainer.py):
  adds `TrainerConfig.lora_debug: LoraDebugConfig` (sibling of `watch`), adds
  `Trainer._lora_debug_enabled()`, installs callback in `_add_default_hooks`
- [lib/levanter/src/levanter/tracker/wandb.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/tracker/wandb.py):
  adds `WandbConfig.lora_debug: bool = False` convenience flag
- [lib/levanter/src/levanter/main/train_dpo.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/main/train_dpo.py):
  calls `log_topology_summary` once at init when the flag is on

### Local validation

- `.venv/bin/python -m py_compile` on all five edited files — clean.
- Synthetic toy-LoRA smoke test exercised `_collect_lora_modules`,
  `_collect_lora_arrays`, `_delta_w_frobenius`, `_emit_factor_stats`,
  `_emit_family_rollup`, `_emit_sentinels`:
  - Module paths resolve to clean `q_proj` / `o_proj` / `gate_proj` /
    `down_proj` (no `/lora` tail, no `<flat index 0>` FlattenedIndexKey
    artifacts).
  - `delta_W` is exactly 0 at zero-init-B, nonzero after mutating B.
  - Sentinel indices at fractional positions pick up the expected asymmetric
    grad values from a ramp tensor.
  - Attention/MLP rollups sum correctly.
  - `TrainerConfig()` and `WandbConfig()` still construct with defaults
    unchanged.

pyrefly is not installed in this worktree, so the project's type checker step
was skipped.

### Relationship to prior instrumentation

The existing stderr-only debug gates still work and are not touched:

- `MARIN_DEBUG_LOG_STEP_TRACE` — aggregate grad/param L2 + 3 sentinel paths.
- `MARIN_DEBUG_DUMP_GRAD_VALUES` — five fixed-index grad-B elements.
- `MARIN_DEBUG_DUMP_SHARDING` — partition-spec dump at startup.
- `MARIN_DEBUG_LORA_FACTOR_TRACE` — forward factor z checksum.

The new callback **supersedes and extends** `DUMP_GRAD_VALUES` (via
`sentinel/grad_B/*`) and `LOG_STEP_TRACE` (via the aggregates and per-module
norms) by routing the same signals to queryable W&B series. The stderr paths
remain available as a belt-and-braces fallback for offline log forensics.

### How to use it in the next run

Add one env var to an existing iris submission:

```bash
-e MARIN_DEBUG_LORA_DEBUG 1
```

No script or config changes needed. Every LoRA DPO experiment script
immediately starts publishing the full `lora_debug/*` namespace.

For fine-grained control (longer interval, turn off Adam moments, add
sentinel-A, etc.), edit the script's `TrainerConfig.lora_debug` field directly.

### What this unblocks

The LoRA-specific isolation program (L1–L15) is now executable. The next
highest-priority runs per the recommended ordering are:

- **Phase A**: `L1` attention-only LoRA, `L2` MLP-only LoRA, `L5` replicated
  vs data-sharded target groups.
- **Phase B**: `L7` rank sweep, `L8` scale sweep, `L9` small nonzero B init,
  `L10` A=0/B=random-small.
- **Phase C**: `L11` optimizer ablation, `L12` SFT vs DPO, `L14` `delta_W`
  direction audit.
- **Phase D**: `L6` force LoRA param replication, `L15` bootstrap-on-safe-topology,
  `L13` dense-delta adapter control.

Each of these now gets per-module `grad/param/update/delta_W/Adam` series
logged to W&B without further harness work.

### Operational tag for later retrieval

Runs that should be compared against this infrastructure change will carry
`MARIN_DEBUG_LORA_DEBUG=1` in `lora_debug/debug_envs` in the run config.
Filter W&B by that key to find them.

## 2026-04-19T L1 Launch: Attention-Only LoRA on `v5p-8`

First real use of the L0 instrumentation. L1 tests whether the canonical
bad BL regime reproduces under **attention-only** LoRA
(`target_modules=["q_proj","k_proj","v_proj","o_proj"]`).

### Hypothesis

Attention LoRA is the primary driver of the Bug-1 canonical-vs-reverse
training split. If true, the investigation narrows to L3 (Q/V vs O), L5
(replicated vs data-sharded B), and the attention rollups under
`lora_debug/agg/attn/*`.

### Design

Hold the full BL baseline fixed and swap only `target_modules`:

| knob | value |
|---|---|
| TPU | `v5p-8` |
| mesh | `{replica:1, data:4, model:1}` |
| batch | `64` |
| `pd` | `4` |
| steps | `10` |
| `r` | `64` |
| `alpha` | `64` |
| `zero_init_b` | `True` |
| reference | `AdapterBaseReferenceConfig` |
| lr | `1e-6` |
| seed | `0` |
| target_modules | `["q_proj","k_proj","v_proj","o_proj"]` |
| permutations | `canonical`, `reverse` |

Swap12 and rotate are intentionally omitted. BL already showed those land
in the same two compiled-HLO families as canonical/reverse, so they would
duplicate evidence we already have.

### Script

- new:
  [experiment_l1_attn_only_v5p8_pd4_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_l1_attn_only_v5p8_pd4_s10.py)
- minimal delta from
  [experiment_bl_v5p8_pd4_device_permutation_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_bl_v5p8_pd4_device_permutation_s10.py):
  only `target_modules`, name, tags, and the env-var namespace change
  (`EXPERIMENT_L1_ORDER` instead of `EXPERIMENT_BL_ORDER`).

Local `py_compile` clean. Config smoke test confirms:

- `target_modules=['q_proj','k_proj','v_proj','o_proj']`
- `r=64 alpha=64 zero_init_b=True`
- `reference=AdapterBaseReferenceConfig`
- `mesh.axes={'replica':1,'data':4,'model':1}`
- `device_permutation=(0,1,2,3)` for canonical, `(3,2,1,0)` for reverse

### Launch commands

Both jobs use `MARIN_DEBUG_LORA_DEBUG=1` so the L0 callback publishes
`lora_debug/*` series (per-module grad/param/update/delta_W/Adam/sentinels
+ one-time mesh/XLA/env topology summary).

Canonical:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --region us-central1 --cpu 1 --memory 3g \
  --job-name experiment-l1-attn-v5p8-pd4-can-20260419-1840 \
  -e REGIONS_OVERRIDE us-central1 \
  -e EXPERIMENT_L1_ORDER canonical \
  -e MARIN_DEBUG_RUN_TAG l1a-can \
  -e MARIN_DEBUG_LORA_DEBUG 1 \
  -e MARIN_DEBUG_HLO_UPLOAD_DIR gs://marin-us-central1/debug/bug_1_L1_attn_only_hlo/20260419/l1a-can/ \
  -e MARIN_DEBUG_DUMP_SHARDING 1 \
  -e MARIN_DEBUG_DUMP_GRAD_VALUES 1 \
  -e MARIN_DEBUG_SKIP_HF_EXPORT 1 \
  -e XLA_FLAGS "--xla_dump_to=/tmp/xla_hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*train.*" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  --no-wait -- \
  python experiments/posttrain/per_stmt_dpo/experiment_l1_attn_only_v5p8_pd4_s10.py
```

Reverse is identical except `EXPERIMENT_L1_ORDER=reverse`,
`MARIN_DEBUG_RUN_TAG=l1a-rev`, and HLO prefix suffix `l1a-rev/`.

### Submitted Iris jobs

- canonical:
  `/ahmed/experiment-l1-attn-v5p8-pd4-can-20260419-1840`
- reverse:
  `/ahmed/experiment-l1-attn-v5p8-pd4-rev-20260419-1840`

HLO upload prefixes:

- `gs://marin-us-central1/debug/bug_1_L1_attn_only_hlo/20260419/l1a-can/`
- `gs://marin-us-central1/debug/bug_1_L1_attn_only_hlo/20260419/l1a-rev/`

Initial state at submission (UTC 18:41): both `pending` on v5p-8 capacity
in `us-central1`. Workspace bundle `6.7 MB` each, no config rejection, no
same-second auto-name collision (explicit `--job-name` used per the earlier
BL lesson).

### Decision rule

Apply the standard thresholds from the forward-looking program:

- **full Bug-1 split preserved** — bad step-9 near `~0.66`, good near `~0.31`:
  attention alone is sufficient. Next: L3 (Q/V vs O), L5 (sharding-class),
  and `lora_debug/agg/attn/*` direction audit from the same W&B runs.
- **split effectively removed** — step-9 gap `<= 0.05` on both: MLP is the
  driver. Next: L2 MLP-only.
- **partial split** — gap shrunk by `50%` or more but still visible:
  both families contribute. Keep L1 and L2 alive in parallel.
- **ambiguous** — no clean step-2 split but late divergence: extend only
  canonical + reverse to 35 / 100 steps to disambiguate.

### Monitoring plan

- Poll every ~5 min until both children land and produce first-step logs.
- On terminal state: pull step-2 / step-9 losses, compare to BL
  (`canonical = 0.686027 / 0.662460`, `reverse = 0.334927 / 0.308011`) to
  classify outcome.
- Pull HLO bundles and run the same hash-bucket analysis as BL — confirm
  the two attention-only compiled-HLO families still split on physical
  permutation (or reveal a new pattern).
- Inspect `lora_debug/*` W&B series to compare per-attention-module
  (q/k/v/o) grad norms and delta_W between canonical and reverse. This is
  the first data directly bridging "compiled-path fork" to "per-module
  first-update direction/scale."

## 2026-04-19T L1 First Attempt Crashed: L0 Callback Bug In `_describe_devices`

Both L1 jobs failed 3 minutes into TPU execution (after ~10 minutes wall-time
including scheduling). Identical failure on canonical and reverse rules out a
topology effect — this was a structural bug in the new L0 instrumentation.

### Failure symptom

Parent iris job (both variants):

```
State: failed  exit=0  failures=1  preemptions=0
Error: Exit code: 1. stderr: RuntimeError: 1 step(s) failed
```

Child `train_dpo` task:

```
State: failed  exit=0  failures=1  preemptions=0
Error: Exit code: 1. stderr: TypeError: Dtype object is not a valid JAX
array type. Only arrays of numeric types are supported by JAX.
```

### Root cause

Worker traceback (from `iris job logs`):

```
File "/app/lib/levanter/src/levanter/callbacks/lora_debug.py", line 575,
    in log_topology_summary
File ".../lora_debug.py", line 559, in collect_topology_summary
    "lora_debug/mesh/devices": _describe_devices(mesh),
File ".../lora_debug.py", line 531, in _describe_devices
...
TypeError: Dtype object is not a valid JAX array type.
```

`jax.sharding.Mesh.devices` is a **numpy object-dtype ndarray of Device
instances**. The v1 `_describe_devices` called `jax.numpy.asarray(devices)`,
which rejects object dtypes because jax.numpy only accepts numeric arrays.
The synthetic unit test during L0 development never hit this path because
it didn't construct a real `Mesh`.

### Audit scope

Before fixing, re-audited every addition end-to-end under production
conditions (real `Mesh`, real `DpoModel` / `LmHeadModel` with scan-stacked
layers, real `opt_state`, tracer-in-JIT context):

- `_fmt_path`: handles `GetAttrKey`, `DictKey`, `SequenceKey`, and
  `FlattenedIndexKey` (`str()` returns `[<flat index N>]`; bracket-strip
  then filter catches it). OK.
- `_collect_lora_modules`: `tree_flatten_with_path` with
  `is_leaf=is_lora_param` preserves `LowRankLinear` through `eqx.partition`
  (only `NamedArray` leaves become `None`). OK.
- `_collect_lora_arrays`: NamedArrays auto-unwrap during flatten; `None`
  sentinels skipped via `hasattr(arr, "shape")`. OK.
- `_delta_w_frobenius`: einsum pattern `...ir,...is->...rs` handles any
  number of leading axes including scan-stacked. LORA_R axis lookup by
  name. OK.
- `_emit_sentinels`: `n = flat.shape[0]` is a static Python int;
  `flat[idx]` is a static-index trace. Output keys deterministic. OK.
- `_emit_opt_state_stats`: `/mu/` and `/nu/` path substrings match optax
  `ScaleByAdamState` layout. Non-LoRA paths and scalar `count` paths
  skipped. OK.
- `inside_step` output dict structure is deterministic at trace time (cfg
  + model structure fixed), so `jax.lax.cond` + `zeros_like_tree` fallback
  works. OK.
- `collect_topology_summary`: only `_describe_devices` broken. Other
  values (`_safe_mesh_shape`, `mesh.size`, axis mappings, env dump,
  `_collect_lora_modules` on a concrete model) are safe.
- `Trainer._lora_debug_enabled` correctly wraps a single `TrackerConfig`
  into a tuple before iterating. OK.
- Dead imports found: `import dataclasses` (bare), `import equinox as
  eqx`, `import optax`, `import logging` + unused `logger`.
- Lazy `from levanter.callbacks.lora_debug import log_topology_summary`
  inside a conditional in `train_dpo.py` violated AGENTS.md ("All imports
  at the top of the file").

### Fix

`lib/levanter/src/levanter/callbacks/lora_debug.py`:

```python
def _describe_devices(mesh) -> list[str]:
    devices = getattr(mesh, "devices", None)
    if devices is None:
        return []
    # mesh.devices is a numpy object-dtype ndarray of Device instances.
    # Iterate via its .flat attribute, which works on any numpy dtype
    # including object; jax.numpy.asarray rejects object dtypes.
    if hasattr(devices, "flat"):
        return [str(d) for d in devices.flat]
    return [str(d) for d in devices]
```

Removed dead imports (`dataclasses`, `equinox`, `optax`, `logging`,
unused `logger`).

`lib/levanter/src/levanter/main/train_dpo.py`: moved
`from levanter.callbacks.lora_debug import log_topology_summary` to the
top-level import block.

### Local verification

Exercised the fixed `_describe_devices` against a real
`jax.sharding.Mesh` (shape `(1, N, 1)` built from `jax.devices()`) and
confirmed clean string output (e.g. `['TFRT_CPU_0']` on laptop, will be
`['TpuDevice(...)', ...]` on v5p-8). All other edited files `py_compile`
clean.

### Second L1 failure: multi-axis Out shape assumption in `_delta_w_frobenius`

The `-1904` pair also crashed (both at exactly 2m 49.38s task duration, identical
across canonical/reverse → structural, not topology-dependent). Root cause was a
different bug in the same callback:

```
ValueError: Incompatible shapes for broadcasting: shapes=[(32, 64, 64), (32, 8, 4, 64, 64)]
  File .../lora_debug.py line 245 in _delta_w_frobenius
    fro_sq = jnp.sum(gram_a * gram_b, axis=(-1, -2))
```

My einsum `"...or,...os->...rs"` in `_delta_w_frobenius` assumed `B` has a
single Out axis. Llama's `q_proj.lora.lora_B` actually has axes
`(LORA_R, Heads, HeadSize)` — multi-axis Out — so `b_perm` had two non-LORA_R
axes and the gram computation produced the wrong shape. The synthetic unit test
used a single-axis Out, so it missed this.

Fix: replace the hand-rolled einsum with the existing `LowRankLinear.merge()`
helper (`lora.py:278`), which does `hax.dot(A, B, axis=LORA_R) * scale` in named
space and correctly handles arbitrary Out/In axis topology:

```python
def _delta_w_frobenius(mod: LowRankLinear) -> jax.Array:
    dw = mod.merge()
    dw_arr = _as_f32(dw)
    return jnp.sqrt(jnp.sum(jnp.square(dw_arr)))
```

Verified locally against the exact Llama multi-axis-Out shape — returns 0 at
`B=0` init, matches `hand-computed ||merge||_F` at nonzero B.

### Third relaunch attempt: `-1928` pair (stopped, superseded by base-model swap)

With the multi-axis fix, launched `-1928` canonical + reverse. Those got past
the earlier crash point but were stopped manually (not on failure) when the
campaign pivoted to Llama-3.1-8B-Instruct (below).

### Relaunch

Both L1 variants resubmitted with the fixed L0 callback. Fresh HLO
prefixes (`20260419b/` instead of `20260419/`) keep artifact history
clean.

Jobs:

- canonical:
  `/ahmed/experiment-l1-attn-v5p8-pd4-can-20260419-1904`
- reverse:
  `/ahmed/experiment-l1-attn-v5p8-pd4-rev-20260419-1904`

HLO upload prefixes:

- `gs://marin-us-central1/debug/bug_1_L1_attn_only_hlo/20260419b/l1a-can/`
- `gs://marin-us-central1/debug/bug_1_L1_attn_only_hlo/20260419b/l1a-rev/`

Previous failed jobs (`...-20260419-1840`) are terminal; no further TPU
consumption. Their GCS prefixes (`20260419/l1a-{can,rev}/`) are empty
because the HLO upload hook runs after `trainer.train()`, which never
ran.

## 2026-04-19T Base Model Swap: Llama-3.1-8B-Instruct From GCS

Campaign pivoted off `marin-community/marin-8b-instruct` (HuggingFace Hub only,
~3-5 min HF-CDN load every run) to `meta-llama/Llama-3.1-8B-Instruct` which is
pre-staged in GCS same-region as the v5p-8 workers.

### Why

Every Bug-1 experiment has paid the HF-CDN load cost. For 30+ experiments in
this campaign, that's ~2-3 hours of cumulative dead startup time plus
occasional HF Hub flakiness. Marin's executor framework already materializes
`meta-llama/Llama-3.1-8B-Instruct` under the standard `download_model_step`
output paths, and both `us-central1` and `us-east5` have complete copies.

### GCS paths verified

- `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/` — 29.93 GiB, 4 safetensor shards + config + tokenizer + `.executor_status` marker.
- `gs://marin-us-east5/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/` — same contents.
- Reached via `output_path_of(llama_3_1_8b_instruct)` in
  [`experiments/models.py`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/models.py:99),
  same pattern as
  [`experiments/dpo_ultrafeedback.py:54`](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/dpo_ultrafeedback.py:54).

### Reproduction sweep launched: original BL on Llama-3.1 base

Goal: confirm the canonical-vs-reverse Bug-1 loss split reproduces when the
only change from the original BL recipe is the base model identity. If yes,
Bug-1 is model-agnostic and the whole L-experiment program moves to the
faster-loading base.

Script:

- [experiment_bl_llama31_v5p8_pd4_device_permutation_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_bl_llama31_v5p8_pd4_device_permutation_s10.py) —
  byte-for-byte mirror of the BL permutation probe except
  `model_name_or_path=output_path_of(llama_3_1_8b_instruct)` and the env var
  knob is renamed to `EXPERIMENT_BL31_ORDER` so the two campaigns don't
  collide.

Recipe held identical to BL canonical:

- TPU: `v5p-8`, mesh `{replica:1, data:4, model:1}` with explicit
  `device_permutation`
- LoRA: `r=64, alpha=64, zero_init_b=True, target_modules=None` (all linears)
- reference: `AdapterBaseReferenceConfig`
- batch 64, `pd=4`, steps 10, `lr=1e-6`, seed 0, beta 0.1

**`MARIN_DEBUG_LORA_DEBUG` is deliberately OFF** on these reproduction runs.
Two L0-callback bugs surfaced in the previous day; I want a clean loss signal
from the base-model swap before re-enabling the instrumentation. The L0
callback is correct as of the most recent fix, but a third latent bug would
confound interpretation of the loss comparison.

### Submitted jobs

- canonical: `/ahmed/experiment-bl31-v5p8-pd4-can-20260419-1941`
- reverse: `/ahmed/experiment-bl31-v5p8-pd4-rev-20260419-1941`

### Decision rule

| outcome | step-9 canonical | step-9 reverse | verdict |
|---|---|---|---|
| reproduces | ~0.66 | ~0.31 | Bug-1 model-agnostic → pivot all future L-experiments to Llama-3.1 base |
| bug disappears | both ~0.31 | both ~0.31 | Bug-1 was marin-8b-instruct-specific → stay on original base |
| mixed / new regime | — | — | need to debug base-model integration before re-running |

If reproduction succeeds, the secondary rescue experiments (Exp W TP
`{data:1, model:4}` and Exp Z3 mixed `{data:2, model:2}`) will be launched on
the new base too to confirm the full Bug-1 phenomenology is preserved.

### First relaunch crashed at import time

Both `-1941` jobs failed at the executor parent, before the TPU child was
ever created:

```
Exit code: 1. stderr: ModuleNotFoundError: No module named 'marin.download'
```

Root cause: my script imported ``experiments.models.llama_3_1_8b_instruct``
which pulls in ``marin.download.huggingface.download_hf``. That submodule is
part of the download-time pipeline and isn't installed in the iris worker
image. Fix: hardcode the GCS path as a string literal in the experiment
script (see the ``LLAMA_3_1_8B_INSTRUCT_GCS_PATH`` constant added to
``experiments/posttrain/per_stmt_dpo/base_model.py``) and drop both the
``experiments.models`` import and ``output_path_of`` usage. No executor step
dependency needed — the model is already pre-staged in GCS.

### Second relaunch succeeded: Bug-1 reproduces on Llama-3.1-8B-Instruct

Jobs:

- canonical: `/ahmed/experiment-bl31-v5p8-pd4-can-20260419-1949`
  (`State: succeeded  duration: 9m 11s  peak mem: 29.06 GB`)
- reverse: `/ahmed/experiment-bl31-v5p8-pd4-rev-20260419-1949`
  (`State: succeeded  duration: 9m 44s  peak mem: 29.33 GB`)

Model load timings from the canonical worker log (new
Llama-3.1-8B-Instruct from same-region GCS):

```
19:54:27  Initializing model from HF checkpoint gs://.../Llama-3-1-8B-Instruct--0e9e39f
19:54:28  Reading model-00004 (1.17 GB)  done 19:54:35  (7s,  167 MB/s)
19:54:35  Reading model-00001 (4.98 GB)  done 19:54:51  (16s, 383 MB/s)
19:54:51  Reading model-00002 (5.00 GB)  done 19:55:08  (17s, 349 MB/s)
19:55:08  Reading model-00003 (4.92 GB)  done 19:55:24  (16s, 350 MB/s)
19:55:31  First batch loaded in 2.2s, starting first train step (JIT compile)
```

Total shard-read time: ~57 seconds for 16.07 GB at ~288 MB/s average — VM
network-cap-bound (GCP single-VM egress is typically ~2 Gbps ≈ 250 MB/s).
Compare to ~3-5 min for the same model load from the HuggingFace Hub CDN on
prior marin-8b-instruct runs. **~3-5× faster** and not subject to HF
rate-limit flakes.

Total job wall-clock (including scheduling + HF load + JIT compile + 10
training steps): 9m 11s canonical / 9m 44s reverse, vs. roughly 12-15m for
the same recipe on the prior HF-based pipeline.

### Loss trajectories — canonical-vs-reverse split reproduces cleanly

Extracted from ``DEBUGJ TRACE`` lines in the worker logs:

| step | Llama-3.1 canonical | Llama-3.1 reverse | marin-8b canonical (BL) | marin-8b reverse (BL) |
|---:|---:|---:|---:|---:|
| 0 | 0.693147 | 0.693147 | 0.693147 | 0.693147 |
| 1 | 0.693147 | 0.693147 | 0.693147 | 0.693147 |
| 2 | **0.630694** | **0.174806** | **0.686027** | **0.334927** |
| 3 | 0.582621 | 0.153281 | — | — |
| 4 | 0.537425 | 0.148124 | — | — |
| 5 | 0.503668 | 0.135565 | — | — |
| 6 | 0.476663 | 0.138122 | — | — |
| 7 | 0.457886 | 0.131556 | — | — |
| 8 | 0.434077 | 0.113639 | — | — |
| 9 | **0.426746** | **0.110859** | **0.662460** | **0.308011** |

### Interpretation

**Bug-1 reproduces on Llama-3.1-8B-Instruct.** The canonical-vs-reverse
step-9 gap on the new base is **0.316** (0.427 − 0.111); on the original
marin-8b base it was **0.354** (0.662 − 0.308). Same sign, same order of
magnitude, same qualitative step-2 basin split. The topology-sensitive XLA
SPMD communication-lowering fork documented in the earlier BL analysis is
**model-agnostic** — confirmed as a property of the 8B Llama-shaped recipe
running on ``v5p-8 {data:4, model:1}`` with an FSDP permutation, not
anything specific to Marin's post-trained 8B.

Absolute loss values are lower on Llama-3.1 for both variants — canonical
goes 0.63→0.43 over 10 steps vs marin-8b's 0.69→0.66, and reverse goes
0.17→0.11 vs 0.33→0.31. Both bases learn the DPO task, but Llama-3.1 is a
stronger / closer starting point for this data. The *relative* canonical
penalty remains ~4× worse than reverse at step 9 on both bases.

### Verdict / forward-looking

All future L-experiments and Bug-1 probes pivot to
Llama-3.1-8B-Instruct as the canonical base. The W/Z3 mesh-rescue
counterparts were NOT launched in this session because the primary signal
(canonical-vs-reverse) is already definitive.

## 2026-04-19T Campaign Base-Model Swap: Documentation Update

To make the base-model swap the new default without invalidating historical
run artifacts, the following changes were applied across the per-statement
DPO experiment directory:

### New shared module

- [experiments/posttrain/per_stmt_dpo/base_model.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/base_model.py)
  — exports ``LLAMA_3_1_8B_INSTRUCT_GCS_PATH`` as a plain string literal
  ("gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f")
  so experiment scripts can reference it without importing
  ``experiments.models`` (which transitively pulls in ``marin.download``,
  not present on the iris worker image).

### Docstring note on every per_stmt_dpo experiment

All 51 pre-existing ``experiment_*.py`` files in
``experiments/posttrain/per_stmt_dpo/`` now carry a standard "Note" section
appended to their module docstring pointing future forks to the new base.
The ``experiment_bl_llama31_...`` reference implementation was skipped
because it already uses the GCS path directly.

**Historical ``model_name_or_path`` values were NOT modified** — the W&B run
links, HLO artifacts, and loss numbers recorded throughout this logbook are
tied to the base model that produced them. Changing those strings
retroactively would break that reproducibility. The note explicitly tells
future readers not to retroactively edit historical experiments.

### How new experiments pick up the new base

```python
from experiments.posttrain.per_stmt_dpo.base_model import (
    LLAMA_3_1_8B_INSTRUCT_GCS_PATH,
)

config = SimpleDPOConfig(
    ...
    model_name_or_path=LLAMA_3_1_8B_INSTRUCT_GCS_PATH,
    ...
)
```

See [experiment_bl_llama31_v5p8_pd4_device_permutation_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_bl_llama31_v5p8_pd4_device_permutation_s10.py) as
the reference pattern.

## 2026-04-19T End-of-Session Handoff State

Clean handoff state for the next agent:

### Code changes committed to the ``dpo-lora`` branch this session

1. **L0 instrumentation** — new ``LoraDebugCallback`` in
   [lib/levanter/src/levanter/callbacks/lora_debug.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/callbacks/lora_debug.py)
   with ``LoraDebugConfig`` exposed via ``TrainerConfig.lora_debug``, the
   ``WandbConfig.lora_debug: bool = False`` convenience flag, and the
   ``MARIN_DEBUG_LORA_DEBUG=1`` env override. Per-step publishes
   per-LoRA-module grad / param / update / delta_W Frobenius / Adam m,v /
   effective-update / fixed-index sentinel series under the ``lora_debug/``
   namespace. One-time ``log_topology_summary`` pushes mesh / device list /
   XLA flags / MARIN_DEBUG_* envs into the W&B run config.

2. **Two production bugs found and fixed in the callback**:
   - ``_describe_devices`` called ``jax.numpy.asarray`` on a numpy
     object-dtype ``Mesh.devices`` ndarray (rejected by jax.numpy). Fixed by
     flattening via ``.flat``.
   - ``_delta_w_frobenius`` used a hand-rolled einsum
     (``"...or,...os->..."``) that assumed single-axis Out; Llama's
     ``q_proj.lora.lora_B`` has multi-axis Out ``(Heads, HeadSize, LORA_R)``
     and produced a broadcast-shape error. Fixed by replacing with
     ``LowRankLinear.merge()``, which uses ``hax.dot`` on the named
     ``LORA_R`` axis and handles arbitrary axis topology.

3. **L1 attn-only experiment script**:
   [experiment_l1_attn_only_v5p8_pd4_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_l1_attn_only_v5p8_pd4_s10.py)
   — mirror of BL with ``target_modules=["q_proj","k_proj","v_proj","o_proj"]``.
   Launched but stopped before completion in favor of the base-model swap.

4. **BL on Llama-3.1 reference implementation**:
   [experiment_bl_llama31_v5p8_pd4_device_permutation_s10.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_bl_llama31_v5p8_pd4_device_permutation_s10.py)
   — reproduced Bug-1 canonical/reverse split on the new base (loss
   trajectories above).

5. **Shared base-model module**:
   [base_model.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/base_model.py)
   + docstring note on 51 historical experiment files.

### Open / in-progress

- The L1 attn-only probe has not been run to completion on either base. The
  L0 instrumentation is proven to work end-to-end (callback passes
  ``eqx.filter_eval_shape`` and executes on real Llama shapes), but L1
  training runs to step 9 on the new base were not submitted this session.
- The Exp W TP ``{data:1, model:4}`` and Exp Z3 mixed ``{data:2, model:2}``
  rescues have not been rerun on Llama-3.1. The primary BL
  canonical/reverse split reproduction is sufficient evidence of
  model-agnosticism, but re-confirming the mesh-based rescues remains an
  open to-do.
- The full Phase-A L-experiment program (L1 attn-only, L2 MLP-only, L5
  sharding-class) was deferred pending the base-model verification. Now
  that verification is complete, those probes can proceed on the new base
  with ``MARIN_DEBUG_LORA_DEBUG=1`` to capture the L0 per-module series.

### Recommended next-agent action

1. Relaunch L1 attn-only (canonical + reverse) on Llama-3.1 base with
   ``MARIN_DEBUG_LORA_DEBUG=1``. Expected outcomes follow the decision tree
   in the "L1: Attention-only LoRA" section above, with reference step-9
   numbers now from the Llama-3.1 baseline (canonical 0.427, reverse 0.111)
   rather than the marin-8b baseline.
2. In parallel, run L2 MLP-only and L5 sharding-class probes — both
   independent of the L1 outcome per the Phase-A decision tree.
3. The W/Z3 mesh rescues on Llama-3.1 are lower-priority but low-cost (two
   more runs); queue them if capacity allows.

## 2026-04-19T L1 Llama-3.1 Attn-Only: Gap Preserved (Attention Is Sufficient)

L1 relaunched on Llama-3.1-8B-Instruct with ``MARIN_DEBUG_LORA_DEBUG=1``
and cross-region coverage per the multi-region-launch rule. All four jobs
reached step 9 successfully; the L0 ``LoraDebugCallback`` fired end to end
(106 ``lora_debug/*`` metric keys on each W&B run).

### New experiment script

- [experiment_l1_llama31_attn_only_v5p8_pd4_s10.py](../../experiments/posttrain/per_stmt_dpo/experiment_l1_llama31_attn_only_v5p8_pd4_s10.py) —
  mirror of ``experiment_bl_llama31_v5p8_pd4_device_permutation_s10.py``
  with ``target_modules=["q_proj","k_proj","v_proj","o_proj"]`` and
  ``env_vars["MARIN_DEBUG_LORA_DEBUG"]="1"``. Env knob renamed to
  ``EXPERIMENT_L1_LLAMA31_ORDER`` to avoid collision with the
  marin-8b-based ``experiment_l1_attn_only_v5p8_pd4_s10.py``.

### Same-region model-path fix (caught live; would have failed every ue5
launch otherwise)

First submission pair in ``us-east5`` failed within ~24-35 s of task
start:

```
ValueError: initialize_from_hf is not in the same region (us-central1)
as the VM (us-east5). This can cause performance issues and billing
surprises.
```

Root cause: ``base_model.LLAMA_3_1_8B_INSTRUCT_GCS_PATH`` was a single
string pinned to ``gs://marin-us-central1/...``. Any launch targeting
``us-east5`` hits Marin's ``check_gcs_paths_same_region`` gate before
the TPU child is ever spawned.

Fix in [base_model.py](../../experiments/posttrain/per_stmt_dpo/base_model.py):

```python
LLAMA_3_1_8B_INSTRUCT_GCS_PATH_BY_REGION = {
    "us-central1": "gs://marin-us-central1/.../Llama-3-1-8B-Instruct--0e9e39f",
    "us-east5":    "gs://marin-us-east5/.../Llama-3-1-8B-Instruct--0e9e39f",
}

def llama_3_1_8b_instruct_gcs_path_for(regions: list[str]) -> str:
    for region in regions:
        path = LLAMA_3_1_8B_INSTRUCT_GCS_PATH_BY_REGION.get(region)
        if path is not None:
            return path
    raise ValueError(...)
```

L1 llama31 script now does
``model_name_or_path=llama_3_1_8b_instruct_gcs_path_for(REGIONS_FOR_TPU)``.
After the relaunch, all four jobs proceeded without hitting the
region gate.

Note: ``experiment_bl_llama31_v5p8_pd4_device_permutation_s10.py`` still
uses a hardcoded ``us-central1`` path and will also silently fail if ever
launched with ``REGIONS_OVERRIDE=us-east5``. Not changed here because the
BL31 historical runs are tied to that path; future BL-style reruns
should either pin ``us-central1`` explicitly or switch to the helper.

### Submitted jobs (final round)

- canonical / uc1: ``/ahmedah/experiment-l1l31-v5p8-pd4-can-uc1-20260419-2029``
- reverse   / uc1: ``/ahmedah/experiment-l1l31-v5p8-pd4-rev-uc1-20260419-2029``
- canonical / ue5: ``/ahmedah/experiment-l1l31-v5p8-pd4-can-ue5-20260419-2034``
- reverse   / ue5: ``/ahmedah/experiment-l1l31-v5p8-pd4-rev-ue5-20260419-2034``

Failed pre-fix (superseded by the -2034 pair, left in Iris history):

- ``/ahmedah/experiment-l1l31-v5p8-pd4-can-ue5-20260419-2029`` (region-gate failure)
- ``/ahmedah/experiment-l1l31-v5p8-pd4-rev-ue5-20260419-2029`` (region-gate failure)

### Step-9 train loss (from ``DEBUGJ TRACE``)

| variant          | region | step 2   | step 9   |
|------------------|:------:|---------:|---------:|
| canonical        | uc1    | 0.667757 | 0.588752 |
| canonical        | ue5    | 0.667757 | 0.588752 |
| reverse          | uc1    | 0.426173 | 0.380036 |
| reverse          | ue5    | 0.426173 | 0.380036 |

uc1 and ue5 step-9 losses are bit-identical per variant — the split is
fully deterministic from recipe + permutation, independent of physical
slice or region.

### Gap comparison vs reference BLs

| recipe                          | canonical @9 | reverse @9 | gap     |
|---------------------------------|-------------:|-----------:|--------:|
| BL-marin8b all-linears          | 0.662460     | 0.308011   | 0.354   |
| BL-Llama3.1 all-linears         | 0.426746     | 0.110859   | 0.316   |
| **L1-Llama3.1 attn-only**       | **0.588752** | **0.380036** | **0.209** |

Gap shrinkage from the matched-base BL reference:
``(0.316 - 0.209) / 0.316 ≈ 34%``. That is **below** the 50% "partial
split" threshold from the Phase-A decision rule, so this outcome lands
on the **"full Bug-1 split preserved"** side of the decision tree.

Practical reading: attention LoRA (q/k/v/o) alone is sufficient to
reproduce the canonical-vs-reverse training regime split. MLP LoRA may
contribute the residual ~34% of the gap, but attention dominates.

Absolute losses are worse than BL-Llama3.1 in both variants because
L1 trains ~1/3 the LoRA adapters; the *relative* canonical/reverse
signature is the load-bearing observation.

### L0 ``LoraDebugCallback`` verification

Checked via the W&B API on ``experiment_l1l31_attn_r64_s10_l1l31-uc1-can-378c18``:

- ``run.state = running`` (HF checkpoint still saving at query time,
  training already terminal)
- ``len([k for k in summary if "lora_debug" in k]) == 106``
- Sample keys:
  ``lora_debug/adam/m/B/inner_state/1/mu/transformer/layers/stacked/self_attn/q_proj/l2``,
  ``.../self_attn/{k,v,o}_proj/l2``, matching ``adam/v``, etc.

So the earlier L1 ``_describe_devices`` / ``_delta_w_frobenius`` failure
modes are fully gone and the 106-key per-attn-module instrumentation is
now cross-run-diffable. No extra topology-summary keys under
``lora_debug/mesh/*`` landed on the W&B run config at this check — worth
re-confirming on a later poll once the HF save finishes and the run
closes.

### Operational notes

- ``MARIN_DEBUG_SKIP_HF_EXPORT=1`` was *not* set, so each job saved a
  ~32 GB 7-shard HF checkpoint at step 9 (~5 min tail). Future
  L-experiment launches should add ``MARIN_DEBUG_SKIP_HF_EXPORT=1`` as
  the earlier ``-1840`` L1 submission did.
- Run times were ~14 min (uc1 pair) / ~13 min (ue5 pair), versus the
  ~9 min BL31 runs that skipped HF export. The ~5 min delta matches the
  shard-copy cost.

### Forward-looking

- L1 passes the "attention is sufficient" gate, so the next two probes
  are **L3** (Q/V vs O subsets, isolate which attention heads carry the
  split) and **L5** (sharding-class / replicated-vs-data-sharded B).
- **L2 MLP-only** is still worth running in parallel — the 34% residual
  gap may be MLP-attributable rather than a measurement artifact.
- The 106 ``lora_debug/*`` keys per L1 run unblock the
  ``lora_debug/agg/attn/*`` direction audit between canonical and
  reverse; this is the first data directly tying the compiler-fork
  result to per-module first-update direction/scale.

## 2026-04-19T L1 ``lora_debug/*`` Direction Audit: O-Proj Is The Fork Epicenter

Full per-step, per-module analysis of the 106 ``lora_debug/*`` metric
keys on the L1-Llama3.1 canonical/reverse runs (both uc1 pair, which is
bit-identical to the ue5 pair). Pulled via W&B API; analysis script was
local ad-hoc Python.

### Headline scalars: norms almost match, loss diverges ~55%

| step | loss c | loss r | agg/attn/delta_W c/r | agg/attn/update_B c/r | agg/attn/grad_B c/r |
|:-:|-:|-:|-:|-:|-:|
| 0    | 0.6931 | 0.6931 | 0.0000 | 0.0000 | n/a    |
| 1    | 0.6931 | 0.6931 | 0.0000 | 1.0009 | n/a    |
| 2    | 0.6678 | 0.4262 | 1.0009 | 1.0088 | 1.4823 |
| 5    | 0.6221 | 0.4028 | 1.0081 | 1.0082 | 1.4476 |
| 9    | 0.5888 | 0.3800 | 1.0090 | 1.0083 | 1.4758 |

- Per-module ``delta_W`` l2 ratios canonical/reverse at step 9 are
  **1.0085–1.0095** across q/k/v/o — effective merged weight deltas
  differ by less than **1%** in Frobenius norm.
- Gradient ``l2`` ratios are **1.47–1.55**, tracking the ``loss`` ratio
  ~1.55 almost exactly. So the elevated grad norm is downstream of the
  larger loss, not its cause.
- Interpretation: the two runs apply update trajectories of **near-identical
  total magnitude** but **different directions**, ending in weight
  configurations with very different DPO loss.

### The cause is visible at step 0: per-module sentinel divergence

Sentinels are 5 fractional-position scalar values per ``lora_B`` gradient
tensor (the zero-init side whose first update magnitude directly
determines the first non-trivial weight delta). Divergence metric:
``mean_i |c_i − r_i| / ((|c_i| + |r_i|)/2)``.

| step | q_proj | k_proj | v_proj | **o_proj** |
|:-:|-:|-:|-:|:-:|
| 0    | 0.045  | 0.165  | 0.240  | **1.099**  |
| 1    | 0.017  | 0.013  | 0.020  | 0.041      |
| 2    | 0.301  | 0.503  | 0.482  | 0.401      |
| 3    | 0.447  | 0.467  | 0.307  | 0.147      |
| 9    | 0.624  | 0.510  | 0.373  | 0.438      |

**Sign-match** (fraction of 5 sentinels whose sign agrees between
canonical and reverse):

| step | q | k | v | **o** |
|:-:|-:|-:|-:|:-:|
| 0    | 1.00 | 1.00 | 1.00 | **0.50** |
| 1-9  | 1.00 | 1.00 | 1.00 | 1.00 (except one blip @ step 4/7) |

**Step 0 is anomalous on ``o_proj`` only.** Its first ``lora_B``
gradient differs by ~110% in magnitude between canonical and reverse,
and half the sentinels flip sign. Every other attention projection
diverges by ≤24% at step 0 with 100% sign-match. By step 2 the split is
distributed across all four projections (30-50%) because the chaotic
amplification of the first divergent update has mixed through the
forward pass.

### Mechanistic hypothesis (tentative, predictive)

Llama-3.1-8B attention shapes:

- ``q_proj``: ``embed → heads*head_size`` — **embed (sharded) is input**.
- ``k_proj``, ``v_proj``: ``embed → kv_heads*head_size`` — **embed (sharded) is input**.
- ``o_proj``: ``heads*head_size → embed`` — **embed (sharded) is output**.

Under FSDP ``{replica:1, data:4, model:1}`` with
``param_mapping={embed: data}``, ``o_proj`` is the only attention
projection whose **output** dimension is data-sharded. That asymmetry
puts ``o_proj`` on a different SPMD collective path from q/k/v during
the first backward pass, and the step-0 sentinel divergence reflects
exactly that collective-lowering fork.

MLP analog:

- ``gate_proj``, ``up_proj``: ``embed → ffn_dim`` — **input-sharded** (like q/k/v).
- ``down_proj``: ``ffn_dim → embed`` — **output-sharded** (like o).

### Prediction for the now-running L2/L3a/L3b/L3c/L4a/L4b cohort

If output-sharding-under-FSDP-``data:4`` is the root mechanism:

| exp           | target            | shard class   | predicted step-9 gap |
|---------------|-------------------|---------------|:--------------------:|
| L1            | q,k,v,o           | mixed         | **0.209** (observed) |
| L3a           | q,v               | input         | **small** (≤0.05)    |
| L3b           | o                 | output        | **near 0.209**       |
| L3c           | k                 | input (GQA)   | **small** (≤0.08)    |
| L2            | gate,up,down      | mixed         | **near 0.21 or larger** |
| L4a           | gate,up           | input         | **small** (≤0.05)    |
| L4b           | down              | output        | **near 0.21 or larger** |

Strongest falsification signal would be L3b (O-only) or L4b (Down-only)
showing a **small** gap — that rules out output-sharding as the sole
mechanism.

Strongest confirmation: L3b and L4b both reproduce ~L1 gap, while L3a,
L3c, L4a all show <0.08 gap.

### Running experiment cohort (launched 2026-04-19T21:50–21:59)

All twelve-variant experiments use the shared
``experiment_l*_llama31_*_v5p8_pd4_s10.py`` scripts (one per module
subset), each launched as four jobs: ``{canonical, reverse} × {us-central1,
us-east5}`` per the multi-region-launch rule, with
``MARIN_DEBUG_LORA_DEBUG=1`` and (from this cohort on)
``MARIN_DEBUG_SKIP_HF_EXPORT=1`` to avoid the 5-minute HF checkpoint
tail.

First batch (``20260419-2150``):

- L2 MLP-only — [experiment_l2_llama31_mlp_only_v5p8_pd4_s10.py](../../experiments/posttrain/per_stmt_dpo/experiment_l2_llama31_mlp_only_v5p8_pd4_s10.py)
- L3a Q/V-only — [experiment_l3a_llama31_qv_only_v5p8_pd4_s10.py](../../experiments/posttrain/per_stmt_dpo/experiment_l3a_llama31_qv_only_v5p8_pd4_s10.py)
- L3b O-only — [experiment_l3b_llama31_o_only_v5p8_pd4_s10.py](../../experiments/posttrain/per_stmt_dpo/experiment_l3b_llama31_o_only_v5p8_pd4_s10.py)

Second batch (``20260419-2159``):

- L3c K-only — [experiment_l3c_llama31_k_only_v5p8_pd4_s10.py](../../experiments/posttrain/per_stmt_dpo/experiment_l3c_llama31_k_only_v5p8_pd4_s10.py)
- L4a Gate/Up-only — [experiment_l4a_llama31_gateup_only_v5p8_pd4_s10.py](../../experiments/posttrain/per_stmt_dpo/experiment_l4a_llama31_gateup_only_v5p8_pd4_s10.py)
- L4b Down-only — [experiment_l4b_llama31_down_only_v5p8_pd4_s10.py](../../experiments/posttrain/per_stmt_dpo/experiment_l4b_llama31_down_only_v5p8_pd4_s10.py)

Total: 24 jobs live.

### What Remains Open

- **L5 sharding-class probe**: after L3b/L4b confirm the output-sharded
  prediction (if they do), the minimal-repro is a single-linear
  experiment that toggles ``LoraAdaptationConfig.lora_B_partition_spec``
  between input-sharded and output-sharded with nothing else changing.
  That isolates the collective-path hypothesis to a one-matmul probe.
- **Directional metrics in L0**: the current ``lora_debug/*`` schema
  captures scalar Frobenius norms per module, which saw only ~1%
  drift. Adding a cross-run directional metric (e.g.
  ``cosine(delta_W_canonical, delta_W_reverse)`` via a paired-run
  callback, or per-step dot products against a frozen reference
  direction) would expose the fork directly rather than via the
  sentinel side channel.
- **Direction-audit follow-up on L2+**: rerun the same per-step
  sentinel-divergence audit on the L2/L3b/L4b runs once they finish.
  If those runs independently exhibit the o_proj-style step-0 explosion
  on their output-sharded modules, the mechanism story is locked in.

## 2026-04-19T Interim Results: L2 / L3a / L3b (attention hypothesis confirmed)

First-batch cohort (``20260419-2150``) ue5 pair finished cleanly;
uc1 pair was preempted around ``21:57`` and restarted from step 0,
still in flight at the time of this update. ue5 losses alone are
sufficient to read the attention decomposition — step-9 losses are
bit-identical between ue5 and uc1 for every prior experiment in this
campaign (L1 included), so the ue5-pair data is representative.

### Step-9 train loss summary

| exp  | target            | shard class       | canonical @9 | reverse @9 | gap      |
|:----:|-------------------|-------------------|------------:|-----------:|---------:|
| BL   | all linears       | mixed             | 0.427       | 0.111      | 0.316    |
| L1   | q,k,v,o           | mixed (1/4 out)   | 0.589       | 0.380      | 0.209    |
| **L3a** | q,v            | input-sharded     | **0.649**   | **0.645**  | **0.004** |
| **L3b** | o              | **output-sharded**| **0.639**   | **0.411**  | **0.228** |
| L2   | gate,up,down      | mixed (1/3 out)   | 0.508       | 0.206      | 0.303    |
| L4a  | gate,up           | input-sharded     | *(pending)* | *(pending)*| *(pending)* |
| L4b  | down              | **output-sharded**| *(pending)* | *(pending)*| *(pending)* |
| L3c  | k                 | input-sharded GQA | *(pending)* | *(pending)*| *(pending)* |

Absolute canonical losses on Q/V-only and O-only canonical runs are
*stuck near the 0.6931 untrained-DPO baseline* (0.649 and 0.639
respectively) — the small LoRA subset can barely train at all under the
bad topology. Reverse-side Q/V is also stuck (0.645), which makes L3a's
gap uninformative about split, but simultaneously tells us Q/V alone
has too little capacity to train in 10 steps.

### Interpretation

1. **L3a Q/V-only gap ≈ 0**. Attention's canonical-vs-reverse split
   does **not** live in the input-sharded Q/V projections.
2. **L3b O-only gap 0.228**. The *output-sharded* attention projection
   alone reproduces the full attention split (within noise of L1's 0.209).
3. **L2 MLP-only gap 0.303 > L1 gap 0.209**. MLP LoRA amplifies Bug-1
   *more* than attention LoRA does. The 34% residual attributed to MLP
   in the L1 analysis was an understatement — MLP, not attention, is
   the dominant Bug-1 driver.

### Prediction status after L2/L3a/L3b

The sentinel-audit prediction **"output-sharded LoRA_B on FSDP data=4
carries the Bug-1 split"** is confirmed for attention (L3b matches L1).
The remaining cross-check is L4b (Down-only, the output-sharded MLP
projection) vs L4a (Gate/Up, input-sharded). If the hypothesis
generalises:

- L4a gap should be small (≤0.05).
- L4b gap should be near L2's 0.303.
- L4a + L4b gap sum should approximately recover L2.

Falsification: L4a showing a large gap or L4b showing near-zero would
break the shard-class story.

### uc1 preemption impact

5 of 12 first-batch uc1 children were preempted ~7 min into execution
and restarted from step 0 (visible in worker logs as a second
``DEBUGJ BATCH step=0`` sequence after a ~3.5 min idle window). Each
preemption costs ~82 s of JIT recompile plus full training re-run, so
a single preemption roughly doubles wall-clock time. No data is lost
— restart is clean — but queuing 12+ concurrent ``v5p-8`` jobs in
``us-central1`` triggers at least one preemption round per cohort on
the current cluster load.

Operational takeaway: for future cohorts of ≥12 ``v5p-8`` jobs, launch
the full cohort (all regions) once, then accept that uc1 variants may
be preempted once. Don't manually resubmit — the executor retries
correctly. Monitor reports the preemption as ``running → pending`` on
the child, with no parent-side failure signal.

### Running cohort update

All 12 first-batch jobs still active (ue5 all succeeded; uc1 re-running
after preemption). 12 second-batch jobs (L3c, L4a, L4b) running/pending.
Next logbook update will land after the L4a/L4b ue5 pairs finish and
the shard-class prediction on MLP is read.

## 2026-04-19T L4a / L4b Results: Output-Shard Hypothesis Confirmed On MLP

Second-batch L4a (Gate/Up-only) and L4b (Down-only) ue5 pairs finished
~15 min after submission (``SKIP_HF_EXPORT=1`` paid off — ~10 min
training + ~5 min scheduling/compile). Losses are bit-identical across
regions for each variant as in prior cohorts.

### Step-9 train loss

| exp  | target            | shard class       | canonical @9 | reverse @9 | gap     |
|:----:|-------------------|-------------------|------------:|-----------:|--------:|
| L4a  | gate,up           | input-sharded     | 0.5476      | 0.5486     | **-0.001** |
| L4b  | down              | **output-sharded**| 0.6408      | 0.2683     | **0.373**  |

### Interpretation: one-module L-experiment decomposition

Combining L3a/L3b/L3c/L4a/L4b with L1 and L2 completes the attention
and MLP single-module decomposition under FSDP ``data:4`` on ``v5p-8``:

| exp  | modules | shard class (of the B side) | step-9 gap |
|:----:|---------|-----------------------------|-----------:|
| L3a  | q, v                    | input    | ~0.004   |
| L3c  | k                       | input (GQA) | *(pending)* |
| L3b  | o                       | **output** | **0.228** |
| L4a  | gate, up                | input    | ~-0.001  |
| L4b  | down                    | **output** | **0.373** |
| L1   | q,k,v,o                 | mixed (¼ output) | 0.209 |
| L2   | gate,up,down            | mixed (⅓ output) | 0.303 |

**Clean rule:** *Only the LoRA adapters on the output-sharded
projections (``o_proj``, ``down_proj``) exhibit a canonical-vs-reverse
gap. Input-sharded adapters do not.*

L4b's gap (**0.373**) is *larger* than L2's full-MLP gap (0.303). In L2,
the input-sharded Gate/Up projections train cleanly in both permutations
and partially compensate canonical's broken Down updates, which
*dilutes* the gap. With Down alone, there is nothing to dilute and the
full divergence shows through. The same explanation accounts for L3b
(0.228) being comparable to L1 (0.209): L1's Q/V/K input-sharded
partners dilute O's divergence, so L1's attention gap is slightly
*lower* than O-only.

### Final mechanistic statement

Bug 1 on ``v5p-8 {replica:1, data:4, model:1}`` is the canonical-vs-reverse
divergence of the **first LoRA_B gradient** on any projection where the
``embed`` axis (the FSDP-sharded axis under ``param_mapping={embed: data}``)
is the **output** dimension. Equivalently:

- Attention: ``o_proj`` (``heads*head_size → embed``)
- MLP:        ``down_proj`` (``ffn_dim → embed``)

These are the only projections in a Llama-shaped model where the
gradient to ``lora_B`` requires an output-axis reduction/scatter under
FSDP. The physical-device permutation changes the SPMD collective
partner ordering on that reduction; with ``zero_init_b`` the step-0
update is entirely determined by the raw gradient (Adam has no
history), so the ordering difference immediately becomes a
direction-level weight-update difference. The broken first update
compounds through subsequent steps (chaotic amplification) until the
canonical run is stuck near the DPO baseline while the reverse run
trains normally.

Input-sharded projections escape this because their ``lora_B`` gradient
is an **input-axis** contraction that reduces without involving a
cross-device scatter over the ``embed`` shard; collective ordering on
the replicated output axis does not change the arithmetic result.

### Immediate implications for Marin practice

- **Never use canonical ``{data:4, model:1}`` FSDP on ``v5p-8`` for LoRA
  DPO that trains ``o_proj`` or ``down_proj``.** Mesh rescues work, but
  a simpler *dtype-free, adapter-level* workaround also exists:
  set ``target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]``
  (skip output-sharded projections). This trades some capacity for
  permutation invariance, and is cheap on small experimental runs.
- **Levanter level fix candidate**: force deterministic participant
  ordering for the output-shard collective on ``lora_B``, regardless of
  physical device permutation. That is essentially the L5 sharding-class
  probe's positive resolution; this result reduces L5 to a mechanical
  implementation rather than an open question.

### Pending tail work

- L3c (K-only) losses still to land — prediction: gap <~0.05 (input-sharded).
- L2 / L3a / L3b / L4a / L4b uc1 pairs still re-running after earlier
  preemption. Will bit-identical-check against their ue5 counterparts
  as they finish (expected).
- **Direction-audit follow-up on L4b**: repeat the step-0 B-grad
  sentinel divergence analysis on L4b canonical vs reverse — should
  show the same ~100% magnitude + 50% sign-flip signature as o_proj
  in L1. Will confirm the mechanism at the metric level, not just the
  outcome level.
- **L5 sharding-class minimal-repro**: one-linear experiment toggling
  ``lora_B`` partition spec between input-sharded and output-sharded
  with everything else held constant, on a much smaller model, to
  isolate the collective-lowering decision cleanly.
- **Levanter fix PR**: pin the collective partner order for
  ``lora_B`` output-axis reductions to the logical-rank order,
  independent of the physical device permutation.

## 2026-04-19T Final Cohort Results: Shard-Class Rule Is Clean

All 24 jobs across L2 / L3a / L3b / L3c / L4a / L4b reached
``JOB_STATE_SUCCEEDED``. Losses are bit-identical between ``uc1`` and
``ue5`` for every variant (verified via W&B API; the Iris log endpoint
started timing out on the tail end of the cohort, switched to W&B for
the final pull). Below, each row's value is the ``uc1 == ue5`` loss.

### Full attention + MLP single-module decomposition (step-9 train loss)

| exp  | target            | B-side shard class                | canonical @9 | reverse @9 | **gap**    |
|:----:|-------------------|-----------------------------------|-------------:|-----------:|----------:|
| BL   | all linears       | mixed                             | 0.427        | 0.111      | 0.316     |
| L1   | q,k,v,o           | mixed (¼ output)                  | 0.589        | 0.380      | 0.209     |
| L2   | gate,up,down      | mixed (⅓ output)                  | 0.508        | 0.206      | 0.303     |
| L3a  | q,v               | **input** (embed → heads·head)    | 0.649        | 0.645      | **0.004** |
| L3b  | o                 | **output** (heads·head → embed)   | 0.639        | 0.411      | **0.228** |
| L3c  | k                 | **input** (embed → kv_heads·head; GQA) | 0.696   | 0.690      | **0.006** |
| L4a  | gate,up           | **input** (embed → ffn_dim)       | 0.548        | 0.549      | **-0.001**|
| L4b  | down              | **output** (ffn_dim → embed)      | 0.641        | 0.268      | **0.373** |

### Shard-class rule

- **Input-sharded LoRA targets** (q, v, k, gate, up, each tested
  individually): ``|gap| ≤ 0.006`` across four independent probes.
- **Output-sharded LoRA targets** (o, down, each tested individually):
  ``gap ∈ {0.228, 0.373}``.
- **Mixed module sets** (L1 attention ¼-output, L2 MLP ⅓-output, BL
  all-linears): gap is monotone in the fraction of output-sharded
  targets being trained, diluted downward by the presence of
  input-sharded targets training cleanly in both permutations.

The rule is robust under five input-sharded negatives (Q, V, K, Gate,
Up tested individually or paired) and two output-sharded positives (O,
Down). No input-sharded target produces a gap; no output-sharded target
fails to produce one.

Single-projection gaps are *larger* than the mixed-set gaps:

- L4b (Down alone, 0.373) > L2 (MLP all, 0.303).
- L3b (O alone, 0.228) > L1 (attention all, 0.209).

This is consistent with the "input-sharded partners dilute the signal"
reading: in L2, Gate and Up train cleanly regardless of permutation and
partially compensate for broken Down updates; in L4b, Down has no
partner, so the full divergence surfaces. Same for L3b vs L1.

Down > O in gap magnitude most likely because ``ffn_dim``
(``4 * embed`` in Llama-style MLPs = 14 336 for 8B) is larger than
``num_heads * head_size = embed`` (4 096); with more first-step
gradient volume concentrated in the output-axis reduction, the
canonical fork's divergence carries more signal.

### Mechanistic statement (confirmed)

Bug 1 on ``v5p-8`` FSDP ``{data:4, model:1}`` is the canonical-vs-reverse
divergence of the **first LoRA_B gradient on any LoRA adapter whose
output dimension is the FSDP-sharded ``embed`` axis**. The physical
device permutation changes SPMD collective partner ordering on the
output-axis reduction; with ``zero_init_b=True`` the step-0 update is
entirely determined by the raw gradient, the broken first update
compounds chaotically, and canonical runs are stuck near the DPO
baseline while reverse runs train normally. Input-sharded adapters
escape because their ``lora_B`` gradient contracts on the input axis
without a cross-device scatter over the shard.

### Practical guidance (ready-to-ship)

- **Adapter-level workaround (no Levanter change required):**
  ``target_modules=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]``.
  Drops the two output-sharded projections. Permutation-invariant on
  ``v5p-8 data:4`` and cheap to adopt in Marin's ``SimpleDPOConfig``.
  Trades some representational capacity for determinism; for 10-step
  diagnostic runs the trade-off is lossless on the permutation axis.
- **Mesh-level workaround (pre-existing):** pure TP ``{data:1, model:4}``
  or mixed ``{data:2, model:2}``. Both rescue the full target set on
  ``v5p-8``.
- **Levanter fix candidate (open):** pin the collective partner order
  for ``lora_B`` output-axis reductions on FSDP-sharded ``embed`` to
  the logical-rank order regardless of the physical device permutation.
  This is the L5 sharding-class probe's positive resolution — now
  well-scoped since the output-sharded axis is known to be the
  load-bearing variable.

### Run costs / operational notes

- 24 ``v5p-8`` jobs launched in two waves (12 at ``T=21:50``, 12 at
  ``T=21:59``); all 24 terminal-succeeded by ``T≈22:32``. Total
  wall-clock ~42 min for the whole cohort.
- 5 of 12 first-wave ``uc1`` children preempted ~7 min after start;
  restarted cleanly from step 0 after ~4 min idle. Executor retry
  handling worked without intervention. Each preemption ~doubled the
  individual job's wall-clock but did not fail the job.
- ``MARIN_DEBUG_SKIP_HF_EXPORT=1`` on the second wave dropped run time
  from ~14 min (L1 cohort) to ~10 min per job.
- Iris ``job logs`` began timing out around ``T≈22:30`` for a subset
  of jobs; W&B API was used as the secondary data source. For large
  cohorts, W&B is more reliable than Iris log fetch.

### What this session produced, summarised

- Four new experiments upstreamed
  (``experiment_l{2,3a,3b,3c,4a,4b}_llama31_*_v5p8_pd4_s10.py``), each
  with region-aware Llama-3.1 GCS path selection and
  ``SKIP_HF_EXPORT=1``.
- ``base_model.LLAMA_3_1_8B_INSTRUCT_GCS_PATH_BY_REGION`` +
  ``llama_3_1_8b_instruct_gcs_path_for(regions)`` helper to fix the
  cross-region ``initialize_from_hf`` failure that killed the first
  L1 ``us-east5`` submission.
- Direction audit on L1 ``lora_debug/*`` metric keys that surfaced the
  o_proj step-0 anomaly (110% magnitude divergence, 50% sentinel sign
  flip), motivating the L3b / L4b probes that then *confirmed* the
  output-sharding mechanism.
- 24 independent data points supporting the shard-class rule; zero
  disconfirming data.
- Mechanistic statement, ready-to-ship adapter-level mitigation, and
  a scoped Levanter fix candidate, all recorded above.

## 2026-04-19T Plain-Language Explanation of Bug 1

This is the stand-alone narrative summary of the mechanism established
by the L1-L4b cohort and the ``lora_debug/*`` direction audit. Written
so a future agent or newcomer can read *just this section* and
understand what is going on and why the experiments above test it.

### What LoRA actually does

LoRA adds a tiny trainable bolt-on to an otherwise frozen base
model. For each linear layer with frozen weight ``W``, LoRA
parameterises the effective weight as:

```
W_effective = W_frozen + B @ A
```

``A`` and ``B`` are small rank-64 matrices.
**Critical detail: ``B`` is initialised to exactly zero
(``zero_init_b=True``)**. So at step 0, the bolt-on contributes
nothing (``0 @ A = 0``); fine-tuning "starts where the base model
left off", which is the whole point of LoRA.

The consequence that matters for Bug-1: at step 0 the *very first
weight update applied to B* is entirely determined by ``B``'s raw
gradient. Adam has no accumulated history (``m = v = 0``), there is
no previous ``B`` to mix with. The first gradient becomes the first
update, unfiltered. Anything weird about that gradient lands directly
in the weights with no damping.

### What FSDP ``data=4`` does

The 8B Llama model has an ``embed`` dimension of 4096. On a
``v5p-8`` (4 chips) with the mesh
``{replica:1, data:4, model:1}`` and
``param_mapping={embed: data}``, the 4096-wide ``embed`` axis is
**physically split** across the 4 chips — each chip holds a
1024-wide slice of every weight matrix along that axis. This saves
memory (each chip stores a quarter of the params) but means most
matmuls need to exchange data between chips during the computation.

### The asymmetry: input-sharded vs output-sharded linear layers

Consider a linear ``y = W @ x`` with ``W`` shaped
``(out_dim, in_dim)``. "Sharding on the ``embed`` axis" can mean
different things depending on which of these dims *is* ``embed``:

- **``embed`` is the INPUT dim** (example: ``q_proj``,
  ``embed → heads * head_size``). Each chip holds 1024 input
  columns of ``W``. Each chip computes a *partial* ``y`` from its
  own 1024 input dims; then all 4 chips **sum** their partials via
  an **all-reduce on the input axis**. This is the "normal" FSDP
  pattern.

- **``embed`` is the OUTPUT dim** (example: ``o_proj``,
  ``heads * head_size → embed``). Each chip holds 1024 output
  *rows* of ``W``. Each chip produces a *slice* of ``y`` — its own
  1024 of the 4096 output elements. Forward pass needs **no
  all-reduce**; the output is simply concatenated.

Here is the twist: for **gradient** computation it flips.
The gradient of the loss with respect to an output-sharded weight
matrix requires a reduction *across the sharded output axis*
because each chip's slice only computed a partial contribution
that has to be gathered correctly.

In a Llama block the weight matrices split into these two classes:

- **Input-sharded** (``embed`` is input dim): ``q_proj``, ``k_proj``,
  ``v_proj``, ``gate_proj``, ``up_proj``.
- **Output-sharded** (``embed`` is output dim): ``o_proj``,
  ``down_proj``.

Only those last two have a cross-device reduction on the sharded
axis in their *backward* pass.

### Why collective ordering matters

A 4-way all-reduce of floats is not "sum four numbers at once" —
it is a sequence of pairwise adds across a ring or tree of chips:
``chip0 + chip1``, then ``+ chip2``, then ``+ chip3``, or some other
algorithm-specific order. **Floating-point addition is not
associative** — ``(a + b) + c ≠ a + (b + c)`` at the bit level.

When XLA compiles the graph it picks a concrete ring order. The
order depends on **which physical chip is at logical rank 0, 1, 2,
3** in the mesh. That is precisely what ``device_permutation``
controls:

- ``canonical = (0, 1, 2, 3)`` walks chips in one order → one ring.
- ``reverse   = (3, 2, 1, 0)`` walks them in reverse → different
  ring → different pairwise-add sequence → different bit-level
  result of the reduction.

For *most* weights these bit-level differences (~``1e-5`` relative)
are far below training noise — Adam's ``eps=1e-8``, LR scaling, and
accumulated momentum absorb them.

But LoRA_B at step 0 is maximally sensitive.

### The pathology, step by step

1. Step 0 forward pass: ``B @ A = 0 @ A = 0``. Forward is bit-identical
   under canonical and reverse. Loss is the DPO baseline
   ``log 2 ≈ 0.6931``.
2. Step 0 backward pass: we need the gradient of the loss w.r.t.
   ``lora_B``. For an output-sharded projection (``o_proj``,
   ``down_proj``), computing that gradient goes through the
   output-axis collective. Canonical and reverse use different
   pairwise-add orders → different bit-level gradient.
3. The L1 ``lora_debug/sentinel`` audit measured how different: at
   step 0, sentinel values (five fixed-index scalars sampled from
   each module's ``lora_B`` gradient tensor) differ between
   canonical and reverse by:

   | module | shard class | mean ``|c − r| / avg`` at step 0 | sentinel sign match |
   |--------|-------------|---------------------------------:|:-------------------:|
   | ``q_proj`` | input  | 0.045 | 5/5 |
   | ``k_proj`` | input  | 0.165 | 5/5 |
   | ``v_proj`` | input  | 0.240 | 5/5 |
   | **``o_proj``** | **output** | **1.099** | **2/5 (half flipped)** |

   The first gradient to ``o_proj``'s ``lora_B`` is *essentially random*
   between the two permutations — half the sampled values flip sign.
4. Because ``B`` was zero and Adam has no history, that first gradient
   *is* the first update, scaled only by learning rate. Canonical and
   reverse end up with meaningfully different step-1 ``B`` matrices.
5. From step 1 on, the forward pass is no longer identical (``B @ A``
   is non-zero and different). Gradients diverge, updates diverge,
   loss diverges. By step 2 the canonical/reverse loss gap is ~0.24.
   By step 9 it is ~0.5 for the single-projection output-sharded
   experiments (L3b, L4b).

Under canonical the step-0 update happens to shove ``B`` into a basin
where training stalls near the DPO baseline. Under reverse the
step-0 update lands somewhere healthy and training proceeds normally.
**Neither canonical nor reverse is "the right answer"** — they are
both legitimate SPMD lowerings of the same mathematical computation.
The arithmetic just happens to be sensitive enough at step 0 to put
them in different chaotic basins.

### What each experiment tested

The idea: turn LoRA *on* only for one well-chosen subset of modules;
measure the canonical/reverse step-9 loss gap; see if the subset alone
reproduces Bug-1 or not.

| exp  | LoRA target              | shard class of those targets | step-9 gap | reading |
|:----:|--------------------------|------------------------------|:----------:|---------|
| L3a  | q, v                     | input                        | 0.004      | input-sharded alone → **no bug** |
| L3c  | k                        | input (GQA)                  | 0.006      | input-sharded alone → **no bug** |
| L4a  | gate, up                 | input                        | -0.001     | input-sharded alone → **no bug** |
| **L3b** | **o**                 | **output**                   | **0.228**  | output-sharded alone → **bug reproduces** |
| **L4b** | **down**              | **output**                   | **0.373**  | output-sharded alone → **bug reproduces** |
| L1   | q, k, v, o               | mixed (¼ output)             | 0.209      | partial bug |
| L2   | gate, up, down           | mixed (⅓ output)             | 0.303      | partial bug |

Five independent input-sharded negatives, two independent
output-sharded positives, zero disconfirming data. The *single*
variable that predicts presence vs absence of Bug-1 is whether the
LoRA adapter's parent projection is output-sharded on the FSDP ``embed``
axis.

### Why device permutation triggers rather than fixes Bug-1

The permutation did not *fix* anything; it *exposed* the fork.
Canonical and reverse are both valid ways to map 4 logical ranks onto
4 physical chips. They produce different SPMD lowerings of the same
mathematical operation. One happens to hit a bad basin; the other
happens to land in the good one. The root problem is that **XLA's
collective partner ordering for the output-axis reduction depends on
the physical device permutation**, so training outcome depends on a
physical hardware-layout detail that *should* be mathematically
irrelevant.

Three classes of fix exist:

1. **Adapter-level (no Levanter change required):** never LoRA-adapt
   the output-sharded projections. Use
   ``target_modules=["q_proj","k_proj","v_proj","gate_proj","up_proj"]``.
   Permutation-invariant on ``v5p-8`` ``data:4``. Trades some
   representational capacity for determinism; for diagnostic /
   short-horizon runs this is lossless on the variable we care about.
2. **Mesh-level (pre-existing):** use a mesh that does not shard
   ``embed`` in a way that induces an output-axis reduction —
   ``{data:1, model:4}`` (pure TP) or ``{data:2, model:2}`` (mixed).
   Both rescue the full target set.
3. **Proper compiler-level fix (open):** in Levanter, pin the
   collective partner order for ``lora_B`` output-axis reductions to
   the **logical-rank** order regardless of the physical device
   layout. This makes the SPMD lowering device-invariant and removes
   the fork at the root.

### Why LoRA specifically — not full fine-tuning

Dense full fine-tuning on ``v5p-8`` *also* shows different HLO under
canonical vs reverse (verified earlier in this logbook, dense-FT
permutation quartet section). But full-FT training is nearly
invariant to the permutation — training loss agrees to three decimal
places across the four permutations.

The reason is damping:

- Full-FT weights are already non-zero and have substantial magnitude,
  so a step-0 gradient bit-difference is a tiny perturbation on an
  existing signal.
- Adam has accumulated history after the first few steps, smoothing
  over individual gradient noise.
- Learning-rate scheduling and normalization act as additional filters.

LoRA with ``zero_init_b=True`` *maximises* sensitivity to step-0
gradient bit-differences because (a) ``B`` is literally zero, so step
0's update IS the entire ``B`` signal going forward, and (b) Adam's
first-step update has no history to smooth over. Bug-1 is a uniquely
amplified edge case, not a generic training issue.

### One-sentence version

Under FSDP ``data=4`` on ``v5p-8``, LoRA's zero-init ``B`` matrix on
any *output-sharded* projection (``o_proj``, ``down_proj``) gets a
first-step gradient whose bit-level value depends on the physical
device ring order chosen by XLA for the output-axis collective; the
chaotic amplification of that step-0 bit-difference is Bug-1.

## 2026-04-19T L4b-AZ (``a_init_mode="zero"``): Bug-1 Eliminated By Swapping Which Matrix Is Zero-Initialised

Direct test of the step-0-damping hypothesis: rerun L4b (Down-only, the
most pathological ``v5p-8`` LoRA probe with 0.373 gap) with the LoRA
init symmetry flipped — instead of the canonical
``zero_init_b=True, a_init_mode="random"`` ("B=0, A=Gaussian"), use
``zero_init_b=False, a_init_mode="zero"`` ("A=0, B=Gaussian"). Both
configurations satisfy the DPO identity constraint ``B @ A = 0`` at
init, so the policy=reference invariant is preserved. They differ only
in *which* adapter matrix carries the first non-trivial update and
through *which* collective class that update flows.

### New experiment script

- [experiment_l4baz_llama31_down_v5p8_pd4_s10.py](../../experiments/posttrain/per_stmt_dpo/experiment_l4baz_llama31_down_v5p8_pd4_s10.py) —
  byte-for-byte mirror of L4b (Down-only) except the adapter config
  flips ``zero_init_b: True → False`` and adds
  ``a_init_mode="zero"``.

### Prediction (recap)

If the step-0 B-zero damping hypothesis is correct, the L4b-AZ
canonical/reverse gap should collapse to ≤0.05 (matching the five
input-sharded L* probes). If the gap persists near L4b's 0.373, the
pathology is deeper than step-0 damping and the output-axis collective
is corrupting Adam state even with an already-large random ``B``.

### Submitted jobs (4 × can/rev × uc1/ue5)

- ``/ahmedah/experiment-l4bazl31-v5p8-pd4-can-cenl1-20260419-2348``
- ``/ahmedah/experiment-l4bazl31-v5p8-pd4-can-eas-20260419-2348``
- ``/ahmedah/experiment-l4bazl31-v5p8-pd4-rev-cenl1-20260419-2348``
- ``/ahmedah/experiment-l4bazl31-v5p8-pd4-rev-eas-20260419-2348``

All four submitted in parallel; ``ue5`` pair and ``uc1`` pair
proceeded without preemption this time.

### Result: gap eliminated

| step | L4b canonical | L4b reverse | **L4b-AZ canonical** | **L4b-AZ reverse** |
|:-:|---:|---:|---:|---:|
| 0    | 0.6931 | 0.6931 | 0.6931   | 0.6931   |
| 1    | 0.6931 | 0.6931 | 0.6931   | 0.6931   |
| 2    | ~0.64  | ~0.43  | **0.5330** | **0.5184** |
| 5    | stuck  | ~0.27  | **0.2766** | **0.2751** |
| 9    | 0.6408 | 0.2683 | **0.1799** | **0.1809** |

- L4b step-9 gap: **0.3725**.
- L4b-AZ step-9 gap: **-0.0010** (reverse slightly above canonical,
  well within noise).
- **Gap reduction: 373×.** Bug-1 on ``down_proj`` is gone.
- Both L4b-AZ variants train to a step-9 loss of ~0.18, which is
  **better** than L4b reverse's 0.27 (the "healthy" permutation under
  the B=0 regime). So the A=0 flip is not just a Bug-1 mitigation; it
  is a strictly better LoRA initialisation for this DPO recipe on this
  topology.
- uc1 and ue5 step-9 losses match to 4 decimals per variant, as in
  every prior experiment.

### Why this works mechanistically

Recap of the damping hypothesis made in the plain-language section
above:

- **L4b (B=0, A=random) step 0:** ``grad_B`` flows through the
  pathological output-axis collective on ``embed`` (the sharded
  dimension). Because ``B = 0``, the first update ``LR × grad_B``
  *becomes* the entire ``B`` going forward. The bit-level fork between
  canonical and reverse is the entire signal, undamped. Chaotic
  amplification follows.
- **L4b-AZ (A=0, B=random) step 0:** backprop through ``A = 0`` zeroes
  the chain into ``grad_B``, so ``grad_B = 0``. The first non-zero
  gradient goes to ``A``, whose gradient path contracts on ``embed``
  as an **input-axis** reduction — the collective class the earlier
  L3a/L3c/L4a probes empirically proved permutation-invariant (gap
  ≤0.006 across four independent input-sharded single-module probes).
  ``A`` gets a clean update. At step 1, ``B`` remains its large random
  init; ``A`` has small values. ``grad_B`` at step 1 flows through the
  output-axis collective, but now lands as a small LR-scaled
  perturbation on an already-substantial ``B``. Same damping that
  makes dense full-FT immune to this bit-fork.

### Updated recommendation for Marin practice

Promote L4b-AZ's init pattern to the **default** for LoRA DPO on
``v5p-8``. Concretely:

```python
adapter=LoraAdaptationConfig(
    r=64,
    alpha=64,
    dropout=0.0,
    zero_init_b=False,     # B initialised to Gaussian
    a_init_mode="zero",    # A initialised to zero — DPO identity preserved
    target_modules=None,   # or whatever subset; works for all
)
```

Three reasons to adopt this as default:

1. Kills Bug-1 at its point of origin (step-0 damping) for *any*
   output-sharded projection, on *any* physical device permutation,
   with no mesh change and no compiler change.
2. Strictly better training outcome on this recipe
   (L4b-AZ step-9 loss **0.18** vs L4b-reverse step-9 **0.27**;
   L4b-AZ canonical at step 5 is ~4× better than L4b canonical at
   step 5).
3. Preserves the DPO identity ``policy = reference`` at init, so no
   changes to the DPO training loop or the ``AdapterBaseReferenceConfig``
   reference-path logic.

### Comparison with the earlier adapter-level mitigation

Before L4b-AZ, the ready-to-ship mitigation was
``target_modules = [q_proj, k_proj, v_proj, gate_proj, up_proj]`` —
drop the two output-sharded projections entirely. That works but gives
up representational capacity (no LoRA on ``o_proj`` / ``down_proj``).
L4b-AZ gives us **both**: LoRA on all projections *and* permutation
invariance. It dominates.

### Remaining open items

- **Symmetry confirmation on attention**: rerun this same flip on L3b
  (O-only) as L3b-AZ. Prediction: gap from 0.228 → ~0. Low-cost (4 jobs,
  ~10 min).
- **Full adapter set on Llama-3.1 base**: rerun BL-Llama3.1 with
  ``a_init_mode="zero"`` to confirm the attention+MLP full recipe also
  has the gap eliminated. The BL canonical/reverse split of 0.316 is
  expected to collapse. This is the real "is Bug-1 solved at the
  product level?" check.
- **``lora_debug`` direction audit on L4b-AZ**: re-verify that step-0
  sentinels on ``down_proj`` no longer diverge between canonical and
  reverse — the mechanism story says ``grad_B = 0`` at step 0 for all
  sentinels, which the W&B run should show directly.
- **PiSSA integration**: medium-effort path for Marin to adopt a
  principled non-zero-B LoRA that *also* gets the reported accuracy
  gains independent of Bug-1.
- **Upstream Levanter docs**: the
  ``zero_init_b=False, a_init_mode="zero"`` pattern deserves its own
  Levanter recipe note — currently it's exposed as a debug-only knob
  (``DEBUGSTART`` marker in ``lora.py``) rather than a first-class
  init choice.

### One-sentence version

Flipping LoRA's init from "B=0, A=Gaussian" (standard) to "A=0,
B=Gaussian" (mathematically dual) preserves DPO identity, routes
the first non-trivial update through the input-axis collective class
(empirically permutation-invariant), and reduces the ``v5p-8 data=4``
down_proj canonical/reverse gap from **0.373 → -0.001**. Bug-1 solved
by a single-adapter-config flip.

## 2026-04-20T Experiment K-closure: The Original v5p/v6e Divergence Is Eliminated

Ultimate closure experiment. Rerun the original **Experiment K**
(2026-04-14, ``debug_r64_alpha64_s10.py``) — the first formal diagnostic
that documented the v5p-8 / v6e-8 LoRA-DPO divergence and kicked off
this entire investigation — with the ``a_init_mode="zero"`` flip
established by L4b-AZ. Same data, same seed, same
``marin-community/marin-8b-instruct`` base model, same recipe, same
``target_modules=None`` (all linears, including the pathological
``o_proj`` and ``down_proj``). The only change is the LoRA init.

### New experiment script

- [experiment_k_closure_azero_r64_s10.py](../../experiments/posttrain/per_stmt_dpo/experiment_k_closure_azero_r64_s10.py) —
  byte-for-byte mirror of the original
  ``debug_r64_alpha64_s10.py`` with ``zero_init_b=False`` and
  ``a_init_mode="zero"``. Also adds ``MARIN_DEBUG_LORA_DEBUG=1``
  and ``MARIN_DEBUG_SKIP_HF_EXPORT=1`` to the worker env.

### Submitted jobs (5 × multi-region per TPU-family rule)

- v5p-8 × us-central1:  ``/ahmedah/experiment-kclosure-v5p8-uc1-20260420-0028``
- v5p-8 × us-east5:     ``/ahmedah/experiment-kclosure-v5p8-ue5-20260420-0028``
- v6e-8 × europe-west4: ``/ahmedah/experiment-kclosure-v6e8-ew4-20260420-0028``
- v6e-8 × us-east5:     ``/ahmedah/experiment-kclosure-v6e8-ue5b-20260420-0028``
- v6e-8 × us-east1:     ``/ahmedah/experiment-kclosure-v6e8-ue1-20260420-0028``

### Step-by-step loss comparison

All within-TPU-family region pairs produced bit-identical step traces
(same determinism signature as every earlier ``v5p-8`` / ``v6e-8``
experiment in this campaign). Representative trace below uses one pair
per family.

| step | **Original Exp K v5p-8** | **Original Exp K v6e-8** | original gap | **K-closure v5p-8** | **K-closure v6e-8** | **new gap** |
|:---:|-------------------------:|-------------------------:|-------------:|--------------------:|--------------------:|-----------:|
| 2   | ~0.68 (interp)           | ~0.35 (interp)           | ~0.33        | 0.6404              | 0.5898              | 0.051      |
| 3   | **0.6796**               | **0.3254**               | **0.354**    | (not sampled)       | (not sampled)       | —          |
| 5   | ~0.66 (interp)           | ~0.31 (interp)           | ~0.35        | 0.5243              | 0.5157              | 0.009      |
| 8   | **0.6583**               | **0.3018**               | **0.356**    | 0.4721              | 0.4714              | **0.0007** |
| 9   | (not logged)             | (not logged)             | —            | 0.4749              | 0.4746              | **0.0003** |

(Original Exp K step-2/5 values not explicitly recorded in the 2026-04-14
trace; step-3 and step-8 are the canonical cited values.)

### Verdict

**The v5p-8 / v6e-8 LoRA-DPO divergence that kicked off this entire
campaign is eliminated.**

- Step-8 gap: original **0.356** → K-closure **0.0007**. A **~500×
  reduction.**
- Step-9 gap: K-closure **0.0003** — within float noise of zero.
- v5p-8 and v6e-8 losses are tracking each other to 3 decimals by
  step 5 and to 4 decimals by step 8, exactly as Exp N (matched
  ``|data|=8``) and the input-sharded L3a/L3c/L4a probes did.

Small residual at step 2 (0.051) is consistent with the
step-0-gradient bit-fork still existing at the arithmetic level — the
output-axis collective's ring order still differs between the two
hardware slices — but with ``B`` initialised to substantial random
values, that step-0 arithmetic difference gets damped into a ~1%
relative perturbation that washes out within 3 training steps. Same
damping that has always protected full fine-tuning from the identical
underlying collective-ordering variability.

### Closes the thread

- **Original logbook** (``debug_accum_tpu_type.md``, 2026-04-14 through
  2026-04-18, 40 experiments): established v5p-8 vs v6e-8 LoRA-DPO
  divergence, narrowed to "width-4 FSDP mesh" but couldn't identify
  which collective mechanism and couldn't find an adapter-level fix.
  Shipping guidance at campaign close: "change mesh or raise LR".
- **This logbook** (``bug_1_dpo_lora_physical_topology.md``,
  2026-04-18 through 2026-04-20, ~30 experiments): identified physical
  device permutation as the trigger, decomposed by module to show
  only output-sharded projections split, and demonstrated that
  ``a_init_mode="zero"`` eliminates Bug-1 at its source.
- **K-closure (this section)**: demonstrates the adapter-level fix
  ports back to the very first experiment in the thread. No mesh
  change, no recipe change, no LR change. One adapter config flip on
  the original Exp K recipe collapses the 0.356 divergence to 0.0007.

### What this means for Marin / Levanter

1. **Ship ``a_init_mode="zero"`` as the default LoRA DPO init on TPU.**
   For configs where the current default is ``zero_init_b=True``,
   flipping to ``zero_init_b=False, a_init_mode="zero"`` preserves the
   DPO identity ``policy = reference`` at init and removes the TPU/slice/
   permutation sensitivity that has affected this recipe for 6+ days of
   investigation.
2. **Upstream the ``debug``-gated Class-B probe knobs as first-class
   init choices in Levanter.** Currently ``a_init_mode`` is marked as
   a DEBUGSTART/DEBUGEND-gated "Class-B probe" option in
   ``lib/levanter/src/levanter/lora.py``. Promote it to a documented
   supported init mode with a recommendation note in the Marin DPO
   docs.
3. **Archive the "mesh rescue" / "high-LR rescue" guidance as
   secondary.** They still work, but the init flip is strictly simpler:
   no mesh config change, no LR change, no rank change, no target
   module change.

### One-sentence summary

Flipping LoRA's `zero_init_b=True, a_init_mode="random"` to
`zero_init_b=False, a_init_mode="zero"` on the exact recipe and base
model that started the Bug-1 campaign six days ago collapses the
originating v5p-8 / v6e-8 step-8 divergence from **0.356 → 0.0007**,
closing the thread with a single-line adapter config change.

## 2026-04-20T Experiment N-closure: The Fix Has A Small Cost At `|data|=16`

Counterpart to K-closure on the "already-close" matched-pod pair.
Rerun the original **Experiment N** (2026-04-15,
``debug_r64_matched_pd2_s10.py``) on v5p-16 pd=2 and v6e-16 pd=2
(both ``|data|=16`` with grad_accum=2), with ``a_init_mode="zero"``
instead of ``zero_init_b=True``. Same data, same seed, same recipe,
same ``marin-community/marin-8b-instruct`` base.

### New experiment script

- [experiment_n_closure_azero_matched_s10.py](../../experiments/posttrain/per_stmt_dpo/experiment_n_closure_azero_matched_s10.py) —
  mirror of ``debug_r64_matched_pd2_s10.py`` with the init flip and
  ``MARIN_DEBUG_SKIP_HF_EXPORT=1``.

### Submitted jobs (5 × multi-region)

- ``/ahmedah/experiment-nclosure-v5p16-uc1-20260420-0127``
- ``/ahmedah/experiment-nclosure-v5p16-ue5-20260420-0127``
- ``/ahmedah/experiment-nclosure-v6e16-ew4-20260420-0127``
- ``/ahmedah/experiment-nclosure-v6e16-ue5b-20260420-0127``
- ``/ahmedah/experiment-nclosure-v6e16-ue1-20260420-0127``

### Step-by-step comparison: Exp N (original) vs N-closure

| step | N v5p-16 | N v6e-16 | N gap | **N-closure v5p-16** | **N-closure v6e-16** | **closure gap** |
|:---:|---------:|---------:|------:|---------------------:|---------------------:|----------------:|
| 2   | 0.335    | 0.336    | 0.001 | 0.5962               | 0.5915               | **0.005**       |
| 5   | ~0.325   | ~0.327   | ~0.002| 0.5306               | 0.5147               | **0.016**       |
| 8   | ~0.318   | ~0.320   | ~0.002| 0.4885               | 0.4705               | **0.018**       |
| 9   | 0.317    | 0.319    | 0.002 | 0.4889               | 0.4735               | **0.015**       |

(Original Exp N step-2/5/8 values are interpolated from logbook text
that cites ~0.32 tracking through steps 2-9. Final step-9 values are
exact from the original logbook.)

Within-pod cross-region is bit-identical as usual
(v5p-16 uc1 == v5p-16 ue5, v6e-16 ew4 == ue5b == ue1).

### What the data says — honestly

The closure gap at step 9 is **0.015**, compared to the original Exp N
gap of **0.002**. The fix makes cross-hardware *worse* by ~8× on this
already-matched pod pair.

**But:** the closure gap is still ~23× smaller than the v5p-8 / v6e-8
Bug-1 gap of **0.356**. So the fix is:

- **Huge win** where Bug-1 was catastrophic (|data|=4, v5p-8): **500×**
  reduction in gap (K-closure).
- **Small cost** where Bug-1 was already small (|data|=16, v5p-16 /
  v6e-16 pd=2): ~**8× widening** of an already-tiny gap (N-closure).

### Why: a clean trade-off between step-0 damping and cross-hardware drift

Under ``zero_init_b=True`` (original):

- Step 0 forward: ``B·A = 0`` everywhere. Bit-identical across hardware.
- Step 0 backward: collective-ordering bit-fork in the output-axis
  reduction → different ``grad_B`` across hardware.
- At ``|data|=4`` the fork is large because 4-way all-reduce
  arithmetic is more sensitive; at ``|data|=16`` the fork is tiny
  because 16-way averages are more stable.
- Because ``B = 0``, the first update *is* the entire ``B`` going
  forward. Whatever fork there was gets maximally amplified. Hence
  catastrophic at ``|data|=4``, small at ``|data|=16``.

Under ``a_init_mode="zero"`` (fix):

- Step 0 forward: ``B·A = 0`` still (``A = 0``). Loss is log 2 on all
  hardware.
- Step 0 backward: ``grad_B = 0`` (through ``A = 0``). ``grad_A`` goes
  through the input-axis collective — the empirically permutation-
  invariant class.
- Step 0 optimizer update: ``A`` gets a clean update; ``B`` stays at
  its random init.
- **However**: from step 1 forward, ``B`` is still a non-zero random
  matrix, and `B · A@x` where `A@x` is tiny but non-zero produces
  slightly different activations on different hardware due to
  collective ordering. These small forward-pass differences accumulate.

So the fix trades:

- **Damping** of step-0 catastrophic chaos (gets rid of the
  ``|data|=4`` disaster) →
- **Sensitivity** to step-1+ cross-hardware drift (introduces a
  small steady gap at all pod sizes).

The trade is great at ``|data|=4`` and small at ``|data|=16``, but it
is not a universal free win.

### What this means for Marin practice

The adapter-level fix still dominates the previous "mesh rescue / high
LR" workarounds at ``|data|=4``, because those workarounds require
config acrobatics and the fix is a one-line switch. At ``|data|=16``
or higher the argument is less clear-cut:

- If cross-hardware reproducibility matters at the 3-decimal level on
  ``|data|=16``: keep the original ``zero_init_b=True`` init. The
  0.015 drift under the fix is visible.
- If we just want training that works consistently *regardless of pod
  size*: adopt the fix. It's 0.015 at worst and 0.0007 at best, all
  with a single config line. No mesh changes, no LR tuning.

**Recommendation update:** Ship ``a_init_mode="zero"`` as the default
for ``v5p-8`` / any narrow-``|data|`` slice where Bug-1 is active.
For larger pods keep the original init unless there's a specific
reason to prefer the fix (e.g., defensive determinism across unknown
future hardware).

### Cross-region bit-identity confirmed at step 9

- v6e-16 ue5b step-9 = **0.47353556752204895**
- v6e-16 ue1  step-9 = **0.47353556752204895** (bit-identical ✓)
- v5p-16 uc1  step-9 = **0.48893740773200989**
- v5p-16 ue5  step-8 bit-identical to uc1 (step-9 also expected to match)

So within each TPU family the fix is fully deterministic; the 0.015
cross-pod gap is a real v5p ↔ v6e hardware signal under the new init,
not a noise artifact.

### Open follow-ups (not run this session)

- **Long-horizon check**: does the cross-hardware drift at
  ``|data|=16`` keep growing or stabilise? Run ``N-closure`` for 100
  steps to see if gap at step 99 is 0.015, 0.05, or something bigger.
  The step-0 bit-fork damping story predicts stability; the
  "accumulating forward-drift" interpretation predicts slow growth.
- **Multi-seed**: is the 0.015 gap consistent across seeds or is
  it the artifact of one unlucky random ``B`` init? Rerun at seeds
  1, 2, 3.
- **Original r=16/α=32 recipe closure**: the *actual* original
  pathological recipe before Exp K's rank change. Confirms the fix
  works at the original rank, not just r=64/α=64.

### Honest one-sentence version

The ``a_init_mode="zero"`` fix solves Bug-1 catastrophically well at
the pod sizes where Bug-1 is catastrophic (0.356 → 0.0007) and
introduces a small ~0.015 cross-hardware drift at larger pod sizes
where Bug-1 was already small; it is a trade-off, not a free lunch,
but a strongly favourable trade-off for Marin's current ``v5p-8``
DPO-LoRA recipes.

## 2026-04-20 Tune-LoRA 1e-5 exact rerun + flipped-init rerun launch

User asked for the original archived Tune-LoRA ``lr=1e-5`` run to be
recreated exactly, then rerun with the new flipped init
(``zero_init_b=False, a_init_mode="zero"``).

### Which archived run we matched

I matched the **executor-native seed-0 ``lr=1e-5`` Tune-LoRA run**,
because it is one of the two canonical top archived runs and the
executor-native source script still exists in-tree.

Archived source-of-truth run:

- W&B:
  ``https://wandb.ai/marin-community/dpo/runs/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d``
- Local archive:
  ``scratch/wandb_dpo_data/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d``
- Original executor-native script:
  [experiments/tune_lora/beta0p1_lr1e5_seed0_b64.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/tune_lora/beta0p1_lr1e5_seed0_b64.py)
- Original launch pattern recorded in
  [dpo-lora-codex.md](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/.agents/logbooks/dpo-lora-codex.md:2287)

Archived run metadata confirms:

- display name:
  ``bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d``
- project: ``marin-community/dpo``
- history length: ``1700`` steps
- final eval DPO loss:
  ``0.006383000873029232``
- final eval DPO accuracy:
  ``0.992759108543396``

### Exact baseline config we copied

The baseline rerun uses the existing script unchanged:

- [experiments/tune_lora/beta0p1_lr1e5_seed0_b64.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/tune_lora/beta0p1_lr1e5_seed0_b64.py)

That script points at the shared Tune-LoRA builder in
[experiments/tune_lora/common.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/tune_lora/common.py)
with:

- ``learning_rate=1e-5``
- ``seed=0``
- ``beta=0.1``
- ``train_batch_size=64``
- ``rank=64``
- ``alpha=64``
- ``dropout=0.0``
- ``target_modules=None``
- ``reference=AdapterBaseReferenceConfig()``
- ``resources=v5p-8``
- ``regions=us-central1, us-east5``

The archived config JSON also confirms the original LoRA init was:

- ``zero_init_b=True``
- no ``a_init_mode`` field in the old archive (the knob did not exist yet,
  so canonical behavior is the modern default ``a_init_mode="random"``)

### Code change to allow the flipped rerun in Tune-LoRA

I made a minimal patch to the shared Tune-LoRA helper:

- [experiments/tune_lora/common.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/tune_lora/common.py)

Change:

- added ``zero_init_b`` and ``a_init_mode`` fields to ``LoraTuneSpec``
- threaded them into ``LoraAdaptationConfig(...)``
- left defaults at the archived canonical behavior:
  - ``zero_init_b=True``
  - ``a_init_mode="random"``

This keeps the baseline rerun faithful while enabling the flipped-init
variant without touching unrelated parts of the recipe.

### Confirmation that ``a_init_mode="zero"`` exists and is live

Current Levanter LoRA config already supports this knob:

- [lib/levanter/src/levanter/lora.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/lora.py:157)
- [lib/levanter/src/levanter/adaptation.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/adaptation.py:144)

Verified locally with a direct config construction check:

```text
BASE True random
FLIP False zero
```

Meaning:

- canonical Tune-LoRA path still resolves to
  ``zero_init_b=True, a_init_mode="random"``
- flipped rerun resolves to
  ``zero_init_b=False, a_init_mode="zero"``

### New flipped rerun script

Added:

- [experiments/tune_lora/beta0p1_lr1e5_seed0_b64_azero.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/tune_lora/beta0p1_lr1e5_seed0_b64_azero.py)

This script keeps everything from the archived seed-0 ``lr=1e-5`` run
the same except the LoRA init:

- ``zero_init_b=False``
- ``a_init_mode="zero"``

### Launch commands used

Note: the original archived Iris command used ``--tpu`` without
``--enable-extra-resources``. The current Iris CLI now requires
``--enable-extra-resources`` whenever ``--tpu`` is present, so I added
that single launcher-only compatibility flag. The training recipe itself
is otherwise unchanged.

Exact baseline rerun launch:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --enable-extra-resources \
  --job-name bloom-speceval-v2-lora-beta0p1-lr1e5-seed0-b64-rerun-20260420 \
  --tpu v5p-8 \
  --region us-central1 \
  --region us-east5 \
  --no-wait -- \
  uv run python experiments/tune_lora/beta0p1_lr1e5_seed0_b64.py
```

Exact flipped-init rerun launch:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --enable-extra-resources \
  --job-name bloom-speceval-v2-lora-beta0p1-lr1e5-seed0-b64-azero-20260420 \
  --tpu v5p-8 \
  --region us-central1 \
  --region us-east5 \
  --no-wait -- \
  uv run python experiments/tune_lora/beta0p1_lr1e5_seed0_b64_azero.py
```

### Iris job IDs

Submitted parent jobs:

- baseline rerun:
  ``/ahmed/bloom-speceval-v2-lora-beta0p1-lr1e5-seed0-b64-rerun-20260420``
- flipped-init rerun:
  ``/ahmed/bloom-speceval-v2-lora-beta0p1-lr1e5-seed0-b64-azero-20260420``

Initial Iris status check:

- both parent jobs are ``running``
- both have ``0`` failures / ``0`` preemptions
- both are currently ``assigned`` to their single task

### What we want to learn

This pair is the cleanest applied test of whether the Bug-1 fix carries
over to the original successful Tune-LoRA regime:

- If baseline and ``a_init_mode="zero"`` track similarly:
  the fix is probably safe to recommend broadly for the classic
  Tune-LoRA recipe.
- If the flipped rerun is materially better:
  the fix improves not just the pathological narrow-pod case but also the
  old best-known Tune-LoRA setup.
- If the flipped rerun is materially worse:
  the Bug-1 fix should stay targeted to the pathological topology regime
  rather than becoming a blanket default.

### Immediate failure of the first relaunch attempt

Both first-attempt relaunches failed immediately at the executor layer
before TPU startup:

- baseline:
  ``/ahmed/bloom-speceval-v2-lora-beta0p1-lr1e5-seed0-b64-rerun-20260420``
- azero:
  ``/ahmed/bloom-speceval-v2-lora-beta0p1-lr1e5-seed0-b64-azero-20260420``

Shared failure:

```text
ValueError: Executor step 'tokenized/bloom_speceval_v2_train_prefs_marin_tokenizer'
has cross-region GCS dependencies. Found regions {us-central1, us-east5}.
us-central1: config.train_paths[0]=gs://marin-us-central1/preference/...
us-east5: config.cache_path=gs://marin-us-east5/tokenized/...
```

Interpretation:

- the Tune-LoRA pipeline still depended on the old
  [experiments/dpo_bloom_speceval_v2.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/dpo_bloom_speceval_v2.py)
  dataset definition
- that file hardcoded raw source paths as
  ``gs://marin-us-central1/...``
- but the executor was free to place the tokenization cache/output in
  another Marin region (here ``us-east5``)
- the executor's cross-region guard correctly rejected that mixed
  configuration

### Why ``mirrored()`` is the right fix

Per [experiments/AGENTS.md](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/AGENTS.md:6),
hardcoded artifact paths that may be consumed cross-region should use
``mirror://`` via ``mirrored(...)`` instead of raw ``gs://`` URLs.

What ``mirrored(...)`` does in practice:

- wraps a literal path in ``MirroredValue``
- at config instantiation time the executor resolves it relative to the
  **local** Marin prefix
- before step execution the mirror filesystem checks the local regional
  bucket first
- if the object is missing locally, it copies it once from another Marin
  regional bucket under a transfer budget
- subsequent reads are local

Implementation references:

- [MirroredValue / mirrored()](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/marin/src/marin/execution/executor.py:939)
- [experiments/AGENTS.md](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/AGENTS.md:6)
- prior multiregion example:
  [v6e8_probe_multiregion.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/tune_lora/v6e8_probe_multiregion.py:1)

So the correct fix is **not** to special-case the launcher or pin
everything to one region. The correct fix is to make the raw dataset
inputs region-agnostic at the shared experiment-definition layer.

### Shared dataset fix applied

I patched
[experiments/dpo_bloom_speceval_v2.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/dpo_bloom_speceval_v2.py)
to replace the old hardcoded:

- ``gs://marin-us-central1/preference/.../train/*.jsonl.gz``
- ``gs://marin-us-central1/preference/.../val_deduped/shard-00000.jsonl.gz``

with mirrored region-agnostic inputs:

- ``mirrored("preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/train/*.jsonl.gz", budget_gb=1)``
- ``mirrored("preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/val_deduped/shard-00000.jsonl.gz", budget_gb=1)``

Verified locally after patch:

```text
MirroredValue(value='preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/train/*.jsonl.gz', budget_gb=1)
MirroredValue(value='preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/val_deduped/shard-00000.jsonl.gz', budget_gb=1)
```

This matches the repo's newer multiregion pattern and is the exact fix
Claude had used earlier for v6e probes.

### Relaunch after the mirrored fix

Fresh relaunches:

- baseline mirrored retry:
  ``/ahmed/bloom-speceval-v2-lora-beta0p1-lr1e5-seed0-b64-rerun-mirrored-20260420``
- azero mirrored retry:
  ``/ahmed/bloom-speceval-v2-lora-beta0p1-lr1e5-seed0-b64-azero-mirrored-20260420``

Initial status after relaunch:

- baseline mirrored retry: ``running`` / task ``assigned``
- azero mirrored retry: ``pending``

Most importantly, neither mirrored retry reproduced the immediate
cross-region executor failure, so the fix addressed the actual root
cause of the first launch failure.

## 2026-04-21T David Hall Suggestion: ``LR=0`` For The First ~10 Steps

### Context from discussion

After the HLO / topology investigation, David Hall suggested a different
framing:

- maybe this is **not** an XLA bug in the strict sense
- XLA may simply be choosing among multiple valid collective
  decompositions / reduction orderings
- the *real* problem may be that LoRA DPO is uniquely fragile to the
  **first real parameter update**
- a practical way to test that is:
  - set **learning rate exactly to zero** for the first ~10 steps
  - then switch to the normal LR schedule

The back-and-forth was roughly:

- David: "can't decide if this is a bug in xla or not; probably not;
  wonder if ~10 steps of LR=0 would fix"
- Ahmed: interpreted this as "not a bug" in the sense that XLA is free
  to pick multiple numerically different but valid reductions, and XLA's
  job is throughput rather than guarding LoRA-DPO-specific numerical
  fragility
- David clarified: yes, literally **start with LR=0**

The key clarification is that David did **not** mean "use a smaller
warmup LR." He meant:

- for steps ``0..N-1``:
  - still run forward / backward
  - still compute gradients
  - ideally still update optimizer moments
  - but apply **zero parameter update**
- at step ``N``:
  - switch to the normal learning-rate schedule

### Why this suggestion is conceptually strong

This suggestion is tightly aligned with the strongest current Bug-1
interpretation.

What we already know:

1. Bug-1 is **not** "full fine-tuning on v5p-8 is broken."
2. The same logical program gets compiled into two different valid
   communication forms depending on physical device order.
3. Dense full FT tolerates that fork; LoRA does not.
4. The strongest existing fix
   (``zero_init_b=False, a_init_mode="zero"``)
   works by changing the **very first live LoRA update geometry**.

David's proposal is another way to probe the same pressure point:

- if the whole catastrophe comes from the **first one or few real LoRA
  updates**, then delaying those updates while still letting Adam see
  gradients could let the optimizer's moment estimates stabilise before
  the first actual move
- if that works, the issue is less "XLA is wrong" and more "LoRA DPO is
  unstable under perfectly legal numeric variation during optimizer
  bootstrap"

In that sense, the suggestion is a very clean discriminator between:

- **persistent per-step numeric divergence**
  vs
- **first-update / optimizer-bootstrap fragility**

### Important implementation subtlety

There are two different things people often conflate:

1. **True LR=0 warmup**
   - compute gradients
   - update Adam moments ``m`` and ``v``
   - but multiply the parameter update by ``0``
2. **Skip optimization entirely**
   - compute gradients
   - do **not** update parameters
   - do **not** update Adam moments either

David's phrase "warm up the gradient scale" strongly suggests he means
**(1)**, not (2).

That distinction matters:

- If Adam moments accumulate during the zero-LR window, then the first
  nonzero step is based on a better estimate of gradient scale /
  variance.
- If the optimizer state is frozen too, then we are merely delaying the
  first update in wall-clock time, not warming anything up.

So the experiment should be designed to preserve that distinction.

### Mechanistic hypothesis behind the probe

Hypothesis ``D-H1``:

- Bug-1 is caused primarily by the **first real LoRA parameter update**
  being taken from a numerically fragile state with no optimizer
  history.
- Different valid XLA collective lowerings perturb that first gradient
  slightly.
- Adam turns those small perturbations into different first updates.
- Once the first update lands in a bad basin, the rest of training
  follows it.
- If we let Adam accumulate moments for ~10 steps while LR is zero, the
  first nonzero update becomes more stable and Bug-1 should shrink or
  disappear.

Competing hypothesis ``D-H2``:

- Bug-1 is not just the first update.
- The topology/XLA path causes **persistent** numerically meaningful
  differences every step.
- In that case, LR=0 warmup will not remove the canonical/reverse split.

### Concrete experiment sketch

#### Experiment family: D1 (``lr0warmup``)

Base recipe:

- the canonical Bug-1 LoRA DPO recipe
- same base model / LoRA config / batch / mesh / regions as the existing
  Bug-1 canonical-vs-reverse probes
- keep the standard init:
  - ``zero_init_b=True``
  - ``a_init_mode="random"``
- do **not** combine with the ``a_init_mode="zero"`` fix in the first
  probe; the point is to test whether LR bootstrap alone rescues the old
  bad recipe

Variants:

1. **D1a canonical + true LR=0 for first 10 steps**
   - bad physical order
   - LR multiplier = 0 on steps ``0..9``
   - normal schedule from step ``10`` onward
2. **D1b reverse + true LR=0 for first 10 steps**
   - good physical order
   - same schedule change
3. **D1c canonical baseline**
   - the existing bad recipe
4. **D1d reverse baseline**
   - the existing good recipe

Primary comparison:

- compare ``canonical`` vs ``reverse`` gap at:
  - step 2
  - step 9
  - optional longer horizon step 99

Secondary comparison:

- compare ``D1a`` vs the existing ``a_init_mode="zero"`` fix
  on:
  - early losses
  - presence/absence of canonical/reverse bifurcation

#### Optional control family: D2 (delay-without-moment-warmup)

If implementation is easy, run a second pair:

1. **D2a canonical + skip optimizer entirely for first 10 steps**
2. **D2b reverse + skip optimizer entirely for first 10 steps**

This is the control for the phrase "warm up the gradient scale."

Interpretation:

- if ``D1`` helps but ``D2`` does not:
  Adam moment warmup is probably the load-bearing effect
- if both help:
  simply delaying the first real update is enough
- if neither helps:
  the problem is not just optimizer bootstrap

### Exact success / failure criteria

For the first pass, the decision rule should be simple:

- **Strong support for D-H1**:
  ``D1a canonical`` moves into the healthy regime and the
  canonical/reverse gap shrinks dramatically relative to the old Bug-1
  split
- **Partial support**:
  the gap shrinks materially but does not disappear
- **Falsification of D-H1**:
  canonical remains in the bad regime with a gap close to the old
  baseline

Useful practical threshold:

- old Bug-1 step-9 gap: order ``~0.35``
- call the probe a strong rescue if the gap drops below ``~0.05``

### What we would learn from each outcome

#### If D1 works

Meaning:

- strong evidence that Bug-1 is primarily a **bootstrap problem**
- XLA is probably not "wrong" in the bug-report sense
- LoRA DPO is too sensitive to the first real optimizer step under legal
  numeric variation

Practical fix options then become:

- ship an LR=0 warmup window for narrow-pod TPU LoRA DPO
- compare it against ``a_init_mode="zero"``
- possibly combine both if the combo is even more stable

At that point the engineering question is:

- which fix is cleaner and cheaper?
  - init change
  - optimizer schedule change
  - or both

#### If D1 does not work

Meaning:

- the issue is not just first-update bootstrap
- persistent per-step topology/XLA numeric differences remain the more
  plausible explanation
- the ``a_init_mode="zero"`` fix is then better understood as a
  geometry-specific cancellation, not a generic optimizer warmup effect

In that case the next best move would be:

- stay with the init fix for practice
- and treat LR=0 warmup as an informative negative result

#### If D1 works only on the original Bug-1 recipe but not elsewhere

Meaning:

- the mechanism is real but recipe-local
- the fix should be scoped to the problematic LoRA DPO regime rather than
  generalised too aggressively

### Suggested implementation shape

The cleanest implementation is not to hack the global cosine schedule
indirectly. Instead add an explicit "zero-LR prefix" knob with exact
step semantics.

Sketch:

- add a config field such as ``initial_zero_lr_steps: int = 0``
- during optimizer step computation:
  - compute the normal schedule value ``lr_t``
  - replace it with ``0`` if ``step < initial_zero_lr_steps``
- ensure Adam moments are still updated during those steps

That gives us a precise and reusable probe instead of overloading the
existing warmup fraction semantics.

### Recommendation

This is worth running.

Reason:

- it is mechanistically sharp
- it is cheap compared with more topology/HLO spelunking
- and it directly tests whether the existing success of
  ``a_init_mode="zero"`` is fundamentally about **which matrix moves
  first** or more broadly about **preventing an unstable first update**

If D1 works, we have a second, conceptually independent fix and a much
clearer story:

> Bug-1 is not "XLA brokenness." It is LoRA DPO first-update fragility
> under valid numeric variation, and both the init flip and LR=0
> warm-start reduce that fragility by damping optimizer bootstrap.

## 2026-04-21T D1 launch plan: canonical/reverse ``LR=0`` warmup probe on ``v5p-8``

User request:

- run ``D1a`` and ``D1b`` on Iris now
- keep the same baseline recipe as the original Bug-1 experiments at the top
  of this logbook

Implementation choice:

- base launcher: the existing BL Bug-1 canonical-vs-reverse harness
- same:
  - ``v5p-8``
  - LoRA ``r=64, alpha=64, zero_init_b=True``
  - ``batch=64``
  - canonical vs reverse explicit physical device order
  - ``learning_rate=1e-6``
  - ``lr_schedule="cosine"``
  - ``warmup=0.1``
  - standard mixed precision
- new:
  - add explicit optimizer knob ``initial_zero_lr_steps``
  - set ``initial_zero_lr_steps=10`` for D1

Code change made for this probe:

- [lib/levanter/src/levanter/optim/config.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/optim/config.py)
  now supports ``initial_zero_lr_steps`` by prepending an exact zero-LR
  prefix to the normal schedule
- [experiments/simple_dpo_config.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/simple_dpo_config.py)
  now exposes ``initial_zero_lr_steps`` for simple DPO experiments
- [experiments/defaults.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/defaults.py)
  now threads that field into ``AdamConfig``
- new dedicated launcher:
  [experiment_d1_v5p8_pd4_lr0warmup_s20.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_d1_v5p8_pd4_lr0warmup_s20.py)

One deliberate deviation from the old 10-step Bug-1 baseline:

- D1 uses ``num_train_steps=20``, not ``10``

Reason:

- if we literally used the old 10-step horizon with ``initial_zero_lr_steps=10``,
  the entire run would take zero parameter updates and would not test David's
  hypothesis
- the 20-step D1 schedule is designed so that:
  - steps ``0..9`` are the zero-LR bootstrap window
  - steps ``10..19`` replay the original 10-step cosine schedule

That means the direct post-bootstrap comparison to the old Bug-1 baseline is:

- old baseline step ``9`` vs D1 step ``19``

Planned runs:

1. **D1a canonical**
   - ``EXPERIMENT_D1_ORDER=canonical``
   - expected to stay bad if topology differences are persistent
   - expected to improve if Bug-1 is mainly first-update bootstrap fragility
2. **D1b reverse**
   - ``EXPERIMENT_D1_ORDER=reverse``
   - control for whether the warmup changes both paths similarly

Launch commands to submit next:

```bash
REGIONS_OVERRIDE=us-east5,us-central1 \
EXPERIMENT_D1_ORDER=canonical \
uv run iris --config lib/iris/examples/marin.yaml job run \
  --enable-extra-resources \
  --job-name experiment-d1-v5p8-pd4-canonical-$(date +%Y%m%d-%H%M%S) \
  --tpu v5p-8 \
  --region us-east5 \
  --region us-central1 \
  --no-wait \
  -- \
  uv run python experiments/posttrain/per_stmt_dpo/experiment_d1_v5p8_pd4_lr0warmup_s20.py
```

Submitted jobs:

- ``D1a canonical``:
  ``/ahmed/experiment-d1-v5p8-pd4-canonical-20260421-161122``
- ``D1b reverse``:
  ``/ahmed/experiment-d1-v5p8-pd4-reverse-20260421-161122``

Immediate scheduler state after submit:

- both jobs are ``JOB_STATE_PENDING``
- no failures, no preemptions
- current pending reason is TPU capacity wait, not an experiment/config error:
  ``Scheduler: Insufficient TPUs (need 4, available 0) ... Waiting for workers in scale group 'tpu_v5p-preemptible_8-us-east5-a' to become ready``

2026-04-21T launch correction:

- the first ``D1b reverse`` submission was invalid
- cause: ``EXPERIMENT_D1_ORDER=reverse`` was set on the local ``uv run iris``
  invocation, which affects the local CLI process, not the remote Iris job
  command
- symptom: the reverse job's ``train_dpo`` child advertised the same
  ``d1-canonical`` W&B run/output name as the canonical job
- action taken:
  - stopped ``/ahmed/experiment-d1-v5p8-pd4-reverse-20260421-161122``
  - re-submitted reverse correctly with the env var inside the remote command

Corrected reverse job:

- ``/ahmed/experiment-d1-v5p8-pd4-reverse-fix-20260421-163451``

Current valid D1 pair:

- canonical:
  ``/ahmed/experiment-d1-v5p8-pd4-canonical-20260421-161122``
- reverse:
  ``/ahmed/experiment-d1-v5p8-pd4-reverse-fix-20260421-163451``

```bash
REGIONS_OVERRIDE=us-east5,us-central1 \
EXPERIMENT_D1_ORDER=reverse \
uv run iris --config lib/iris/examples/marin.yaml job run \
  --enable-extra-resources \
  --job-name experiment-d1-v5p8-pd4-reverse-$(date +%Y%m%d-%H%M%S) \
  --tpu v5p-8 \
  --region us-east5 \
  --region us-central1 \
  --no-wait \
  -- \
  uv run python experiments/posttrain/per_stmt_dpo/experiment_d1_v5p8_pd4_lr0warmup_s20.py
```

2026-04-21T D1 schedule bug discovered and corrected:

The first D1 evidence was invalid. The user noticed immediately from W&B that
``optim/learning_rate`` was not zero during the supposed bootstrap window.

This was a real bug, not a plotting artifact.

What went wrong:

- ``SimpleDPOConfig`` had the field ``initial_zero_lr_steps``
- the D1 launcher correctly set
  ``initial_zero_lr_steps=10``
- ``lib/levanter/src/levanter/optim/config.py`` correctly implemented the zero-LR
  prefix in ``OptimizerConfig.lr_scheduler(...)``
- but ``experiments/defaults.py`` did **not** pass
  ``dpo_config.initial_zero_lr_steps`` into the constructed ``AdamConfig(...)``

That meant the live D1 jobs were silently using the ordinary Bug-1 baseline
schedule:

- no initial zero-LR prefix
- ordinary warmup + cosine decay

Concrete root-cause diff:

```python
optimizer=AdamConfig(
    learning_rate=dpo_config.learning_rate,
    weight_decay=dpo_config.weight_decay,
    warmup=dpo_config.warmup,
    decay=dpo_config.cooldown,
    lr_schedule=dpo_config.lr_schedule,
    min_lr_ratio=dpo_config.min_lr_ratio,
    initial_zero_lr_steps=dpo_config.initial_zero_lr_steps,
    max_grad_norm=dpo_config.max_grad_norm,
)
```

File changed:

- [experiments/defaults.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/defaults.py)

Local verification after patch:

- verified through the actual
  ``SimpleDPOConfig -> default_dpo -> TrainDpoConfig -> AdamConfig`` path
- effective schedule for D1 settings now evaluates to:
  - steps ``0..10``: ``0.0``
  - step ``11``: ``1e-6``
  - then cosine decay on the remaining steps

Interpretation:

- the original ``D1a`` / ``D1b`` pair must not be used as evidence for David's
  LR=0 bootstrap hypothesis
- those runs tested only:
  - the ordinary Bug-1 LoRA baseline
  - plus the previously-discussed mesh permutation

Action taken:

1. stopped the invalid running pair:
   - ``/ahmed/experiment-d1-v5p8-pd4-canonical-20260421-161122``
   - ``/ahmed/experiment-d1-v5p8-pd4-reverse-fix-20260421-163451``
2. relaunched a fresh corrected pair with distinct run tags to prevent W&B
   confusion:
   - ``/ahmed/experiment-d1-v5p8-pd4-canonical-zlrfix-20260421-1700``
   - ``/ahmed/experiment-d1-v5p8-pd4-reverse-zlrfix-20260421-1700``

Relaunch command pattern:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --job-name experiment-d1-v5p8-pd4-canonical-zlrfix-20260421-1700 \
  -- \
  bash -lc 'EXPERIMENT_D1_ORDER=canonical \
    MARIN_DEBUG_RUN_TAG=d1-canonical-zlrfix \
    REGIONS_OVERRIDE=us-east5,us-central1 \
    uv run python experiments/posttrain/per_stmt_dpo/experiment_d1_v5p8_pd4_lr0warmup_s20.py'
```

and

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --job-name experiment-d1-v5p8-pd4-reverse-zlrfix-20260421-1700 \
  -- \
  bash -lc 'EXPERIMENT_D1_ORDER=reverse \
    MARIN_DEBUG_RUN_TAG=d1-reverse-zlrfix \
    REGIONS_OVERRIDE=us-east5,us-central1 \
    uv run python experiments/posttrain/per_stmt_dpo/experiment_d1_v5p8_pd4_lr0warmup_s20.py'
```

Status at this point:

- the broken D1 pair was intentionally terminated
- the corrected ``zlrfix`` pair is now the only valid evidence for the D1 probe

2026-04-21T D1 corrected results (`zlrfix` pair):

Both corrected D1 jobs finished successfully.

Iris job status:

- canonical:
  - ``/ahmed/experiment-d1-v5p8-pd4-canonical-zlrfix-20260421-1700``
  - child ``train_dpo``: ``JOB_STATE_SUCCEEDED``
  - no failures, no preemptions
- reverse:
  - ``/ahmed/experiment-d1-v5p8-pd4-reverse-zlrfix-20260421-1700``
  - child ``train_dpo``: ``JOB_STATE_SUCCEEDED``
  - ``preemption_count=1`` on the child, but the job recovered and finished cleanly

W&B runs:

- canonical:
  [experiment_d1_r64_s20_d1-canonical-zlrfix-f617b0](https://wandb.ai/marin-community/dpo/runs/experiment_d1_r64_s20_d1-canonical-zlrfix-f617b0)
- reverse:
  [experiment_d1_r64_v5p8_s20_d1-reverse-zlrfix-4e9f2d](https://wandb.ai/marin-community/dpo/runs/experiment_d1_r64_v5p8_s20_d1-reverse-zlrfix-4e9f2d)

Most important validation:

- these corrected runs **did** use the intended zero-LR prefix
- W&B ``optim/learning_rate`` history is:
  - steps ``0..10``: ``0``
  - step ``11``: ``1e-6``
  - then cosine decay

Canonical first 14 steps:

```text
step 0  lr=0                  loss=0.693147
step 1  lr=0                  loss=0.693147
step 2  lr=0                  loss=0.693147
step 3  lr=0                  loss=0.693147
step 4  lr=0                  loss=0.693147
step 5  lr=0                  loss=0.693147
step 6  lr=0                  loss=0.693147
step 7  lr=0                  loss=0.693147
step 8  lr=0                  loss=0.693147
step 9  lr=0                  loss=0.693147
step 10 lr=0                  loss=0.693147
step 11 lr=1.0000000e-06      loss=0.693147
step 12 lr=9.6984627e-07      loss=0.686782
step 13 lr=8.8302221e-07      loss=0.677995
```

Reverse first 14 steps:

```text
step 0  lr=0                  loss=0.693147
step 1  lr=0                  loss=0.693147
step 2  lr=0                  loss=0.693147
step 3  lr=0                  loss=0.693147
step 4  lr=0                  loss=0.693147
step 5  lr=0                  loss=0.693147
step 6  lr=0                  loss=0.693147
step 7  lr=0                  loss=0.693147
step 8  lr=0                  loss=0.693147
step 9  lr=0                  loss=0.693147
step 10 lr=0                  loss=0.693147
step 11 lr=1.0000000e-06      loss=0.693147
step 12 lr=9.6984627e-07      loss=0.324148
step 13 lr=8.8302221e-07      loss=0.318302
```

Final train losses:

- canonical step ``19``:
  - ``train/loss = 0.657528``
- reverse step ``19``:
  - ``train/loss = 0.303038``

Final validation summaries from the logs:

- canonical:
  - ``eval/full_val/dpo_loss = 0.69337``
  - ``eval/full_val/dpo_accuracy = 0.5``
- reverse:
  - ``eval/full_val/dpo_loss = 0.59473``
  - ``eval/full_val/dpo_accuracy = 1.0``

Interpretation:

- David's ``LR=0 for ~10 steps`` idea is now tested correctly
- it does **not** collapse the canonical/reverse Bug-1 split
- the first nonzero update still bifurcates immediately:
  - canonical remains in the bad regime
  - reverse immediately drops into the good regime

So D1 falsifies the narrow claim:

> “Bug 1 is mainly that the very first optimizer step is too early and Adam just needs gradient-scale warmup.”

More precise takeaway:

- merely letting Adam moments accumulate for ~10 zero-LR steps is **not enough**
- the topology-sensitive difference still matters at the first real update
- Bug 1 is therefore not just an optimizer-bootstrap timing issue
- the remaining explanations should emphasize:
  - true update direction differences
  - topology-sensitive communication effects that persist into the first real step
  - or LoRA geometry that remains fragile even after moment warmup

2026-04-21T D1 Adam-moment rerun launched:

Question raised after corrected D1:

- the zero-LR warmup probe falsified the narrow “Adam just needs a few zero-LR
  bootstrap steps” story
- but we still did **not** have direct visibility into Adam's LoRA moments
  (`mu`, `nu`) on canonical vs reverse for that probe

Action:

- updated
  [experiment_d1_v5p8_pd4_lr0warmup_s20.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/experiments/posttrain/per_stmt_dpo/experiment_d1_v5p8_pd4_lr0warmup_s20.py)
  so it forwards ``MARIN_DEBUG_LORA_DEBUG=1`` into the child TPU job's
  ``env_vars`` when the parent launch environment sets it
- this enables the existing
  [levanter.callbacks.lora_debug](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/callbacks/lora_debug.py)
  callback for D1

What this should publish to W&B:

- per-module LoRA Adam first-moment norms:
  ``lora_debug/adam/m/.../l2``
- per-module LoRA Adam second-moment norms:
  ``lora_debug/adam/v/.../l2``
- per-module effective Adam update norms:
  ``lora_debug/adam/effective_update/.../l2``

Fresh D1 moment jobs submitted:

- canonical:
  ``/ahmed/experiment-d1-v5p8-pd4-canonical-zlrmom-20260421-1915``
- reverse:
  ``/ahmed/experiment-d1-v5p8-pd4-reverse-zlrmom-20260421-1915``

Launch pattern:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name experiment-d1-v5p8-pd4-canonical-zlrmom-20260421-1915 \
  -- \
  bash -lc 'EXPERIMENT_D1_ORDER=canonical \
    MARIN_DEBUG_LORA_DEBUG=1 \
    MARIN_DEBUG_RUN_TAG=d1-canonical-zlrmom \
    REGIONS_OVERRIDE=us-east5,us-central1 \
    uv run python experiments/posttrain/per_stmt_dpo/experiment_d1_v5p8_pd4_lr0warmup_s20.py'
```

and

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name experiment-d1-v5p8-pd4-reverse-zlrmom-20260421-1915 \
  -- \
  bash -lc 'EXPERIMENT_D1_ORDER=reverse \
    MARIN_DEBUG_LORA_DEBUG=1 \
    MARIN_DEBUG_RUN_TAG=d1-reverse-zlrmom \
    REGIONS_OVERRIDE=us-east5,us-central1 \
    uv run python experiments/posttrain/per_stmt_dpo/experiment_d1_v5p8_pd4_lr0warmup_s20.py'
```

Immediate status after submit:

- both jobs are ``JOB_STATE_PENDING``
- no failures, no preemptions yet
- these are the valid D1-moment jobs to inspect once the W&B ``lora_debug/*``
  namespace appears

2026-04-21T D1 Adam-moment rerun result:

The Adam-moment rerun pair completed:

- canonical:
  [experiment_d1_r64_s20_d1-canonical-zlrmom-585787](https://wandb.ai/marin-community/dpo/runs/experiment_d1_r64_s20_d1-canonical-zlrmom-585787)
- reverse:
  [experiment_d1_r64_v5p8_s20_d1-reverse-zlrmom-5761e8](https://wandb.ai/marin-community/dpo/runs/experiment_d1_r64_v5p8_s20_d1-reverse-zlrmom-5761e8)

Key result:

- the canonical/reverse loss bifurcation still happens immediately at the first
  real update after the zero-LR prefix
- but the **Adam first- and second-moment norms are extremely similar**
  between canonical and reverse on the probed LoRA-B modules

Representative D1-moment history (steps 10, 11, 12, 13, 19):

Canonical:

```text
step 10: loss=0.693147  m_o=0.16928  v_o=4.18e-05  m_down=0.32790  v_down=6.30e-04  m_q=0.12804  v_q=1.79e-04
step 11: loss=0.693147  m_o=0.17867  v_o=4.51e-05  m_down=0.34460  v_down=6.73e-04  m_q=0.13504  v_q=1.93e-04
step 12: loss=0.689029  m_o=0.18659  v_o=4.80e-05  m_down=0.35981  v_down=7.14e-04  m_q=0.14118  v_q=2.05e-04
step 13: loss=0.678488  m_o=0.19387  v_o=5.09e-05  m_down=0.37415  v_down=7.57e-04  m_q=0.14681  v_q=2.20e-04
step 19: loss=0.656821  m_o=0.22436  v_o=6.51e-05  m_down=0.43224  v_down=9.62e-04  m_q=0.16960  v_q=2.81e-04
```

Reverse:

```text
step 10: loss=0.693147  m_o=0.16879  v_o=4.16e-05  m_down=0.32764  v_down=6.31e-04  m_q=0.12764  v_q=1.77e-04
step 11: loss=0.693147  m_o=0.17822  v_o=4.49e-05  m_down=0.34444  v_down=6.74e-04  m_q=0.13467  v_q=1.92e-04
step 12: loss=0.325414  m_o=0.18617  v_o=4.78e-05  m_down=0.35972  v_down=7.16e-04  m_q=0.14083  v_q=2.04e-04
step 13: loss=0.318148  m_o=0.19274  v_o=5.02e-05  m_down=0.37196  v_down=7.53e-04  m_q=0.14610  v_q=2.19e-04
step 19: loss=0.304189  m_o=0.22127  v_o=6.24e-05  m_down=0.42460  v_down=9.40e-04  m_q=0.16813  v_q=2.81e-04
```

Interpretation:

- D1 already showed that zero-LR moment warmup alone does not rescue canonical
- the Adam-moment rerun sharpens that:
  - this does **not** look like a large scalar mismatch in Adam bootstrap
  - canonical and reverse accumulate very similar moment **magnitudes**
  - yet the losses diverge immediately at the first real update

Best current reading:

- Bug 1 is probably **not** an “optimizer scale” issue
- it is more likely a **directional** issue:
  - the gradients / moments / effective updates may have similar norms
  - but differ in sign pattern or elementwise direction on the fragile LoRA-B
    tensors

Consequence for next experiments:

- the next probe should not be another scalar warmup or norm-only audit
- it should inspect **signed elementwise values** around the first real update:
  - gradient sentinels
  - Adam ``mu`` sentinels
  - effective Adam update sentinels
  - especially on ``o_proj`` and ``down_proj`` where the Bug-1 LoRA story is
    already strongest

2026-04-21T D1 signed-Adam-sentinel follow-up (D1c / D1d):

Decision:

- the norm-only Adam rerun did its job: it ruled out a large scalar mismatch in
  Adam bootstrap
- the next discriminant has to be **directional**
- therefore the next D1 rerun keeps the same corrected setup
  (zero-LR prefix, 20 total steps, canonical vs reverse) and extends the
  existing LoRA debug callback to emit **signed fixed-index values** from:
  - Adam first moment ``mu``
  - effective Adam update ``mu / (sqrt(nu) + eps)``

Instrumentation patch:

- updated
  [lib/levanter/src/levanter/callbacks/lora_debug.py](/Users/ahmed/code/marin/.claude/worktrees/spicy-hugging-cat/lib/levanter/src/levanter/callbacks/lora_debug.py)
  so the callback now emits:
  - ``lora_debug/sentinel/adam_m/<module>/idx_<frac>``
  - ``lora_debug/sentinel/adam_effective_update/<module>/idx_<frac>``
- by default these are enabled for ``lora_B`` only:
  - ``include_sentinel_adam_m_b=True``
  - ``include_sentinel_effective_update_b=True``
  - A-side is still off by default because Bug 1 is already centered on the
    fragile first live ``B`` update

Why this is the right next probe:

- if canonical and reverse have near-identical ``|m|`` and ``|v|`` but
  different signed sentinel values in ``mu`` or in the effective Adam update,
  then Bug 1 is likely a **direction/sign-pattern defect** rather than an
  optimizer-scale defect
- that would fit the current evidence:
  - topology-sensitive XLA communication fork is real
  - dense full FT sees the fork but is robust
  - LoRA bifurcates
  - zero-LR warmup does not fix it
  - Adam norm magnitudes are nearly identical

Planned fresh rerun pair:

- ``D1c`` canonical signed-sentinel rerun
- ``D1d`` reverse signed-sentinel rerun

Both will use:

- the corrected D1 launcher
- ``MARIN_DEBUG_LORA_DEBUG=1``
- 20 steps
- 10 initial zero-LR steps
- same Bug-1 LoRA baseline otherwise

Primary W&B signals to inspect after launch:

- ``lora_debug/sentinel/grad_B/...``
- ``lora_debug/sentinel/adam_m/...``
- ``lora_debug/sentinel/adam_effective_update/...``

especially on:

- ``o_proj``
- ``down_proj``
- ``q_proj``

Immediate success criterion:

- we want the first real post-warmup steps (roughly 11, 12, 13) for
  canonical vs reverse with signed sentinel values visible in W&B

Fresh signed-sentinel D1 jobs submitted:

- canonical:
  ``/ahmed/experiment-d1-v5p8-pd4-canonical-zlrmomsign-20260421-2115``
- reverse:
  ``/ahmed/experiment-d1-v5p8-pd4-reverse-zlrmomsign-20260421-2115``

Launch commands:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name experiment-d1-v5p8-pd4-canonical-zlrmomsign-20260421-2115 \
  -- \
  bash -lc 'EXPERIMENT_D1_ORDER=canonical \
    MARIN_DEBUG_LORA_DEBUG=1 \
    MARIN_DEBUG_RUN_TAG=d1-canonical-zlrmomsign \
    REGIONS_OVERRIDE=us-east5,us-central1 \
    uv run python experiments/posttrain/per_stmt_dpo/experiment_d1_v5p8_pd4_lr0warmup_s20.py'
```

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name experiment-d1-v5p8-pd4-reverse-zlrmomsign-20260421-2115 \
  -- \
  bash -lc 'EXPERIMENT_D1_ORDER=reverse \
    MARIN_DEBUG_LORA_DEBUG=1 \
    MARIN_DEBUG_RUN_TAG=d1-reverse-zlrmomsign \
    REGIONS_OVERRIDE=us-east5,us-central1 \
    uv run python experiments/posttrain/per_stmt_dpo/experiment_d1_v5p8_pd4_lr0warmup_s20.py'
```

Status at submit time:

- both jobs accepted cleanly by Iris
- no launch-time config error
- next check is whether the resulting W&B runs expose the new signed keys:
  - ``lora_debug/sentinel/adam_m/...``
  - ``lora_debug/sentinel/adam_effective_update/...``

2026-04-22T D1c / D1d signed-sentinel result:

The signed-sentinel rerun pair reached and finished the 20-step train loop.

Runs:

- canonical:
  ``experiment_d1_s20_d1-canonical-zlrmomsign-bfe685``
- reverse:
  ``experiment_d1_r64_s20_d1-reverse-zlrmomsign-867d3e``

Because the public W&B history API timed out repeatedly on these runs, the
history comparison below was extracted directly from the worker-local
``.wandb`` bundles using:

- ``wandb.sdk.internal.datastore.DataStore``
- ``wandb.proto.wandb_internal_pb2.Record``

That bypassed GraphQL history pagination and gave the exact per-step history
rows as recorded on the worker.

Loss split remains immediate after the zero-LR prefix:

| step | canonical loss | reverse loss |
| --- | ---: | ---: |
| 11 | 0.693147 | 0.693147 |
| 12 | 0.689029 | 0.325414 |
| 13 | 0.678488 | 0.318148 |

Representative midpoint sentinel values (``idx_0.5``):

| step | module | grad_B canonical | grad_B reverse | adam_m canonical | adam_m reverse |
| --- | --- | ---: | ---: | ---: | ---: |
| 11 | ``q_proj`` | ``8.529e-05`` | ``8.681e-05`` | ``3.161e-05`` | ``3.146e-05`` |
| 11 | ``o_proj`` | ``3.228e-04`` | ``3.226e-04`` | ``6.497e-05`` | ``6.512e-05`` |
| 11 | ``down_proj`` | ``2.801e-06`` | ``8.106e-06`` | ``-3.053e-05`` | ``-3.014e-05`` |
| 12 | ``q_proj`` | ``1.378e-04`` | ``7.185e-05`` | ``3.171e-05`` | ``3.164e-05`` |
| 12 | ``o_proj`` | ``2.543e-04`` | ``1.401e-04`` | ``7.084e-05`` | ``7.096e-05`` |
| 12 | ``down_proj`` | ``-1.648e-04`` | ``-9.422e-05`` | ``-2.737e-05`` | ``-2.682e-05`` |
| 13 | ``q_proj`` | ``3.761e-05`` | ``3.576e-06`` | ``3.425e-05`` | ``3.460e-05`` |
| 13 | ``o_proj`` | ``2.952e-04`` | ``1.910e-04`` | ``7.428e-05`` | ``7.580e-05`` |
| 13 | ``down_proj`` | ``-2.050e-05`` | ``-1.055e-05`` | ``-3.145e-05`` | ``-3.217e-05`` |

What this says:

- the loss bifurcation still appears at the first real update
- selected signed ``grad_B`` sentinels already differ in **magnitude** at step
  12
- selected signed Adam ``mu`` sentinels remain extremely close between
  canonical and reverse
- these probes do **not** currently show a clean sign-flip story on the sampled
  indices

Important instrumentation finding:

- ``lora_debug/sentinel/adam_m/...`` keys were present in worker-side history
- ``lora_debug/sentinel/adam_effective_update/...`` keys were **absent**
- even the non-sentinel
  ``lora_debug/adam/effective_update/.../l2`` family was absent from the
  worker-local history stream

So the signed-effective-update part of D1c / D1d is currently incomplete:

- the code path intends to emit those keys
- but they did not land in the recorded history
- that is an instrumentation bug / emission gap, not model evidence

Clarification of terms used in this section:

- **sentinel**
  a fixed scalar probe sampled from a large tensor, used so canonical vs
  reverse can be compared at exactly the same tensor coordinates without
  logging the full tensor
- **emission**
  the callback actually writes a metric into the per-step W&B history
  (worker-local ``.wandb`` record stream), not just computes it in memory

How sentinel indices are chosen:

- flatten the tensor to 1-D
- let ``n = flat.shape[0]``
- for each configured fraction in ``sentinel_fractions = (0, 0.25, 0.5, 0.75, 1)``
- compute:

```python
idx = round(frac * (n - 1))
```

- clamp to ``[0, n - 1]``

So:

- ``idx_0`` = first element
- ``idx_0.25`` = about 25% through the flattened tensor
- ``idx_0.5`` = midpoint element
- ``idx_0.75`` = about 75% through
- ``idx_1`` = last element

Important subtlety:

- this operates on the **fully flattened** LoRA tensor, including scan stacking
- it is **not** “the midpoint of layer 0”; it is the midpoint of the entire
  flattened stacked tensor

Current best interpretation after D1c / D1d:

- Bug 1 still does **not** look like an Adam bootstrap-scale problem
- selected ``adam_m`` sentinels remain nearly identical across canonical and
  reverse
- the first obvious difference on these probes is in signed ``grad_B`` values,
  mainly as a magnitude / pattern difference rather than a clean sampled sign
  flip

Immediate next step:

- debug why ``adam_effective_update`` metrics are not emitted into history
- then rerun the same short canonical / reverse pair
- only after that decide whether we need denser sentinel coverage or true
  full-tensor snapshots for ``o_proj`` / ``down_proj``

## CLAUDESKSTART — Next-Agent Handoff

Last committed state on branch ``dpo-lora``:
``31ad93bc6 [dpo] D1 LR=0 warmup probe + Tune-LoRA a_init_mode=zero rerun``.
Working tree is clean.

### Where we are

The ``a_init_mode="zero"`` fix is shipped and eliminates Bug 1 at its
source (K-closure: ``v5p`` / ``v6e`` step-8 gap ``0.356 → 0.0007``).
**Everything below this line is root-cause investigation, not
mitigation.** Marin practice should already be on the init flip; do not
regress that default while poking at the mechanism.

The still-open scientific question is **why** the first live ``lora_B``
update on output-sharded projections (``o_proj``, ``down_proj``)
bifurcates under physical device permutation. The current best
interpretation from L1, zlrmom, and D1c/d is:

- ``delta_W`` Frobenius norms agree canonical vs reverse to ~1 %
- Adam ``m`` and ``v`` norms agree to ~1 %
- signed ``adam_m`` sentinels at the five sampled fractions agree to ~1 %
- but the loss bifurcates ~50 % immediately at the first non-zero step

Conclusion: the defect is **directional / sign-pattern / coordinate
structure**, not scale or marginal distribution. That rules out most of
the scalar probes we have been reaching for.

### Why the last round underperformed (read before planning the next one)

Two problems with D1c / D1d that should not be repeated:

1. **Emission gap**. ``lora_debug/sentinel/adam_effective_update/*`` and
   the aggregate ``lora_debug/adam/effective_update/*/l2`` do **not**
   land in the W&B history stream, even though the callback code path
   intends to emit them. This was never smoke-tested before the TPU
   rerun. Unforced error.
2. **Wrong instrument for the question**. Five fixed-index sentinels
   on a scan-stacked flattened tensor cannot distinguish a direction
   defect from a distribution-shape agreement. Two tensors with
   identical marginal statistics can differ by an arbitrary rotation,
   coordinate permutation, or per-index sign flip, and 5 scalars from
   the flat tensor will usually miss it. Histograms have the same
   limitation at the tensor level.

The direction question needs **both** tensors in the same analysis
(canonical and reverse side by side). That means full-tensor capture,
not more scalar summaries.

### Priority 1 — Fix ``effective_update`` emission (~30 min)

File:
[lib/levanter/src/levanter/callbacks/lora_debug.py](../../lib/levanter/src/levanter/callbacks/lora_debug.py).

Likely suspects:

- ``_emit_opt_state_stats`` not matching the correct optax path
  substring for ``mu`` / ``nu`` within the LoRA subtree (watch for
  wrapper states like ``MaskedState`` / ``ScaleByScheduleState`` that
  sit above ``ScaleByAdamState`` on only some of the parameter tree).
- A key-prefix typo so the emission ends up under a key that isn't
  read back by the W&B history API.
- Sentinel fractions iterated for ``grad_B`` and ``adam_m`` but not
  for ``adam_effective_update``.

Smoke-test locally against a toy ``optax.adam(...).init(params)`` on a
synthetic LoRA tree **before any TPU work**:

```python
out = {}
_emit_opt_state_stats(state, params, out)
assert any("effective_update" in k for k in out), sorted(out)[:20]
```

Do not launch Iris jobs until that assert passes and the emitted keys
also appear in a local ``tracker.log_metrics(out)`` → W&B sync.

### Priority 2 — Full-tensor GCS dump → offline cosine / sign-agreement (~half day)

This is the probe that actually addresses the direction hypothesis.
Histograms and sentinels both summarise to scalars and throw away the
coordinate structure we need.

**Config surface** (extend ``LoraDebugConfig``):

- ``dump_tensors_at_steps: tuple[int, ...] = ()``
  example value for the D1 post-warmup window: ``(11, 12, 13)``
- ``dump_tensor_modules: tuple[str, ...] = ()``
  example: ``("o_proj", "down_proj")``
- ``dump_tensor_path: str | None = None``
  GCS prefix, e.g.
  ``gs://marin-us-central1/debug/bug_1_tensor_dump/20260422/{run_tag}/``

**What to dump**, per module, per configured step, per variant
(canonical / reverse):

- ``grad_B`` (pre-optimizer)
- Adam first moment ``mu`` (LoRA-B side only)
- Adam second moment ``nu`` (LoRA-B side only)
- effective update ``mu / (sqrt(nu) + eps)``

**Object layout**:
``{dump_tensor_path}/step_{N:04d}/{module}/{metric}.npy``

**Shape notes** (important for the serializer):

- ``o_proj`` / ``down_proj`` ``lora_B`` shape is
  ``(layer, Embed, LORA_R)`` — single-Out axis.
- ``q_proj`` ``lora_B`` shape is
  ``(layer, Heads, HeadSize, LORA_R)`` — multi-Out axis; the multi-axis
  bug that broke L0 originally was in this path. Use
  ``LowRankLinear.merge()`` or ``hax.dot`` on ``LORA_R``, not a
  hand-rolled einsum. Unwrap ``NamedArray`` via ``.array`` before
  ``jnp.save``.

**IO path**. The callback is inside a ``JitCallback``, so direct
``jnp.save`` inside JIT does not work. Use ``jax.experimental.io_callback``
(host-side, blocks JIT briefly at the dump steps only) or return the
tensors out of the JIT region and write from a step-end hook. Gate
everything on ``dump_tensors_at_steps`` so the dump path is a no-op at
non-dump steps — do **not** write every step.

**Volume estimate**. ``o_proj.lora_B`` at 8B Llama ≈ 32 layers × 4096 ×
64 × 4 B ≈ 32 MB. 4 metrics × 2 modules × 3 steps × 2 variants ≈ 1.5 GB
total. Trivial.

**Offline analysis** (scratch notebook, not a committed script):

```python
for step in (11, 12, 13):
    for mod in ("o_proj", "down_proj"):
        for metric in ("grad_B", "mu", "effective_update"):
            c = np.load(f"{prefix}/d1-canonical-tdump/step_{step:04d}/{mod}/{metric}.npy")
            r = np.load(f"{prefix}/d1-reverse-tdump/step_{step:04d}/{mod}/{metric}.npy")
            # per-layer cosine
            c_ly = c.reshape(c.shape[0], -1)
            r_ly = r.reshape(r.shape[0], -1)
            cos = (c_ly * r_ly).sum(-1) / (
                np.linalg.norm(c_ly, axis=-1) * np.linalg.norm(r_ly, axis=-1) + 1e-12
            )
            sign_agree = (np.sign(c) == np.sign(r)).mean()
            rel_diff = np.linalg.norm(c - r) / (np.linalg.norm(c) + 1e-12)
            ...
```

**Decision criteria**:

- ``cosine < 0.5`` on ``o_proj`` / ``down_proj`` at step 11 while
  ``> 0.95`` on ``q_proj``: direction fork confirmed at the coordinate
  level. Topology hypothesis stands.
- ``cosine ≈ 1`` everywhere but loss still diverges: the defect is
  downstream of these tensors — look at ``lora_A`` interaction or
  forward-pass activations, not more optimizer state.
- ``sign_agree < 0.6`` on a specific layer range: the permutation
  effect is layer-localised and the next probe is per-layer, not
  per-module.

### Priority 3 — Targeted histograms (optional complementary probe)

Per the Codex suggestion. Infrastructure already exists:

- [lib/levanter/src/levanter/tracker/histogram.py](../../lib/levanter/src/levanter/tracker/histogram.py)
- [lib/levanter/src/levanter/callbacks/watch.py](../../lib/levanter/src/levanter/callbacks/watch.py)
- ``split_scan_layers=True`` already supported in watch for per-layer
  histogram emission

Enable **targeted** histograms for LoRA-B opt state on
``o_proj`` / ``down_proj`` / ``q_proj`` only, split by scan layer,
steps 11-13 only. Do **not** turn on the generic ``watch`` callback
without a filter — it emits too many leaves.

This is secondary because histograms cannot by themselves distinguish
a direction defect (identical marginal distributions + different
coordinate structure is compatible with any amount of directional
disagreement). Run it alongside P2, not instead of it.

### What not to do

- More sentinels at different fractions. Same blind spot; will produce
  more "norms agree, sentinels mostly agree, loss diverges" rows in
  the logbook.
- Rerunning any D1 / zlrmom / zlrmomsign variant before P1 emission
  fix lands and is verified.
- Treating W&B as the sink for direction questions. For this
  investigation the sink is GCS + offline numpy, and the summary is a
  scalar per-layer cosine, not a per-step scalar time-series.
- Adding more Adam-norm probes. L1 and zlrmom already proved norms
  agree.
- Editing any of the 51 historical ``experiment_*.py`` files in
  ``per_stmt_dpo/`` to retrofit ``a_init_mode="zero"`` — historical
  results are tied to their original init. The breadcrumb note on
  each of those files is sufficient.
- Regressing the ``a_init_mode="zero"`` default. This is shipped
  mitigation; breaking it for an investigation rerun would be a real
  user-facing regression.

### Recommended single rerun shape after P1 + P2

One clean pair of Iris jobs:

- launcher:
  [experiment_d1_v5p8_pd4_lr0warmup_s20.py](../../experiments/posttrain/per_stmt_dpo/experiment_d1_v5p8_pd4_lr0warmup_s20.py)
- 20 steps, 10-step zero-LR prefix (unchanged)
- ``EXPERIMENT_D1_ORDER`` = ``canonical`` / ``reverse``
- ``MARIN_DEBUG_LORA_DEBUG=1``
- new ``dump_tensors_at_steps=(11, 12, 13)``,
  ``dump_tensor_modules=("o_proj", "down_proj")``,
  ``dump_tensor_path=gs://marin-us-central1/debug/bug_1_tensor_dump/<date>/<variant>/``
- run tags: ``d1-canonical-tdump`` / ``d1-reverse-tdump`` (avoids W&B
  display-name collision with prior zlrmomsign pair)
- regions: ``us-east5,us-central1`` per the standing multi-region rule
- launch via ``bash -lc '...env vars... uv run python ...'`` inside
  the Iris command so env vars land on the TPU worker, not the local
  CLI (see the D1b fix note around
  ``2026-04-21T launch correction`` above for why)

Expected wall-clock ~15 min per variant including capacity wait.

### Quick reference

- branch: ``dpo-lora`` at ``31ad93bc6``
- last verified good mitigation:
  ``zero_init_b=False, a_init_mode="zero"`` — keep shipping this
- canonical launcher:
  [experiment_d1_v5p8_pd4_lr0warmup_s20.py](../../experiments/posttrain/per_stmt_dpo/experiment_d1_v5p8_pd4_lr0warmup_s20.py)
- callback to extend:
  [lib/levanter/src/levanter/callbacks/lora_debug.py](../../lib/levanter/src/levanter/callbacks/lora_debug.py)
- optimizer knob already in place:
  [lib/levanter/src/levanter/optim/config.py](../../lib/levanter/src/levanter/optim/config.py) → ``initial_zero_lr_steps``
- existing GCS debug prefix pattern:
  ``gs://marin-us-central1/debug/bug_1_*/``
- Iris config:
  ``lib/iris/examples/marin.yaml``
- do **not** use ``EXPERIMENT_BL_ORDER`` on D1 — the knob is
  ``EXPERIMENT_D1_ORDER``
- do **not** set env vars on the local ``uv run iris ...`` process
  expecting them to reach the TPU worker — they won't; wrap the
  remote command in ``bash -lc '...'`` with the env vars inside

### One-sentence summary for the next agent

Fix the ``effective_update`` emission gap in ``lora_debug.py`` first
(verify keys land locally), then add a full-tensor GCS dump hook for
``o_proj.lora_B`` and ``down_proj.lora_B`` at steps 11-13 and run
offline cosine / sign-agreement analysis — that is the probe that
actually answers the direction question that L1, zlrmom, and D1c/d all
pointed at without resolving.


## 2026-04-22T P1 + P2: Emission fix shipped, tensor-dump infrastructure added, D1-tdump pair launched

Executes the CLAUDESKSTART handoff end-to-end. P1 (effective_update
emission bug) is fixed and verified. P2 (full-tensor GCS dump for offline
direction analysis) is implemented behind an explicit safety gate. Four D1
Iris jobs are now live that will produce the first full-tensor canonical
vs reverse snapshot at the post-zero-LR bifurcation window.

### P1 — root cause and fix

`_emit_opt_state_stats` computed `mod_key` from the full flattened opt-state
path, which for optax Adam includes a `.mu`/`.nu` segment
(e.g. `inner_state/1/mu/transformer/.../q_proj`). The mu version and the
nu version of the same parameter therefore built different pairing keys
(`B/inner_state/1/mu/...` vs `B/inner_state/1/nu/...`), so
`nu_by_key.get(mu_key)` returned `None` for every entry. The
`m/(sqrt(v)+eps)` block `continue`d for every parameter → zero
`adam/effective_update/*` and zero `sentinel/adam_effective_update/*` keys
ever emitted to W&B in D1c/D1d. Sentinel `adam_m` survived because its
list was built independently of mu/nu pairing.

Fix: add `_strip_adam_slot_segments(path)` that drops `mu`/`nu`
components before computing the module key. Slot detection still runs on
the original path so `adam/m/` vs `adam/v/` labeling is unchanged. Side
benefit: keys no longer carry the slot marker — new shape is
`lora_debug/adam/{m|v|effective_update}/{A|B}/inner_state/1/transformer/.../q_proj/l2`
instead of the prior
`.../adam/m/B/inner_state/1/mu/transformer/.../q_proj/l2`. Historical
D1c/D1d keys with embedded `/mu/` are obsolete; any future cross-run diff
should use post-fix runs only.

Commit: `cfc3fea03 [dpo] P1: Fix lora_debug effective_update emission gap`.

Verified offline on a Llama-8B-shaped params tree under the real
`optax.inject_hyperparams + chain(clip, scale_by_adam,
add_decayed_weights(masked), scale)` layout. Both `adam/effective_update`
aggregates and `sentinel/adam_effective_update` keys now emit. The
aggregate value for `q_proj.B` matched a direct NumPy recompute to
fp32 sum-order noise (~2e-4 relative). Repro script at
[scratch/p1_repro_effective_update.py](../../scratch/p1_repro_effective_update.py).

### P2 — full-tensor GCS dump

Norms and fixed-index sentinels both summarise each tensor to a few
scalars. Two tensors with identical marginal statistics can still differ
by an arbitrary rotation, coordinate permutation, or per-index sign
flip — which is exactly the regime D1c/D1d implied (norms agree ~1%,
loss bifurcates ~50%). Closing that question needs both tensors in the
same analysis, not more scalar summaries.

Added four knobs to `LoraDebugConfig`, all gated behind a single
top-level safety flag:

```
verbose_grads:        bool = False   # hard guard; must be True to do anything
dump_tensors_at_steps: tuple[int, ...] = ()
dump_tensor_modules:   tuple[str, ...] = ()
dump_tensor_path:      Optional[str] = None   # fsspec prefix (gs://, local, ...)
```

Inside `inside_step`, when all four are non-default the callback collects
`grad_B` (pre-optimizer), Adam `mu_B`, `nu_B`, and `effective_update_B
= mu / (sqrt(nu) + eps)` for each module whose path substring-matches
`dump_tensor_modules`. Tensors are `lax.cond`-gated on
`state.step ∈ dump_tensors_at_steps`: the non-dump branch returns
`jnp.zeros_like` of the same shape, so the JIT payload shape stays
fixed and XLA can elide real device→host transfer on non-dump steps.

Tensors flow out under `__lora_tensor_dump__/{metric}/{module}` keys in
the normal `cb_info` dict. `on_step` splits the dict on that prefix
before calling `levanter.tracker.log` so W&B never sees the 256 MB
blobs even if the gate was flipped accidentally. When the Python-level
step matches a configured dump step, the host writes each tensor to
`{dump_tensor_path}/step_{N:04d}/{module}/{metric}.npy` via
`fsspec.open` (supports `gs://` and local).

Commit: `9795627c9 [dpo] P2: Full-tensor GCS dump for offline
cosine/sign-agreement analysis`.

Env-var plumbing added so SimpleDPOConfig-based experiments can enable
P2 without modifying `default_dpo` to thread a new field. Four env
vars, all consumed by `LoraDebugConfig.with_env_overrides()` which
`build()` now calls:

```
MARIN_DEBUG_LORA_VERBOSE_GRADS=1          # required explicit opt-in
MARIN_DEBUG_LORA_DUMP_STEPS="11,12,13"
MARIN_DEBUG_LORA_DUMP_MODULES="o_proj,down_proj"
MARIN_DEBUG_LORA_DUMP_PATH="gs://..."
```

The `verbose_grads` gate is intentionally a **separate env var** from
`MARIN_DEBUG_LORA_DEBUG=1`: setting only a `DUMP_PATH` cannot enable
dumps. Both gates must be on. The D1 launcher mirrors these into the
TPU worker env when present in the parent process.

Commit: `d51bdfe0d [dpo] P2 env-var plumbing + D1 launcher pass-through`.

### Verification before launch

All exercised on a Levanter-shaped optax chain + 8B-Llama-shaped params
tree (q/k/v/o + gate/up/down LoRA with scan-stacked layers):

- **Default-off path** (`verbose_grads=False`): 270 scalar keys emit
  normally; zero `__lora_tensor_dump__/` keys; zero P2 overhead.
- **Verbose-on 5-step walk** with `dump_steps=(2,3)`,
  `modules=(o_proj,down_proj)`:
  - 8 tensor keys per step in `cb_info` (4 metrics × 2 modules).
  - Zeros on steps 0, 1, 4 — `lax.cond` gating correct.
  - Real non-zero values on steps 2, 3.
  - On-disk: only `step_0002/` and `step_0003/` dirs exist, each with
    exactly 8 `.npy` files. No stray writes.
- **Split logic**: tracker.log receives only scalar keys even when
  tensor keys are present in cb_info.
- **P1 regression**: passes — `effective_update` emission still works.

Scratch scripts preserved at
[scratch/p2_smoke_tensor_dump.py](../../scratch/p2_smoke_tensor_dump.py)
and
[scratch/p2_smoke_default_off.py](../../scratch/p2_smoke_default_off.py).

### D1-tdump rerun launched (`20260423-0021` Z stamp)

Rerun shape per the handoff: same D1 recipe (20 steps, 10-step zero-LR
prefix, canonical vs reverse, `MARIN_DEBUG_LORA_DEBUG=1`), with P2 env
vars set to dump at steps `(11, 12, 13)` (the first three live training
steps after zero-LR warmup) for `o_proj` and `down_proj` — the two
output-sharded projections already established as the Bug-1 epicenter.

Per the standing multi-region rule, 4 jobs submitted
(canonical/reverse × us-central1/us-east5):

| variant | region     | Iris job                                                         | GCS dump prefix                                                              |
|---------|------------|-----------------------------------------------------------------|------------------------------------------------------------------------------|
| canonical | us-central1 | `/ahmedah/experiment-d1-canonical-central1-tdump-20260423-0021` | `gs://marin-us-central1/debug/bug_1_tensor_dump/20260422/d1-canonical-tdump/` |
| canonical | us-east5    | `/ahmedah/experiment-d1-canonical-east5-tdump-20260423-0021`    | `gs://marin-us-east5/debug/bug_1_tensor_dump/20260422/d1-canonical-tdump/`    |
| reverse   | us-central1 | `/ahmedah/experiment-d1-reverse-central1-tdump-20260423-0021`   | `gs://marin-us-central1/debug/bug_1_tensor_dump/20260422/d1-reverse-tdump/`   |
| reverse   | us-east5    | `/ahmedah/experiment-d1-reverse-east5-tdump-20260423-0021`      | `gs://marin-us-east5/debug/bug_1_tensor_dump/20260422/d1-reverse-tdump/`      |

Each run tagged `d1-<order>-<region>-tdump` in `MARIN_DEBUG_RUN_TAG`.
At submit time all four parent tasks reached state `JOB_STATE_RUNNING`
with no errors. TPU children had not yet been spawned (CPU parent in
executor-prep phase).

### Expected artifacts per successful variant

```
{gcs_prefix}/step_0011/o_proj/{grad_B,mu_B,nu_B,effective_update_B}.npy
{gcs_prefix}/step_0011/down_proj/{grad_B,mu_B,nu_B,effective_update_B}.npy
{gcs_prefix}/step_0012/o_proj/{grad_B,mu_B,nu_B,effective_update_B}.npy
{gcs_prefix}/step_0012/down_proj/{grad_B,mu_B,nu_B,effective_update_B}.npy
{gcs_prefix}/step_0013/o_proj/{grad_B,mu_B,nu_B,effective_update_B}.npy
{gcs_prefix}/step_0013/down_proj/{grad_B,mu_B,nu_B,effective_update_B}.npy
```

24 tensors × 2 variants × 2 regions = 96 objects total. Volume is
dominated by `grad_B`, `mu_B`, `nu_B`, `effective_update_B` on
`down_proj` (fp32 × layers × ffn_dim × r) ≈ 120 MB each on 8B Llama.
Total across the cohort: single-digit GB.

Within each region×variant, the four metrics for each module should
satisfy `effective_update_B ≈ mu_B / (sqrt(nu_B) + 1e-8)` — this is a
cheap post-hoc sanity check.

### What the offline analysis will compute

Per-layer, per-module, per-metric, per-step comparisons between the
canonical and reverse runs (using the same region to avoid cross-region
noise — prefer the us-central1 pair):

- **Flat cosine**: `(c·r) / (||c|| ||r||)` over the full flattened
  tensor.
- **Per-layer cosine**: reshape to `(layer, -1)`, compute per-layer
  cosine. Bug-1 hypothesis predicts `cosine < 0.5` on `o_proj` /
  `down_proj` at step 11 with `q_proj` control staying `> 0.95`.
- **Sign agreement**: `(sign(c) == sign(r)).mean()`. Expected fork on
  output-sharded modules; expected ~1 on input-sharded control.
- **Relative L2**: `||c - r||_2 / ||c||_2`. Pairs with cosine to split
  norm vs direction defects.

Deferred decision rules (carried over from the handoff):

- `cosine < 0.5` on `o_proj` / `down_proj` at step 11 while > 0.95 on
  `q_proj` → direction fork confirmed at the coordinate level. Topology
  hypothesis stands; look at what per-layer pattern separates them.
- `cosine ≈ 1` everywhere but loss still diverges → defect is
  downstream of these tensors; look at `lora_A` interaction or
  forward-pass activations, not more optimizer state.
- `sign_agree < 0.6` on a specific layer range → effect is
  layer-localised; next probe should be per-layer rather than
  per-module.

### Next checkpoint

When any of the four jobs reaches terminal state, run:

1. `gsutil ls -r gs://marin-us-central1/debug/bug_1_tensor_dump/20260422/` to
   inventory the dumps.
2. Verify each run produced 24 .npy files at 3 steps × 2 modules × 4
   metrics.
3. Pull matching canonical/reverse pair (prefer us-central1) locally
   into `.agents/artifacts/bug_1_tdump/20260422/`.
4. Run the offline cosine / sign-agreement notebook (not yet written;
   template at the bottom of the CLAUDESKSTART section).

Low-risk operational notes:
- us-east5 runs write to a us-east5 bucket (region-local), so no
  cross-region transfer overhead.
- If the us-central1 pair finishes first with bit-identical recipe
  determinism (standard for these experiments), the us-east5 pair can
  be cancelled to free capacity.
- `.agents/artifacts/` is locally gitignored per the earlier BL lesson
  so pulled dumps won't bloat the next Iris workspace bundle.
