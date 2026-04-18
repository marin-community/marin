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
