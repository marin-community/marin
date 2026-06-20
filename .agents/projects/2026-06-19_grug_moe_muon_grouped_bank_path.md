# Grug MoE MuonH Grouped-Bank Path

Status: 2026-06-20.

## Goal

Make MuonH practical for Grug MoE on CoreWeave while keeping `model_axis=1`:

1. Keep train-state/master parameters in the model's FSDP layout when that is
   required by training semantics.
2. Compute Newton-Schulz over a grouped optimizer layout, especially for routed
   expert weights.
3. Avoid per-leaf collective explosions at the grouped-to-model boundary.
4. Validate on single-node and scaled CoreWeave update gates before another
   full training profile.

## Current Evidence

The Muon-only grouped expert path is fast when the grouped representation is
preserved:

| path | shape | time | compiled collectives |
| --- | --- | ---: | --- |
| single-node full-production H3 update-only | L26, E8 | about 0.617 s | AG/AR/RS = 0/0/0 |
| R2/D2/E8 bf16-NS expert grouped apply-only | L26, 4 nodes | about 0.173 s | AG/AR/RS/A2A = 0/0/0 |
| R4 persistent grouped-2D apply | L26, 4 nodes | about 0.175-0.177 s | AG/AR/RS = 0/0/0 |

The production `ns_compute_dtype=bf16` knob also works for fp32 inputs. On a
one-node L4 production-shaped A/B, bf16 Newton-Schulz improved
`full_production_muonh_optimizer_apply_h3` from about 0.190 s to about
0.096 s, a 1.98x speedup, with compiled AG/AR/RS/A2A = 0/0/0/0. The equivalent
L26 one-node fp32-input attempts OOMed during timing, so use smaller layer
counts or a memory-focused setup when validating fp32-input production paths.

The FSDP-master grouped-boundary path has not met the bar:

| boundary | shape | result |
| --- | --- | --- |
| XLA-inferred grouped-to-FSDP restore/apply | R2/D2/E8, L26 | replicate-then-partition warnings or OOMs |
| explicit all-gather restore-only | R2/D2/E8, L26 | bf16 fits at about 0.305 s; fp32 full tree OOMs |
| explicit all-gather fused restore+apply | R2/D2/E8, L26 | fits, but about 0.611 s and 112 compiled all-gathers |
| grouped chunk 8/16 fused restore+apply | R2/D2/E8, L26 | all-gather count drops, runtime stays about 0.614 s |
| explicit replica all-gather + data all-to-all apply | R2/D2/E8, L26 | about 0.619 s, compiled 112 all-gathers plus 98 all-to-alls |
| `jax.custom_partitioning` restore stub | R2/E8, L26, 2 H100 nodes | lowered as 26 custom calls, but compiled GPU HLO inlined back to 26 all-gathers |
| direct restore-and-apply at FSDP leaves | R2/E8, L26, 2 H100 nodes | compiled to the same 8 all-gathers as restore-then-`optax.apply_updates` |
| persistent grouped expert params/apply | R2/E8, L26, 2 H100 nodes | compiled with zero AG/A2A/AR/RS/CP |
| grouped MuonH plus grouped expert-bank consumer | R2/E8, L26, 2 H100 nodes | compiled with zero AG/A2A/AR/RS/CP and 175 GPU GEMM custom calls |
| grouped MuonH plus public grouped-MoE consumer | R2/E8, L26, 2 H100 nodes | compiled with same AG/RS/A2A as standalone grouped-MoE consumer: 11/1/0 |

Conclusion: preserving grouped params/updates/results is fast; converting
grouped updates back to ordinary per-layer FSDP leaves in the hot path is not.
The two latest H100 compile-only checks tighten this conclusion:

- `jax.custom_partitioning` is only a lowered-HLO insertion point here. XLA GPU
  still inlines the body and emits the same all-gathers as an explicit tuple
  restore.
- Leaf-local direct `apply_updates` compiles faster than restore-then-apply,
  but it does not reduce the grouped-to-FSDP collective count. It should stay a
  diagnostic path unless a runtime profile shows a separate overlap/memory
  benefit.
- A grouped expert-bank consumer avoids the grouped-to-FSDP restore entirely in
  the synthetic gate. On 2 H100 nodes, the strict
  `--require-no-boundary-collectives` gate compiled with zero boundary
  collectives, which is the first authoritative evidence that the persistent
  grouped-bank representation can survive both MuonH and an expert-like
  consumer.
- A grouped MuonH plus public `grouped_moe_mlp` consumer gate is the closer
  model-path proxy. On 2 H100 nodes, MuonH+grouped-MoE compiled with the same
  communication counts as standalone grouped-MoE (`AG/RS/A2A = 11/1/0`) while
  adding the expected NS/GEMM custom calls. The legitimate EP collectives mean
  this gate cannot use the global zero-collective strict check; compare against
  the standalone grouped-MoE baseline instead.

The harness now reports boundary byte estimates for these rows. For the
R2/D2/E8 May shape, the FSDP output shard is already 2x the grouped input shard
per device, because the grouped layer axis starts sharded over
`replica_dcn,data` while the FSDP result replicates layer across those axes and
only shards one matrix axis over `data`. The all-gather+slice helper transiently
materializes 4x the grouped input shard per device before discarding the
unneeded `data` shard. The data all-to-all variant targets the lower-byte
boundary, but the measured result stayed flat/slightly worse, so packed or
reshaped versions of the same grouped-to-FSDP conversion are unlikely to be
enough unless they also reduce the required FSDP output materialization or
overlap it with useful work.

## OSS Implementation Clues

Public Muon implementations mostly avoid this hot per-leaf conversion rather
than solving it with a generic optimizer-tree restore:

- Megatron layer-wise Muon is the closest conceptual model. It separates
  model-visible parameter layout from optimizer ownership: whole Muon-managed
  matrices are owned/updated by selected ranks and then synchronized back to a
  model-consumable parameter/bucket layout.
- Dion/Axolotl is the closest FSDP2 transport analogue. It batches compatible
  DTensor parameters, all-to-alls shards so each rank owns complete matrices for
  Newton-Schulz, then all-to-alls back. This validates coarse grouped transport,
  but copied per leaf it would reproduce the XLA collective-explosion problem.
- KellerJordan/modded-nanogpt uses architecture-specific parameter banks and
  bank-level reduce-scatter/all-gather around batched Newton-Schulz. This is a
  useful grouping clue if Grug expert banks become first-class model objects.
- NeMo's Emerging Optimizers are useful for batched Newton-Schulz and
  MuonHyperball API shape, but do not directly solve the FSDP ownership
  boundary.

Design rule from these implementations: Muon ownership is allowed to be a
different representation from model consumption, but the synchronization
boundary must be explicit and coarse. Avoid per-leaf gather/scatter or per-leaf
all-to-all even if PyTorch/FSDP2 examples tolerate it.

## Why The Boundary Is Bad

The model consumes routed expert weights as ordinary per-layer
`MoEExpertMlp.w_gate_up` and `MoEExpertMlp.w_down` arrays:

- `lib/levanter/src/levanter/grug/grug_moe.py`: `MoEExpertMlp.__call__`
  passes `self.w_gate_up` and `self.w_down` directly to `moe_mlp`.
- `moe_mlp` currently expects rank-3 expert arrays `[E, D, I2]` and
  `[E, I, D]`.
- `experiments/grug/moe/model.py`: each `Block` owns one `MoEMLP`, and each
  `MoEMLP` owns one `MoEExpertMlp`.

If grouped Muon stores expert weights as `[group, E, D, I2]` and
`[group, E, I, D]` sharded over `replica_dcn,data`, selecting one layer's
rank-3 leaf reintroduces a grouped-to-FSDP conversion. The harness now shows
that conversion costs about 0.61 s at the May shape even when expressed
explicitly.

## Next Integration Gate

## FSDP-Master Objective Audit

The conservative objective is still:

1. keep train-state/master expert params in ordinary FSDP/model layout;
2. compute MuonH in an NS-friendly grouped optimizer layout;
3. convert grouped updates back to FSDP leaves before ordinary
   `optax.apply_updates`;
4. prove the bridge avoids the per-leaf collective explosion and preserves
   performance on the target CoreWeave shapes.

Current evidence does **not** prove this objective complete.

| Requirement | Current evidence | Status |
| --- | --- | --- |
| FSDP params at the API boundary | `real_expert_fsdp_grouped_muonh_optimizer_*` and `expert_fsdp_grouped_trace_muonh_apply` keep input params in FSDP-shaped leaves. | partial |
| Grouped NS-friendly MuonH compute | Grouped-update and grouped-trace harnesses compute MuonH in `P(("replica_dcn", "data"), "expert", None, None)` or the R2/D1 specialization and reach about 50% nominal peak in update-only runs. | achieved in harness |
| Convert grouped updates back before ordinary `apply_updates` | Explicit slice-first restore/apply, target-layout reshard, tuple-returning `shard_map`, direct restore/apply, and grouped-trace restore/apply all produce ordinary FSDP-shaped results before apply. | achieved semantically |
| Avoid per-leaf collective explosion | Not achieved. H100 compiled HLO reintroduces grouped-to-FSDP all-gathers for every JAX-level bridge tested so far. Direct target-layout reshard lowered cleanly but compiled back to the same 26 all-gathers on the R2/D1 gate; packed variants reduced lowered all-gathers but introduced worse compiled A2A/CP patterns or OOMs. | failed so far |
| Preserve end-to-end performance | Not achieved for the FSDP-master bridge. Best update-only grouped-trace/FSDP-master path is a useful partial win, but still pays the compiled all-gather boundary. Persistent grouped-bank consumers remove that boundary, but they are a different representation contract until the model consumes grouped banks. | incomplete |

This means the FSDP-master path has a clear remaining blocker, not a missing
bookkeeping step: the grouped-to-FSDP bridge must be lower-level than current
JAX reshards/`shard_map`, or the hot path must avoid restoring ordinary FSDP
leaves. A lower-level bridge should be judged against the current explicit
slice-first baseline: fewer compiled all-gathers, no new all-to-all/collective
permute explosion, and comparable or better H100 runtime.

New harness rows also report:

- `estimated_boundary_replica_fanout_factor`
- `estimated_boundary_requires_replica_fanout`
- `estimated_boundary_replica_fanout_min_extra_per_device_bytes`
- `estimated_boundary_replica_fanout_min_total_receive_bytes`

These fields capture the inherent part of the bridge: grouped MuonH ownership
over `replica_dcn` must fan out to FSDP leaves because expert FSDP params do not
name `replica_dcn` in their sharding spec. They do not excuse the current
compiled per-layer all-gather pattern; they separate the unavoidable fanout from
avoidable compiler-induced fragmentation.

Do not launch another full training profile from this path until this gate is
ported into the real model path. The harness now has an initial
`expert_grouped_bank_consumer` bench for this purpose: it consumes grouped expert
banks with grouped
`[group, expert, token, hidden]` activations and runs the gate/up and down
expert MLP matmuls without restoring per-layer FSDP leaves.

The synthetic `expert_grouped_muonh_bank_consumer` compile-only gate has passed
the strict H100 gate on 2 nodes for R2/E8/L26 with
`boundary_collectives_required_absent=true` and zero compiled AG/A2A/AR/RS/CP.
The closer `expert_grouped_muonh_moe_mlp_consumer` gate has also passed on 2
H100 nodes with no additional compiled collectives beyond standalone
`grouped_moe_mlp`. The remaining integration work is to make the real
`MoEExpertMlp`/`MoEMLP` path consume this representation, or to add an explicit
coarse transport if we keep ordinary FSDP leaves.

The first production-facing adapter now exists:
`GroupedMoEExpertMlp.from_layers(layers)` stacks ordinary per-layer
`MoEExpertMlp` modules into a grouped expert bank and rejects mixed
implementation/activation/capacity/remat settings. Its model-level regression
test compares grouped execution against independent per-layer calls. This is a
small but important invariant: a future grouped `MoEMLP` or grouped block can
start from today's per-layer module state without using benchmark-only helpers.

Build the real model-consumer path that keeps expert weights grouped through the
consumer boundary:

1. Represent routed expert weights as grouped banks:
   - `blocks[*].mlp.expert_mlp.w_gate_up`: `[group, E, D, I2]`
   - `blocks[*].mlp.expert_mlp.w_down`: `[group, E, I, D]`
   - group axis sharded over `replica_dcn,data` when divisible, padded when not.
2. Add a consumer that invokes the existing MoE local/EP kernels for every layer
   in a group without first materializing a tuple of ordinary rank-3 FSDP leaves.
3. Compile the gate for R2/D2/E8 and assert:
   - grouped expert params preserve `P(("replica_dcn", "data"), "expert", None, None)`;
   - compiled all-gather/reduce-scatter/all-reduce are zero inside the grouped
     optimizer/apply path;
   - any required per-layer selection happens inside the grouped consumer, not
     at `optax.apply_updates`.
4. Only after the synthetic gate passes, port the representation into the real
   `MoEExpertMlp`/`MoEMLP` path.

The immediate next patch should be a grouped `MoEMLP`/block consumer that:

1. runs each layer's router normally to produce grouped routed inputs,
2. calls `GroupedMoEExpertMlp` directly over those grouped routed inputs, and
3. keeps `GroupedMoEExpertMlp.layer()` out of the hot path except for parity
   tests.

The gate should explicitly fail any production-candidate path whose compiled
HLO contains grouped boundary collectives. Use the harness
`--require-no-boundary-collectives` flag for candidate rows and reserve
`--allow-boundary-collectives` only for attribution/decomposition runs.

## Implementation Sketch

The likely code shape is a separate grouped expert module rather than changing
`MoEExpertMlp` in place:

```python
class GroupedMoEExpertMlp(eqx.Module):
    w_gate_up: jax.Array  # [G, E, D, I2]
    w_down: jax.Array     # [G, E, I, D]
    valid_group_size: int = eqx.field(static=True)

    def layer(self, local_layer_index: int) -> MoEExpertMlp:
        ...
```

The `layer()` API is useful for correctness tests, but the performance-critical
path should avoid calling `layer()` in the hot loop if it lowers to the same
all-gather boundary. The useful consumer is closer to:

```python
def grouped_moe_blocks(x_group, selected_experts_group, combine_weights_group, grouped_experts):
    # Keep the group axis present while calling the MoE backend for each local
    # layer in the group, then return grouped block outputs.
    ...
```

That may require moving the block loop from Python tuple-of-blocks toward a
grouped block kernel for the expert MLP portion. Attention, router, and
non-expert parameters can stay per-layer initially; the first gate only needs
to prove the routed expert bank can be consumed without restoring the expert
weights to FSDP leaves before apply.

## Stop Criteria

Continue the FSDP-master path only if a new boundary reduces the R2/D2/E8 L26
explicit apply cost far below 0.61 s or removes the H100 compiled boundary
collectives entirely. The current all-gather, group-size, all-to-all,
custom-partition, and direct-apply variants do not.

Otherwise, focus on grouped-bank consumers. The first success criterion is a
compiled synthetic gate with the grouped expert bank preserved and no compiled
boundary collectives. The second success criterion is a short single-node train
profile that materially closes the gap to the SGD reference.
