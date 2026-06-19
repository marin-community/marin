# Grug MoE MuonH Grouped-Bank Path

Status: 2026-06-19.

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

The FSDP-master grouped-boundary path has not met the bar:

| boundary | shape | result |
| --- | --- | --- |
| XLA-inferred grouped-to-FSDP restore/apply | R2/D2/E8, L26 | replicate-then-partition warnings or OOMs |
| explicit all-gather restore-only | R2/D2/E8, L26 | bf16 fits at about 0.305 s; fp32 full tree OOMs |
| explicit all-gather fused restore+apply | R2/D2/E8, L26 | fits, but about 0.611 s and 112 compiled all-gathers |
| grouped chunk 8/16 fused restore+apply | R2/D2/E8, L26 | all-gather count drops, runtime stays about 0.614 s |
| explicit replica all-gather + data all-to-all apply | R2/D2/E8, L26 | about 0.619 s, compiled 112 all-gathers plus 98 all-to-alls |

Conclusion: preserving grouped params/updates/results is fast; converting
grouped updates back to ordinary per-layer FSDP leaves in the hot path is not.

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

Do not launch another full training profile from this path until this gate
passes.

Build a synthetic model-consumer gate that keeps expert weights grouped through
the consumer boundary:

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
explicit apply cost far below 0.61 s. The current all-gather, group-size, and
all-to-all variants do not.

Otherwise, focus on grouped-bank consumers. The first success criterion is a
compiled synthetic gate with the grouped expert bank preserved and no compiled
boundary collectives. The second success criterion is a short single-node train
profile that materially closes the gap to the SGD reference.
