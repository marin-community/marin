# Debugging log for Grug no-scan layer storage

Determine whether the full-MoE no-scan probe incorrectly retained array-stacked layer parameters, and validate a true per-module representation.

## Initial status

The no-scan Toolbox run configured `scan_layers=false` and `remat_mode=recompute_all`, but failed before step 0 on an 863.96 GiB temporary allocation. `Transformer.init` always created `ArrayStacked[Block]`; the no-scan call path then iterated over `ArrayStacked.unstacked()` views.

## Hypothesis 1

Unstacking views from one array-stacked pytree is not equivalent to presenting XLA with independently initialized layer-module leaves. The no-scan model must initialize a tuple of `Block` modules, while the scan model retains `ArrayStacked[Block]`.

## Changes to make

- Add a regression test for the parameter-storage contract selected by `scan_layers`.
- Initialize independent `Block` modules when `scan_layers=false`.
- Preserve per-layer `eqx.filter_checkpoint` calls.
- Compare state-dict keys and small-model outputs between the two representations.

## Results

The regression test failed before the change because both configurations lacked an independent `blocks` field. After selecting the representation during initialization, the no-scan model contains one `Block` pytree per layer and no `ArrayStacked` container; the scan model retains only `ArrayStacked`. A real CPU execution initialized both representations from the same key and produced identical logits. The focused storage, output-parity, and variant one-step lowering tests pass.

The corrected exact-shape Toolbox probe used 16 four-GPU GB200 hosts, `scan_layers=false`, and `remat_mode=recompute_all`. It spent about 26 minutes in the first `jit_train_step`, then failed before step 0 when CUDA requested 972,028,334,224 bytes (905.27 GiB) against a 138.22 GiB allocator limit. It produced no MFU or throughput metrics and no null-module or in-memory-CUBIN signature. Independent modules therefore preserve the same computation, but they do not make this full unrolled graph fit; the earlier 863.96 GiB result from array-stacked views was not a valid test of the intended representation.

[W&B run](https://wandb.ai/marin-community/marin_moe/runs/jax-toolbox-7507-toolbox-noscan-modules-a-20260722-1640)

## Conclusion

The intended no-scan representation is now tested. JAX-Toolbox gets past the previously observed CUBIN failure surface for this probe, but XLA requests an infeasible temporary arena before execution, so the run cannot establish MFU or prove that Toolbox resolves CUBIN failures in general.
