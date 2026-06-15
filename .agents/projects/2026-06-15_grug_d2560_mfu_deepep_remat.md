# Grug d2560 MFU and DeepEP remat PR spec

## Problem
The Grug d2560 CoreWeave experiments needed a reproducible launch path for
direct global-batch/topology probes and a DeepEP MoE path that works with
rematerialization enabled.

Two correctness issues blocked those runs:

- `experiments/grug/moe/launch.py` built the default trainer mesh before the
  Grug compact mesh was installed, so reduced direct batches such as B64 on
  16 nodes could fail validation when default batch shards exceeded the global
  batch.
- The DeepEP FFI dispatch/combine calls were inside full-block remat regions.
  JAX rejects checkpoint partial evaluation with `FfiEffect`, so
  `MAY_REMAT=save_moe` failed before runtime. After splitting the checkpoint
  boundary, the H100 runtime still compiled the upstream DeepEP `intranode.cu`
  launch pattern even though the cache key used patched source bytes.

The optimizer diagnostics also showed first-step OOMs that were insensitive to
smaller true global batch, so the branch adds an explicit routed-expert optimizer
knob to isolate MuonH expert-stack memory behavior from the rest of the recipe.

## Approach
The launcher now passes a trainer `MeshConfig` whose batch mapping matches the
Grug compact mesh (`replica_dcn`, `data`, `expert`). The shell launcher exposes
the topology, remat, DeepEP, and routed-expert optimizer switches used by the
CoreWeave run log.

DeepEP remat is split at the MoE boundary: side-effecting dispatch/combine FFI
calls stay outside remat, while pure local expert compute can still be
checkpointed. The experiment model avoids full-block remat for DeepEP and
checkpoints attention separately.

The DeepEP transport build now writes generated `intranode.cu` whenever the
patched source bytes differ from upstream bytes. This keeps the compiled CUDA
source aligned with the hashed build cache contents on H100.

The optimizer path adds `expert_3d_optimizer`, defaulting to the previous MuonH
behavior, with an `adamh` diagnostic option for routed expert 3D weights. The
scale-invariant hyperball update is algebraically rewritten to avoid
materializing the full post-step parameter value before subtracting the original
parameter.

Run evidence is captured on the dedicated research logbook branch
`codex/research-grug-moe-d2560-mfu-logbook`, keeping this branch focused on
durable code and spec changes.

## Key code
The DeepEP transport source preparation now compiles the patched launch source
whenever the patch changes upstream bytes:

```python
def _prepare_intranode_source(build_dir: Path, deepep_root: Path) -> Path:
    source = _intranode_source(deepep_root)
    source_bytes = source.read_bytes()
    patched_bytes = _intranode_source_bytes(deepep_root)
    if patched_bytes == source_bytes:
        return _intranode_source(deepep_root)
    patched_source = build_dir / "generated" / "intranode.cu"
    patched_source.parent.mkdir(parents=True, exist_ok=True)
    patched_source.write_bytes(patched_bytes)
    return patched_source
```

The launcher validation mesh is made explicit so direct global-batch probes use
the same batch axes as the Grug compact mesh:

```python
mesh=MeshConfig(
    axes={
        "data": -1,
        "expert": config.grug_trainer.expert_axis_size,
        "model": config.grug_trainer.model_axis_size,
    },
    dcn_axes={"replica_dcn": -1},
    compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
)
```

The routed-expert optimizer diagnostic keeps the default recipe unchanged while
allowing targeted AdamH routing for expert 3D weights:

```python
if ".mlp.expert_mlp.w_" in path_lower and hasattr(param, "ndim") and param.ndim == 3:
    return expert_3d_optimizer
if hasattr(param, "ndim") and param.ndim in (2, 3):
    return "muonh"
return "adam"
```
