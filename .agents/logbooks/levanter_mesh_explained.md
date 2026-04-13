# How Levanter Sets Up the TPU Mesh

## Overview

Levanter doesn't hardcode any knowledge of TPU types. It asks JAX what devices exist, JAX asks the hardware, and Levanter builds the mesh from two numbers: **total device count** and **number of hosts (slices)**.

## Step-by-Step: How the Mesh Gets Built

### Step 1: JAX discovers devices

When a TPU program starts, JAX calls into the TPU runtime and gets back a list of device objects. Each device has:
- `id`: unique device ID
- `platform`: "tpu"
- `device_kind`: "TPU v5p", "TPU v6e", etc.
- `slice_index`: which host this device belongs to (only present on multi-host setups)

### Step 2: Levanter counts hosts

```python
# lib/levanter/src/levanter/trainer.py, line 960
num_slices = max(getattr(device, "slice_index", 0) for device in jax.devices()) + 1
```

If `slice_index` doesn't exist on the device (single host), `num_slices = 1`.

### Step 3: Levanter computes chips per host

```python
# lib/levanter/src/levanter/trainer.py, line 966
per_slice = jax.device_count() // num_slices
```

### Step 4: The mesh axes are computed

The defaults (from `lib/levanter/src/levanter/utils/mesh.py`):
```python
DEFAULT_ICI_AXIS_SPEC = {"data": -1, "replica": 1, "model": 1}   # within-host
DEFAULT_DCN_AXIS_SPEC = {"replica_dcn": -1}                       # across-host
```

The `-1` means "absorb whatever's left." So:
- `data = per_slice / (replica × model) = per_slice` (all chips within a host)
- `replica_dcn = num_slices` (one slot per host)

### Concrete examples

| TPU config | `jax.device_count()` | `num_slices` | `per_slice` | Mesh: `data` | Mesh: `replica_dcn` |
|------------|---------------------|-------------|-------------|--------------|---------------------|
| v5p-8 (1 host, 4 chips) | 4 | 1 | 4 | 4 | 1 |
| v5p-32 (4 hosts, 4 chips each) | 16 | 4 | 4 | 4 | 4 |
| v6e-4 (1 host, 4 chips) | 4 | 1 | 4 | 4 | 1 |
| v6e-128 (32 hosts, 4 chips each) | 128 | 32 | 4 | 4 | 32 |

**v5p-8 and v6e-4 look identical to Levanter** — both are 4 chips, 1 host. The only difference is HBM size (95 GB vs 31 GB), which Levanter doesn't check.

**v5p-32 and v6e-128** both have `per_slice=4` — same within-host FSDP depth. The difference is 4 vs 32 hosts, and 95 vs 31 GB per chip.

## How FSDP Sharding Depth Is Determined

The critical config (line 40 in mesh.py):
```python
param_mapping: {"embed": "data"}
```

This means: **shard the model's embed dimension across the `data` axis only.**

Since `data` is an ICI axis (within-host), FSDP only shards within a host. The `replica_dcn` axis is used for data parallelism (batch distribution + gradient averaging) but NOT for parameter sharding.

### What this looks like on v6e-128

```
Host 0:  [chip0: 1/4 model] [chip1: 1/4 model] [chip2: 1/4 model] [chip3: 1/4 model]
Host 1:  [chip0: 1/4 model] [chip1: 1/4 model] [chip2: 1/4 model] [chip3: 1/4 model]
...
Host 31: [chip0: 1/4 model] [chip1: 1/4 model] [chip2: 1/4 model] [chip3: 1/4 model]
```

Each chip holds 1/4 of the model (sharded within-host). All 32 hosts hold identical copies. Adding more hosts doesn't reduce per-chip model memory.

Per-chip model storage: `8B params × 4 bytes (f32) / 4 = 8 GB`

On v6e (31.25 GB HBM), 8 GB is 25% of the chip — before activations, optimizer state, or XLA temp buffers.

## How to Shard Across All Chips

Change the param_mapping in the YAML config:

```yaml
trainer:
  mesh:
    param_mapping:
      embed: [replica_dcn, data]
```

This tells Levanter: "shard the embed dimension across both `replica_dcn` AND `data`." On v6e-128 that's 32 × 4 = 128-way sharding.

Per-chip model storage: `8B params × 4 bytes / 128 = 250 MB`

### The tradeoff

**Default (`embed: data`):**
- All-gathers stay within-host (fast ICI, microseconds)
- Each host has a full model copy (high memory per chip)
- Good for v5p (95 GB HBM — plenty of room)

**Cross-host (`embed: [replica_dcn, data]`):**
- All-gathers go across hosts (slow DCN, milliseconds)
- Each chip holds 1/128 of the model (low memory per chip)
- Necessary for v6e (31 GB HBM — too small for 1/4 of 8B model in f32)

### Why this is especially OK for LoRA

| Communication type | Full fine-tuning | LoRA |
|-------------------|-----------------|------|
| Forward all-gather (base weights, per layer) | ~864 MB × 64 layers | Same (unavoidable) |
| Gradient all-reduce (across hosts) | **32 GB** (all params) | **620 MB** (LoRA only, 50× less) |
| Optimizer step | All params | LoRA only (tiny) |

The forward all-gather is the same cost regardless. But gradient all-reduce — which is the OTHER expensive cross-host communication — is 50× cheaper with LoRA. So the performance penalty of cross-host FSDP is much smaller for LoRA than for full fine-tuning.

## ICI vs DCN: The Two Networks

TPU pods have two interconnects:

- **ICI (Inter-Chip Interconnect)**: Connects chips within a host. Very fast, very high bandwidth. This is the `data`, `replica`, and `model` axes.
- **DCN (Data Center Network)**: Connects hosts to each other. Much slower, lower bandwidth. This is the `replica_dcn` axis.

The default design philosophy: keep parameter communication on ICI, use DCN only for gradient averaging. This is optimal when chips have enough HBM. When they don't (v6e), you trade DCN latency for memory savings.

## MeshConfig: The Full Configuration Surface

From `lib/levanter/src/levanter/utils/mesh.py`:

```python
@dataclass(frozen=True)
class MeshConfig:
    axes: {"data": -1, "replica": 1, "model": 1}     # ICI axis sizes (-1 = absorb remaining)
    dcn_axes: {"replica_dcn": -1}                      # DCN axis sizes (-1 = absorb remaining)
    batch_axis_name: "batch"                           # logical name for the batch axis
    shared_mapping: {}                                  # logical → physical (shared by compute + params)
    compute_mapping: {}                                 # logical → physical (compute only)
    param_mapping: {"embed": "data"}                   # logical → physical (params + optimizer)
```

Resolved mappings:
- **Parameters**: `{"mlp": "model", "heads": "model", "embed": "data"}` — FSDP on `data`, TP on `model`
- **Compute**: same + `{"batch": ("replica_dcn", "replica", "data")}` — batch spans ALL axes

Everything is configurable via the trainer YAML:
```yaml
trainer:
  mesh:
    axes:
      data: -1
      model: 1
    dcn_axes:
      replica_dcn: -1
    param_mapping:
      embed: [replica_dcn, data]   # cross-host FSDP
    compute_mapping:
      batch: [replica_dcn, replica, data]
```

## TPU Generation Differences

Levanter's mesh code is **generation-agnostic**. It doesn't detect v5p vs v6e. The differences that matter:

| Property | v5p | v6e |
|----------|-----|-----|
| HBM per chip | 95.74 GB | 31.25 GB |
| Chips per host | 4 | 4 |
| ICI bandwidth | High | High |
| DCN bandwidth | Medium | Medium |
| Levanter mesh | Identical | Identical |

The only code that checks TPU generation is the Pallas kernel tuning (`tuned_block_sizes.py`), which detects `"v5p" in device_kind` for fused cross-entropy block sizes.

## Git History

The mesh code was refactored in commit `896a390de` (Dec 18, 2025) by David Hall and William Held: "Mesh refactor in support of context parallelism." This introduced the clean ICI/DCN separation and the configurable `MeshConfig`. The default of `embed: data` (FSDP within-host only) was an intentional performance choice, not a limitation — it's configurable.
