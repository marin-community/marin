# XLA Collective Primer

> Target: someone who knows JAX + basic collective ops (all-reduce, all-gather,
> reduce-scatter) and wants to understand what XLA does with them *under the
> hood*, enough to reason about the Bug 1 "physical topology" question.

## 0. The short version

1. You write JAX code. JAX converts it to **HLO** (High-Level Optimizer IR), XLA's
   intermediate representation.
2. XLA takes HLO, optimizes it, and **compiles it for a specific hardware target**
   (TPU v5p, GPU H100, whatever).
3. `jax.lax.psum` / `jax.lax.all_gather` / etc. become **HLO collective ops** with
   attributes describing which devices participate.
4. XLA then **chooses an algorithm** for each collective based on the number and
   physical layout of the participating chips — ring, tree, recursive-halving,
   etc.
5. Different physical layouts → different algorithms → different order of
   floating-point additions → potentially different numerical results.

That last step is the crux of the "device reordering" question.

---

## 1. What XLA actually is

**XLA = Accelerated Linear Algebra.** It's the compiler backend shared by JAX,
TF, and PyTorch/XLA. It sits between "my program, as a graph of array
operations" and "actual TPU/GPU instructions."

The JAX pipeline:

```
Python/JAX source
    ↓   (jax.jit tracing)
jaxpr                   ← JAX's IR, Python-ish, one level above HLO
    ↓   (JAX → HLO lowering)
StableHLO / HLO         ← XLA's IR
    ↓   (XLA optimization passes)
Optimized HLO
    ↓   (XLA backend: TPU / GPU / CPU)
Hardware-specific code  ← actual kernel dispatch + collective protocol
```

You can think of HLO as "a graph of array ops with shapes and dtypes, plus
annotations for parallelism and sharding." Every `jnp.matmul`, every
`jax.lax.psum`, every `jax.lax.with_sharding_constraint` turns into an HLO op.

### What JAX controls vs. what XLA controls

| Concern | JAX decides | XLA decides |
|---|---|---|
| What ops run | yes | no |
| Which devices are in the mesh | yes | no |
| The logical name of each axis (`'data'`, `'model'`) | yes | no |
| Which chips form a replica group for a collective | partially | partially |
| Which collective algorithm (ring vs tree vs …) | **no** | **yes** |
| Whether bf16 or f32 is used inside the kernel | partially | partially |
| How multiple collectives are fused / reordered | no | yes |

The key mental model: **JAX describes *what* the collective does; XLA decides
*how* it runs on real hardware.**

---

## 2. HLO and collectives

Every collective in JAX lowers to an HLO op. The most common ones:

| JAX op | HLO op | What it does |
|---|---|---|
| `jax.lax.psum(x, axis)` | `all-reduce` | sum across chips along `axis`, everyone gets the result |
| `jax.lax.pmean` / `pmax` | `all-reduce` (with different reduction) | same, different reduction op |
| `jax.lax.all_gather(x, axis)` | `all-gather` | concatenate all chips' shards, everyone gets the concat |
| `jax.lax.with_sharding_constraint` (often) | `all-to-all` / `collective-permute` | rearrange data across chips |
| reduce-scatter (implicit in FSDP) | `reduce-scatter` | sum + shard in one op |

The important point: **each of these HLO ops carries a `replica_groups`
attribute**, which is the list of which physical device IDs participate in the
collective.

For a `{data:4, model:1}` mesh on v5p-8, a data-axis all-reduce would be
compiled with `replica_groups={{0,1,2,3}}` (one group, all four devices). If
the model axis were size 2, you'd see `replica_groups={{0,1},{2,3}}` (two
independent groups) for a model-axis collective.

**You can dump HLO and inspect these yourself:**

```bash
XLA_FLAGS="--xla_dump_to=/tmp/hlo_dump --xla_dump_hlo_as_text" python my_script.py
grep -A2 "all-reduce" /tmp/hlo_dump/*optimized*.txt
```

You'll see entries like:

```
all-reduce.123 = bf16[512,4096] all-reduce(input.122), replica_groups={{0,1,2,3}}, to_apply=add_bf16
```

(Aside: `AB` in our logbook did exactly this — dumped HLO under `p=f32,c=f32`
and read off `to_apply=add_f32` to prove the collective was f32, not bf16.)

---

## 3. How XLA compiles a collective

Once XLA has the HLO op with its `replica_groups`, the backend (TPU or GPU)
decides **what protocol to actually execute**.

Possible algorithms for an all-reduce over N devices:

- **Ring** — pipeline elements around a ring topology in N-1 steps. Bandwidth
  optimal when the ring matches the physical link topology.
- **Bi-directional ring** — two rings going opposite directions, 2× throughput.
- **Recursive halving + doubling** — log N steps, good for small payloads or
  unusual counts.
- **Tree / hierarchical** — combine in a tree. Good when participants are on a
  hierarchy (multi-host, mixed networks).
- **Direct (pair exchange)** — for N=2, just swap and add.

The compiler picks one based on:

1. **Payload size.** Tiny reductions prefer direct/tree; large reductions prefer
   ring.
2. **Number of participants.** Powers of 2 favor recursive halving; rings work
   for any size.
3. **Physical topology.** If the participants form a natural ring/torus
   segment, ring is cheap. If they're scattered, ring costs more hops.

The TPU backend has a dedicated library (`tpu::collective::...`) with
hand-tuned implementations. Depending on the participant set, **different
implementations kick in**. That's where topology awareness shows up.

### Why this is non-trivial on TPU

TPU pods are **3D tori**. A v5p-8 isn't "8 chips in a line" — it's something
like a `2×2×2` cube of chips connected by a grid of high-bandwidth ICI
(inter-chip interconnect) links. v5p-16 is `2×2×4` or similar. The physical
topology matters for every collective, because the hardware can only do
nearest-neighbor ICI hops in one wire-time.

When XLA compiles `all-reduce` over 4 participants on a v5p-8, it:

1. Looks up which 4 specific chips are in `replica_groups` (e.g., `{0,1,2,3}`).
2. Queries the physical topology: where are those chips in the torus?
3. Checks if they form a nice substructure (a 2×2 face, a 4-element ring, …).
4. Picks an algorithm optimized for that substructure.

If the 4 chips **are** a 2×2 face, a dedicated 2×2 reduction kernel runs and
does the all-reduce in a very specific order (e.g., reduce along one axis
first, then the other). The order of floating-point additions is fully
determined by that algorithm.

If the 4 chips **aren't** a clean face (say chips `{0,2,4,6}` on v5p-8), the
compiler falls back to a more general ring or tree algorithm — different
additions, different order.

---

## 4. The Mesh → device-ID → chip mapping

This is the part that confused you, so let me be pedantic.

### Step A: JAX sees devices

```python
print(jax.devices())
# [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0),
#  TpuDevice(id=1, process_index=0, coords=(1,0,0), core_on_chip=0),
#  TpuDevice(id=2, process_index=0, coords=(0,1,0), core_on_chip=0),
#  ...]
```

Each device has an `id` (what JAX calls it) and `coords` (where it sits in the
physical torus). The mapping from `id` to `coords` is set by the TPU runtime
and is stable within a job.

### Step B: You build a mesh from an array of devices

```python
from jax.sharding import Mesh
from jax.experimental import mesh_utils

# Canonical way — mesh_utils picks a physically-sensible layout
devices = mesh_utils.create_device_mesh((4, 1))   # shape (data=4, model=1)
# devices is a 2D ndarray of TpuDevice objects
mesh = Mesh(devices, ('data', 'model'))
```

`mesh_utils.create_device_mesh` is smart: given a requested shape, it tries to
find a physical chip assignment that puts each axis on a contiguous physical
direction (so collective along `data` runs along a physical line of chips, not
jumping across the torus).

**But you can override it.** For example:

```python
# Just use device IDs 0..3 in declaration order
devices = np.array(jax.devices()[:4]).reshape(4, 1)
mesh = Mesh(devices, ('data', 'model'))

# Reverse the order
devices = np.array(jax.devices()[:4][::-1]).reshape(4, 1)
mesh = Mesh(devices, ('data', 'model'))

# Pick non-contiguous devices
devices = np.array([jax.devices()[i] for i in [0, 2, 4, 6]]).reshape(4, 1)
mesh = Mesh(devices, ('data', 'model'))
```

All three produce a **mesh with the same logical shape `(data=4, model=1)`**
but **different physical participants** in each data-axis collective.

### Step C: XLA sees those participants in `replica_groups`

When your jitted function runs an all-reduce along the `data` axis, XLA sees
`replica_groups={{0,1,2,3}}` in the first case, `{{3,2,1,0}}` in the second,
`{{0,2,4,6}}` in the third. Based on those IDs + the known physical topology,
it picks an algorithm.

In XLA-speak, different `replica_groups` can legitimately compile to
**different collective kernels**, and those kernels can have **different
numerical outputs** even at fixed input values. The difference is almost
always below your normal precision budget, but it IS real.

---

## 5. Floating-point non-associativity, briefly

This is why "same values, different order" can produce different bits.

Float addition is not associative: `(a + b) + c` can differ from
`a + (b + c)` at the last 1–2 bits. For bf16 that's ~2^-7 relative; for f32
it's ~2^-23 relative. Usually invisible.

But:
- When millions of numbers get summed across 100+ HLO ops, bit differences
  compound.
- When Adam's first step is essentially sign-based (`update ≈ lr × sign(grad)`),
  a small perturbation to grad that flips one or two element signs can nudge
  the whole trajectory.
- When you're training near a symmetric fixed point (LoRA `zero_init_b`,
  `adapter_out = 0`), **the direction at step 1 matters a lot**. Any perturbation
  that gives the gradient a slightly different direction at step 1 can pick a
  different descent path.

That's why "same logical math, different collective algorithm" can, in
principle, produce a different training trajectory — especially near-symmetric.

**Important caveat**: Exp AB already verified the actual collective was **f32**
and emitted identical `replica_groups`, and the bug persisted. So the
"different algorithm gives different numerics" story alone is looking thin. The
interesting question now is whether *which specific chips participate* (not
just the order) makes a difference. That's where the mesh-permutation probe
comes in.

---

## 6. Why permute physical devices in a Bug 1 probe

Recall the question: is Bug 1 bound to **width-4 logically** (abstract), or to
**this specific v5p-8 physical 4-chip subset** (hardware-specific)?

There are two *kinds* of permutation, and they answer different questions:

### 6a. Permute rank order within the SAME 4-chip subset

Use the same 4 physical chips (say `{0,1,2,3}`) but assign them to different
*logical* data-axis ranks: `[0,1,2,3]` vs. `[3,2,1,0]` vs. `[1,3,0,2]`.

- **What this tests**: Does the XLA collective algorithm pick differently
  based on the **order of participant IDs** in `replica_groups`? In theory it
  shouldn't (the algorithm depends on *which* chips, not the order listed).
  In practice, there can be quirks.
- **Likely outcome**: bit-identical loss for all orders. Because XLA usually
  sorts or canonicalizes `replica_groups`.
- **If DIFFERENT**: ordering is load-bearing, which would be a serious XLA
  reproducibility bug. Noteworthy but probably not the Bug 1 root cause.

### 6b. Use a DIFFERENT 4-chip physical subset out of the 8 available

On v5p-8, pick 4 different chips for each run:
- `{0,1,2,3}` — likely a 2×2 face of the torus.
- `{4,5,6,7}` — the other 2×2 face.
- `{0,2,4,6}` — even chips, scattered.
- `{0,1,4,5}` — a non-contiguous 2×2.

- **What this tests**: Does *which specific 4 chips* participate matter?
- **Likely outcome**: `{0,1,2,3}` and `{4,5,6,7}` should behave similarly
  (symmetric position in torus); the non-contiguous picks might behave
  differently.
- **If DIFFERENT**: the bug is pinned to specific chip subsets / specific
  physical substructures. Big find — it says "width-4 on physical 4-chip
  topology-A is bad, but on topology-B is fine." Narrows root cause toward
  topology.
- **If IDENTICAL and bad everywhere**: bug is "width-4 FSDP + LoRA on v5p-8,
  regardless of which 4 chips." Different flavor of topology claim.

**(6b) is the more informative version.** That's what I'd actually run.

### How you'd construct the mesh in Levanter/Haliax

Haliax meshes are built from a device list passed to
`jax.sharding.Mesh(...)`. To permute, you'd construct the device array
explicitly before building the mesh:

```python
import jax
import numpy as np

# canonical
device_ids = [0, 1, 2, 3]

# reverse
# device_ids = [3, 2, 1, 0]

# disjoint 4-chip subset
# device_ids = [4, 5, 6, 7]

# scattered
# device_ids = [0, 2, 4, 6]

devices = np.array([jax.devices()[i] for i in device_ids]).reshape(4, 1)
mesh = jax.sharding.Mesh(devices, ('data', 'model'))
```

You'd plumb this through the ResourceConfig / Trainer setup. Concretely in
Marin/Levanter it usually goes through a `mesh_override` or similar parameter
on the trainer config, or by patching the device list used by
`mesh_utils.create_device_mesh`.

---

## 7. Practical: how to inspect what XLA actually did

You generally cannot force XLA to use a specific algorithm from Python. But
you *can* read back what it chose.

### HLO dump

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*" \
    python train.py
```

Then:

```bash
ls /tmp/xla/*.txt
grep -nE "all-reduce|reduce-scatter|all-gather" /tmp/xla/*optimized*.txt | head
```

Look at:

- `replica_groups={...}` — which devices participate.
- `to_apply=` — which reduction op (and at what dtype).
- `channel_id=` — which collective channel (each distinct collective gets an id).
- `use_global_device_ids=true/false` — whether IDs in the group are global
  (multi-host) or local.

### TPU-specific debugging

Under `XLA_FLAGS="--xla_dump_hlo_as_proto"` you also get a `.pb` file with the
after-backend-lowering form. That one shows the actual TPU algorithm selection
(look for `AllReduceDoneSchedulingInfo`, `HloScheduleInfo`, etc.) but it's
less human-friendly.

Most of the time, the **text HLO pre- and post-optimization** plus the
`replica_groups` and dtypes are enough to tell you what's happening at the
logical level.

---

## 8. How this ties back to Bug 1

Recap of what we know:

1. Bug 1 appears on v5p-8 with mesh `{replica:1, data:4, model:1}` under a
   LoRA DPO recipe. Loss stuck at log(2) through 10 steps at lr=1e-6.
2. Changing mesh to `{data:1, model:4}` (pure TP) or `{data:2, model:2}`
   (mix) rescues completely.
3. Changing compute dtype (c=f32), storage dtype (p=bf16,c=bf16), reference
   path, or CE kernel tile sizes does NOT rescue.
4. **Exp AB** verified the actual all-reduce is f32 under the AC recipe, so
   the narrow "bf16 collective rounding" story is falsified.
5. **Exp Z1** showed post-reduce LoRA gradient values differ element-wise
   between bad v5p-8 width-4 and good v6e-8 width-8 at matched init. But
   that's across both width and hardware, so it doesn't isolate either.

Given #1–#5, the plausible remaining mechanisms are:

- **Specific collective algorithm at width-4 on v5p-8's physical topology**
  that produces a systematic bias in the LoRA gradient (not just rounding,
  but algorithmic reduction order). Possibly compounded by LoRA's rank-64
  bottleneck making small grad values especially sensitive.
- **FSDP update-path interaction** at 4 participants — e.g., reduce-scatter
  + all-gather patterns that emit different buffers at width-4 vs. others.
- **Something about physical chip identity on v5p-8** — unlikely-but-not-impossible
  hardware variance or routing bug.

The mesh-permutation probe (6b) cuts across these: if *which* 4 chips matters,
we're in a topology-specific lane. If the bug survives every 4-chip
combination, we're in an algorithm-at-width-4 lane.

The larger-slice probe (width-4 on v5p-16) cuts differently: if width-4 on
v5p-16 is *good*, we rule out "width-4 anywhere" and pin it to v5p-8. If
width-4 on v5p-16 is *bad*, we've got width-4 as a general XLA/algorithm
issue.

They're complementary. Running both is cheap.

---

## 9. Glossary

- **HLO** — XLA's intermediate representation. High-Level Optimizer.
- **replica_groups** — per-collective attribute listing which devices
  participate together.
- **TPU ICI** — inter-chip interconnect, the high-bandwidth links between
  adjacent TPU chips in a pod.
- **TPU DCN** — data-center network, the slower links between multi-host
  (cross-slice) communication.
- **Torus** — the physical wiring topology of a TPU pod (wrapped grid).
- **Shard** — a partition of a tensor across devices along a named axis.
- **FSDP** — Fully-Sharded Data Parallelism; weights, grads, and opt state
  are all sharded along a "data" axis; requires reduce-scatter during
  backward and all-gather during forward.
- **TP (tensor parallelism)** — a different sharding where layer weights are
  split along model-internal dims, not across data batch.
- **Mixed precision (`p=`, `c=`)** — `p` = param storage dtype,
  `c` = activation/compute dtype, per haliax/jmp policies.

---

## 10. Further reading

- JAX sharding tutorials: <https://jax.readthedocs.io/en/latest/sharded-computation.html>
- XLA HLO semantics: <https://openxla.org/xla/operation_semantics>
- The `jax.experimental.mesh_utils.create_device_mesh` source is short and
  readable and shows exactly how JAX assigns devices to a requested mesh shape.
