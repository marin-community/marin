# Expert-Chunked EP Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trace-gated expert-chunked ragged-all-to-all benchmark that compares serialized and depth-2 schedules with exact routing semantics and QuACK grouped GEMMs.

**Architecture:** Start from `origin/research/jaxpp-grug-moe`, which has the current QuACK forward and weight-gradient bindings. Add routing, dispatch, expert compute, combine, and window scheduling to a disposable experiment module; do not refactor or expose the production ragged backend during the trace gate. Use the production unchunked function as the numerical oracle.

**Tech Stack:** Python 3.12, JAX explicit meshes and `shard_map`, `jax.lax.ragged_all_to_all`, QuACK/CuTe TVM-FFI kernels, pytest, XPlane/xprof.

## Global Constraints

- Existing SonicMoE, QuACK, and EP branches are evidence and source material, not compatibility constraints.
- Compute global routing and receiver-capacity clipping once; expert chunks must preserve the unchunked accepted routes and drop count exactly.
- Keep `num_expert_chunks` separate from `pipeline_depth`.
- Support pipeline depths 1 and 2 only in this investigation.
- Do not expose the experimental path through Grug model configuration.
- Do not hard-code a communication SM or CTA carveout.
- Use QuACK only through a compute boundary that ignores trailing capacity padding in forward and backward.
- Peak saved state must be independent of `num_expert_chunks` at fixed `pipeline_depth`.
- A performance result counts as pipelining only when HLO and an XPlane trace show communication for chunk `n + 1` overlapping expert GEMMs for chunk `n`.
- Use `./infra/pre-commit.py`; do not invoke pre-commit directly.

---

### Task 1: Partition globally accepted routes by local expert slice

**Files:**
- Create: `experiments/grug/moe/ragged_ep_pipeline.py`
- Create: `lib/levanter/tests/grug/test_ragged_ep_pipeline.py`

**Interfaces:**
- Consumes: `_clip_receiver_group_sizes`, `_expert_prefix_keep_mask`, and clipped counts shaped `[ep_size, num_experts]`.
- Produces: `_ExpertChunkSpec`, `_expert_chunk_specs(local_experts, num_expert_chunks)`, and `_counts_for_expert_chunk(clipped_group_sizes, spec)`.

- [ ] **Step 1: Write failing behavior tests for expert chunk validation and partitioning**

Add imports for the new helpers and these tests:

```python
def test_expert_chunk_specs_partition_local_experts() -> None:
    assert _expert_chunk_specs(local_experts=8, num_expert_chunks=2) == (
        _ExpertChunkSpec(index=0, start=0, size=4),
        _ExpertChunkSpec(index=1, start=4, size=4),
    )


@pytest.mark.parametrize(
    ("local_experts", "num_chunks", "match"),
    [(7, 2, "must divide"), (8, 0, "positive"), (8, 9, "must not exceed")],
)
def test_expert_chunk_specs_reject_invalid_partitions(local_experts, num_chunks, match) -> None:
    with pytest.raises(ValueError, match=match):
        _expert_chunk_specs(local_experts=local_experts, num_expert_chunks=num_chunks)


def test_expert_chunk_counts_partition_globally_clipped_counts() -> None:
    clipped = jnp.array(
        [
            [3, 0, 1, 2, 4, 1, 0, 2, 1, 5, 2, 0, 3, 1, 4, 2],
            [0, 2, 4, 1, 1, 0, 3, 2, 5, 1, 0, 2, 1, 3, 2, 4],
        ],
        dtype=jnp.int32,
    )
    chunks = [
        _counts_for_expert_chunk(clipped, spec, local_experts=8)
        for spec in _expert_chunk_specs(local_experts=8, num_expert_chunks=2)
    ]
    np.testing.assert_array_equal(np.asarray(chunks[0] + chunks[1]), np.asarray(clipped))
    assert np.count_nonzero(np.asarray(chunks[0])[:, [4, 5, 6, 7, 12, 13, 14, 15]]) == 0
    assert np.count_nonzero(np.asarray(chunks[1])[:, [0, 1, 2, 3, 8, 9, 10, 11]]) == 0
```

- [ ] **Step 2: Run the tests and confirm they fail because the helpers do not exist**

Run:

```bash
uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_ragged_ep_pipeline.py \
  -k 'expert_chunk_specs or expert_chunk_counts' -q
```

Expected: collection fails with an import error for `_ExpertChunkSpec` or the first test fails because the helper is absent.

- [ ] **Step 3: Implement the immutable chunk description and masked-count helper**

Add these top-level definitions:

```python
@dataclass(frozen=True)
class _ExpertChunkSpec:
    index: int
    start: int
    size: int


def _expert_chunk_specs(*, local_experts: int, num_expert_chunks: int) -> tuple[_ExpertChunkSpec, ...]:
    if num_expert_chunks <= 0:
        raise ValueError(f"num_expert_chunks must be positive, got {num_expert_chunks}")
    if num_expert_chunks > local_experts:
        raise ValueError(
            f"num_expert_chunks={num_expert_chunks} must not exceed local_experts={local_experts}"
        )
    if local_experts % num_expert_chunks:
        raise ValueError(
            f"num_expert_chunks={num_expert_chunks} must divide local_experts={local_experts}"
        )
    size = local_experts // num_expert_chunks
    return tuple(_ExpertChunkSpec(index=index, start=index * size, size=size) for index in range(num_expert_chunks))


def _counts_for_expert_chunk(
    clipped_group_sizes: Int[Array, "S E"],
    spec: _ExpertChunkSpec,
    *,
    local_experts: int,
) -> Int[Array, "S E"]:
    local_expert = jnp.arange(clipped_group_sizes.shape[1], dtype=jnp.int32) % local_experts
    selected = jnp.logical_and(local_expert >= spec.start, local_expert < spec.start + spec.size)
    return jnp.where(selected[None, :], clipped_group_sizes, 0)
```

- [ ] **Step 4: Run the focused tests and the existing EP helper tests**

Run the command from Step 2, then:

```bash
uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_ragged_ep_pipeline.py \
  -k 'clip_receiver or expert_prefix or shard_a2a or expert_chunk' -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the routing partition**

```bash
git add experiments/grug/moe/ragged_ep_pipeline.py \
  lib/levanter/tests/grug/test_ragged_ep_pipeline.py
git commit -m "[grug] Partition accepted EP routes by expert chunk"
```

### Task 2: Implement experimental ragged dispatch, compute, and combine stages

**Files:**
- Modify: `experiments/grug/moe/ragged_ep_pipeline.py`
- Modify: `lib/levanter/tests/grug/test_ragged_ep_pipeline.py`

**Interfaces:**
- Consumes: `_ExpertChunkSpec` and the existing full-batch routing helpers.
- Produces: `_RaggedA2APlan`, `_DispatchedExpertChunk`, `_prepare_ragged_a2a_plan`, `_dispatch_expert_chunk`, `_compute_expert_chunk_ragged`, and `_combine_expert_chunk`.
- Uses `_moe_mlp_ep_ragged_a2a_local(...) -> (out_local, dropped_total)` as the unchanged oracle.

- [ ] **Step 1: Add a GPU behavior test for the experimental one-chunk path**

In `test_ragged_ep_pipeline.py`, build the same small explicit EP mesh used by
`test_moe_mlp_ragged_matches_ring_with_ep_axis_when_available`. Compare the
experimental one-chunk path with `_moe_mlp_ep_ragged_a2a_local` using a fixed
cotangent. Check output, dropped count, and gradients for `x`, combine weights,
`w13`, and `w2` at `rtol=1e-5, atol=1e-5`. Skip when fewer than two GPUs are
available or the ragged collective has no runtime implementation.

- [ ] **Step 2: Run the focused test before refactoring**

Run:

```bash
uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_ragged_ep_pipeline.py \
  -k 'experimental_one_chunk' -q
```

Expected on CPU: one skip. Expected on a multi-GPU worker: pass. Record the baseline result before changing the backend.

- [ ] **Step 3: Introduce explicit stage records and prepare the global plan once**

Use frozen dataclasses for static structure. Keep JAX arrays as fields and do not register a new public pytree type because the records remain inside the experimental local function:

```python
@dataclass(frozen=True)
class _RaggedA2APlan:
    sorted_x: jax.Array
    sorted_indices: jax.Array
    group_sizes: jax.Array
    clipped_group_sizes: jax.Array
    shard_id: jax.Array
    assignments_per_shard: int
    tokens_per_shard: int
    topk: int
    local_experts: int
    receiver_capacity: int


@dataclass(frozen=True)
class _DispatchedExpertChunk:
    inputs: jax.Array
    local_sort_indices: jax.Array
    local_group_sizes: jax.Array
    valid_rows: jax.Array
    keep_mask: jax.Array
    shard_counts: jax.Array
```

`_prepare_ragged_a2a_plan` performs `_permute_by_global_expert`, the count `all_gather`, and `_clip_receiver_group_sizes` exactly once. It does not compact `sorted_x`, because each chunk has a different keep mask.

- [ ] **Step 4: Implement chunk dispatch with exact global clipping**

For a chunk:

```python
chunk_counts = _counts_for_expert_chunk(
    plan.clipped_group_sizes,
    spec,
    local_experts=plan.local_experts,
)
sender_counts = chunk_counts[plan.shard_id]
keep_mask = _expert_prefix_keep_mask(
    plan.group_sizes.astype(jnp.int32),
    sender_counts,
    total_size=plan.assignments_per_shard,
)
send_rows = _compact_by_keep_mask(plan.sorted_x, keep_mask)
shard_counts = jnp.sum(
    chunk_counts.reshape(ep_size, ep_size, plan.local_experts),
    axis=2,
)
```

Dispatch into `[receiver_capacity, hidden_dim]`. Add `_local_permute_expert_chunk_from_counts` that slices `[all senders, spec.size]` counts for the receiving rank, sorts valid rows by chunk-local expert, returns unpadded group sizes, and reports `valid_rows` separately.

- [ ] **Step 5: Implement the reference compute and reverse combine stages**

The reference ragged-dot compute pads only its group-size vector so the dense oracle has defined trailing zeros:

```python
padded_group_sizes = local_group_sizes.at[-1].add(inputs.shape[0] - valid_rows)
w13_out = ragged_dot(inputs, w13_chunk, padded_group_sizes)
gate, up = jnp.split(w13_out, [w2_chunk.shape[1]], axis=-1)
out = ragged_dot(activation_fn(gate) * up, w2_chunk, padded_group_sizes)
return jnp.where(jnp.arange(inputs.shape[0])[:, None] < valid_rows, out, 0)
```

The combine stage reverses the local permutation, performs the transposed ragged all-to-all, expands with the chunk keep mask, runs `_unpermute_from_global_expert`, and returns a token-shaped contribution. Summing all chunk contributions must equal the unchunked output.

- [ ] **Step 6: Add an experimental one-chunk composition**

Add `_moe_mlp_ep_ragged_a2a_experimental_local` in the experiment module. Use `_ExpertChunkSpec(index=0, start=0, size=local_experts)` and call prepare, dispatch, reference compute, and combine. Compute `dropped_total` once from the full plan. Compare this private experimental composition with the unchanged production function.

- [ ] **Step 7: Run the existing abstract lowering and GPU parity tests**

```bash
uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_ragged_ep_pipeline.py \
  lib/levanter/tests/grug/test_grugformer_moe.py \
  -k 'ragged_all_to_all or ragged_matches_ring_with_ep_axis or shard_a2a' -q
```

Expected: CPU-capable tests pass, GPU-only runtime tests skip locally, and the abstract mesh lowering still succeeds.

- [ ] **Step 8: Commit the experimental stages**

```bash
git add experiments/grug/moe/ragged_ep_pipeline.py \
  lib/levanter/tests/grug/test_ragged_ep_pipeline.py
git commit -m "[grug] Add staged ragged EP experiment"
```

### Task 3: Make QuACK varlen MLP ignore trailing capacity rows

**Files:**
- Modify: `lib/levanter/src/levanter/grug/_moe/sonic_quack.py:529-703`
- Modify: `lib/levanter/tests/grug/test_grugformer_moe.py:500-580`
- Modify: `experiments/grug/moe/ragged_ep_pipeline.py`

**Interfaces:**
- Consumes: `quack_mlp_varlen(x, w13, w2, group_sizes)` where `sum(group_sizes) <= x.shape[0]`.
- Produces: zero output and zero `dx` for rows at or beyond `sum(group_sizes)`; exact weight gradients over valid rows only.
- Produces: `_compute_expert_chunk_quack(...)` with the same result shape as the reference compute stage.

- [ ] **Step 1: Write a GPU regression test for trailing capacity padding**

Add a test beside `test_quack_mlp_varlen_matches_ragged_dot_output_and_vjp_on_gpu`:

```python
def test_quack_mlp_varlen_ignores_trailing_capacity_rows_on_gpu() -> None:
    _skip_without_quack_gpu_runtime()
    capacity = 256
    valid_rows = 96
    group_sizes = jnp.array([0, 17, 31, 48], dtype=jnp.int32)
    key_x, key_w13, key_w2 = jax.random.split(jax.random.key(73), 3)
    x = jax.random.normal(key_x, (capacity, 128), dtype=jnp.bfloat16)
    w13 = 0.02 * jax.random.normal(key_w13, (4, 128, 256), dtype=jnp.bfloat16)
    w2 = 0.02 * jax.random.normal(key_w2, (4, 128, 128), dtype=jnp.bfloat16)

    def loss(x, w13, w2):
        return jnp.sum(quack_mlp_varlen(x, w13, w2, group_sizes).astype(jnp.float32))

    output = jax.jit(quack_mlp_varlen)(x, w13, w2, group_sizes)
    dx, _, _ = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(x, w13, w2)
    np.testing.assert_array_equal(np.asarray(output[valid_rows:]), 0)
    np.testing.assert_array_equal(np.asarray(dx[valid_rows:]), 0)
```

- [ ] **Step 2: Run the regression on a QuACK-capable GPU and confirm the old wrapper fails**

Run:

```bash
uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_grugformer_moe.py \
  -k 'quack_mlp_varlen_ignores_trailing_capacity_rows' -q
```

Expected before the fix: nonzero or undefined trailing output/`dx`, or a kernel contract failure because the final offset is smaller than `x.shape[0]`.

- [ ] **Step 3: Mask padding in the QuACK forward and custom VJP**

Add:

```python
def _mask_rows_after_group_sizes(values: jax.Array, group_sizes: jax.Array) -> jax.Array:
    valid_rows = jnp.sum(group_sizes, dtype=jnp.int32)
    return jnp.where(jnp.arange(values.shape[0], dtype=jnp.int32)[:, None] < valid_rows, values, 0)
```

Apply it to the public forward body and `_quack_mlp_fwd` output. Apply it to `dx` in `_quack_mlp_bwd`. Do not pad `group_sizes`; the CuTe scheduler must see only accepted rows.

- [ ] **Step 4: Add the QuACK expert-chunk compute stage**

Slice `w13` and `w2` with static `spec.start/spec.size`, call `quack_mlp_varlen` with unpadded local group sizes, and preserve the `moe_up_down/chunk_<index>` named scope. Reject non-BF16 inputs and non-SiLU activation with the same concrete errors as the existing QuACK ring boundary.

- [ ] **Step 5: Run QuACK output/gradient tests and ragged EP tests**

```bash
uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_grugformer_moe.py \
  -k 'quack_mlp_varlen or ragged_all_to_all' -q
```

Expected on a QuACK GPU: existing full-capacity parity and the new padded-capacity test pass. Expected locally: GPU tests skip and CPU/abstract tests pass.

- [ ] **Step 6: Commit padding-safe QuACK compute**

```bash
git add lib/levanter/src/levanter/grug/_moe/sonic_quack.py \
  experiments/grug/moe/ragged_ep_pipeline.py \
  lib/levanter/tests/grug/test_grugformer_moe.py
git commit -m "[grug] Ignore padded rows in QuACK expert GEMMs"
```

### Task 4: Add serialized and two-slot expert-chunk schedules

**Files:**
- Modify: `experiments/grug/moe/ragged_ep_pipeline.py`
- Modify: `lib/levanter/tests/grug/test_ragged_ep_pipeline.py`

**Interfaces:**
- Produces: `_moe_mlp_ep_ragged_a2a_chunked_local(..., num_expert_chunks: int, pipeline_depth: int, compute: Literal["ragged", "quack"])`.
- Keeps the function private and absent from `MoeImplementation` and model configuration.

- [ ] **Step 1: Write schedule validation tests**

```python
@pytest.mark.parametrize(
    ("depth", "match"),
    [(0, "pipeline_depth must be 1 or 2"), (3, "pipeline_depth must be 1 or 2")],
)
def test_ragged_expert_pipeline_rejects_unsupported_depth(depth, match) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_expert_pipeline(num_expert_chunks=2, pipeline_depth=depth)


def test_ragged_expert_pipeline_rejects_depth_larger_than_chunks() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        _validate_expert_pipeline(num_expert_chunks=1, pipeline_depth=2)
```

- [ ] **Step 2: Run and confirm the validation tests fail**

Use the focused command from Task 1 with `-k 'expert_pipeline'`. Expected: missing helper import or assertion failure.

- [ ] **Step 3: Implement a windowed schedule with an explicit dependency between windows**

Add `_validate_expert_pipeline`. Build static chunk specs in Python. For each window of `pipeline_depth`:

1. dispatch every chunk in the window;
2. compute every dispatched chunk;
3. combine every computed chunk;
4. sum token-shaped contributions into the output accumulator; and
5. feed the accumulator through `jax.lax.optimization_barrier` with the next window's dispatch input.

The barrier helper must return the unchanged dispatch input and make the next window depend on the preceding combines:

```python
def _after_pipeline_window(value: jax.Array, dependency: jax.Array) -> jax.Array:
    value, _ = jax.lax.optimization_barrier((value, dependency))
    return value
```

Wrap the per-window body in `jax.checkpoint(..., prevent_cse=False)`. Use static named scopes `expert_chunk_<index>/dispatch`, `expert_chunk_<index>/compute`, and `expert_chunk_<index>/combine`.

- [ ] **Step 4: Write a multi-GPU values-and-gradients parity test**

Parameterize over `pipeline_depth=(1, 2)`, `compute=("ragged", "quack")`, and balanced/hot-expert routing. Compare the experimental function with `_moe_mlp_ep_ragged_a2a_local` for output, dropped count, `dx`, combine-weight gradients, `dw13`, and `dw2`. Use exact dropped counts, `1e-5` tolerances for ragged compute, and the existing QuACK BF16 tolerances for QuACK compute.

- [ ] **Step 5: Run the new test on the local backend and a multi-GPU worker**

```bash
uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_ragged_ep_pipeline.py \
  -k 'ragged_expert_pipeline' -q
```

Expected locally: validation tests pass and runtime tests skip. Expected on GPU: all depth/compute/routing cases pass.

- [ ] **Step 6: Inspect HLO for the intended dependency difference**

Lower depth 1 and depth 2 with two chunks. Save HLO text and confirm:

- both contain two dispatch and two combine ragged collectives;
- depth 1 contains the window dependency between chunk 0 combine and chunk 1 dispatch; and
- depth 2 has no data dependency from chunk 0 compute/combine to chunk 1 dispatch.

Do not claim runtime overlap from HLO alone.

- [ ] **Step 7: Commit the experimental schedules**

```bash
git add experiments/grug/moe/ragged_ep_pipeline.py \
  lib/levanter/tests/grug/test_ragged_ep_pipeline.py
git commit -m "[grug] Add two-slot expert-chunk EP schedule"
```

### Task 5: Add the trace-gated standalone benchmark

**Files:**
- Create: `experiments/grug/moe/benchmark_ep_ragged_pipeline.py`
- Create: `lib/levanter/tests/grug/test_benchmark_ep_ragged_pipeline.py`

**Interfaces:**
- CLI inputs: `--ep-size`, `--microbatch-size`, `--sequence-length`, `--hidden-dim`, `--intermediate-dim`, `--num-experts`, `--top-k`, `--capacity-factor`, `--num-expert-chunks`, `--pipeline-depth`, `--compute`, `--routing`, `--skew-alpha`, `--seed`, `--warmup`, `--iterations`, `--lower-only`, `--hlo-dir`, `--profile-dir`, and `--output`.
- JSON output: shape/configuration, routing counts, dropped routes, parity metrics, forward and value-and-gradient timing, HLO paths, and profile path.

- [ ] **Step 1: Write parser and validation tests**

Cover these behaviors:

```python
def test_parser_defaults_to_serialized_two_chunk_quack():
    args = _parser().parse_args([])
    assert args.num_expert_chunks == 2
    assert args.pipeline_depth == 1
    assert args.compute == "quack"


def test_validate_args_requires_expert_divisibility():
    args = _parser().parse_args(["--ep-size", "8", "--num-experts", "64", "--num-expert-chunks", "3"])
    with pytest.raises(ValueError, match="must divide"):
        _validate_args(args, device_count=8)


def test_validate_args_rejects_depth_two_with_one_chunk():
    args = _parser().parse_args(["--num-expert-chunks", "1", "--pipeline-depth", "2"])
    with pytest.raises(ValueError, match="cannot exceed"):
        _validate_args(args, device_count=8)
```

Also test deterministic balanced and skew routing without asserting incidental call counts or output strings.

- [ ] **Step 2: Run and confirm the benchmark tests fail because the module is absent**

```bash
uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_benchmark_ep_ragged_pipeline.py -q
```

- [ ] **Step 3: Implement the focused benchmark CLI**

Import `_parity_metrics`, `_parity_status`, and `_time` from
`experiments.grug.moe.benchmark_ep_ring`; do not copy those implementations.
Keep the new benchmark focused on the unchunked function and the requested
chunked schedule:

```text
unchunked_ragged
chunked_depth_<pipeline_depth>
```

Call `DistributedConfig().initialize()` before reading `jax.devices()` so the
same entry point works under Iris, Slurm, and a local process. Construct a global
explicit mesh from `jax.devices()[:ep_size]`. Initialize large activation and
weight arrays directly with their target `NamedSharding` through jitted random
initializers so E512 weights are never materialized in full on one host.

Compile forward and `jax.value_and_grad` variants. Use unchunked ragged compute as the exact routing/drop oracle and plain-ragged chunk compute as the numerical oracle for QuACK.

- [ ] **Step 4: Add HLO and profile capture**

When `--hlo-dir` is set, write forward and value-and-gradient HLO for each selected schedule. When `--profile-dir` is set, run warmup first, then capture iterations with:

```python
with jax.profiler.trace(profile_dir, create_perfetto_trace=True):
    for _ in range(iterations):
        jax.block_until_ready(compiled(*inputs))
```

Only process 0 writes human/JSON output and profile metadata in distributed runs.

- [ ] **Step 5: Run benchmark unit tests and a lower-only smoke**

```bash
uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_benchmark_ep_ragged_pipeline.py -q

XLA_FLAGS=--xla_force_host_platform_device_count=8 \
uv run --package marin-levanter --group test python \
  experiments/grug/moe/benchmark_ep_ragged_pipeline.py \
  --ep-size 8 --num-experts 64 --hidden-dim 128 --intermediate-dim 128 \
  --microbatch-size 1 --sequence-length 128 --top-k 2 \
  --num-expert-chunks 2 --pipeline-depth 2 --compute ragged \
  --lower-only --output json
```

Expected: tests pass and the smoke emits valid JSON listing all lowered functions and HLO artifacts requested by the command.

- [ ] **Step 6: Commit the benchmark**

```bash
git add experiments/grug/moe/benchmark_ep_ragged_pipeline.py \
  lib/levanter/tests/grug/test_benchmark_ep_ragged_pipeline.py
git commit -m "[grug] Benchmark expert-chunked ragged EP overlap"
```

### Task 6: Verify locally and run the overlap gate

**Files:**
- Modify only if a failing check exposes a behavior bug covered by a new regression test.
- Produce ephemeral artifacts under `/tmp/expert-chunked-ep/`.

**Interfaces:**
- Consumes: the benchmark and tests from Tasks 1-5.
- Produces: local verification output, HLO files, XPlane summaries, and a go/pivot decision.

- [ ] **Step 1: Run the focused Levanter test set**

```bash
uv run --package marin-levanter --group test pytest \
  lib/levanter/tests/grug/test_benchmark_ep_ragged_pipeline.py \
  lib/levanter/tests/grug/test_benchmark_ep_ring.py \
  lib/levanter/tests/grug/test_ragged_ep_pipeline.py \
  lib/levanter/tests/grug/test_grugformer_moe.py \
  -m 'not slow' -q
```

Expected: all CPU tests pass; GPU-specific tests skip on CPU.

- [ ] **Step 2: Run required lint and type checks**

```bash
./infra/pre-commit.py --changed-files --fix
uv run pyrefly
```

Expected: both commands exit zero. Review any formatter changes before continuing.

- [ ] **Step 3: Run the one-node GPU correctness matrix**

Run the new padded QuACK and pipeline parity tests on eight GPUs. Expected: exact drop parity; ragged values/gradients within `1e-5`; QuACK values/gradients within the existing BF16 thresholds.

- [ ] **Step 4: Capture serialized and depth-2 traces**

Run E512/top-8 BF16 on EP16 or EP64 with `num_expert_chunks=2`, once at depth 1 and once at depth 2. Keep routing assignments, placement, compiler flags, warmup, and iteration count identical. Save HLO and XPlane profiles under distinct `/tmp/expert-chunked-ep/` directories.

- [ ] **Step 5: Summarize the profiles**

```bash
uv run --with xprof --with protobuf python lib/marin/tools/profile_summary.py summarize \
  --profile-dir /tmp/expert-chunked-ep/depth1 \
  --output /tmp/expert-chunked-ep/depth1.json

uv run --with xprof --with protobuf python lib/marin/tools/profile_summary.py summarize \
  --profile-dir /tmp/expert-chunked-ep/depth2 \
  --output /tmp/expert-chunked-ep/depth2.json

uv run python lib/marin/tools/profile_summary.py compare \
  --before /tmp/expert-chunked-ep/depth1.json \
  --after /tmp/expert-chunked-ep/depth2.json \
  --strict-provenance
```

- [ ] **Step 6: Apply the trace gate**

Proceed with ragged all-to-all only if the depth-2 XPlane timeline contains an interval where chunk `n + 1` NCCL send/receive executes concurrently with chunk `n` QuACK GEMMs and forward-plus-backward median time improves by at least 5%. Also verify peak saved-state memory is flat as `num_expert_chunks` increases at fixed depth.

If the gate fails, preserve the benchmark, HLO, and profile evidence and stop adding scheduling abstractions. Write the next design against explicit asynchronous NCCL_EP transport.

- [ ] **Step 7: Run verification-before-completion and report the result**

Re-run the exact focused tests and lint commands after any fixes. Record commands, pass/skip counts, GPU shape, timings, memory, and overlap evidence. Do not describe the implementation as pipelined unless the trace gate passed.
