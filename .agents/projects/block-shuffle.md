# Blockwise Shuffle Plan (2026-02-20)

## Tracking

- GitHub issue: https://github.com/marin-community/marin/issues/2917
- Latest status comment: https://github.com/marin-community/marin/issues/2917#issuecomment-3932446706
- Sync rule: keep the issue as the short status/decision summary and keep this document as the detailed plan.

## Status (2026-02-20)

- Phase 1: complete
  - Coalesced contiguous reads implemented in `TokenSeqDataset.get_batch`.
  - Lightweight read-efficiency counters added (`examples`, `unique_examples`, `ranges`, `read_ops`).
  - Focused tests added for order/duplicates/coalescing and negative-index guard.
  - Data-only benchmark + metric dump workflow added:
    - `lib/levanter/scripts/bench/bench_data_loading_block_shuffle.py`
    - reports per-strategy elapsed time, examples/tokens per second, and read-op deltas vs estimated old path.
- Phase 2: complete
  - `BlockShufflingDataset` implemented and exported.
  - `AsyncDataset.block_shuffle(...)` API added.
  - Tail block policy implemented: tail stays at end and can be permuted within the tail region.
- Phase 3: complete
  - Unified config surface: `LmDataConfig.shuffle` now supports `BlockShuffleConfig`.
  - Marin tokenize helper type signatures updated to match.
  - Integration tests added for config-driven block shuffle application.
- Remaining
  - Phase 4: instrumentation/benchmarks and Nemotron ablations.

## Permutation Quality Characterization (2026-02-20)

### Reproduction Commands

```bash
cd lib/levanter
uv run python scripts/bench/bench_permutation_quality.py \
  --n 8192 --repeats 8 --seq-len 2048 \
  --strategies full,era,block --era-length 1024 --window-blocks 8 --batch-size 2048
```

```bash
cd lib/levanter
uv run python scripts/bench/bench_data_loading_block_shuffle.py \
  --steps 20 --batch-size 128 --prefetch-batches 16 \
  --num-examples 16384 --seq-len 2048 \
  --strategies full,era,block --era-length 1024
```

### Aggregate Shuffle Metrics (`perm_type=feistel`, mean across 8 seeds)

| Strategy | mean_abs_displacement_norm | inversion_fraction | Spearman rho | same_block_transition_rate |
|---|---:|---:|---:|---:|
| `full` | 0.3325 | 0.4997 | 0.0010 | 0.0107 |
| `era` | 0.0417 | 0.0625 | 0.9844 | 0.3106 |
| `block` | 0.3319 | 0.4980 | 0.0088 | 0.3063 |

Reference baselines for this setup (`n=8192`, `io_block_size=128`):

- random expected `mean_abs_displacement_norm`: `0.3334`
- random expected `inversion_fraction`: `0.5000`
- random expected `same_block_transition_rate`: `0.0155`
- no-shuffle `same_block_transition_rate`: `0.9923`

Interpretation:

- `block` is near `full` on global randomness metrics (`displacement`, `inversions`, low `rho`).
- `block` keeps locality close to `era` (`same_block_transition_rate` ~`0.306` vs `0.311`).
- This is the intended quality/cost shape: full-like mixing with era-like locality.

### Quick Visuals

Global mixing score (higher is better, using `mean_abs_displacement_norm / random_expected`):

- `full`: 0.997 `####################`
- `era`: 0.125 `###`
- `block`: 0.996 `####################`

Locality score (higher means more contiguous block reuse, using `same_block_transition_rate`):

- `full`: 0.011 `#`
- `era`: 0.311 `###############`
- `block`: 0.306 `###############`

### Read-Op Snapshot (same day, synthetic cache)

From `bench_data_loading_block_shuffle.py` with `seq_len=2048`, `steps=20`, `batch_size=128`, `prefetch=16`:

| Strategy | reads_issued | examples | reads/example | estimated_read_reduction |
|---|---:|---:|---:|---:|
| `full` | 35,650 | 40,960 | 0.8704 | 12.96% |
| `era` | 20 | 40,960 | 0.00049 | 99.95% |
| `block` | 287 | 40,960 | 0.00701 | 99.30% |

## Problem

Current training data loading assumes cheap random access, but our token caches are TensorStore-on-GCS.

- `DataLoader` prefetches many examples (`lib/levanter/src/levanter/data/loader.py`) and calls `dataset.get_batch(indices)` once per prefetch group.
- For text pretraining, `TokenSeqDataset.get_batch` currently does one TensorStore read per sequence (`lib/levanter/src/levanter/data/text/datasets.py`).
- Full permutation (`PermutationDataset`) makes indices globally random (`lib/levanter/src/levanter/data/permutation.py`), so read locality is poor and GCS read-op count is high.
- Existing `era_shuffle` improves locality but is only single-level and does not directly target zarr chunk boundaries.

This is expensive (many remote reads) and becomes a bottleneck as model scale increases.

## Goals

1. Reduce remote read requests and improve input throughput without introducing distributed cache coordination.
2. Preserve deterministic, resumable behavior (stateless index mapping from key + index).
3. Keep shuffle quality close to full permutation.
4. Make defaults align with TensorStore/zarr physical chunk layout.

## Non-Goals

1. No shared cache service between hosts/processes in v1.
2. No change to model/trainer math.
3. No attempt to exactly reproduce historical linear-permutation runs.

## Recommendation

Use a combined approach:

1. Add **hierarchical block shuffle** (new dataset wrapper) to make access locality explicit.
2. Add **coalesced contiguous reads** in `TokenSeqDataset.get_batch` so locality turns into fewer TensorStore reads.
3. Add **host-local block cache** only (small, bounded).

Block shuffle alone helps locality, but coalesced reads are what directly reduce request count.

## Why This Over Alternatives

### A. Only increase prefetch
Already done. It overlaps latency but does not reduce read ops; cost remains high.

### B. Shared cross-host cache
Could help further, but adds complexity/coordination/failure modes we want to avoid.

### C. Keep full shuffle, sort requests only
Low risk but limited upside: physical indices are still close to random under full permutation.

### D. Streaming shuffle queue rewrite
Potentially strong, but much larger redesign and resume semantics work.

### Chosen path
Hierarchical block shuffle + coalesced reads gives most of the win with limited architectural change and keeps random-access determinism.

## Proposed Design

### 1) New `BlockShufflingDataset` in `lib/levanter/src/levanter/data/permutation.py`

Add a dataset wrapper that maps logical index -> physical index using two levels:

1. Permute block IDs globally.
2. Within each window of `K` blocks, permute examples.

Terminology:

- `io_block_size`: examples per physical block (default chunk-aligned)
- `window_blocks` (`K`): number of blocks read/mixed together
- `window_size = io_block_size * window_blocks`

High-level mapping sketch:

```python
# Pseudocode only
block_perm = Permutation.make(perm_type, num_blocks, key_block)

for window_id:
    block_positions = range(window_id * K, min((window_id + 1) * K, num_blocks))
    physical_blocks = [block_perm(p) for p in block_positions]
    local_perm = Permutation.make(perm_type, window_len_examples, fold_in(key_local, window_id))
```

Behavior targets:

- Deterministic across restarts.
- Bijection for finite datasets.
- Handles tail block/window correctly.
- Finite-only in v1 (fail fast for infinite datasets).

### 2) Chunk-aligned default block size

For token sequences, align `io_block_size` to one zarr read chunk in `JaggedArrayStore.data`.

Current chunk constants in `lib/levanter/src/levanter/store/jagged_array.py`:

- `DEFAULT_CHUNK_SIZE = 256 * 1024` elements (tokens)

For `seq_len = 2048`, default block size is roughly:

- `256 * 1024 / 2048 = 128` examples/block

This means your “~1024 example locality target” is naturally represented as:

- `io_block_size = 128`, `window_blocks = 8`

### 3) Coalesced reads in `TokenSeqDataset.get_batch`

Replace one-read-per-example with contiguous range coalescing.

```python
# Pseudocode only
sorted_unique = sorted(set(indices))
runs = coalesce_consecutive(sorted_unique)  # [(start_idx, end_idx), ...]

with ts.Batch():
    futs = [data[start * seq_len : end * seq_len].read() for (start, end) in runs]

# split each run into seq_len-sized rows and restore original order
```

This is orthogonal and useful even outside block shuffle.

### 4) Host-local cache only

Add small LRU cache of decoded blocks/windows in the block-shuffle wrapper.

- Cache key: `(window_id)` or `(physical_block_id)`
- Bounded size (e.g. 4–16 windows)
- No inter-host communication

This avoids introducing shared services while still removing repeated reads on each host.

### 5) Config plumbing

Add explicit block-shuffle config in text data config path (`lib/levanter/src/levanter/data/text/datasets.py`).

Proposed shape:

```python
@dataclass(frozen=True)
class BlockShuffleConfig:
    io_block_size: int | None = None   # None => derive from chunk and seq_len
    window_blocks: int = 8
    perm_type: Literal["feistel", "linear"] = "feistel"
    cache_windows: int = 8
```

Then in `LmDataConfig.train_sets(...)`:

- If `block_shuffle` is set, apply block shuffle wrapper to each train component dataset.
- Keep existing `shuffle`/`era_shuffle` behavior unchanged when `block_shuffle` is not set.
- Reject ambiguous config combinations explicitly.

## Multi-Reader Strategy

V1 strategy: **no shared cache and no reader communication**.

Rationale:

1. Keep failure surface small.
2. Deterministic global mapping already prevents duplicate examples; overlap is at chunk level only.
3. Host-local cache captures most immediate reuse.

If cross-host chunk overlap is still costly, phase 2 can add process-aware block affinity, but that should come after quality validation.

## Implementation Plan

### Phase 0: Baseline instrumentation

1. Add debug counters for:
- `examples_requested`
- `tensorstore_reads_issued`
- `coalesced_ranges`
- `reads_per_example`

2. Add a lightweight data-only benchmark script (new) under `lib/levanter/scripts/bench/`.

Deliverable: baseline metrics for full shuffle vs era shuffle on existing path.

### Phase 1: Coalesced reads

1. Implement contiguous-range coalescing in `TokenSeqDataset.get_batch`.
2. Add tests for ordering and exact output equivalence.

Success criteria:

- Exact sample equivalence vs old implementation.
- Lower `tensorstore_reads_issued` on clustered indices.

### Phase 2: Block shuffle dataset wrapper

1. Implement `BlockShufflingDataset` in `permutation.py`.
2. Add `AsyncDataset.block_shuffle(...)` in `dataset.py`.
3. Export in `lib/levanter/src/levanter/data/__init__.py`.

Success criteria:

- Deterministic mapping, bijection, tail correctness.
- Pass unit and loader integration tests.

### Phase 3: Config integration

1. Add `BlockShuffleConfig` to `LmDataConfig`.
2. Thread config through `lib/marin/src/marin/processing/tokenize/data_configs.py` helpers.
3. Add config tests for serialization and conflict checks.

### Phase 4: Training ablations + rollout gate

1. Run Nemotron-mixture ablations.
2. Compare quality and throughput to full shuffle.
3. Decide default policy.

## Testing Plan

### Unit tests

1. `lib/levanter/tests/test_newdataset.py`
- block-shuffle determinism
- bijection for finite datasets
- tail-window correctness
- out-of-bounds behavior

2. New tests for coalescing helper
- preserves order
- handles duplicate and unsorted indices
- exact output equality

### Integration tests

1. `lib/levanter/tests/test_new_loader.py`
- DataLoader + block shuffle on 1 and multi-device setups
- sharded consistency still holds

2. `lib/levanter/tests/test_text.py`
- `LmDataConfig` applies block shuffle only when configured

### Performance regression tests (bench harness)

Track and report:

1. examples/sec from data loader
2. reads/example proxy
3. wall-clock batch latency p50/p95

## Nemotron Ablation Plan

Use existing Nemotron mixture definitions from:

- `experiments/tootsie/exp1295_32b.py` (`nemotron_mix`)
- optionally `experiments/isoflop_sweep.py` (same mixture in scaling workflows)

Create one dedicated ablation experiment script (new), using a moderate model (for faster iteration), e.g. `llama_300m` from `experiments/llama.py`, fixed sequence length and fixed optimizer.

Arms:

1. `full_shuffle`: current baseline (`shuffle=True`)
2. `era_shuffle`: existing locality baseline (`shuffle=<era_length>`)
3. `block_shuffle_k8`: `io_block_size=chunk_aligned`, `window_blocks=8`
4. `block_shuffle_k32`: same block size, larger window for quality stress-test

Run plan:

1. Short throughput pass: 1k–2k steps each, 1 seed
2. Quality pass: 10k+ steps each, 2 seeds

Metrics:

1. train loss and validation bpb/perplexity
2. tokens/sec
3. reads/example proxy from instrumentation

Acceptance gate:

1. Throughput/read-cost improvement is material (target: >=2x fewer reads/example and meaningful tokens/sec uplift).
2. Quality regression vs full shuffle is small (target: within normal seed variance; if not, increase `window_blocks`).

## Suggested Defaults

Initial default proposal for 2k-context token training:

1. `io_block_size = 128` (chunk-aligned at current default chunking)
2. `window_blocks = 8` (effective local shuffle span 1024 examples)
3. `perm_type = feistel`
4. `cache_windows = 8`

If quality gap appears, increase `window_blocks` before changing `io_block_size`.

## Risks and Mitigations

1. Risk: quality drift from reduced global mixing.
- Mitigation: include full-shuffle baseline and tune `window_blocks`.

2. Risk: complexity in index mapping around tails.
- Mitigation: bijection tests + property checks on many small lengths.

3. Risk: improvements depend heavily on `TokenSeqDataset` only.
- Mitigation: keep generic block-shuffle wrapper and add dataset-specific fast path where needed.

4. Risk: no cross-host cache leaves some duplicated chunk fetches.
- Mitigation: first validate gains; only then consider process-aware affinity.

## Open Questions

1. Should block shuffle be enabled per-component before mixture (current shuffle placement), or after mixture as a single wrapper?
2. Do we want finite-only support initially, or an era-style infinite fallback in the same class?
3. Should we expose separate knobs for `io_block_size` and an optional larger logical `shuffle_block_size`?

## Immediate Next Step

Run Phase 4 Nemotron ablations (`full`, `era`, `block`) and capture quality/throughput/read-op deltas in the issue thread for default-policy decision.
