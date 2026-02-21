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

## Uncheatable Eval Snapshot (2026-02-21)

Using the exp2917 sweep runs with default validation on, and comparing at the shared eval horizon (`global_step=1000`):

| Strategy | Uncheatable macro loss | Uncheatable micro loss | Macro vs full | Micro vs full |
|---|---:|---:|---:|---:|
| `full` | 4.2613 | 4.3572 | baseline | baseline |
| `era` | 4.7481 | 4.8700 | +11.43% | +11.77% |
| `block_w8` | 4.3364 | 4.4457 | +1.76% | +2.03% |
| `block_w16` | 4.3007 | 4.4114 | +0.92% | +1.25% |
| `block_w32` | 4.3167 | 4.4258 | +1.30% | +1.58% |

Artifacts:

- `.agents/analysis/exp2917/uncheatable_eval_macro_loss_by_step.png`
- `.agents/analysis/exp2917/uncheatable_eval_micro_loss_by_step.png`
- `.agents/analysis/exp2917/uncheatable_eval_summary.json`

Note on robustness:

- These eval plots are preemption-safe by construction (deduped by `global_step`, no cumulative counters).

## Final Trio Snapshot (2026-02-21, step 2061)

Runs:

- `full_shuffle`: `exp2917_shuffle_block_150m_2026-02-20-full_shuffle-e638fd`
- `era_shuffle`: `exp2917_shuffle_block_150m_2026-02-20-era_shuffle-d4b30e`
- `block_shuffle` (16GiB TensorStore cache): `exp2917_shuffle_block_150m_2026-02-20-block_shuffle-5dab59`

| Strategy | tokens/s | eval loss | uncheatable macro | gcs_read_count_total | gcs_bytes_read_total |
|---|---:|---:|---:|---:|---:|
| `full` | 882,216 | 3.8418 | 3.8546 | 1,932,354 | 621,914,747,700 |
| `era` | 869,611 | 4.1593 | 4.4147 | 4,899 | 2,608,161,636 |
| `block` | 874,715 | 3.9114 | 3.9424 | 9,137 | 2,642,644,048 |

Relative to `full`:

- `era`: `-1.43%` tokens/s, `-99.58%` GCS bytes, `+14.53%` uncheatable macro loss.
- `block`: `-0.85%` tokens/s, `-99.58%` GCS bytes, `+2.28%` uncheatable macro loss.

Tail loss stability (`step >= 1000`, same logging horizon):

| Strategy | train loss std | jitter std of first diff |
|---|---:|---:|
| `full` | 0.0862 | 0.03443 |
| `era` | 0.0821 | 0.03222 |
| `block` | 0.0890 | 0.03696 |

Artifacts:

- `.agents/analysis/exp2917/report_latest.md`
- `.agents/analysis/exp2917/metrics_summary_latest.json`
- `.agents/analysis/exp2917/loss_loglog_vs_tokens_latest.png`
- `.agents/analysis/exp2917/throughput_tokens_per_s_by_step_latest.png`
- `.agents/analysis/exp2917/loading_time_by_step_latest.png`
- `.agents/analysis/exp2917/tensorstore_gcs_reads_total_by_step_latest.png`
- `.agents/analysis/exp2917/uncheatable_eval_macro_loss_by_step_latest.png`
- `.agents/analysis/exp2917/uncheatable_eval_micro_loss_by_step_latest.png`

## Block-Window Report (w8/w16/w32)

Runs:

- `block_w8`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_20260220-190441-exp2-block_shuffle_w8-1d74bd
- `block_w16`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_20260220-190441-exp-block_shuffle_w16-c50e53
- `block_w32`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_20260220-190441-exp-block_shuffle_w32-01502f

### Final metrics (step 2061)

| run | train_loss | eval_loss | uncheatable_macro | uncheatable_micro | tokens/s | tensorstore_reads_total |
|---|---:|---:|---:|---:|---:|---:|
| block_w8 | 3.6543 | 3.9034 | 3.9164 | 4.0272 | 871570 | 4465 |
| block_w16 | 3.6524 | 3.8860 | 3.8931 | 4.0068 | 868988 | 3089 |
| block_w32 | 3.5904 | 3.8841 | 3.8801 | 3.9859 | 873077 | 4577 |

### Relative to w16 (%)

| run | tokens/s | eval_loss | uncheatable_macro | uncheatable_micro | tensorstore_reads_total |
|---|---:|---:|---:|---:|---:|
| block_w8 | +0.30% | +0.45% | +0.60% | +0.51% | +44.55% |
| block_w32 | +0.47% | -0.05% | -0.33% | -0.52% | +48.17% |

### Tail loss stability (step >= 1000)

| run | loss_std | jitter_std_diff |
|---|---:|---:|
| block_w8 | 0.110942 | 0.045568 |
| block_w16 | 0.112688 | 0.045551 |
| block_w32 | 0.125579 | 0.036995 |

### Takeaways

- `w16` minimizes read count in this sweep (`3089` total reads).
- `w32` has best final quality metrics (lowest eval and uncheatable losses) and slightly best throughput, but read count is highest (`4577`).
- `w8` is in-between on throughput/quality, with read count above `w16`.

### Artifacts

- `.agents/analysis/exp2917/block_window_summary_latest.json`
- `.agents/analysis/exp2917/block_window_loss_loglog_vs_tokens_latest.png`
- `.agents/analysis/exp2917/block_window_throughput_tokens_per_s_by_step_latest.png`
- `.agents/analysis/exp2917/block_window_tensorstore_reads_total_by_step_latest.png`
- `.agents/analysis/exp2917/block_window_uncheatable_macro_by_step_latest.png`
- `.agents/analysis/exp2917/block_window_uncheatable_micro_by_step_latest.png`

### Read-Op Snapshot (same day, synthetic cache)

From `bench_data_loading_block_shuffle.py` with `seq_len=2048`, `steps=20`, `batch_size=128`, `prefetch=16`:

| Strategy | reads_issued | examples | reads/example | estimated_read_reduction |
|---|---:|---:|---:|---:|
| `full` | 35,650 | 40,960 | 0.8704 | 12.96% |
| `era` | 20 | 40,960 | 0.00049 | 99.95% |
| `block` | 287 | 40,960 | 0.00701 | 99.30% |

## Large-Window Report (w64/w128/w256/w512/w1024)

Baselines:

- `full_shuffle`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-20-full_shuffle-e638fd
- `legacy_block` (small-window): https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-20-block_shuffle-5dab59

Runs:

- `w64`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-21-block_shuffle_w64-27229c
- `w128`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-21-block_shuffle_w128-53ab01
- `w256`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-21-block_shuffle_w256-daeeaa
- `w512`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-21-block_shuffle_w512-b8c967
- `w1024`: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-21-block_shuffle-8ba290

### Final metrics (step 2061)

| window | eval_loss | uncheatable_macro | uncheatable_micro |
|---|---:|---:|---:|
| `w64` | 3.876324 | 3.862838 | 3.975510 |
| `w128` | 3.868693 | 3.862938 | 3.976528 |
| `w256` | 3.865820 | 3.875195 | 3.983520 |
| `w512` | 3.850497 | 3.853288 | 3.962978 |
| `w1024` | 3.846557 | 3.860804 | 3.966571 |

### Relative to full shuffle (%)

| window | eval_loss | uncheatable_macro | uncheatable_micro |
|---|---:|---:|---:|
| `w64` | +0.899% | +0.213% | +0.564% |
| `w128` | +0.700% | +0.216% | +0.590% |
| `w256` | +0.625% | +0.534% | +0.767% |
| `w512` | +0.226% | -0.034% | +0.247% |
| `w1024` | +0.124% | +0.160% | +0.338% |

### Relative to legacy small-window block (%)

| window | eval_loss | uncheatable_macro | uncheatable_micro |
|---|---:|---:|---:|
| `w64` | -0.897% | -2.018% | -1.914% |
| `w128` | -1.092% | -2.015% | -1.889% |
| `w256` | -1.166% | -1.704% | -1.716% |
| `w512` | -1.557% | -2.260% | -2.223% |
| `w1024` | -1.658% | -2.069% | -2.135% |

### Takeaways

- This sweep supports a clear quality trend with larger window sizes.
- `w512` nearly closes the full-shuffle validation gap and is best on uncheatable macro in this set.
- `w1024` is similarly close to full and slightly improves eval loss vs `w512`, but is slightly worse on uncheatable macro/micro than `w512`.
- Practical guidance: target `window_blocks ~= batch_size_examples` as a default starting point.
- Follow-up in flight: `w1024` run launched to test the `~2x batch-size` window regime.
- `w1024` Ray job: `ray-run-dlwh-exp2917_shuffle_block-20260221-104457`
- `w1024` W&B run: https://wandb.ai/marin-community/marin/runs/exp2917_shuffle_block_150m_2026-02-21-block_shuffle-8ba290

### Artifacts

- `.agents/analysis/exp2917/block_window_large_summary_latest.json`
- `.agents/analysis/exp2917/block_window_large_report_latest.md`
- `.agents/analysis/exp2917/block_window_large_eval_gap_vs_full_latest.png`

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
