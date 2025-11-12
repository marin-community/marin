**Splice Document Dataset**

- Goal: train on a single chosen document but present it to the model at many absolute positions along the positional axis so the model learns its structure invariant to splice position. For a document of length L and model seq_len S, we emit examples where the same document tokens appear starting at offset s for s in [0, S − L], typically stride 1. Only the document tokens contribute to loss; padding/filler tokens do not. Doc token order is unchanged; only absolute positions shift.

**How Levanter Builds LmExamples Today**

- Token caches: Training data is tokenized to a TreeCache with a JaggedArray store under `input_ids` (flat `data`, plus `offsets` per original record). See `TreeCache`, `JaggedArrayStore`.
- Fixed-length sequences: `TokenSeqDataset` reads contiguous, fixed-length slices of `data` of length `Pos.size` (the model’s seq_len) at multiples of `Pos.size`, ignoring original document boundaries.
  - Ref: src/levanter/data/text.py: TokenSeqDataset
- Mapping to LmExample: `CausalLmDataset` wraps a token sequence and produces `LmExample.causal(tokens)` with a full causal loss mask (all tokens except final position), optionally respecting `ignore_id` and building segment breaks from `eos_id`.
  - Ref: src/levanter/data/text.py: CausalLmDataset; src/levanter/models/lm_model.py: LmExample.causal
- Packing (eval harness and supervised/chat): `packing.py` constructs packed `LmExample`s with explicit segment IDs and custom loss masks (e.g., only completion tokens count). This shows how to shape loss and attention independent of raw token positions.
  - Ref: src/levanter/data/packing.py: SequencePacker, PromptCompletion, pack_prompt_completions

**Problem With Standard Training**

- Standard `TokenSeqDataset` ignores document boundaries and does not shift a single document across absolute positions. Even if you select a single document upstream, you still get a fixed left-aligned view of that document unless you alter seq_len or perform synthetic packing.

**Design: SpliceDocumentDataset**

- Summary: A new AsyncDataset that, given the tokens of a single selected document (length L) and the model axis `Pos` (length S), yields `LmExample`s where the document (or a suffix/subspan of it) is “spliced” at absolute offset `s`, with:
  - tokens[0:s] = filler (pad or EOS)
  - tokens[s:s+L] = document tokens (truncated if L > S − s)
  - tokens[s+L:S] = filler
  - loss_mask = 1 only on document token positions that have an in-document next token (mask last in-doc position and all filler).
  - segment_ids: break attention at the splice boundary so the doc does not attend to filler to its left (prefix segment 0, doc+trailer segment 1).

**Document Selection**

- Source: Either a single dataset (`SingleDatasetLMConfigBase`) or one component of an `LMMixtureDatasetConfig` identified by name.
- Deterministic selection: Accept `doc_index` (int) or `doc_key` (string predicate) to fetch one record from the training cache. Optionally honor `restrict_to_training_subset` in the same way as inner-loop P(z) does (compute training token cutoff from `max_train_batches` and `initial_batch_size`).
- Implementation sketch:
  - Build or load the training cache(s) with the underlying source config(s): `build_caches("train")`.
  - For the chosen dataset name, read `input_ids.offsets` to compute `(start, end)` for the selected `doc_index` and fetch `data[start:end]` as the document tokens (np.int32).

**Document Length Filtering**

- Goal: ensure the selected document is long enough for downstream eval windows (e.g., P(z) chunking) without manual doc indexing.
- Config (SpliceSingleDocumentLMConfig):
  - `min_doc_length: int | null` Only consider docs with length ≥ this value.
  - `max_doc_length: int | null` Only consider docs with length ≤ this value.
  - `doc_select_mode: first|longest|random` Policy among candidates that meet the thresholds. If none meet them, fall back to the longest doc overall.
  - `doc_index: int | null` If provided but violates length thresholds, fall back to the policy above; otherwise honor `doc_index`.
- Efficiency: compute lengths from `input_ids.offsets` once and select an index; only materialize tokens for the chosen doc.
- YAML example:
  - data:
    dataset_name: common_pile/wikimedia
    min_doc_length: 1024
    doc_select_mode: longest
    doc_index: null

**Example Construction**

- Given `doc` (length L), model axis `Pos` (size S), and a splice offset `s` (0 ≤ s ≤ max_offset):
  - `tokens = np.full(S, pad_id, dtype=int32)`
  - `copy_len = min(L, S - s)`
  - `tokens[s:s+copy_len] = doc[:copy_len]`
  - `loss_mask = np.zeros(S, dtype=int32)`
  - If `copy_len >= 2`, set `loss_mask[s : s + copy_len - 1] = 1` so only next tokens within the doc contribute.
  - `segment_ids = np.zeros(S, dtype=int32)` and `segment_ids[s:] = 1` to prevent attention from doc tokens to prefix filler. (Trailing filler is future w.r.t. doc tokens and thus not attended under causal masking.)
  - Create `LmExample`:
    - Either call `LmExample(tokens, loss_mask, attn_mask=AttentionMask.causal().with_segment_ids(segment_ids))`, or use `LmExample.causal(tokens, loss_mask=..., segment_ids=segment_ids)`.
    - Optional alternative: set `tokens[s-1] = eos_id` when `s > 0` and use `LmExample.causal(..., eos_id=eos_id)` to auto-derive the segment break.

**Offset/Content Schedule (Splice Policy)**

- Absolute offsets (placement): `s = 0, 1, 2, …, max_offset` where `max_offset = max(0, S - K)` for a chosen content length `K` (see below). Configurable stride `k_s` (default 1).
- In-document content start: `t = 0, 1, 2, …, L - 1` choosing where within the document to start copying from. Configurable stride `k_t` (default 1).
- Content length: `K` = number of tokens copied from the document starting at `t`. Typically `1 ≤ K ≤ S`; default behavior from the initial design is `K = L` (clamped by `S - s`).
- Determinism: enumerate or seed-permute the Cartesian product of feasible pairs `(t, s)` subject to `copy_len = min(K, L - t, S - s) ≥ 2` (so at least one next-token loss position exists). Shard deterministically across workers.
- Cycling/epochs: after exhausting the `(t, s)` pairs, repeat (optionally reshuffling either axis per epoch).

**Edge Cases**

- L ≥ S: Provide a mode to emit “sliding content windows” instead of “absolute splice”. Emit windows of length S across the doc: start positions `w = 0..(L − S)` with stride `k_window` and standard `LmExample.causal` loss across the full S tokens. This is the classic sliding-window training.
- L ≪ S: The default splice behavior works as above; the doc occupies a contiguous block of length L anywhere in the S-length frame.
- Optional right-side filler variant: set `segment_ids[s:s+L] = 1` and leave trailing filler as segment 0 if you want to also prevent doc tokens from attending to trailing filler (usually unnecessary under causal masking).

**Class and Config Sketch**

- New config type: `SpliceSingleDocumentLMConfig` (subclass of `LMTaskConfig`).
  - Parameters:
    - `base`: an existing `SingleDatasetLMConfigBase` or a minimal source spec (e.g., `UrlSingleDatasetLMConfig` or name of a component from an `LMMixtureDatasetConfig`), used solely to build caches and pick the target doc.
    - `dataset_name`: when wrapping an `LMMixtureDatasetConfig`, the component key to read doc tokens from.
    - `doc_index: int` (or `doc_selector: {min_len, max_len, hash, ...}` as future extension).
    - `pad_token_id`, `eos_token_id` (default from tokenizer).
    - Placement schedule:
      - `offset_stride: int` (aka `k_s`, default 1), `shuffle_offsets: bool|int`, `offset_seed: int`.
    - Content schedule:
      - `content_length: int | null` (aka `K`; default null means “whole doc”),
      - `content_stride: int` (aka `k_t`, default 1), `shuffle_content: bool|int`, `content_seed: int`.
      - `content_start_mode: {anchor_start, slide_within, cyclic_roll}` where:
        - anchor_start: always `t=0` (original design)
        - slide_within: iterate/sampled `t` in `[0..L-1]` (suffix exposure)
        - cyclic_roll: treat doc as circular buffer for augmentation (optional)
    - `restrict_to_training_subset: bool`, `initial_batch_size: Optional[int]` for parity with training-head gating.
    - `mode: "splice" | "slide"` (splice = place whole doc at offset; slide = classic fixed-window over long docs).
  - Implements:
    - `build_caches(split)`: delegate to the wrapped source config(s) to obtain `TreeCache`.
    - `train_set(Pos, batch_schedule, *, key, epochs)`:
      1) materialize the single doc tokens from the training cache;
      2) construct an `AsyncDataset` over `(t, s)` pairs that yields `LmExample`s as specified, with `K` and strides;
      3) wrap with optional `EpochPermutationDataset` if either axis should reshuffle per epoch.
    - `validation_sets(Pos)`: either empty or provide the same doc with a fixed small sample of offsets for sanity checks.
  - Dataset type: `SpliceDocumentDataset(AsyncDataset[LmExample])` that implements `async_len` (number of offsets × epochs), `get_batch(indices)` by building the corresponding splices on the fly (vectorized where possible) and returns `LmExample`s.

**How Examples Become LmExamples (Concrete)**

- We explicitly produce `LmExample` with:
  - `tokens`: length `Pos.size`, dtype int32.
  - `loss_mask`: binary, 1 at document positions except the final in-document position; 0 elsewhere.
  - `attn_mask`: `AttentionMask.causal().with_segment_ids(segment_ids)` with a segment break at `s`.
  - This mirrors the packing flow in `packing.py` (explicit segment IDs and loss masks) and bypasses the default full-causal-loss behavior in `CausalLmDataset`.

**Interplay With Position Embeddings**

- Because the same document tokens are emitted at different absolute indices on the `Pos` axis, the model’s RoPE embeddings expose the document content under different absolute positions. Introducing in-document start `t` also exposes later suffixes (e.g., `doc[1:]`, `doc[2:]`), so the model learns the document’s structure regardless of splice point and starting location.

**Worked Examples**

- Notation: `S` = seq_len = 5, document tokens `doc = [0,1,2,3,4]` (so `L=5`). `Null` denotes filler/pad.

- Whole-doc splice (original behavior): `K = L` (clamped), `t = 0`.
  - `s = 0` → copy_len = min(5, 5, 5) = 5 → tokens = [0,1,2,3,4]
  - `s = 1` → copy_len = min(5, 5, 4) = 4 → tokens = [Null,0,1,2,3]
  - `s = 2` → copy_len = min(5, 5, 3) = 3 → tokens = [Null,Null,0,1,2]
  - This never yields `[1,2,Null,Null,Null]` because content always starts at `t=0`.

- Suffix splice (expose later starts): choose `K=2`, `t=1`, `s=0`.
  - copy_len = min(2, L−t=4, S−s=5) = 2 → tokens = [1,2,Null,Null,Null]
  - Loss positions are on indices 0..(0+2−2)=0 (just token 1→2), matching next-token causal loss.

- Another suffix splice with absolute shift: `K=3`, `t=2`, `s=1`.
  - copy_len = min(3, L−t=3, S−s=4) = 3 → tokens = [Null,2,3,4,Null]

- Combined coverage: iterate over `(t,s)` with `K` fixed (e.g., `K=3`) to expose many views:
  - `(t,s) ∈ {(0,0),(0,1),(0,2),(1,0),(1,1),(2,0)}`, pruned where `copy_len < 2`.
  - This includes your desired later-start slice `[1,2,Null,Null,Null]` at `(t=1,s=0,K=2)`.

**slide_within Mode (Detailed)**

- Intent: expose later in-document starts (suffixes/subspans) to keep perplexity low on “later splices”, and also place those subspans at different absolute positions to train invariance to splice position.

- Controls:
  - `content_start_mode: slide_within` enables varying in-document start `t`.
  - `content_length = K` fixes the number of document tokens copied per example (unless truncated by document end or frame end). Choose `K ≥ 2` so at least one next-token prediction exists.
  - `content_stride = k_t` increments `t` by `k_t` (default 1).
  - `offset_stride = k_s` increments absolute placement `s` by `k_s` (default 1).
  - Allowed `(t, s)` pairs satisfy `copy_len = min(K, L − t, S − s) ≥ 2`.
  - Tight bounding for each `t`: `s ∈ [0, S − min(K, L − t)]`. This allows longer `s` ranges when the remaining suffix is shorter than `K`.

- Deterministic scheduling:
  - Default enumeration order: outer loop over `t` (ascending), inner loop over `s` (ascending). Alternative: seed-permute either axis each epoch.
  - Sharding: assign every Nth pair to process `rank` (e.g., enumerate pairs to a flat index and use `idx % world_size == rank`).

- Loss and attention per example (unchanged from above):
  - `loss_mask = 1` on the in-frame document tokens except the last copied token; `0` elsewhere.
  - `segment_ids`: 0 on `[0:s)`, 1 on `[s:S)`. Causal mask prevents attending to the future; segment boundary prevents doc tokens from attending to the prefix filler.

- Step-by-step example A (S=5, L=5, K=3, slide_within):
  - Document: `doc = [0,1,2,3,4]`.
  - For `t=0`, remaining suffix length is 5, so `min(K, L − t) = 3` and `s ∈ [0..S−3] = [0..2]`:
    - (t=0, s=0): copy_len=3 → tokens = [0,1,2,Null,Null]
    - (t=0, s=1): copy_len=3 → tokens = [Null,0,1,2,Null]
    - (t=0, s=2): copy_len=3 → tokens = [Null,Null,0,1,2]
  - For `t=1`, `min(K, L − t) = 3`, `s ∈ [0..2]`:
    - (t=1, s=0): [1,2,3,Null,Null]
    - (t=1, s=1): [Null,1,2,3,Null]
    - (t=1, s=2): [Null,Null,1,2,3]
  - For `t=2`, `min(K, L − t) = 3`, `s ∈ [0..2]`:
    - (t=2, s=0): [2,3,4,Null,Null]
    - (t=2, s=1): [Null,2,3,4,Null]
    - (t=2, s=2): [Null,Null,2,3,4]
  - For `t=3`, remaining suffix is 2, so `min(K, L − t) = 2` and `s ∈ [0..S−2] = [0..3]`:
    - (t=3, s=0): [3,4,Null,Null,Null]
    - (t=3, s=1): [Null,3,4,Null,Null]
    - (t=3, s=2): [Null,Null,3,4,Null]
    - (t=3, s=3): [Null,Null,Null,3,4]
  - For `t=4`, remaining suffix is 1 → `copy_len=1` < 2; skip all s.
  - This schedule presents many later-start splices and absolute placements. For example, `(t=3, s=0)` yields `[3,4,Null,Null,Null]` and `(t=1, s=0)` yields `[1,2,3,Null,Null]`.

- Step-by-step example B (exact match to user slice, K=2):
  - Same `S=5, L=5, doc=[0,1,2,3,4]`, set `K=2`.
  - For `t=1`, `min(K, L − t) = 2`, `s ∈ [0..S−2] = [0..3]`:
    - (t=1, s=0): [1,2,Null,Null,Null]  ← desired
    - (t=1, s=1): [Null,1,2,Null,Null]
    - (t=1, s=2): [Null,Null,1,2,Null]
    - (t=1, s=3): [Null,Null,Null,1,2]
  - Loss mask for (t=1, s=0): 1 at idx 0, 0 elsewhere (predict 2 from 1), ensuring low perplexity on this later splice after training exposure.

- Counting pairs per epoch:
  - Total examples = sum over `t=0..L−1` of `(S − min(K, L − t) + 1)`, excluding `t` where `min(K, L − t) < 2`.
  - With S=5, L=5, K=3: 3 + 3 + 3 + 4 + 0 = 13 examples per full pass (as listed in Example A).

**Performance Considerations**

- Memory: With `S` large and stride 1, you generate `S − L + 1` examples per cycle. Adjust stride to limit compute.
- Throughput: Example construction is simple array slicing/filling; we can pre-allocate and reuse buffers per batch in `get_batch`.
- Multi-host sharding: Use modulo or range-splitting across JAX processes to avoid duplicate offsets.

**Why `content_length (K)` Is A Parameter, And Recommended Defaults**

- Compute vs supervision: Each example always has `S` tokens (model seq_len). Attention and forward cost scale with `S`, not with `K`. `K` only controls how many of those `S` positions contribute to loss. Very small `K` is usually wasteful because you pay full compute for few supervised positions. Therefore:
  - Recommended default for training: set `content_length: null` (meaning “use as many document tokens as fit”), effectively `K = min(L − t, S − s)`, or equivalently set `K >= S` and let `copy_len` clamp. This maximizes useful supervision per example.
  - Guard-rail: require `K ≥ 2` (we already prune pairs with `copy_len < 2`).

- Why expose `K` anyway?
  - Exact left-anchored later splices: To emit examples like `[1,2,Null,Null,Null]` (left-anchored short suffix), you need `K=2, t=1, s=0`. With `K=S` you’d get `[1,2,3,4,Null]` (longer content) or with a late placement `[Null,Null,Null,1,2]` (right-anchored). If you care about that specific left-anchored pattern for evaluation, ablations, or curriculum, `K` must be adjustable.
  - Curriculum or budgeted supervision: You might down-weight long-range targets by capping `K` (e.g., warmup with smaller `K`, then increase). While compute is unchanged, the per-step gradient comes from fewer positions, which can stabilize early training in some regimes.
  - Matching external eval windows: P(z) style or other evals often use fixed window sizes (e.g., 100 tokens). Being able to mirror that window for training experiments can be valuable.
  - Balancing coverage: Smaller `K` increases the number of valid `(t, s)` placements for late `t` (since `S − min(K, L − t)` grows as `K` shrinks). If you want more absolute-position diversity near the document end without changing `S`, a lower `K` can be used strategically (though note it is supervision-inefficient).

- Practical guidance:
  - Use `content_length: null` (or `>= S`) for standard training; combine with `slide_within` over `t` and a broad range of `s` to get both later-start exposure and positional invariance efficiently.
  - Use small `K` sparingly for evaluation or targeted ablations (e.g., to exactly reproduce `[1,2,Null,Null,Null]` style slices). Avoid small `K` for full training runs unless you have a specific reason.

**Validation and Debugging**

- Add a debug mode to log: selected dataset name, doc_index, doc_len, seq_len, stride, number of offsets.
- Optionally log an entropy or tiny eval on a few offsets to confirm nonzero loss only on doc positions.
- Add a quick unit test: build a tiny fake doc (e.g., [1,2,3]) and S=5, verify tokens, loss_mask, and segment_ids for offsets 0..2.

**Configuration Example (YAML Sketch)**

- data:
  type: splice_single_document
  base:
    type: url_single_dataset
    cache_dir: gs://.../tokenized/common_pile/wikimedia-...
    train_urls: [gs://.../wikimedia_filtered-...]
  dataset_name: common_pile/wikimedia
  doc_index: 12345
  # Placement
  offset_stride: 1
  # Content
  content_length: 3     # e.g., K=3; or null to use whole document
  content_stride: 1
  content_start_mode: slide_within
  mode: splice
  restrict_to_training_subset: true
  initial_batch_size: 128

**Proposed Plan**

- Add new config + dataset
  - Implement `SpliceSingleDocumentLMConfig(LMTaskConfig)` and `SpliceDocumentDataset(AsyncDataset[LmExample])`.
  - Reuse tokenizer from `LMTaskConfig.the_tokenizer`.
  - Implement cache lookup + doc materialization (via `TreeCache.store.tree["input_ids"]`).
- Example generation logic
  - Build splices with explicit `loss_mask` and `segment_ids` as above.
  - Support `mode: splice|slide`, deterministic sampling/enumeration over `(t,s)` with `content_length K`, and sharding.
- Trainer integration
  - Ensure `train_set()` returns finite dataset length (count of `(t,s)` pairs per epoch); support `epochs` repeat logic.
  - Validation: optional fixed `(t,s)` grid for sanity checks.
- Testing
  - Unit tests for loss mask and segmentation correctness, including the `[1,2,Null,Null,Null]` case.
  - Small end-to-end smoke run (tiny model) confirming low perplexity on later splices after training.
- Documentation
  - Keep this docs page updated and add a runnable example config.

**Alternatives Considered**

- Reusing `TokenSeqDataset` with special caches: cumbersome; loses explicit control of segment boundaries and loss mask.
- Leveraging `pack_prompt_completions`: close, but we want absolute-position splicing over a single doc; direct construction is simpler and clearer.

**Multi-Document Splice Extension**

**Motivation**

The single-document splice dataset is powerful for memorization studies and position-invariance experiments on one text. However, many training scenarios require fine-tuning on multiple documents:
- Multi-document memorization: Study how models learn to distinguish and recall N documents simultaneously.
- Domain adaptation: Fine-tune on a curated set of documents (e.g., 10 longest legal texts, 20 random scientific papers).
- Controlled data scaling: Systematically vary the number of training documents (1, 5, 10, 50, ...) to study memorization scaling laws.
- Document-level curriculum: Start with a few documents, gradually introduce more.

**Design Overview**

Extend the splice architecture with two new classes that maintain the same splice semantics (coverage-balanced placement, segment IDs, loss masking) while distributing examples across multiple documents:

1. `MultiSpliceDocumentDataset(AsyncDataset[LmExample])`: Core dataset that generates splice examples from a list of documents.
2. `SpliceMultiDocumentLMConfig(LMTaskConfig)`: Configuration wrapper that selects N documents from a base dataset and constructs the multi-document splice dataset.

**MultiSpliceDocumentDataset Class**

Signature and core parameters:
```python
class MultiSpliceDocumentDataset(AsyncDataset[LmExample]):
    def __init__(
        self,
        *,
        Pos: Axis,
        doc_tokens_list: List[np.ndarray],       # List of tokenized documents
        doc_metadata: Optional[List[dict]] = None,  # Optional metadata per doc
        pad_token_id: int,
        eos_token_id: Optional[int] = None,
        content_length: Optional[int] = None,    # K (same as single-doc)
        content_stride: int = 1,                 # t stride
        offset_stride: int = 1,                  # s stride (ignored in balanced mode)
        content_start_mode: str = "coverage_balanced",
        min_copy_len: int = 2,
        alpha: float = 0.0,                      # Edge upsampling
        balance_mode: str = "by_coverage",       # NEW: balancing policy
        rng_key=None,
    ):
```

**Key Difference: Balance Modes**

The critical new parameter is `balance_mode`, which controls how examples are distributed across documents:

- **`by_coverage` (default)**: Each document contributes examples proportional to its length.
  - Document of length L_i generates ~L_i splice pairs (with stride and edge upsampling).
  - Longer documents → more training examples.
  - Natural distribution: model sees more of longer documents.
  - Total examples per epoch = sum(L_i / content_stride) across all documents.
  - Use case: General fine-tuning where document importance correlates with length.

- **`by_document`**: Each document contributes equal number of examples regardless of length.
  - Find max pairs across all documents: `max_pairs = max(num_pairs_i)`.
  - Replicate each document's pairs to reach `max_pairs` (with wraparound).
  - Shorter documents get upsampled (their splice positions repeat).
  - Ensures uniform representation of all documents.
  - Total examples per epoch = num_docs * max_pairs.
  - Use case: Memorization experiments where each document should be learned equally well.

**Example: by_coverage vs by_document**

Given 3 documents with lengths [10K, 5K, 2K], content_stride=1, S=4096, K=4096:
- Document 0: ~10K splice pairs
- Document 1: ~5K splice pairs
- Document 2: ~2K splice pairs

With `balance_mode="by_coverage"`:
- Total pairs per epoch: 10K + 5K + 2K = 17K
- Document 0 contributes 59% of examples
- Document 1 contributes 29% of examples
- Document 2 contributes 12% of examples

With `balance_mode="by_document"`:
- Total pairs per epoch: 3 * 10K = 30K
- Document 0 contributes 33% of examples (10K unique pairs)
- Document 1 contributes 33% of examples (5K unique pairs, 5K duplicates)
- Document 2 contributes 33% of examples (2K unique pairs, 8K duplicates)

**Critique: Multi-Document Approach**

- Exact replication for by_document is inefficient and risky.
  - Replicating splice pairs to equalize counts produces many identical examples, increasing gradient correlation and overfitting risk without adding information.
  - It also inflates epoch length, hurting wall-clock without improving coverage. Prefer sampling with weights over deterministic replication.
- Fixed placement s=S−K across all documents reduces absolute-position diversity.
  - While coverage_balanced ensures uniform coverage over in-document starts t, it pins content to one absolute offset, eliminating invariance benefits along s.
  - Add lightweight offset jitter so each t is seen at a small set of absolute positions without reintroducing heavy offset sweeps.
- Short documents (L < K) are dropped under the current description.
  - This systematically biases selection toward longer docs. When the use case allows, adapt K per-document to K_i = min(K, S, L_i) so shorter docs still contribute.
- Determinism and sharding need a precise mapping.
  - A naive enumerate-then-replicate strategy leads to large lists and awkward rank partitioning. Define a deterministic, seed-based sampler over documents and t that yields disjoint sample streams per rank.
- Edge upsampling via duplication is ad hoc.
  - Prefer continuous per-t weights with weighted sampling over t rather than Bernoulli duplication. This yields smoother coverage near edges without discrete duplicates.
- Memory and I/O considerations.
  - Materializing all selected docs may be fine for N≈10 with L≈50K, but does not scale to very long or many docs. Favor lazy reads or memory-mapped token arrays, with background prefetch.
- Metrics and observability.
  - Without doc_idx tagging in outputs, it’s hard to verify balancing. Carry doc_idx in dataset tags or logs and report per-doc coverage and perplexity distributions.

**Refined Design (Multi-Document)**

- Unify balancing as temperature sampling over document length.
  - Sample documents with probability P(doc_i) ∝ (L_i)^τ where τ∈[0,1]. Special cases: τ=1 → by_coverage; τ=0 → by_document (equal per doc) without hard replication.
  - Expose `balance_mode: by_temperature` and `balance_tau` instead of separate modes whenever possible. Retain legacy modes for clarity but recommend temperature.
- Adaptive content length per document.
  - Use K_i = min(K, S, L_i) when `adaptive_k=True`. This includes short docs without pathological duplication. Keep `min_copy_len` to prune extremely short spans.
- Lightweight absolute-position diversity.
  - Add `offset_jitter` to place each copied span at s' ∈ {S−K_i−j | j∈J} for a small J (e.g., J={0,1,2,3}) selected per example.
  - Use a low-discrepancy sequence or per-epoch seeded permutation for reproducible coverage over s.
- Streaming sampler instead of pair replication.
  - Define an epoch length E (optional). For each sample index n, deterministically map to (doc_idx, t, s) via:
    - Sample doc_idx using an alias table for P(doc)
    - Sample t from {0..L_i−K_i} with stride k_t, either uniformly or with edge-weight w(t)∝(K_i/min(d(t),K_i))^α
    - Compute s=S−K_i and apply `offset_jitter` if enabled
  - Seed the sampler with (global_seed, epoch, rank) so ranks produce disjoint, reproducible sequences.
- Enumeration mode (small N) remains available.
  - For small doc sets, you can still enumerate pairs for auditing. But training defaults should use the streaming sampler to avoid large replicated lists.

**Internal Structure**

The dataset stores `(doc_idx, t, s)` triples instead of `(t, s)` pairs:
- `doc_idx`: which document (0..num_docs-1)
- `t`: in-document start position
- `s`: absolute offset in model frame (always `S - K` in balanced mode)

Pair generation (`_enumerate_multi_doc_pairs`):
1. Deterministic quotas (implemented): compute per‑doc integer quotas from P(doc_i)∝(L_i)^τ and epoch_length using largest‑remainder rounding; for each doc, select t positions by quantile mapping over the per‑doc CDF (edge‑weighted if alpha>0). Concatenate all (doc_idx, t) pairs to a fixed list and rely on `PermutationDataset`/`EpochPermutationDataset` for shuffling across steps and epochs. This avoids per‑sample RNG in the dataset and keeps ordering reproducible.
2. Streaming (alternative): draw (doc_idx, t, s) on the fly using a seeded sampler. This reduces upfront list construction but is not used in the current implementation.
3. If strict equalization is required for analysis, prefer τ≈0 to bias quotas rather than hard replication.

Example construction (`_make_example`):
- Identical to `SpliceDocumentDataset._make_example`, but select `doc = self.docs[doc_idx]`.
- Same loss_mask, segment_ids, attention masking semantics.

**SpliceMultiDocumentLMConfig Class**

Configuration interface for multi-document splice training:

```python
@dataclass(frozen=True)
class SpliceMultiDocumentLMConfig(LMTaskConfig):
    # Base dataset (reused from single-doc config)
    base: Optional[LMMixtureDatasetConfig] = None
    dataset_name: Optional[str] = None

    # Multi-document selection
    num_docs: int = 10
    min_doc_length: Optional[int] = None
    max_doc_length: Optional[int] = None
    doc_select_mode: str = "longest"  # "longest", "shortest", "random", "first"

    # Splice parameters (same as single-doc)
    content_length: Optional[int] = None
    content_stride: int = 1
    offset_stride: int = 1
    content_start_mode: str = "coverage_balanced"
    min_copy_len: int = 2
    alpha: float = 0.0
    adaptive_k: bool = True                  # NEW: clamp K to min(K,S,L_i)
    offset_jitter: int = 0                   # NEW: number of alternate s placements to cycle/sample
    jitter_mode: str = "uniform"              # NEW: "uniform" or "low_discrepancy"

    # Multi-doc balancing
    balance_mode: str = "by_temperature"     # NEW: unify balancing via temperature over lengths
    balance_tau: float = 1.0                 # τ=1→by_coverage, τ≈0→by_document

    # Streaming/epoch sizing
    epoch_length: Optional[int] = None       # If set, draw this many samples per epoch

    # Training config
    restrict_to_training_subset: bool = False
    initial_batch_size: Optional[int] = None
```

**Document Selection Policies**

The `doc_select_mode` parameter controls how to choose `num_docs` documents from candidates that meet `min_doc_length` and `max_doc_length` criteria:

- **`longest`**: Select N longest documents. Useful for studying long-context memorization.
- **`shortest`**: Select N shortest documents. Useful for studying compact memorization.
- **`random`**: Deterministically random selection (seeded from dataset hash + num_docs). Reproducible across runs with same data.
- **`first`**: Select first N documents in cache order. Simple and deterministic.

Selection algorithm (`_select_multiple_docs`):
1. Read `input_ids.offsets` from cache to compute document lengths (no token materialization yet).
2. Filter documents by `min_doc_length <= length <= max_doc_length`.
3. Apply selection policy to choose `num_docs` indices.
4. Materialize tokens only for selected documents.
5. Return `(doc_tokens_list, doc_indices)` for logging and debugging.

**Memory and Performance**

- **Memory**: Prefer adaptive materialization. For typical num_docs≈10, direct materialization is fine; for large L_i or large N, memory‑map token arrays or lazily read spans on demand with a small prefetch buffer.
- **Scaling**: Streaming sampler avoids constructing giant pair lists and scales to large corpora; enumeration remains available for debugging.
- **Determinism**: Seed doc and t samplers from a run key plus `(epoch, rank)` to get reproducible per-rank disjoint samples.
- **Sharding**: Use the same alias table on all ranks but independent, non-overlapping sample indices (e.g., leapfrogging by world_size) to prevent duplicate examples across hosts.

**Configuration Example (YAML)**

```yaml
data:
  type: splice_multi_document
  tokenizer: meta-llama/Meta-Llama-3.1-8B

  # Base dataset
  base:
    type: mixture
    tokenizer: meta-llama/Meta-Llama-3.1-8B
    configs:
      common_pile/wikimedia:
        cache_dir: gs://.../tokenized/common_pile/wikimedia-...
        train_urls: [gs://.../wikimedia_filtered-...]
        format: {text_key: text}
    train_weights: {common_pile/wikimedia: 1.0}
    shuffle: false
    shuffle_per_epoch: true

  dataset_name: common_pile/wikimedia

  # Multi-document selection
  num_docs: 10
  min_doc_length: 10000
  max_doc_length: 50000
  doc_select_mode: longest

  # Splice parameters
  content_length: 4096
  content_stride: 1
  content_start_mode: coverage_balanced
  min_copy_len: 128
  alpha: 0.8

  # Balancing (refined): temperature sampling over lengths
  balance_mode: by_temperature
  balance_tau: 0.7  # 1.0≈length-proportional, 0.0≈equal per doc

  # Optional absolute-position jitter (adds position diversity cheaply)
  offset_jitter: 2
  adaptive_k: true

  shuffle_per_epoch: true  # Infinite streaming
```

**Usage Example (Python/Marin)**

```python
from levanter.data.splice_dataset import SpliceMultiDocumentLMConfig
from levanter.data.text import LMMixtureDatasetConfig, UrlDatasetSourceConfig

# Base dataset config
wikimedia = UrlDatasetSourceConfig(
    cache_dir="gs://.../tokenized/common_pile/wikimedia-...",
    train_urls=["gs://.../wikimedia_filtered-..."],
    format=TextLmDatasetFormat(text_key="text"),
)

base_mixture = LMMixtureDatasetConfig(
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    configs={"common_pile/wikimedia": wikimedia},
    train_weights={"common_pile/wikimedia": 1.0},
    shuffle=False,
    shuffle_per_epoch=True,
)

# Multi-document splice config
data = SpliceMultiDocumentLMConfig(
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
    base=base_mixture,
    dataset_name="common_pile/wikimedia",

    # Select 10 longest documents between 10K-50K tokens
    num_docs=10,
    min_doc_length=10_000,
    max_doc_length=50_000,
    doc_select_mode="longest",

    # Equal examples per document
    balance_mode="by_document",

    # Splice parameters (coverage-balanced, edge upsampling)
    content_length=4096,
    content_stride=1,
    content_start_mode="coverage_balanced",
    min_copy_len=128,
    alpha=0.8,

    shuffle_per_epoch=True,
)

# Use in TrainLmConfig
config = TrainLmConfig(
    data=data,
    model=LlamaConfig(...),
    trainer=TrainerConfig(...),
    optimizer=AdamConfig(...),
)
```

**Debug Output**

When constructing the dataset, it prints diagnostic info:

```
[multi_splice] num_docs=10 doc_lengths=[45231, 43890, 42105, 38234, 35678, 33421, 31209, 29834, 28456, 27891]
               Pos.size=4096 content_length=4096 mode=coverage_balanced
               balance_mode=by_document total_pairs=452310
[multi_splice] pairs_per_doc=[45231, 45231, 45231, 45231, 45231, 45231, 45231, 45231, 45231, 45231]
[multi_splice] Selected doc_idx=12345, length=45231
[multi_splice] Selected doc_idx=23456, length=43890
...
```

With `balance_mode="by_coverage"`:
```
[multi_splice] pairs_per_doc=[45231, 43890, 42105, 38234, 35678, 33421, 31209, 29834, 28456, 27891]
```

With `balance_mode="by_temperature" (tau=0.7)`:
```
[multi_splice] doc_probs ~ length^0.7 normalized
[multi_splice] expected_doc_fraction=[0.44, 0.42, 0.40, ...]  # illustrative
```

**Comparison with Single-Document Splice**

| Aspect | Single-Document | Multi-Document |
|--------|-----------------|----------------|
| Documents | 1 | N (configurable) |
| Selection | `doc_index` or policy (longest/random/first) | `num_docs` + policy + length filters |
| Balancing | N/A | `by_coverage` or `by_document` |
| Example structure | `(t, s)` pairs | `(doc_idx, t, s)` triples |
| Memory | One document's tokens (~50-100KB) | N documents' tokens (~500KB-5MB) |
| Use case | Single-doc memorization, position invariance | Multi-doc memorization, scaling studies |

**Worked Example: by_document Balancing**

Setup: `num_docs=3`, `S=8`, `K=4`, `content_stride=1`, `balance_mode="by_document"`.

Documents:
- Doc 0: length 7, tokens `[A,B,C,D,E,F,G]`
- Doc 1: length 5, tokens `[H,I,J,K,L]`
- Doc 2: length 3, tokens `[X,Y,Z]`

Base pair generation (coverage-balanced, s_fixed = S - K = 4):
- Doc 0: t ∈ [0..7-4] = [0..3] → 4 pairs: `[(0,0,4), (0,1,4), (0,2,4), (0,3,4)]`
- Doc 1: t ∈ [0..5-4] = [0..1] → 2 pairs: `[(1,0,4), (1,1,4)]`
- Doc 2: t ∈ [0..3-4] = invalid (L < K) → 0 pairs

After balancing (max_pairs=4):
- Doc 0: keep 4 pairs as-is
- Doc 1: replicate 2 pairs → `[(1,0,4), (1,1,4), (1,0,4), (1,1,4)]` (4 total)
- Doc 2: skipped (no valid pairs)

Total: 8 pairs per epoch, evenly split between Doc 0 and Doc 1.

Example outputs for Doc 1 pairs (tokens at s=4):
- `(doc=1, t=0, s=4)`: `[Null,Null,Null,Null, H,I,J,K]`
- `(doc=1, t=1, s=4)`: `[Null,Null,Null,Null, I,J,K,L]`

Loss mask: `[0,0,0,0, 1,1,1,0]` (predict within doc, exclude last token).
Segment IDs: `[0,0,0,0, 1,1,1,1]` (prevent attention to prefix padding).

**Validation and Testing**

Recommended tests for multi-document splice:
1. **Unit test: pair generation**: With 3 tiny docs, verify `(doc_idx, t, s)` triples match expectations for both balance modes.
2. **Unit test: balancing**: Confirm `by_document` mode equalizes pair counts; `by_coverage` mode respects length proportions.
3. **Unit test: example construction**: Verify tokens, loss_mask, segment_ids for each doc_idx.
4. **Integration test**: Train tiny model on 3 documents for 100 steps, verify loss decreases and documents are distinguishable.
5. **Smoke test: document selection**: With mock cache, verify `longest`, `shortest`, `random`, `first` policies select correct documents.

**Limitations and Future Extensions**

Current design limitations:
- **Memory**: All selected documents loaded into RAM. For 1000s of documents, consider lazy loading.
- **No per-document weights**: All documents treated equally (modulo balancing). Could add `doc_weights: List[float]` parameter.
- **No document-aware callbacks**: No built-in per-document perplexity tracking. Could extend evaluation to report metrics per doc_idx.
- **No dynamic document sets**: Document selection is fixed at dataset construction. Could add document curriculum (add docs over time).

Potential extensions:
- **Per-document weights**: Custom importance weights in mixture (e.g., `[1.0, 0.5, 2.0]` for 3 docs).
- **Document-aware evaluation**: Track memorization P(z) per document, report which docs are learned fastest.
- **Lazy loading**: Stream documents from cache per batch for very large N.
- **Document augmentation**: Different content_stride or alpha per document.
- **Curriculum learning**: Start with 1 document, add 1 every K steps until reaching N.
- **Hierarchical balancing**: Group documents by metadata (e.g., source, length bin) and balance across groups.

**Integration with Existing Splice Infrastructure**

The multi-document classes extend (not replace) the single-document classes:
- `SpliceDocumentDataset` remains unchanged; used for single-doc experiments.
- `MultiSpliceDocumentDataset` reuses the same splice construction logic (`_make_example`).
- Both configs implement `LMTaskConfig` interface; compatible with Levanter/Marin training pipelines.
- Evaluation callbacks (P(z), visualizations) work with multi-doc datasets (operate per example, agnostic to source document).

**Performance Characteristics**

Expected throughput (relative to standard Levanter training):
- **By coverage mode**: Similar to standard training (total examples ~ sum of doc lengths).
- **By temperature mode**: Controls concentration on long docs smoothly without hard replication; epoch length is explicit via `epoch_length` or trainer schedule.
- **Shuffling**: With `shuffle_per_epoch=True` and `EpochPermutationDataset`, examples permuted across all documents each epoch.
- **Sharding**: Compatible with Levanter's data sharding (each worker enumerates full pair set, samples subset by rank).

**Recommended Defaults**

For most use cases:
- `num_docs`: 10 (enough for multi-doc memorization studies without excessive memory)
- `min_doc_length`: 4096 (ensures at least one full-context example per document)
- `max_doc_length`: 100000 (avoid extremely long documents that dominate training)
- `doc_select_mode`: "longest" (study long-context behavior) or "random" (representative sample)
- `balance_mode`: "by_temperature", `balance_tau`: 0.7 (softly downweights very long docs)
- `content_length`: Same as `Pos.size` (maximize supervision per example)
- `content_stride`: 1 (dense coverage) or 10 (faster epochs)
- `alpha`: 0.8 (moderate edge upsampling)

Add when multiple absolute positions are relevant:
- `offset_jitter`: 2–4 (adds s diversity at low cost)
- `adaptive_k`: true (include short docs, prune with `min_copy_len`)

**Summary**

The multi-document splice extension enables training on N documents with controlled balancing:
- **Core classes**: `MultiSpliceDocumentDataset` and `SpliceMultiDocumentLMConfig`.
- **Key feature**: `balance_mode` controls whether examples are proportional to document length (`by_coverage`) or equal per document (`by_document`).
- **Document selection**: Flexible policies (`longest`, `shortest`, `random`, `first`) with length filtering.
- **Compatibility**: Extends existing single-doc splice without breaking changes; integrates with Levanter/Marin infrastructure.
- **Use cases**: Multi-document memorization, scaling laws, domain adaptation, curriculum learning.

This design maintains the splice dataset's core benefits (position invariance, segment-aware attention, explicit loss masking) while scaling to multiple documents with configurable balancing semantics.
