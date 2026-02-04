# KV Cache in Levanter Inference: How It Works Today, Why It Slows Down, and How to Restructure It

This document is a "from-zero" explanation of how Levanter's inference-time KV caching works, as implemented today.
It is intentionally detailed so a newcomer can follow the full dataflow and understand how it relates to:

- **Sequence length scaling** (why decode gets slower as generation proceeds)
- **Out-of-memory / resource exhaustion failures** (HBM, TPU `vmem`, allocator exhaustion)
- The suspicion that "**KV cache cleanup is broken**" (what "cleanup" means here, what exists, and what is not wired up)

Code references (current implementation):

- `src/levanter/inference/engine.py` (host orchestration, JIT entrypoints)
- `src/levanter/inference/jit_scheduler.py` (DecodeState, SequenceTable, TokenQueue, cleanup primitives)
- `src/levanter/inference/page_table.py` (PageTable + PageBatchInfo wire format)
- `src/levanter/layers/kv_cache.py` (PageCache/KvPageCache storage + update kernel)
- `src/levanter/layers/attention.py` (paged attention; TPU kernel vs reference fallback)
- `src/levanter/models/llama.py` (how model decode uses paged_decode; representative of other models)

---

## 1. What "KV Cache" Means (Transformer Inference 101)

For autoregressive generation, at step `t` the model predicts token `x[t]` using all earlier tokens `x[0:t-1]`.
In a Transformer, that means each layer computes attention where the **query** is the current token's representation
and the **keys/values** are the representations of all previous tokens.

If we naively re-ran the full forward pass over the entire prefix for every new token, we would do `O(t^2)` work.
The standard trick is to **cache** each layer's computed keys/values for previous tokens:

- After processing token `t`, store its per-layer **K[t]** and **V[t]**
- When processing token `t+1`, reuse cached K/V for `0..t`

This reduces compute per token from "recompute the whole prefix" to "compute Q/K/V for the new token, then attend
to the cached K/V".

However:

- You still have to attend over all prior tokens.
- So even with KV cache, **cost per token grows with sequence length**.

That "growing per token cost" is the core reason throughput drops as sequences get longer (more on that later).

---

## 2. Why Levanter Uses a *Paged* KV Cache

Levanter does not store KV as a simple dense tensor like `[batch, max_seq_len, heads, head_dim]` for decode.
Instead it uses a **paged** representation:

- KV memory is a global pool split into fixed-size **pages**.
- Each page holds KV for `page_size` tokens.
- Each sequence is assigned a list of page IDs (`page_indices`) that represent its KV storage.

This is conceptually similar to a paged virtual memory system:

- A sequence grows over time (unknown final length).
- Pages are allocated as needed when the sequence crosses a page boundary.
- Pages can be shared between sequences in special cases (cloning).

Main motivations:

1. **Dynamic lengths**: don't pay for `max_seq_len` per sequence up front.
2. **Cloning support**: share full pages across "n_generations" clones cheaply.
3. **Kernel-friendly layout**: attention kernels can iterate pages rather than require a dense `[seq_len]` axis.

---

## 3. The Key Data Structures (Device-Side vs Host-Side)

There are several layers of state. The confusion usually comes from mixing:

- **Metadata** (which page belongs to which sequence, what length is each sequence)
- **Storage** (the actual KV tensors)
- **Scheduling** (which tokens are next to be decoded)
- **Host bookkeeping** (request IDs, mapping local slot -> request choice, result accumulation)

### 3.1 Device-side storage: `PageCache` / `KvPageCache`

File: `src/levanter/layers/kv_cache.py`

`KvPageCache` is the physical KV storage:

- `KvPageCache.kv_pages: NamedArray  # [Page, Slot, 2 * KVHeads, Embed]`

Important details:

- **"Page" axis** is the global page pool (`num_pages`).
- **"Slot" axis** is the position within a page (`page_size` tokens).
- Heads are interleaved as `(k0, v0, k1, v1, ...)` (see `_interleave_kv`).
- This cache is per-layer; multi-layer models use `ListCache[KvPageCache]` (one per layer).

`KvPageCache.update(batch_info, new_k, new_v)` writes new token KV into the correct `[page, slot]` locations.

How it decides where to write:

- `batch_info.new_token_dests` encodes a "linear destination" per token: `dest = page * page_size + slot`.
- `PageBatchInfo.pages_and_slots()` converts those linear destinations into `(t_pages, t_slots)` arrays.
- `kv_update_unified_prefix` then iterates `i=0..K-1` and does `dynamic_update_slice` into the KV buffer.

This is correct but not particularly efficient (it's essentially a per-token loop).
The attention kernel dominates runtime for long sequences, but KV updates can still matter.

### 3.2 Device-side page allocator: `PageTable`

File: `src/levanter/inference/page_table.py`

`PageTable` is the allocator's global state:

- It stores only `page_ref_counts: i32[Page]`
- A page is considered **free** if `ref_count == 0`.

It does **not** store per-sequence page lists; those live in `SequenceTable`.

Allocation policy (implemented in `SequenceTable.allocate_for_seq`):

- If any `ref_count == 0`, pick a free page (currently via `argmin(ref_counts)`; with free pages, that selects a 0).
- Increment refcount and assign that page to the sequence's `page_indices`.

### 3.3 Device-side per-sequence metadata: `SequenceTable`

File: `src/levanter/inference/jit_scheduler.py`

`SequenceTable` holds per-(local) sequence slot metadata:

- `seq_lens: i32[Seq]` - current logical length per local slot
- `page_indices: i32[Seq, PagePerSeq]` - mapping from sequence -> page list
- `kv_pages: i32[Seq, PagePerSeq]` - currently redundant mirror of `page_indices` (see below)
- `clone_sources: i32[Seq]` - clone parent (or INVALID)
- `used_mask: bool[Seq]` - whether this slot is active
- `page_size` - static

Conceptually:

- A "sequence" is a **local slot ID** (`slot_id`), `0..max_seqs-1`.
- Each local slot can be assigned to a request/generation choice on the host.
- A local slot "owns" a set of pages in the KV cache.

The most important method is:

- `SequenceTable.allocate_for_seq(page_table, token_slot_ids, token_pos_ids) -> (new_seq_table, new_page_table, PageBatchInfo)`

This is how Levanter bridges from "we are about to decode these tokens" to "ensure pages exist and compute write destinations".

Key invariant:

- For every token with `(slot_id=s, pos_id=p)`:
  - It must be assigned to some page `page_indices[s, p // page_size]`
  - It will write into `slot = p % page_size`

### 3.4 Device-side decode state: `DecodeState`

File: `src/levanter/inference/jit_scheduler.py`

`DecodeState` combines:

- `sequences: SequenceTable`
- `page_table: PageTable`
- Buffers for tokens/logprobs: `tokens[seq, position]`, `logprobs[seq, position]`
- Per-seq decoding params: `max_num_tokens`, `stop_tokens`, `temperature`, `prng_keys`
- The work queue: `tqueue: TokenQueue`
- A cached finished flag: `finished[seq]`

So DecodeState is the "single source of truth" for:

- which sequences exist
- how long they are
- what pages they use
- what token should be decoded next

It also contains convenience operations:

- `reserve_slot`, `assign_seq` (host-driven initial admission)
- `allocate_for_seq` (ensure pages and produce `PageBatchInfo`)
- `update_tokens` (append new sampled tokens, mark finished, enqueue next decode work)
- `clone_pages_from` and free methods

### 3.5 Device-side scheduling: `TokenQueue`

File: `src/levanter/inference/jit_scheduler.py`

`TokenQueue` is a "flat" queue of tokens to be decoded, not grouped by sequence.
It stores parallel arrays:

- `queued_tokens[position]`
- `queued_slot_ids[position]`
- `queued_pos_ids[position]` (absolute position in the sequence buffer)
- `num_queued_tokens`

Decode loop uses:

- `pack_next_sequence(max_tokens)` which:
  - takes the first `num=min(num_queued_tokens, max_tokens)` entries
  - shifts the remaining queue entries left ("roll")
  - sorts the packed slice by slot_id (so slot IDs are contiguous)
  - returns a `PackedSequence(tokens, slot_ids, pos_ids, num_tokens)`

That "sorted by slot_id" property is relied upon by `SequenceTable.allocate_for_seq` which assumes tokens for a given
slot are contiguous.

In the current design, in steady-state decode there is typically **one queued token per active sequence**.
(The queue exists largely to make it JIT-friendly to pack and process a bounded number of tokens per iteration.)

### 3.6 Host-side orchestration: `InferenceEngine`

File: `src/levanter/inference/engine.py`

Host responsibilities:

- Map user requests -> local slots (`self.free_slots`, `self.local_map`, `self.sequences`)
- Build "work packets" for the device:
  - `PrefillWork` for prefill (prompt ingestion)
- Run the JIT functions:
  - `_run_prefill(gen_state, model, sampler, work, max_seqs_in_prefill)`
  - `_run_generation_loop(gen_state, model, sampler, max_tokens_per_round, max_rounds)`
- Pull `_DecodeOutputs` back to host and assemble results

There is a `GenState` container holding:

- `cache: PageCache`
- `decode_state: DecodeState`

---

## 4. The Full Dataflow: Init -> Prefill -> Decode -> Outputs

This is the most important mental model. I'll describe the "happy path" with one request, then note batching/clones.

### 4.1 Initialization

`InferenceEngine.from_model_with_config(...)` does:

1. Compute `max_pages` if not provided (budgeting via free HBM; see `_infer_max_pages_from_hbm` in `engine.py`).
2. Create `PageTable`:
   - `PageTable.init(max_pages, max_seqs, page_size, max_pages_per_seq)`
3. Create model cache:
   - `cache = named_jit(model.initial_cache)(table.spec(), dtype=compute_dtype)`
4. Create empty `DecodeState`:
   - `DecodeState.init(table, max_stop_seqs, max_stop_tokens, max_queued_tokens)`

From now on, KV cache memory is allocated and lives on device.

### 4.2 Prefill ("ingest the prompt")

Host builds `PrefillWork`:

- `PrefillWork.queue`: a *local* TokenQueue containing **prompt tokens** with:
  - `queued_tokens = prompt tokens`
  - `queued_slot_ids = chosen slot_id repeated`
  - `queued_pos_ids = 0..len(prompt)-1`
- `new_slot_ids`: list of slots to create
- `prompt_tokens`: full `[max_slots, max_seq_len]` prompt buffer used by `assign_seq`
- `prompt_lengths`: prompt length per slot
- `seq_params`: per-slot decoding params

Device executes `_run_prefill`:

1. `_apply_prefill_work`:
   - `reserve_slot(slot_id)`
   - `assign_seq(local_slot_id, tokens=prompt_tokens[i], seq_len=prompt_len, kv_pages=None, ...)`
   - this sets:
     - `DecodeState.tokens[slot, 0:prompt_len] = prompt tokens`
     - `DecodeState.sequences.seq_lens[slot] = prompt_len`
     - `used_mask[slot] = True`
     - stop tokens, temps, prng key, etc.
   - clones (`n_generations > 1`) use `GenState.clone_sequence(...)` which:
     - copies prompt tokens
     - shares page indices
     - adjusts page refcounts via `clone_pages_from`
     - copies the last partial KV page if necessary

2. `_prefill_kernel`:
   - reads prompt tokens from `work.queue`
   - calls `decode_state.allocate_for_seq(token_slot_ids, token_pos_ids)`:
     - ensures page_indices exist for every token position
     - builds `PageBatchInfo` describing:
       - which sequences are in this batch (`slot_ids`)
       - their page lists (`page_indices`)
       - their lengths (`seq_lens`)
       - where each token should write its KV (`new_token_dests`)
       - cumulative query lengths (`cu_q_lens`)
   - calls `model.decode(tokens, cache, batch_info, pos_ids)`
     - attention layer computes q/k/v for each token
     - `kv_cache.update(batch_info, k, v)` writes KV into the proper pages
     - attention reads KV back (paged attention) and produces outputs/logits
   - samples the "next token" at prompt boundaries (one per sequence)
   - `decode_state.update_tokens(new_tokens, new_slot_ids, log_probs, num_new_tokens)`:
     - appends the sampled token to the per-seq token buffer
     - increments `seq_lens`
     - sets `finished` if `max_num_tokens` or stop sequence reached
     - enqueues that sampled token into the *main* `DecodeState.tqueue`
       - except if it finished immediately (then it is purged and not enqueued)
   - returns a `_DecodeOutputs` containing those sampled tokens for host extraction

At the end of prefill:

- KV cache contains prompt KV
- Decode queue contains the first sampled continuation token (per active sequence)
- Host has the first generated token(s) available for output

### 4.3 Decode loop ("generate until done")

Device executes `_run_generation_loop`, a `jax.lax.while_loop`:

Each iteration:

1. `decode_state.pack_next_sequence(max_tokens_per_round)`:
   - take up to `max_tokens_per_round` queued tokens
   - sort by slot_id so each slot's tokens are contiguous

2. `decode_state.allocate_for_seq(token_slot_ids=slot_ids, token_pos_ids=pos_ids)`:
   - This is crucial:
   - Those queued tokens have absolute `pos_id` values; the sequence length must be at least `pos_id+1`.
   - It allocates **new pages** when tokens cross page boundaries.
   - It builds `PageBatchInfo` for the batched tokens.

3. `model.decode(tokens, cache, batch_info, pos_ids)`:
   - computes q/k/v for those tokens
   - writes KV for those tokens into the paged cache
   - attends to the cached prefix via `ragged_paged_attention(...)`
   - returns logits and updated cache

4. Sample next tokens:
   - `_compute_sample_indices` selects the last token per sequence inside this packed batch.
   - For each sequence, sample a new token from its boundary logits.

5. `decode_state.update_tokens(...)`:
   - append new tokens
   - mark finished if needed
   - enqueue the newly sampled tokens back to the TokenQueue

6. Append tokens to `_DecodeOutputs` for host.

Stop condition:

- stop after `max_rounds`, or queue empty, or all sequences finished.

Host calls `jax.device_get` on the returned `_DecodeOutputs`, appends tokens into Python results.

---

## 5. How the "Paged" Addressing Works (The Core Mapping)

Everything in paged KV caching reduces to a single mapping:

For a token in sequence slot `s` at absolute position `p`:

1. Compute the per-sequence page index:
   - `page_idx = p // page_size`
2. Compute within-page offset:
   - `slot = p % page_size`
3. Look up the global page ID:
   - `page = page_indices[s, page_idx]`
4. KV memory location:
   - `kv_pages[page, slot, ...]`

`SequenceTable.allocate_for_seq` is responsible for ensuring that:

- `page_indices[s, page_idx]` is valid (allocating if needed)
- `batch_info.new_token_dests` encodes the `(page, slot)` destination for every token

`KvPageCache.update` is responsible for writing new K/V values to those destinations.

`ragged_paged_attention` is responsible for reading those pages and masking out tokens beyond `kv_len`.

---

## 5.1 A Toy Example (Page Mapping in Practice)

This example ignores heads/values and focuses only on page mapping.

Assume:

- `page_size = 4` (each page stores KV for 4 token positions)
- Global page pool has pages `0..9`
- One active slot: `slot_id = 3`
- Sequence positions are 0-indexed

### The sequence grows

If the sequence has length 6 (positions `0..5`), it needs:

- `page_idx 0` -> positions `0..3`
- `page_idx 1` -> positions `4..7` (but only `4..5` are valid so far)

So `SequenceTable.page_indices[3]` might look like:

```text
page_indices[slot=3] = [2, 7, INVALID, INVALID, ...]
```

Meaning:

- positions `0..3` live in global page `2`
- positions `4..7` live in global page `7`

### Writing KV for token at position 5

Token `(slot_id=3, pos_id=5)` maps to:

- `page_idx = 5 // 4 = 1`
- `slot = 5 % 4 = 1`
- `page = page_indices[3, 1] = 7`

So KV writes go to:

```text
kv_pages[page=7, slot=1, ...] = (K, V)
```

### Attention masking prevents stale reads

The last page (`page=7`) has slots `0..3`.
If we've only generated up to position 5 (`kv_len = 6`), then slots corresponding to positions 6 and 7 are invalid and
must be masked out by attention (`kv_tok < kv_len`). That's why pages can be reused safely without zeroing: stale KV in
the masked region does not affect results.

---

## 5.2 How `SequenceTable.allocate_for_seq` Actually Works (Detailed)

File: `src/levanter/inference/jit_scheduler.py`, method:

- `SequenceTable.allocate_for_seq(...)`

This is the "allocator + batch planner" and is the heart of paged caching.

Inputs:

- `page_table.page_ref_counts[Page]`
- `self.page_indices[Seq, PagePerSeq]`
- `token_slot_ids[position]` and `token_pos_ids[position]` describing the tokens we are about to decode

Outputs:

- updated `SequenceTable` (seq lengths and page_indices may change)
- updated `PageTable` (refcounts incremented when new pages allocated)
- `PageBatchInfo` describing the batch to the model/attention code

The algorithm in phases:

### Phase A: normalize and group tokens by slot

1. Replace INVALID slot IDs with a sentinel "out of range" so segmentation ops won't crash:
   - `token_slot_ids = where(token_slot_ids < 0, max_seqs, token_slot_ids)`
2. Compute `unique_ids` and `dense_ids` via `get_unique_in_order(...)`
   - goal: map arbitrary slot IDs into a dense 0..K-1 index space so `segment_sum/max` work
3. Compute per-slot counts and max position:
   - `segment_lengths[seq] = count(tokens for that seq)`
   - `max_pos_per_seq[seq] = max(token_pos_ids for that seq)`

Key invariant:

- `token_slot_ids` must have tokens for a given slot be *contiguous* (the queue sorts by slot id for this reason).

### Phase B: compute new sequence lengths

4. Determine which "updated slots" are valid:
   - `valid_updated = is_valid(updated_seqs) & (updated_seqs < max_seqs)`
5. For each updated slot, compute the implied new length:
   - `masked_seq_len = (max_pos + 1)` for active slots
6. Update `seq_lens`:
   - `new_lens[slot] = max(old_len, masked_seq_len)`

This is subtle but crucial:

- The *token positions* we're decoding define how long the sequence must be.
- For causal attention, the token at position `p` must see KV length at least `p+1` (to include itself).

### Phase C: determine how many pages are needed and allocate new ones

7. Compute "how many pages are needed" given the new length:
   - `new_num_pages_needed = ceil(new_len / page_size)`
8. Compute "how many pages are already allocated" by counting valid `page_indices` entries:
   - `old_num_pages_needed = sum(is_valid(page_indices[slot, :]))`
9. For each updated slot, allocate pages from `[old_needed .. new_needed)`:
   - For each page to allocate:
     - check there exists `ref_counts == 0`
     - choose a free page (currently `argmin(ref_counts)`; with free pages, that returns an index with refcount 0)
     - increment that refcount
     - write it into `page_indices[slot, page_idx]`

Important properties:

- Allocation happens only when crossing page boundaries.
- Allocation is deterministic given identical refcounts and identical order of updated slots.
- Failure mode is "no free pages", which raises via `eqx.error_if`.

### Phase D: build `PageBatchInfo` (the interface to attention/cache update)

`PageBatchInfo` fields:

- `slot_ids[seq]`: which slots are present in this batch (packed to length `max_seqs`, with INVALID padding)
- `page_indices[seq, page]`: gathered rows of `SequenceTable.page_indices` for those slots
- `seq_lens[seq]`: the corresponding lengths (kv_lens)
- `cu_q_lens[seq]`: cumulative query lengths (prefix sums of how many query tokens each seq has in this batch)
- `num_seqs`: number of valid seqs in this batch
- `new_token_dests[position]`: for each query token, the linear destination `page*page_size + slot`

How `new_token_dests` is computed:

- For each token `(slot_id, pos_id)`:
  - compute `page_idx = pos_id // page_size`
  - lookup `page = page_indices[slot_id, page_idx]`
  - compute `dest = page * page_size + (pos_id % page_size)`
  - if anything is invalid, set `dest=INVALID`

This `PageBatchInfo` is exactly what `KvPageCache.update(...)` and `ragged_paged_attention(...)` consume.

---

## 6. Clones (`n_generations > 1`): Refcounts and Partial Pages

Levanter supports generating multiple samples from the same prompt ("clones") efficiently.

Problem:

- If you have a prompt of length `L`, all clones share the same prefix KV.
- If you duplicate KV buffers naively, you multiply memory by `n_generations`.

Paged caching makes sharing possible:

- Fully filled pages can be **shared** safely (immutable once full).
- The last partial page cannot be shared if clones diverge (they will write different tokens into it).

Implementation details:

- `GenState.clone_sequence` in `engine.py`:
  - reserves a new slot for the child
  - assigns child's tokens/seq_len/page_indices to match parent initially
  - records `clone_sources[child] = parent`
  - calls `decode_state.clone_pages_from(parent, child)`:
    - increments refcounts for shared pages
    - if the last page is partial, allocate a new page for the child's last page
  - if last page is partial, physically copy its KV content:
    - `cache.copy_page(src_page, dst_page)`

This design implies a "page immutability contract":

- Pages that are fully filled are never written again for that sequence.
- Therefore sharing full pages is safe.

The contract is enforced by the logic in `clone_pages_from`:

- It shares all pages up to (but not including) the last partial page.
- It allocates a unique page for the last partial page.

---

## 7. What "Cleanup" Means, and Why It's Fair to Say It's "Broken"

When someone says "cleanup of the KV cache is broken", they usually mean one of these:

1. **Logical cleanup (allocator)**:
   - finished sequences should have their pages returned to the free pool
   - refcounts should be decremented
   - local slots should be marked unused
2. **Physical cleanup (memory overwrite / zeroing)**:
   - overwrite KV pages with zeros so old data cannot affect future runs
   - (usually not necessary if masking and metadata are correct)

In Levanter today:

- The *primitives for logical cleanup exist*:
  - `SequenceTable.free_pages(seq_id)` and `free_pages_for_finished(finished_mask)`
  - `DecodeState.free_pages_for_finished(finished_mask)`
- But **InferenceEngine does not call them during generation**.
  - In `engine.py`, there are comments "Device-side release of finished sequences", but no code.
  - `_ingest_outputs` claims to "apply host-side release" but does not.

Instead, `InferenceEngine.generate(...)` currently does a full reset at the start:

- `self.reset()` calls a jitted `GenState.reset()` which does:
  - `cache.reset()` -> `jnp.zeros_like(...)` over the entire KV cache
  - `decode_state.reset()` -> resets PageTable refcounts and all metadata

That means:

- **Within a single `generate(...)` call**, pages are never freed as sequences finish.
  - This is okay if you don't need to reuse pages until the end.
  - It becomes a real issue if you want streaming admission or very large prompt sets.
- **Across calls**, the engine "cleans up" by doing the most expensive thing possible: zeroing the entire KV cache.

Why this matters:

- If you want to run many prompts sequentially (like `sample_lm_multihost.py` does), you pay the cost of zeroing
  a gigantic KV buffer each time.
- If you want a long-lived inference service, you can't just "free finished sequences and keep going" efficiently;
  you have to reset, which defeats the purpose of a paged allocator.

So "cleanup is broken" is accurate in the sense that:

- There is a page allocator and free logic
- It is not wired into the normal execution path
- The current "cleanup" strategy is a brute-force full reset

---

## 8. Why Decode Gets Slower Over Time (Sequence Length Scaling)

Even with perfect cleanup, **a single long sequence will slow down as it grows**.
This is inherent to attention: each new token must attend over more cached tokens.

Where the scaling comes from in code:

- Attention uses `ragged_paged_attention(...)` in `src/levanter/layers/attention.py`.
- The relevant inputs are:
  - `kv_lens` (per-seq length)
  - `page_indices` (which pages belong to the sequence)
  - `kv_pages` (the storage)

As `kv_lens` increases:

- The kernel reads more KV positions.
- The number of page blocks processed increases roughly as `kv_lens / page_size`.
- So per-token time grows ~linearly in `kv_lens` (ignoring caching/tiling effects).

Important nuance: TPU backend choice changes *how bad* the scaling looks.

### 8.1 TPU kernel vs reference fallback

`ragged_paged_attention` has two paths:

1. TPU kernel (`tpu_ragged_paged_attention`) when `use_tpu_ragged_paged_attention=True`
2. Reference implementation (`default_ragged_paged_attention`) otherwise

The reference implementation is explicitly labeled "not optimized".
It uses nested `fori_loop`s over sequences, query blocks, and KV blocks.
It will show a noticeable throughput decline as sequences get longer.

This matches the "decode iter tok/s decreases steadily as we generate more tokens" logs.

### 8.2 The decode loop structure also amplifies the appearance of slowdown

In `InferenceEngine.generate`, each "Decode iter":

- launches `_run_generation_loop(...)` (async)
- then does `jax.device_get(decode_outputs)` in `_extract_outputs`, which blocks until compute is done

So the logged "extract time" largely includes device compute time.
This can make it look like "host is getting slower", but it's just where the synchronization happens.

This does not cause the slowdown; it mainly affects how we *observe* it in logs.

---

## 9. How This Relates to OOM / RESOURCE_EXHAUSTED Failures

There are multiple "OOM-like" failure modes here, and it helps to separate them.

### 9.1 HBM OOM (device memory for KV cache)

HBM is where the KV cache lives.
HBM pressure comes from:

- `max_pages` (global page pool size)
- Per-page KV size (model-dependent: layers, kv_heads, head_dim, dtype)

`InferenceEngineConfig.max_seq_len` affects:

- `pages_per_seq = ceil(max_seq_len / page_size)`
- The shape of per-seq metadata arrays (`page_indices[seq, pages_per_seq]`)
- The max length of `DecodeState.tokens` buffer (`page_size * pages_per_seq`)

But HBM KV storage size is mostly controlled by `max_pages`:

- Bigger `max_pages` -> bigger `kv_pages` tensor -> more HBM used immediately at engine creation.

If cleanup (freeing pages) is missing in a long-lived engine, you can also hit allocator exhaustion:

- pages are "in use" (refcount > 0)
- new sequences can't allocate pages -> `Out of free pages during allocation`

This is not always reported as a clean Python traceback in multi-host settings; it can surface as a generic failure.

### 9.2 TPU `vmem` OOM (kernel scratch memory)

TPU has a small per-core "vmem" scratch space (~16MB in the reported failures).
Some kernels allocate scratch buffers there.

`tpu_ragged_paged_attention` can fail with:

- `RESOURCE_EXHAUSTED: Ran out of memory in memory space vmem ... allocating on stack for ragged_paged_attention`

This is **not** fixed by "cleanup" because it is scratch allocation, not persistent KV storage.

The practical consequence:

- If you disable TPU ragged paged attention to avoid `vmem` OOM, you fall back to the reference implementation.
- The reference implementation is slower and shows stronger throughput degradation with increasing sequence length.

That ties the OOM story directly to the "speed decreases over time" story:

- Avoiding the kernel OOM currently means accepting a much slower attention path for long sequences.

---

## 10. Suggested Simplifications and Refactors (With Tradeoffs)

This section is intentionally opinionated: it's a set of changes that would make this subsystem easier to reason about,
enable proper cleanup, and reduce accidental costs (like resetting a giant KV buffer for every prompt).

I'm grouping suggestions by scope.

### 10.1 Wire up logical cleanup (minimal correctness improvement)

Goal:

- When a sequence is finished, free its pages and return its slot to the free pool *without resetting the whole engine*.

What exists already:

- `DecodeState.free_pages_for_finished(finished_mask)` (device-side)
- `TokenQueue.purge_queue_of_slot(slot_id)` (device-side)
- Host has `free_slots` tracking local slot IDs.

What's missing:

- `engine.py` never calls the "free pages for finished" path.
- Host never reclaims `free_slots` based on device-side slot release.

Refactor direction:

- Add a post-step cleanup call in `_run_generation_loop` or right after it:
  - Identify which slots finished in this step (from `decode_state.finished` snapshot).
  - Purge any queued tokens for them (defensive).
  - Call `decode_state.free_pages_for_finished(mask)` so refcounts decrement and slots are released.

Tradeoffs:

- Requires care with clones: refcounts must remain correct (the current implementation supports this).
- Adds additional device work per iteration (but likely small compared to attention).
- Changes semantics: "finished sequences remain in state until reset" would become "finished sequences disappear".
  - Host mapping must handle this (it already tracks `dr.done`, so it's plausible).

### 10.2 Stop zeroing the entire KV cache on reset (big performance win for sequential prompts)

Today:

- `PageCache.reset()` calls `jnp.zeros_like(...)` for every KV page in every layer.
- For large models and large `max_pages`, this is a very large device write.

But correctness does not require zeroing if:

- When a slot is released, its `page_indices` are set to INVALID.
- When a page is reused, attention only considers tokens `< kv_len` (it does).

Refactor direction:

- Redefine "reset" into two operations:
  1. **Logical reset**:
     - reset `PageTable.page_ref_counts` to 0
     - clear `SequenceTable` metadata (`used_mask=False`, `seq_lens=0/INVALID`, page_indices=INVALID)
     - clear TokenQueue
     - (optionally) clear per-seq token/logprob buffers
  2. **Physical zero** (optional debug):
     - a separate method `zero_cache()` that explicitly zeros KV pages

Tradeoffs:

- Debuggability: zeroing can help catch bugs where you accidentally attend to stale KV.
- Security: zeroing prevents old data from lingering (rarely relevant in typical ML infra).
- Performance: skipping zeroing is a large win.

Given the "attention masks by kv_len" invariant, logical reset should be safe and much faster.

### 10.3 Remove redundant state (`SequenceTable.kv_pages` vs `page_indices`)

`SequenceTable` stores both:

- `page_indices[seq, page]`
- `kv_pages[seq, page]`

and `allocate_for_seq` sets `kv_pages = page_indices`.

If no code requires them to diverge (it doesn't appear so), this is pure duplication:

- extra memory
- extra state to keep consistent
- more surface area for subtle bugs

Refactor direction:

- Remove `kv_pages` from `SequenceTable` and use `page_indices` everywhere.

Tradeoffs:

- Small API churn: update call sites that refer to `decode_state.kv_pages`.
- Simpler invariants: one canonical mapping.

### 10.4 Simplify scheduling: consider eliminating `TokenQueue` for standard decode

Current design is a generalized "token work queue".
For typical autoregressive decode with 1 token/sequence/step, this is more machinery than needed.

Alternative:

- Represent the "next decode token" as `tokens[seq, seq_lens-1]` for each active seq.
- Each iteration, decode exactly one token per active sequence (or a fixed small block).

Benefits:

- Fewer moving parts (no rolling queue, no sorting)
- Easier to reason about "what is active"

Costs:

- Less flexibility if you want to:
  - interleave prefill and decode
  - process variable numbers of tokens per sequence per step
  - do "burst" decode for some sequences but not others

If the near-term goal is "robust and fast long-seq sampling", simplifying scheduling may reduce bug surface area.

### 10.5 Performance-focused refactors (bigger projects)

If you need 4096-token generation to be fast, the main lever is attention.

Options:

1. Fix/replace the TPU ragged paged attention kernel that triggers `vmem` OOM.
2. Implement a more efficient non-kernel fallback than `default_ragged_paged_attention`.
3. Use a different backend (flash attention variants) with a dense KV layout for small batch sizes.

Tradeoffs:

- Kernel work is high effort but gives the best perf.
- Better fallback can be moderate effort and improve "safe path" performance.
- Dense KV can be simpler but may blow up memory for large batch sizes.

---

## 11. Concrete Debug/Validation Ideas (To Confirm or Refute "Cleanup Is Broken")

If we want to validate the senior engineer's hypothesis ("cleanup broken") against reality, here are concrete checks:

1. **Track free pages over time** inside a long run:
   - log `count(page_ref_counts == 0)`
   - log `max(page_ref_counts)` and `sum(page_ref_counts)` (approx "pages in use" accounting for sharing)

2. **Track allocated pages per slot**:
   - for each active slot, `num_valid_pages = sum(is_valid(page_indices[slot, :]))`
   - correlate with `seq_lens[slot]`

3. **Introduce a mode that runs sequential prompts without `engine.reset()`**, but with proper per-seq cleanup:
   - if free page count never recovers after sequences finish, cleanup is genuinely broken
   - if it recovers, the cleanup primitives work and we just weren't using them

4. **Add sanity assertions in debug builds** (device-side):
   - after freeing, ensure released slot has `used_mask=False`, `page_indices=INVALID`
   - ensure no `page_ref_counts` goes negative (already clamped with `maximum(..., 0)`)

These checks would turn "cleanup broken" from a hunch into an actionable diagnosis.

---

## 12. Summary (The One-Paragraph Version)

Levanter implements a paged KV cache where sequences map `(slot_id, position)` to `(page, offset)` via a `SequenceTable`
and a global `PageTable` allocator that tracks per-page refcounts; model decode updates KV pages and runs paged attention
using `PageBatchInfo`. The codebase already has device-side primitives to free pages when sequences finish, but the
InferenceEngine does not wire them into the decode loop and instead performs a heavy-handed reset that zeros the entire
KV cache. Throughput decreases during long generation because attention work grows with sequence length, and this effect
is much stronger when the TPU ragged paged attention kernel is disabled (falling back to a slow reference implementation),
which was done to avoid TPU `vmem` OOM. A restructure should separate "logical reset/free" from "physical zeroing", wire
up `free_pages_for_finished`, simplify redundant state, and (for performance) restore an efficient attention kernel path.
