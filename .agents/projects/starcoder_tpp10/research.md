# Research inputs

Effort: focused local implementation and metadata audit, 9 September 2026. The requested experiment fixes the scientific question; this investigation resolves executable controls and costs.

## Evidence

The previous [completed refinement](../starcoder_epoch_matching/refinement/results/RESULTS.md) found target regrets 0.031826 BPB for unmatched and 0.063770 for matched. Those results remain unchanged. Their total-parameter TPP mismatch motivated the new control, but model, tokenizer and subset changes prevent attributing any outcome difference across experiments to that control alone.

The [TinyLlama tokenizer](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/tree/59f6f375b26bde864a6ca194a9a3044570490064) is pinned at revision `59f6f375b26bde864a6ca194a9a3044570490064`. Only four tokenizer files were downloaded, totaling about 2.34 MB. The runtime loader reports vocabulary 32,000, BOS 1, EOS 2; the native text preprocessor appends BOS and EOS. A local code/newline/indentation example passed that path. Files and digests are in `experiments/domain_phase_mix/starcoder_tpp10_assets/`; no model weights are used.

GCS metadata confirms all selected raw objects exist in `marin-us-central1`. [The inventory](source_inventory.json) pins object names, generations, compressed sizes and CRC32C. StarCoder uses all 49 Dolma shards. The six web inventories contain 2,755 / 8,353 / 2,454 / 9,678 / 4,287 / 1,964 gzip shards; select the 16 lowest seeded URI hashes in each complete inventory. Paloma uses all 100 programming-language validation shards (9,499,025 compressed bytes). These were metadata reads; no remote training payload was downloaded locally. Actual token yields and production cache construction still require regional preparation.

The raw StarCoder shards are large (203.8 GB compressed in total), but each contributes only about 3.88M tokens. A source reader stops at its exact quota, with a compressed-byte limit of four times that quota or at least 8 MiB, capped at the object size. GCS may prefetch one extra 8 MiB chunk per source. An insufficient source quota or exhausted byte budget fails explicitly. There is no automatic extra shard, larger budget, or cross-region fallback.

## Reused code and relevant behavior

- `lib/levanter/src/levanter/models/qwen.py`: Qwen3 always adds Q/K RMSNorm. Its inherited `LlamaConfig.total_trainable_params` omits those weights; abstract model trees give 16,587,008 and 301,241,344 total parameters.
- `lib/marin/src/marin/experiment/train.py`: `train_lm` assembles the existing training/checkpoint stack. The new wrapper sets the two batch sizes, data seed, stable source names and explicit shuffle keys.
- `lib/levanter/src/levanter/data/text/formats.py`: use the same native text preprocessor for preparation and training metadata.
- `lib/levanter/src/levanter/store/cache.py`: `SerialCacheWriter`, finished ledgers and `consolidate_shard_cache_ledgers` support resumable finite caches without copying web token arrays during consolidation.
- `lib/levanter/src/levanter/data/text/datasets.py`: `TokenSeqDataset` has length `total_tokens // seq_len`. Physical finite parents and matched subsets avoid runtime support caps. Explicit named shuffle keys avoid the historical zero-weight endpoint bug.
- `lib/levanter/src/levanter/data/mixture.py`: the block allocator fixes counts within each block and permutes ordering. The audit evaluates its public `get_batch` at a full block and the final partial block, using the exact data key.
- `experiments/domain_phase_mix/launch_starcoder_epoch_matching.py`: reuse successful-artifact/fingerprint checks and immutable plan persistence. The old design and measured results are not edited.
- `lib/levanter/src/levanter/optim/muonh.py`: matrix and vector/embedding routing, groupwise clipping and norm-constrained updates. The new protocol screens proxy batch size at a fixed mixture before target training.

The local checkout is dirty and shared. Review packets record selected source-file hashes and the repository lock rather than claiming the current commit alone captures the implementation. Echo search for prior tokenizer/cache work returned HTTP 403; local source and Fieldbook supplied the evidence.

## Remaining empirical questions

The new tokenizer requires fresh bounded CPU caches. The small model's optimization stability, evaluation quality, actual chip-hours, and target turnover require live runs. Three subset draws are conditional on one finite parent; one target seed permits descriptive selection comparisons only. Additional off-match supports or a second evaluation domain would broaden the claim but are outside this three-curve release.
