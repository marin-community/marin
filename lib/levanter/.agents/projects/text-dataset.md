# Text Dataset Refactor Plan

Context: simplify `src/levanter/data/text.py` so all LM data flows through a single “mixture of components” path. Components have unique names, optional sources (HF or URL), per-split caches (including source-less components that require a prebuilt cache), and expose train/validation datasets. Cache formats: (1) text tokens (`input_ids`, add `loss_weight`) and (2) chat (`input_ids`, `assistant_masks`); formats should support concat, packing, and future padding mode.

## Design Direction
- Single concept: `DatasetComponent` encapsulating name, cache location, source (optional), format (text/chat), packing/concat preferences, and metadata tags. No single/mixture subclass hierarchy.
- One builder that, given a component + split, loads cache if present or builds from source; if neither exists, fail loudly.
- Format-aware preprocessing: text gets loss weights (all ones) and optional BOS/EOS; chat uses template + assistant mask. Packing strategy is orthogonal to format.
- Mixture as the only training API: even one component is a mixture of size 1. Train/validation maps remain per-component to preserve tagging and metrics.

```python
@dataclass
class DatasetComponent:
    name: str
    format: LmDatasetFormatBase = TextLmDatasetFormat()
    source: DatasetSource | None = None  # UrlSource | HFSource; None == cache-only component
    cache_dir: str | None = None
    pack: bool | int | Literal["pad"] | None = None  # None -> per-format default (text=concat, chat=pack); True/False/int/"pad" override
    tags: list[str] | None = None
```

## Refactor Steps
1) **Extract primitives into modules**
   - Move format + processor classes to `data/text/formats.py`.
   - Move cache helpers (`build_lm_dataset_cache`, `load_lm_dataset_cache`, ledger utils) to `data/text/cache.py`.
   - Move dataset assembly to `data/text/datasets.py` (forwarded via `data/text/__init__.py`); trim file size and clarify ownership.

2) **Introduce `DatasetComponent` + `DatasetSource`**
   - Define `DatasetSource` protocol with `get_shard_source(split) -> ShardedDataSource | None` and concrete `HFSource`, `UrlSource`.
   - Support source-less components: if `source is None`, cache must already exist per split or we error at build time.
   - Components own `cache_dir`; mixture config supplies a default `cache_root` used when component cache is unset.

3) **Rebuild config surface**
   - Replace `SingleDatasetLMConfigBase`, `UrlSingleDatasetLMConfig`, `HfSingleDatasetLMConfig` with `LmDataConfig` holding `components: dict[str, DatasetComponent]`, mixing weights, stop strategy, shuffle, budgets.
   - Default `pack=None` defers to per-format defaults (text → concat, chat → pack).
   - Update `LMMixtureDatasetConfig` to be a thin wrapper around `LmDataConfig` or merge into it; train/validation builders always iterate components.

4) **Unify cache build/load path**
   - `build_component_cache(component, split, tokenizer, cache_root, options, enforce_eos)` handles: resolved cache path, load-if-exists, build from source via processor, error if no cache and no source (covers cache-only components).
   - Add loss weights when writing text caches; carry assistant masks for chat. Ensure metadata records `pack`/`pad` choice and format.

5) **Generalize dataset creation**
   - `dataset_for_component(component, cache, Pos)` chooses between packed vs concat vs (future) pad for either format: packed uses `GreedyPrepackedDataset`, concat uses `TokenSeqDataset`, pad would batch with padding. Chat packing uses `assistant_masks` to derive loss weights; text packing derives loss weights as ones (shifted for causal).
   - `pack` may be None (use format default: text concat, chat pack), bool, int (max segments), or `"pad"` (future).

6) **Mixture assembly**
   - `build_train_sets` returns `{name: dataset_for_component(...)};` mixing weights remain dict or schedule. Keep era/perm shuffle per component.
   - Validation uses explicit validation split cache; if missing, error unless `num_validation_sequences` is configured from train cache.

7) **Metrics + utilities**
   - Update `count_corpus_sizes` and other helpers to operate on components dict; simplify tagging to use `component.tags or [name]`.
   - Drop `ignore_token_id`; rely on loss masks for both text and chat paths.

8) **Compatibility & migration**
   - Remove old config classes; add lightweight loader that can interpret legacy configs (optional, if desired) or raise clear error with migration note.
   - Update recipe/docs references in `docs/` (data formats, config examples) and adjust any config files under `config/` and tests.

9) **Tests**
   - Unit tests for: cache build/load fallbacks (source present vs missing cache, cache-only components), text vs chat pack/concat, loss_weight correctness, assistant mask masking, missing cache error when no source, mixture weight schedule integration, placeholder for future pad mode.
   - Update golden/fixture configs to use new `LmDataConfig`.

## Open Questions / Decisions
- Default packing via `pack=None`: chat packs; text concats (matches existing behavior).
- Remove `ignore_token_id`; loss masks cover both formats.
- Assume tokenizer is global for now; per-component tokenizer could be added later for prebuilt caches.

## Execution Order (safe landing)
1. Land new modules (`formats.py`, `cache.py`) + `DatasetComponent`/`DatasetSource` definitions, no behavior change.
2. Port cache build/load to new helpers; keep legacy configs calling through adapters.
3. Rewrite mixture config to consume components; delete single-dataset configs; adjust dataset creation to pack/concat switch.
4. Update utilities/tests/config examples; remove dead code and reroute imports.

## Verification
- Run `uv run pytest tests/data/test_text.py tests/data/test_mixture.py` (or equivalent) plus targeted new tests.
- Spot-check cache metadata files for loss_weight/assistant_masks fields and pack flag.
