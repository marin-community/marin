# Delphi TPP40 Europe historical Stack rebuild review

## Decision requested

Review the exact compatibility implementation and launch command below. Return
`GO` only if it is safe to rebuild Europe-local Stack token caches for the
cross-accelerator bridge; otherwise return `NO-GO` with concrete blockers.

Do not assess the later full TPP40 launch. This review covers only the isolated
Europe cache rebuild.

## Why the rebuild is needed

The repaired Europe raw Stack payload has the same 167,063,162 rows as East5,
but the first Europe tokenization has 134,076,475,061 tokens versus
134,071,054,270 in East5, a +5,420,791-token difference spread across all 15
languages.

East5 was built with March 2026 semantics: `marin-community/marin-tokenizer`
had no normalizer, so the old conditional long-string workaround left each
document intact. Current code forced a split at 10,000 characters, changing BPE
boundaries. On one complete 20,000-row Europe C shard:

- Hugging Face/full-document historical encoding: 15,302,042 tokens.
- New historical production mode: 15,302,042 tokens, exact ID agreement.
- Current forced-split mode: 15,302,518 tokens, +476.
- 822 rows exceed 10,000 characters; 360 rows change token count.
- The newer homogeneous-run safety splitter contributes zero tokens of this
  sample's difference; the forced 10k split accounts for all of it.

## Implementation to inspect

- `lib/levanter/src/levanter/tokenizers.py`
- `lib/marin/src/marin/processing/tokenize/_core.py`
- `lib/marin/src/marin/processing/tokenize/tokenize.py`
- `lib/marin/src/marin/processing/tokenize/__init__.py`
- `experiments/pretraining_datasets/dolma3_pool.py`
- `experiments/domain_phase_mix/prepare_delphi_tpp40_europe_historical_stack_caches.py`
- `lib/levanter/tests/test_text.py`
- `tests/test_prepare_delphi_tpp40_europe_historical_stack_caches.py`

The compatibility mode is represented by a dedicated config subtype and a
new cache namespace. Normal `TokenizeConfig` objects retain default splitting.
All 15 compatibility workers are Europe-local, use 10 GB RAM, and SQL uses
20 GB. Existing hydrated raw outputs are reused; no training corpus is copied
between regions.

Focused results:

- Levanter tokenizer tests: 3 passed.
- Europe current and historical graph tests: 15 passed.
- Ruff on all touched implementation/test files: passed.
- Region-local launch safety on the exact command: passed.

## Exact proposed command

```bash
UV_FROZEN=1 uv run python -m marin.run.iris_run --config lib/iris/config/marin.yaml --working-dir-exclude .agents/ --working-dir-exclude .github/ --working-dir-exclude docs/ --working-dir-exclude scripts/ --working-dir-exclude experiments/domain_phase_mix/exploratory/ --working-dir-exclude experiments/domain_phase_mix/manifests/ --working-dir-exclude experiments/domain_phase_mix/starcoder_wsd80_lr_onset_dense_surface_design_20260825.json.gz --working-dir-exclude checkpoints/ --working-dir-exclude tests/ --working-dir-exclude infra/grafana/ --working-dir-exclude .experiments/ --working-dir-exclude .experiments.zip -- --no-wait --no-preemptible --job-name dm-delphi-tpp40-europe-stack-historical-full-document-v1-20260830 --region europe-west4 --zone europe-west4-b --priority interactive --enable-extra-resources --cpu 1 --memory 16GB --disk 16GB --timeout 259200 --extra cpu -e MARIN_PREFIX gs://marin-eu-west4 -- python -m experiments.domain_phase_mix.prepare_delphi_tpp40_europe_historical_stack_caches --force_run_failed true --max_concurrent 15
```

## Questions

1. Does disabling both split layers faithfully reproduce historical
   full-document BPE semantics without changing short-document behavior?
2. Is the compatibility mode sufficiently isolated from normal cache identity
   and normal tokenization behavior?
3. Are the 10 GB/20 GB worker envelopes and 15-way concurrency defensible for
   unsplit documents?
4. Is the graph idempotent and strictly region-local?
5. Is anything missing that should block the exact command above?

## Post-review hardening and canary command

The first review returned `GO` and recommended three changes that are now in
the tree:

- The historical subtype is a frozen dataclass with a materialized
  `historical_full_document_tokenization=True` marker. Step identity remains
  protected by the dedicated historical namespace.
- `BatchTokenizer.long_string_workaround` is an explicit property; historical
  mode raises if the text-format processor is not a `BatchTokenizer`.
- The parent memory request is raised from 4 GB to 16 GB.

Before the full command, run the smallest Stack partition, Ruby, into the same
final historical namespace and require its stats to equal the frozen East5
count of 1,388,223,902 tokens:

```bash
UV_FROZEN=1 uv run python -m marin.run.iris_run --config lib/iris/config/marin.yaml --working-dir-exclude .agents/ --working-dir-exclude .github/ --working-dir-exclude docs/ --working-dir-exclude scripts/ --working-dir-exclude experiments/domain_phase_mix/exploratory/ --working-dir-exclude experiments/domain_phase_mix/manifests/ --working-dir-exclude experiments/domain_phase_mix/starcoder_wsd80_lr_onset_dense_surface_design_20260825.json.gz --working-dir-exclude checkpoints/ --working-dir-exclude tests/ --working-dir-exclude infra/grafana/ --working-dir-exclude .experiments/ --working-dir-exclude .experiments.zip -- --no-wait --no-preemptible --job-name dm-delphi-tpp40-europe-stack-historical-ruby-canary-v1-20260830 --region europe-west4 --zone europe-west4-b --priority interactive --enable-extra-resources --cpu 1 --memory 16GB --disk 16GB --timeout 86400 --extra cpu -e MARIN_PREFIX gs://marin-eu-west4 -- python -m experiments.domain_phase_mix.prepare_delphi_tpp40_europe_historical_stack_canary --force_run_failed true --max_concurrent 1
```

If the count passes, rerun the full command above with `--memory 16GB`; it
reuses the completed Ruby step by identity. Both commands pass the Europe
region-local launch-safety validator.
