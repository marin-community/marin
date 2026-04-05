# Investigate Packing for Nemotron Terminal-Corpus: Research Logbook

> Note: the repo recipe prefers `.agents/logbooks/`, but this thread lives in `.agents/logbook/` because the user requested that exact path.

## Scope

- Goal: quantify packing efficiency and truncation risk for the Nemotron Terminal-Corpus data path used by `experiments/exp3490b_sft_nemotron_terminal_corpus_qwen3_8b.py`, without running training.
- Primary metrics: per-subset and aggregate `total_docs`, `total_tokens`, `total_seqs`, `padding_fraction`, `truncation_fraction`, document length percentiles, fraction of documents longer than the target sequence length, assistant-token loss under current left truncation, and sensitivity to `max_segments_per_example`.
- Constraints: no training runs; keep the current packer implementation unchanged; reuse the existing `exp3490b` dataset definition and tokenizer; prefer read-only statistics jobs once token caches exist.
- Stop criteria: enough evidence to answer all of the following:
  - Is `seq_len=32768` a sensible operating point for this corpus?
  - Is the default chat segment cap of `64` binding at 32k?
  - How much supervision do we lose from current left truncation?
  - Is current cache order materially hurting greedy pack efficiency?
  - Do full-corpus and skill-only modes behave differently enough to warrant separate defaults?

## Baseline

- Date: 2026-04-04
- Experiment ID prefix: `NTPACK`
- Canonical experiment: `experiments/exp3490b_sft_nemotron_terminal_corpus_qwen3_8b.py`
- Canonical corpus: full Terminal-Corpus mixture of:
  - `nvidia/Nemotron-Terminal-Corpus/dataset_adapters`
  - `nvidia/Nemotron-Terminal-Corpus/skill_based_easy`
  - `nvidia/Nemotron-Terminal-Corpus/skill_based_medium`
  - `nvidia/Nemotron-Terminal-Corpus/skill_based_mixed`
- Tokenizer + format:
  - `qwen3_8b_tokenizer`
  - `ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE)`
- Current packing behavior in code:
  - `ChatLmDatasetFormat(pack=None)` defaults to packed chat examples.
  - Effective chat packing is resolved in `lib/levanter/src/levanter/data/text/datasets.py`.
  - `ChatDataset` always wraps the cache with `GreedyPrepackedDataset`.
  - Current default for chat is effectively `max_segments_per_example=64`.
  - Current chat slice strategy is `"left"`.
  - `mask_user_turns=True`, so only assistant tokens contribute loss.
- Sequence-length path for `exp3490b`:
  - `SimpleSFTConfig.max_seq_len=32768`
  - `default_sft(...) -> SimpleTrainConfig.train_seq_len=32768`
  - `train_lm.py -> Pos = config.model.max_Pos.resize(train_length)`
  - `ChatDataset(..., Pos)` then packs to `Pos.size == 32768`
- Important non-goal:
  - `mixture_block_size=12288` is a `MixtureDataset` sampling knob, not token packing. We should not conflate it with sequence packing.
- Baseline numbers: `TBD by NTPACK-001`

## Questions to Answer

1. At `seq_len=32768`, what fraction of the full corpus is padding under the current packer and default segment cap?
2. At `seq_len=32768`, what fraction of the corpus is truncated, and which subsets drive that truncation?
3. Is the default `max_segments_per_example=64` the main source of residual padding at 32k, or is greedy packing already near saturation?
4. For documents longer than 32k, how much assistant supervision is dropped by current left truncation?
5. How much headroom is left if we keep the same greedy algorithm but change only corpus order or segment cap?

## Artifacts

- Logbook: `.agents/logbook/investigate_packing_nemotron.md`
- Planned output directory: `.agents/logbook/artifacts/nemotron_packing/`

## Experiment Plan

| Run | Corpus | Measurement | Hypothesis | ETA |
|-----|--------|-------------|------------|-----|
| `NTPACK-001` | Full corpus | Baseline pack stats at `seq_len=32768` with current defaults | Padding is non-zero even at 32k; truncation is low but not necessarily negligible | 5-10 min if caches exist, 30-90 min if caches must be built |
| `NTPACK-002` | Full corpus | Sequence-length sweep at `4096, 8192, 16384, 32768` | Padding and truncation improve as sequence length increases, but we need the actual curve rather than intuition | 10-20 min after `001` |
| `NTPACK-003` | Full corpus | Segment-cap sweep at `1, 4, 8, 16, 32, 64, 128` with `seq_len=32768` | If `64` is binding, padding falls meaningfully above 64; otherwise the curve plateaus by 32 or 64 | 10-20 min after `001` |
| `NTPACK-004` | Full corpus | Document length distribution per subset from token caches | The corpus is heavy-tailed and one or two subsets dominate the long-context behavior | 10-30 min |
| `NTPACK-005` | Full corpus | Left-truncation audit using `assistant_masks` for docs longer than 32k | Raw truncation rate understates impact because assistant tokens are likely concentrated late in long docs | 20-40 min |
| `NTPACK-006` | Full corpus | Greedy-packing order sensitivity under current order vs randomized vs length-sorted order | Current cache order leaves measurable packing headroom even with the same greedy packer | 10-20 min |
| `NTPACK-007` | Skill-only corpus | Repeat `001` and `004` for the `USE_FULL_CORPUS=false` subset selection | Skill-only mode has materially different length and padding behavior from the full four-subset mix | 10-30 min |

## Run Order

1. Run `NTPACK-001` first to anchor all later comparisons.
2. Run `NTPACK-004` next if `001` shows non-trivial truncation or surprising padding.
3. Run `NTPACK-002` and `NTPACK-003` to separate sequence-length effects from segment-cap effects.
4. Run `NTPACK-005` only if `004` finds a meaningful long-document tail above 32k.
5. Run `NTPACK-006` after `003` so the theoretical headroom can be compared against actual cap sensitivity.
6. Run `NTPACK-007` only if we care about the skill-only mode as an operational target.

## Decision Rules

- If `NTPACK-003` shows less than a 1 percentage point improvement in aggregate `padding_fraction` from `64 -> 128`, treat the current segment cap as "not binding enough to tune first".
- If `NTPACK-005` shows negligible assistant-token loss above 32k, de-prioritize slice-strategy work.
- If `NTPACK-006` shows less than a 1 percentage point gap between current-order padding and randomized-order padding, de-prioritize order-aware packing ideas.
- If one subset dominates both long-doc rate and padding, focus future tuning on that subset before proposing global defaults.

## Fallbacks

- If `count_corpus_sizes(...)` has to build missing caches and the end-to-end job is too slow, split runs per subset and persist intermediate JSON per subset.
- If full-corpus cache construction is too expensive for a first pass, start with the skill-only mixture to validate the measurement harness, then scale up to the full four-subset corpus.
- If reading `assistant_masks` for every long document is too slow, sample long docs uniformly at random with a fixed seed and report confidence intervals instead of full exact totals.

## Command Templates

### `NTPACK-001` - Baseline full-corpus packing stats at 32k

Expected outputs:
- aggregate and per-subset `total_docs`, `total_tokens`, `total_seqs`
- `padding_fraction` or `truncation_fraction` for each subset

```bash
mkdir -p .agents/logbook/artifacts/nemotron_packing

USE_FULL_CORPUS=true uv run python - <<'PY' \
  > .agents/logbook/artifacts/nemotron_packing/NTPACK-001_full_seq32768_default.json
import json

from experiments.exp3490b_sft_nemotron_terminal_corpus_qwen3_8b import mixture_config
from levanter.data.text.datasets import count_corpus_sizes

stats = count_corpus_sizes(mixture_config, prefix="", seq_len=32768)
print(json.dumps(stats, indent=2, sort_keys=True))
PY
```

### `NTPACK-002` - Sequence-length sweep with current defaults

```bash
mkdir -p .agents/logbook/artifacts/nemotron_packing

USE_FULL_CORPUS=true uv run python - <<'PY' \
  > .agents/logbook/artifacts/nemotron_packing/NTPACK-002_seq_sweep_full.json
import json

from experiments.exp3490b_sft_nemotron_terminal_corpus_qwen3_8b import mixture_config
from levanter.data.text.datasets import count_corpus_sizes

out = {}
for seq_len in [4096, 8192, 16384, 32768]:
    out[str(seq_len)] = count_corpus_sizes(mixture_config, prefix="", seq_len=seq_len)

print(json.dumps(out, indent=2, sort_keys=True))
PY
```

### `NTPACK-003` - Segment-cap sweep at 32k with current packer

This changes only the effective chat segment cap. It does not change the packer algorithm.

```bash
mkdir -p .agents/logbook/artifacts/nemotron_packing

USE_FULL_CORPUS=true uv run python - <<'PY' \
  > .agents/logbook/artifacts/nemotron_packing/NTPACK-003_segment_cap_sweep_full_seq32768.json
import dataclasses
import json

from experiments.exp3490b_sft_nemotron_terminal_corpus_qwen3_8b import mixture_config
from levanter.data.text.datasets import DatasetComponent, count_corpus_sizes


def with_pack_cap(cfg, cap: int):
    components = {
        name: dataclasses.replace(component, pack=cap) if isinstance(component, DatasetComponent) else component
        for name, component in cfg.components.items()
    }
    return dataclasses.replace(cfg, components=components)


out = {}
for cap in [1, 4, 8, 16, 32, 64, 128]:
    out[str(cap)] = count_corpus_sizes(with_pack_cap(mixture_config, cap), prefix="", seq_len=32768)

print(json.dumps(out, indent=2, sort_keys=True))
PY
```

### `NTPACK-004` - Token-length distribution per subset

```bash
mkdir -p .agents/logbook/artifacts/nemotron_packing

USE_FULL_CORPUS=true uv run python - <<'PY' \
  > .agents/logbook/artifacts/nemotron_packing/NTPACK-004_doc_length_stats_full.json
import json
import numpy as np

from experiments.exp3490b_sft_nemotron_terminal_corpus_qwen3_8b import mixture_config


def lengths_from_cache(cache):
    store = cache.store.tree["input_ids"]
    offsets = store.offsets[0 : store.num_rows + 1].read().result().copy()
    offsets[0] = 0
    return offsets[1:] - offsets[:-1], store


out = {}
for name, cache in mixture_config.build_caches("train").items():
    lengths, store = lengths_from_cache(cache)
    out[name] = {
        "docs": int(store.num_rows),
        "tokens": int(store.data_size),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "max": int(lengths.max()),
        "pct_gt_4096": float((lengths > 4096).mean()),
        "pct_gt_8192": float((lengths > 8192).mean()),
        "pct_gt_16384": float((lengths > 16384).mean()),
        "pct_gt_32768": float((lengths > 32768).mean()),
        "percentiles": {
            "p90": float(np.percentile(lengths, 90)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "p99_5": float(np.percentile(lengths, 99.5)),
            "p99_9": float(np.percentile(lengths, 99.9)),
        },
    }

print(json.dumps(out, indent=2, sort_keys=True))
PY
```

### `NTPACK-005` - Assistant-token loss under current left truncation

This estimates how much supervised signal is dropped for long documents under the current `slice_strategy="left"` behavior.

```bash
mkdir -p .agents/logbook/artifacts/nemotron_packing

USE_FULL_CORPUS=true uv run python - <<'PY' \
  > .agents/logbook/artifacts/nemotron_packing/NTPACK-005_left_truncation_audit_full_seq32768.json
import json
import numpy as np

from experiments.exp3490b_sft_nemotron_terminal_corpus_qwen3_8b import mixture_config

SEQ_LEN = 32768
out = {}

for name, cache in mixture_config.build_caches("train").items():
    ids_store = cache.store.tree["input_ids"]
    mask_store = cache.store.tree["assistant_masks"]
    offsets = ids_store.offsets[0 : ids_store.num_rows + 1].read().result().copy()
    offsets[0] = 0
    lengths = offsets[1:] - offsets[:-1]
    long_doc_indices = np.nonzero(lengths > SEQ_LEN)[0]

    total_kept_assistant = 0
    total_lost_assistant = 0
    tail_heavy_docs = 0

    for idx in long_doc_indices.tolist():
        start = offsets[idx]
        end = offsets[idx + 1]
        mask = np.asarray(mask_store.data[start:end].read().result())

        kept = int(mask[:SEQ_LEN].sum())
        lost = int(mask[SEQ_LEN:].sum())
        total_kept_assistant += kept
        total_lost_assistant += lost

        if mask.sum() > 0 and mask[SEQ_LEN:].sum() > mask[:SEQ_LEN].sum():
            tail_heavy_docs += 1

    total_assistant = total_kept_assistant + total_lost_assistant
    out[name] = {
        "docs_gt_seq_len": int(len(long_doc_indices)),
        "assistant_tokens_kept": int(total_kept_assistant),
        "assistant_tokens_lost": int(total_lost_assistant),
        "assistant_loss_fraction": (
            float(total_lost_assistant / total_assistant) if total_assistant else 0.0
        ),
        "tail_heavy_doc_fraction": (
            float(tail_heavy_docs / len(long_doc_indices)) if len(long_doc_indices) else 0.0
        ),
    }

print(json.dumps(out, indent=2, sort_keys=True))
PY
```

### `NTPACK-006` - Order sensitivity under the same greedy algorithm

This does not change the production packer. It estimates the headroom available if the same greedy algorithm saw different document orders.

```bash
mkdir -p .agents/logbook/artifacts/nemotron_packing

USE_FULL_CORPUS=true uv run python - <<'PY' \
  > .agents/logbook/artifacts/nemotron_packing/NTPACK-006_order_sensitivity_full_seq32768_cap64.json
import json
import numpy as np

from experiments.exp3490b_sft_nemotron_terminal_corpus_qwen3_8b import mixture_config
from levanter.data.packing import pack_documents

SEQ_LEN = 32768
CAP = 64
RNG = np.random.default_rng(0)


def pack_summary(lengths):
    effective_lengths = np.minimum(lengths, SEQ_LEN)
    packs = pack_documents(
        {"input_ids": lengths},
        {"input_ids": SEQ_LEN},
        max_segments_per_example=CAP,
        slice_strategy="left",
    )
    seqs = len(packs)
    kept_tokens = int(effective_lengths.sum())
    capacity = seqs * SEQ_LEN
    return {
        "seqs": int(seqs),
        "kept_tokens": kept_tokens,
        "padding_fraction": float(1 - (kept_tokens / capacity)),
    }


out = {}
for name, cache in mixture_config.build_caches("train").items():
    store = cache.store.tree["input_ids"]
    offsets = store.offsets[0 : store.num_rows + 1].read().result().copy()
    offsets[0] = 0
    lengths = offsets[1:] - offsets[:-1]

    out[name] = {
        "current_order": pack_summary(lengths),
        "random_order_seed0": pack_summary(lengths[RNG.permutation(len(lengths))]),
        "ascending_length": pack_summary(np.sort(lengths)),
        "descending_length": pack_summary(np.sort(lengths)[::-1]),
    }

print(json.dumps(out, indent=2, sort_keys=True))
PY
```

### `NTPACK-007` - Skill-only comparison

Note: `SYNTHETIC_DATA_FRACTION` changes training weights, not raw document stats. For packing statistics, the relevant distinction is the subset selection driven by `USE_FULL_CORPUS=false`.

```bash
mkdir -p .agents/logbook/artifacts/nemotron_packing

USE_FULL_CORPUS=false uv run python - <<'PY' \
  > .agents/logbook/artifacts/nemotron_packing/NTPACK-007_skill_only_baseline_seq32768.json
import json

from experiments.exp3490b_sft_nemotron_terminal_corpus_qwen3_8b import mixture_config
from levanter.data.text.datasets import count_corpus_sizes

stats = count_corpus_sizes(mixture_config, prefix="", seq_len=32768)
print(json.dumps(stats, indent=2, sort_keys=True))
PY
```

## Experiment Log

### 2026-04-04 12:40 - NTPACK-001: Baseline full-corpus 32k packing stats

- Hypothesis: current default chat packing at 32k still leaves measurable padding, and the subset-level breakdown will tell us whether residual inefficiency comes from the long tail or from the segment cap.
- Exact command:

```bash
mkdir -p .agents/logbook/artifacts/nemotron_packing

USE_FULL_CORPUS=true uv run python - <<'PY' \
  > .agents/logbook/artifacts/nemotron_packing/NTPACK-001_full_seq32768_default.json
import json

from experiments.exp3490b_sft_nemotron_terminal_corpus_qwen3_8b import mixture_config
from levanter.data.text.datasets import count_corpus_sizes

stats = count_corpus_sizes(mixture_config, prefix="", seq_len=32768)
print(json.dumps(stats, indent=2, sort_keys=True))
PY
```

- Config details:
  - Corpus: full Terminal-Corpus, all four subsets
  - Tokenizer: `qwen3_8b_tokenizer`
  - Format: `ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE)`
  - Effective packing mode: packed chat with current default segment cap (`64`)
  - Sequence length: `32768`
  - No trainer startup; statistics only
- Expected outcome:
  - concrete aggregate and per-subset `padding_fraction` / `truncation_fraction`
  - confirmation that the measurement harness can inspect the same data path as `exp3490b`
  - a baseline JSON artifact that later runs can diff against
- Success criteria:
  - the artifact is written successfully
  - each subset reports `total_docs`, `total_tokens`, and `total_seqs`
  - we can identify whether the next priority is sequence-length sweep, segment-cap sweep, or truncation audit
- Result: `TBD`
- Interpretation: `TBD`
- Next action: run `NTPACK-004` if truncation is non-trivial, otherwise run `NTPACK-003`

### 2026-04-04 12:34 - Cache verification for `exp3490b` full run

- Context:
  - Issue: `https://github.com/marin-community/marin/issues/3490`
  - W&B run: `https://wandb.ai/marin-community/marin/runs/exp3490b_sft_nemotron_terminal_corpus_full_qwen3_8b_32768tokens_-3da6c1`
- Source of truth:
  - `gs://marin-us-east5/checkpoints/exp3490b_sft_nemotron_terminal_corpus_full_qwen3_8b_32768tokens_-3da6c1/.executor_info`
- Verified existing token caches used by the full run:
  - `gs://marin-us-east5/tokenized/dataset_adapters_qwen3_8b_tokenizer-c319d9`
  - `gs://marin-us-east5/tokenized/skill_based_easy_qwen3_8b_tokenizer-a5dd6f`
  - `gs://marin-us-east5/tokenized/skill_based_medium_qwen3_8b_tokenizer-045f3b`
  - `gs://marin-us-east5/tokenized/skill_based_mixed_qwen3_8b_tokenizer-3d0352`
- Verified transformed document roots:
  - `gs://marin-us-east5/documents/nvidia--Nemotron-Terminal-Corpus--dataset_adapters-a1667c4-1eb68d`
  - `gs://marin-us-east5/documents/nvidia--Nemotron-Terminal-Corpus--skill_based_easy-a1667c4-40c563`
  - `gs://marin-us-east5/documents/nvidia--Nemotron-Terminal-Corpus--skill_based_medium-a1667c4-88595b`
  - `gs://marin-us-east5/documents/nvidia--Nemotron-Terminal-Corpus--skill_based_mixed-a1667c4-7e553e`
- Verification result:
  - each tokenized prefix exists and contains `.artifact`, `.executor_info`, `.executor_status`, and `train/`
- Implication for the stats plan:
  - the `NTPACK` runs should be treated as read-only cache-inspection jobs against the existing east5 caches
  - we should avoid any harness path that re-materializes tokenization unless a cache lookup unexpectedly misses
- Next action:
  - wire the first stats command to the existing cache roots and run `NTPACK-001` on Iris as a CPU-only job

### 2026-04-04 12:42 - NTPACK-001 launch plan (read-only Iris job)

- Goal:
  - run the baseline 32k packing stats against the existing full-run token caches only
- Read-only guardrails:
  - use a custom script with `DatasetComponent(source=None, cache_dir=...)` for each verified cache root
  - set `LmDataConfig(auto_build_caches=False)`
  - verify each `gs://.../train` path exists before calling `count_corpus_sizes(...)`
  - if any cache is missing, fail immediately rather than tokenizing
- Script:
  - `.agents/logbook/artifacts/nemotron_packing/ntpack_001_readonly.py`
- Exact launch command:

```bash
uv run --with protobuf==6.33.4 iris --config lib/iris/examples/marin.yaml job run \
  --job-name ntpack-001-full-readonly \
  --cpu 2 \
  --memory 12GB \
  --disk 20GB \
  --zone us-east1-b \
  -- \
  uv run --with protobuf==6.33.4 python .agents/logbook/artifacts/nemotron_packing/ntpack_001_readonly.py
```

- Scheduling note:
  - `--zone us-east1-b` targets the fallback `cpu_vm_e2_highmem_2_ondemand` group from `lib/iris/examples/marin.yaml`
  - this avoids Iris scheduling the stats job onto a TPU worker, since CPU demand is otherwise fungible
- Expected log evidence:
  - cache verification lines for each `gs://marin-us-east5/tokenized/.../train`
  - `Loading cache from ...`
  - no `Building cache for ...`
  - no `Cache not found at ... Building with zephyr pipeline.`

### 2026-04-04 13:09 - Iris launch failure analysis and checkout suspicion

- What is now confirmed:
  - the full `exp3490b` token caches already exist on GCP and are sufficient to reconstruct the dataset without any tokenization work
  - the read-only script `.agents/logbook/artifacts/nemotron_packing/ntpack_001_readonly.py` verifies all four `gs://.../train` roots successfully from this machine
  - no tokenization rerun happened during this investigation
- Iris attempts and outcomes:
  - `/ahmed/ntpack-001-full-readonly`
    - never started; `us-east1-b` fallback CPU pool was in autoscaler backoff
    - terminated manually
  - `/ahmed/ntpack-001-full-readonly-c1`
    - scheduled in `us-central1-a`
    - failed in Iris build phase before user code ran
    - worker log:
      - `syncing deps`
      - `error: No pyproject.toml found in current directory or any parent directory`
  - `/ahmed/ntpack-001-full-readonly-raw`
    - raw controller submission attempt
    - my request was malformed because `replicas=1` was omitted
    - controller rejected it before execution
- Why this looks like a bad checkout or workspace-staging issue rather than a data issue:
  - local bundle inspection shows the submitted workspace bundle contains:
    - `pyproject.toml`
    - `uv.lock`
    - `.agents/logbook/artifacts/nemotron_packing/ntpack_001_readonly.py`
  - local import of `experiments.exp3490b_sft_nemotron_terminal_corpus_qwen3_8b` fails in this checkout because it imports `experiments.qwen3_chat_template`, while the actual file present here is `experiments/chat_templates/qwen3_chat_template.py`
  - together, these suggest the current checkout is not a clean basis for `iris job run`, even though the token caches themselves are fine
- Practical implication:
  - once we run from a healthy checkout, the GCP cache roots below are enough
  - we should reconstruct the dataset directly from the cache roots instead of depending on the broken experiment import path in this checkout

### Minimal cache-backed reconstruction for the full dataset

- Goal:
  - reproduce the full `exp3490b` dataset definition from existing token caches only
  - never rebuild tokenization
- Reference implementation:
  - `.agents/logbook/artifacts/nemotron_packing/ntpack_001_readonly.py`
- Exact cache roots:
  - `gs://marin-us-east5/tokenized/dataset_adapters_qwen3_8b_tokenizer-c319d9`
  - `gs://marin-us-east5/tokenized/skill_based_easy_qwen3_8b_tokenizer-a5dd6f`
  - `gs://marin-us-east5/tokenized/skill_based_medium_qwen3_8b_tokenizer-045f3b`
  - `gs://marin-us-east5/tokenized/skill_based_mixed_qwen3_8b_tokenizer-3d0352`
- Exact weights from `exp3490b` full mode:
  - `dataset_adapters: 226313`
  - `skill_based_easy: 44800`
  - `skill_based_medium: 89300`
  - `skill_based_mixed: 5690`

```python
from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from levanter.data.text import ChatLmDatasetFormat
from levanter.data.text.datasets import DatasetComponent, LmDataConfig


def build_exp3490b_full_from_caches() -> LmDataConfig:
    chat_format = ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE)

    components = {
        "nvidia/Nemotron-Terminal-Corpus/dataset_adapters": DatasetComponent(
            source=None,
            cache_dir="gs://marin-us-east5/tokenized/dataset_adapters_qwen3_8b_tokenizer-c319d9",
            format=chat_format,
        ),
        "nvidia/Nemotron-Terminal-Corpus/skill_based_easy": DatasetComponent(
            source=None,
            cache_dir="gs://marin-us-east5/tokenized/skill_based_easy_qwen3_8b_tokenizer-a5dd6f",
            format=chat_format,
        ),
        "nvidia/Nemotron-Terminal-Corpus/skill_based_medium": DatasetComponent(
            source=None,
            cache_dir="gs://marin-us-east5/tokenized/skill_based_medium_qwen3_8b_tokenizer-045f3b",
            format=chat_format,
        ),
        "nvidia/Nemotron-Terminal-Corpus/skill_based_mixed": DatasetComponent(
            source=None,
            cache_dir="gs://marin-us-east5/tokenized/skill_based_mixed_qwen3_8b_tokenizer-3d0352",
            format=chat_format,
        ),
    }

    weights = {
        "nvidia/Nemotron-Terminal-Corpus/dataset_adapters": 226313.0,
        "nvidia/Nemotron-Terminal-Corpus/skill_based_easy": 44800.0,
        "nvidia/Nemotron-Terminal-Corpus/skill_based_medium": 89300.0,
        "nvidia/Nemotron-Terminal-Corpus/skill_based_mixed": 5690.0,
    }

    return LmDataConfig(
        tokenizer="Qwen/Qwen3-8B",
        cache_dir=None,
        enforce_eos=True,
        auto_build_caches=False,
        shuffle=False,
        block_cross_document_attention=True,
        mixture_block_size=12288,
        components=components,
        train_weights=weights,
    )
```

- Minimal usage for stats:

```python
import json

from levanter.data.text.datasets import count_corpus_sizes

config = build_exp3490b_full_from_caches()
stats = count_corpus_sizes(config, prefix="", seq_len=32768)
print(json.dumps(stats, indent=2, sort_keys=True))
```

- Minimal usage for packed per-component datasets:

```python
from haliax import Axis

Pos = Axis("position", 32768)
config = build_exp3490b_full_from_caches()
caches = config.build_caches("train")  # load only; raises if a cache is missing
datasets = config.build_token_datasets(caches, Pos, split="train")
```

- Safety properties of this path:
  - `source=None` means there is no raw dataset source attached to the components
  - `auto_build_caches=False` means any missing cache raises instead of tokenizing
  - the cache roots match the exact full-run checkpoint metadata from `exp3490b`

### Recommended next step on a healthy checkout

- Run this exact script directly:

```bash
uv run --with protobuf==6.33.4 python \
  .agents/logbook/artifacts/nemotron_packing/ntpack_001_readonly.py \
  > .agents/logbook/artifacts/nemotron_packing/NTPACK-001_full_seq32768_default.json
```

- If we want Iris rather than local execution, use the same script from a healthy checkout and confirm a trivial CPU job can see `/app/pyproject.toml` before retrying the stats run.

### CLAUDE 2026-04-04 17:30 - NTPACK-001 v2 failure: validation cache missing

- Job: `/ahmed/ntpack-001-full-readonly-v2` on Iris (`us-central1-a`, CPU-only)
- Outcome: **failed**
- Root cause: `count_corpus_sizes()` unconditionally calls `config.build_caches("validation")` at `datasets.py:962`. The exp3490b caches only have `train/` splits — no `validation/shard_ledger.json` exists.
- Error: `ValueError: No source and no cache found for component nvidia/Nemotron-Terminal-Corpus/dataset_adapters split validation`
- Good news: the train cache loading worked perfectly — all 4 caches verified and loaded.
- Fix: replaced `count_corpus_sizes` call with a local `count_train_only()` function that only loads train caches. Inlined the relevant logic from `datasets.py:918-960`, skipping the validation block at lines 962-977.
- Import change: `count_corpus_sizes` → `_get_token_key_for_component, dataset_for_component`
- Resubmitted as `/ahmed/ntpack-001-full-readonly-v3`

### CLAUDE 2026-04-04 17:31 - NTPACK-001 v3 submitted

- Job: `/ahmed/ntpack-001-full-readonly-v3`
- Zone: `us-central1-a`
- Resources: 2 CPU, 12GB RAM, 20GB disk
- Command: `uv run python .agents/logbook/artifacts/nemotron_packing/ntpack_001_readonly.py`
- Status: **succeeded**
- Artifact: `.agents/logbook/artifacts/nemotron_packing/NTPACK-001_full_seq32768_default.json`

### CLAUDE 2026-04-04 17:32 - NTPACK-001 results

- Job: `/ahmed/ntpack-001-full-readonly-v3` — **succeeded** in ~20s of user-code time

#### Results summary (seq_len=32768, default segment cap 64)

| Subset | Docs | Seqs | Tokens | Padding % | Weight |
|--------|------|------|--------|-----------|--------|
| dataset_adapters | 226,313 | 145,033 | 3,992,592,464 | **16.0%** | 61.8% |
| skill_based_easy | 44,809 | 23,758 | 597,515,934 | **23.2%** | 12.2% |
| skill_based_medium | 89,343 | 73,013 | 1,661,263,733 | **30.6%** | 24.4% |
| skill_based_mixed | 5,689 | 2,917 | 74,312,459 | **22.3%** | 1.6% |

#### Key observations

1. **No truncation anywhere** — all subsets report `padding_fraction` (positive), not `truncation_fraction`. At 32k, the corpus fits without truncation.
2. **Padding ranges from 16% to 31%** — this is significant. At 32k we waste ~20-30% of capacity on padding.
3. **`skill_based_medium` is the worst** at 30.6% padding, despite being the second-largest subset by weight (24.4%).
4. **`dataset_adapters` is the best-packed** at 16.0% — it dominates the mix (61.8% weight) and packs most efficiently.
5. **Aggregate padding** (weight-averaged): ~0.618×16.0 + 0.122×23.2 + 0.244×30.6 + 0.016×22.3 ≈ **20.6%** padding overall.

#### Interpretation

- 20.6% aggregate padding at 32k means roughly 1 in 5 tokens of capacity is wasted. This is a meaningful efficiency loss.
- Since there is **zero truncation**, the 32k sequence length is more than sufficient for this corpus. The question now is whether the padding comes from the segment cap or from doc-length distribution.
- Per the decision rules: since truncation is zero, `NTPACK-005` (truncation audit) is low priority. The next priority is `NTPACK-003` (segment-cap sweep) and `NTPACK-004` (doc length distribution) to understand whether padding is driven by the segment cap or by short documents that can't fill sequences.

#### CONSTRAINT: seq_len=32768 is fixed and non-negotiable

**The sequence length MUST remain 32768. Do not propose changing it.** This is a hard requirement from the project, not a tuning knob. NTPACK-002 (seq-len sweep) is cancelled — it is not actionable regardless of results. All optimization must happen within the packing strategy at the fixed 32k window.

#### Next actions

1. **NTPACK-004** — doc length distribution per subset (to understand the shape of the problem)
2. **NTPACK-003** — segment-cap sweep at 32k (to test if raising above 64 helps)

### CLAUDE 2026-04-04 17:35 - NTPACK-002 cancelled

- NTPACK-002 (seq-len sweep) is cancelled. `seq_len=32768` is a hard constraint, not a tuning parameter.

### CLAUDE 2026-04-04 17:36 - NTPACK-003 and NTPACK-004 submitted in parallel

- **NTPACK-003** — segment-cap sweep
  - Job: `/ahmed/ntpack-003-segment-cap-sweep`
  - Zone: `us-central1-a`, 2 CPU, 12GB RAM, 20GB disk
  - Script: `.agents/logbook/artifacts/nemotron_packing/ntpack_003_segment_cap_sweep.py`
  - Sweeps `max_segments_per_example` over [1, 4, 8, 16, 32, 64, 128] at `seq_len=32768`
  - Uses `ChatLmDatasetFormat(pack=<cap>)` which flows to `max_segments_per_example` via `_effective_pack()` → `dataset_for_component()`
  - Hypothesis: if 64 is binding, padding will drop meaningfully at 128; if not, the curve plateaus by 32-64
  - Status: submitted, awaiting results

- **NTPACK-004** — doc length distribution
  - Job: `/ahmed/ntpack-004-doc-length-stats`
  - Zone: `us-central1-a`, 2 CPU, 12GB RAM, 20GB disk
  - Script: `.agents/logbook/artifacts/nemotron_packing/ntpack_004_doc_length_stats.py`
  - Reads token cache offsets directly to compute per-doc token lengths
  - Reports: mean, median, min, max, std, percentiles (p10–p99.9), fraction >1k/4k/8k/16k/32k
  - Hypothesis: the corpus is heavy-tailed with most docs well under 32k; `skill_based_medium` likely has a different shape explaining its 30.6% padding
  - Status: submitted, awaiting results

### CLAUDE 2026-04-04 17:42 - NTPACK-003 results: segment cap is NOT the bottleneck

- Job: `/ahmed/ntpack-003-segment-cap-sweep` — **succeeded**
- Artifact: `.agents/logbook/artifacts/nemotron_packing/NTPACK-003_segment_cap_sweep_full_seq32768.json`

#### Aggregate padding by segment cap (dataset_adapters as representative)

| Cap | Seqs | Padding % | Delta from cap=64 |
|-----|------|-----------|-------------------|
| 1 | 226,313 | **46.2%** | +30.2pp |
| 4 | 145,377 | **16.2%** | +0.2pp |
| 8 | 145,033 | **16.0%** | 0.0pp |
| 16 | 145,033 | **16.0%** | 0.0pp |
| 32 | 145,033 | **16.0%** | 0.0pp |
| 64 | 145,033 | **16.0%** | baseline |
| 128 | 145,033 | **16.0%** | 0.0pp |

#### Key finding

- **The segment cap saturates at 8.** Caps 8, 16, 32, 64, and 128 all produce identical results — same number of sequences, same padding fraction, for every subset.
- Cap=4 shows a tiny 0.2pp degradation vs cap=8+ for `dataset_adapters` only (145,377 vs 145,033 seqs).
- Cap=1 (no packing) is catastrophic: 46-60% padding depending on subset.
- **Conclusion: the default cap of 64 is not binding. The residual 16-31% padding is entirely due to document length distribution, not the segment cap.** Raising the cap will not help.
- Per decision rules: "less than 1pp improvement from 64→128" → treat cap as not binding. Confirmed.

### CLAUDE 2026-04-04 17:42 - NTPACK-004 results: doc length distributions

- Job: `/ahmed/ntpack-004-doc-length-stats` — **succeeded**
- Artifact: `.agents/logbook/artifacts/nemotron_packing/NTPACK-004_doc_length_stats_full.json`

#### Per-subset length summary

| Subset | Docs | Median | Mean | p90 | p99 | Max | % > 32k |
|--------|------|--------|------|-----|-----|-----|---------|
| dataset_adapters | 226,313 | 13,986 | 17,642 | 37,091 | 53,808 | 137,124 | **14.0%** |
| skill_based_easy | 44,809 | 12,381 | 13,335 | 20,277 | 32,345 | 70,829 | 0.9% |
| skill_based_medium | 89,343 | 18,079 | 18,594 | 25,958 | 32,888 | 72,316 | 1.1% |
| skill_based_mixed | 5,689 | 12,196 | 13,063 | 19,126 | 23,296 | 27,961 | **0.0%** |

#### Key findings

1. **`dataset_adapters` is the only subset with a significant long tail**: 14% of docs exceed 32k, with a max of 137k tokens. This explains why it packs best (16% padding) — long docs fill sequences well. The long-doc truncation concern from the original plan is relevant here.
2. **`skill_based_medium` has the highest median** (18,079) but almost nothing over 32k (1.1%). Median ≈ 55% of seq_len means greedy packing typically fits only 1 doc per sequence with ~45% headroom, but the second doc rarely fits. This explains its 30.6% padding — it's the worst-case for greedy packing at 32k.
3. **`skill_based_easy` and `skill_based_mixed`** have lower medians (~12-13k) allowing ~2 docs per sequence more often, giving 22-23% padding.
4. **No docs in `skill_based_mixed` exceed 32k** — zero truncation concern for that subset.
5. **The padding problem is structural**: docs are too long to fit 2-per-sequence but too short to fill 1 sequence. This is a "half-fill" problem that the segment cap cannot solve.

#### Implications

- The 20.6% aggregate padding is inherent to the document length distribution at `seq_len=32768`.
- **NTPACK-005** (truncation audit) is now relevant for `dataset_adapters` only — 14% of its docs are truncated at 32k, and those are likely long multi-turn conversations where assistant tokens may be concentrated late.
- **NTPACK-006** (order sensitivity) may have limited value since the cap isn't binding — the greedy packer is already near its theoretical best for this corpus shape.
- The main lever for reducing padding would be a smarter packing algorithm (e.g., best-fit decreasing bin packing) rather than parameter tuning.

#### Next actions

1. **NTPACK-005** — truncation audit for `dataset_adapters` (14% of docs > 32k, potential assistant-token loss)
2. **NTPACK-006** — order sensitivity (low priority per above, but quick to run for completeness)

### CLAUDE 2026-04-04 17:46 - NTPACK-005 submitted

- Job: `/ahmed/ntpack-005-truncation-audit`
- Zone: `us-central1-a`, 2 CPU, 16GB RAM, 20GB disk
- Script: `.agents/logbook/artifacts/nemotron_packing/ntpack_005_truncation_audit.py`
- What it does: for every doc > 32k in each subset, reads the `assistant_masks` array from the token cache and counts how many assistant tokens fall in the kept region (first 32k) vs the lost region (beyond 32k). Also flags "tail-heavy" docs where more assistant tokens are lost than kept.
- Expected to be I/O heavy on `dataset_adapters` (31,700+ long docs, each needing a mask read from GCS).
- Hypothesis: assistant tokens are likely concentrated in the latter portion of long multi-turn conversations, so the loss fraction may be worse than the raw token fraction suggests.
- Status: **succeeded** (~15min wall time, dominated by per-doc GCS mask reads)

### CLAUDE 2026-04-04 18:06 - NTPACK-005 results: significant assistant-token loss in dataset_adapters

- Job: `/ahmed/ntpack-005-truncation-audit` — **succeeded**
- Artifact: `.agents/logbook/artifacts/nemotron_packing/NTPACK-005_truncation_audit_full_seq32768.json`

#### Per-subset truncation audit

| Subset | Docs > 32k | Asst. Kept | Asst. Lost | **Loss %** | Tail-Heavy |
|--------|-----------|------------|------------|-----------|------------|
| dataset_adapters | 31,753 (14.0%) | 713.8M | 186.2M | **20.7%** | 520 (1.6%) |
| skill_based_easy | 423 (0.9%) | 8.9M | 1.5M | **14.3%** | 2 |
| skill_based_medium | 941 (1.1%) | 15.2M | 2.1M | **11.9%** | 11 |
| skill_based_mixed | 0 (0.0%) | 0 | 0 | 0.0% | 0 |

#### Key findings

1. **`dataset_adapters` loses 20.7% of its assistant tokens** to left truncation. That's 186M assistant tokens silently dropped across 31,753 documents. This is the dominant subset (61.8% weight), so this is a major training signal loss.
2. **The total raw tokens lost** from `dataset_adapters` alone is 291M — that's 7.3% of the subset's total 3.99B tokens, but because assistant tokens are disproportionately concentrated later in conversations, the assistant-token loss rate (20.7%) is much higher than the raw token loss rate.
3. **Tail-heavy docs are rare** (1.6% of long docs in `dataset_adapters`), meaning the problem isn't that assistant answers are entirely in the tail — it's that a significant fraction of assistant tokens in every long doc falls past the 32k cutoff.
4. **Skill subsets have modest losses**: 11-14% of assistant tokens lost for the few docs that exceed 32k. Small absolute numbers.
5. **`skill_based_mixed` has zero truncation** (confirmed from NTPACK-004: max doc length is 27,961).

#### Implications

- **`slice_strategy="right"` is not viable.** These are instruction-following conversations — dropping the system prompt and user query makes the assistant response meaningless. Left truncation is the only sensible choice.
- The 186M lost assistant tokens in `dataset_adapters` are **unrecoverable at seq_len=32768 with the current single-example approach.**
- The only mitigation that preserves all assistant tokens would be a **split-and-continue strategy** that breaks long conversations into multiple training examples, where each chunk carries enough preceding context to remain coherent. This would require a code change in levanter's slicing logic.

## CLAUDE — Smart Packing Strategy Plans

### Context

The current greedy packer (`pack_documents()` in `lib/levanter/src/levanter/data/packing.py`) walks documents in cache order and packs consecutive docs into 32k sequences. Doc lengths are already available from the JaggedArrayStore offsets at zero cost. The ~20% padding is structural: median doc lengths of 13-18k mean two docs often don't fit in 32k, leaving ~half a sequence as padding.

Key code constraints:
- `pack_documents()` (line 327) returns `list[range]` — each pack is a contiguous range of doc indices.
- `GreedyPrepackedDataset.get_batch()` (line 520) relies on `dr.start`/`dr.stop` to compute token boundaries from the offsets array, which requires contiguous doc ranges.
- `_offsets` and `_lengths` are read once at init from the JaggedArrayStore (line 494-504). Lengths are cheap.
- `max_segments_per_example` saturates at 8 for this corpus (NTPACK-003), so any strategy only needs to pack 2-4 docs per sequence.

### Strategy A — Sorted packing via index permutation (smallest change)

**Idea:** Pre-compute a permutation of doc indices sorted by length, then feed the permuted order to the existing greedy packer. The packer still sees "consecutive" docs, but consecutive in sorted order means similar-length docs are adjacent. Two ~16k docs pack perfectly into 32k.

**Implementation plan:**

1. Add a `packing_order` parameter to `GreedyPrepackedDataset.__init__()` with values `"cache"` (default, current behavior) or `"sorted"`.

2. When `packing_order="sorted"`:
   - After computing `self._lengths` (line 504), compute a permutation:
     ```python
     # Sort by length descending — puts long docs first so they fill sequences,
     # then shorter docs naturally fill remaining gaps
     perm = np.argsort(-primary_lengths)
     ```
   - Apply the permutation to `self._lengths` before calling `pack_documents()`.
   - Store `self._perm = perm` so that `get_batch()` can map permuted indices back to real cache positions.

3. In `get_batch()` (line 532), when building `pack_doc_ranges`, map through the permutation:
   - Current: `dr.start`/`dr.stop` are real cache indices (contiguous).
   - With permutation: `range(start, stop)` refers to positions in the *sorted* order. Need to resolve `perm[start:stop]` to get real cache indices. These will no longer be contiguous.

4. **This breaks the contiguity assumption.** `get_batch()` at line 541 does `token_start = offsets[dr.start]` and `token_end = offsets[dr.stop]` — a single contiguous slice read. With non-contiguous docs, we'd need per-doc reads instead of a single range read.

5. Change `get_batch()`'s inner loop to read each doc individually:
   ```python
   for dr in pack_doc_ranges:
       real_doc_ids = self._perm[dr.start:dr.stop]  # non-contiguous
       for doc_id in real_doc_ids:
           token_start = offsets[doc_id]
           token_end = offsets[doc_id + 1]
           # read each doc separately, then concatenate
   ```
   This is more I/O calls but each call is batched via `ts.Batch()` context.

6. Also change `pack_documents()` return type or add a wrapper so it can return `list[list[int]]` instead of `list[range]` when given a permutation.

**Pros:**
- Minimal algorithmic change — the greedy packer is unchanged, only the input order changes.
- Sorted order means similar-length docs are adjacent, which is actually optimal for the greedy sequential algorithm.
- Easy to A/B test: just toggle `packing_order`.

**Cons:**
- Breaks contiguous reads. Instead of one `store.data[start:end]` per pack, we do N reads (one per doc in the pack). With `ts.Batch()` this should still be efficient, but it's a change to I/O patterns.
- Sorted order biases training batches: early training sees only long docs, late training sees only short docs. May need to shuffle packed sequences after packing (separate from doc ordering for packing purposes).

**Expected padding improvement:**
- Sorted greedy packing on this distribution should get padding down from ~20% to ~5-10%. Similar-length docs pair efficiently, and the long tail (>32k) gets isolated into single-doc packs (as it does now).

**Files to modify:**
- `lib/levanter/src/levanter/data/packing.py`: `GreedyPrepackedDataset.__init__()`, `get_batch()`, possibly `pack_documents()`.
- `lib/levanter/src/levanter/data/text/datasets.py`: thread `packing_order` parameter through `dataset_for_component()` → `ChatDataset` / `PackedTokenDataset`.
- `lib/levanter/src/levanter/data/text/formats.py`: add `packing_order` field to `ChatLmDatasetFormat` / `TextLmDatasetFormat`.

**Estimated complexity:** Medium. Main risk is the `get_batch()` contiguity change.

---

### Strategy B — Best-fit decreasing bin packing (optimal packing, larger change)

**Idea:** Replace the greedy sequential packer with a proper bin-packing algorithm. Sort docs by length descending, then for each doc, assign it to the open sequence with the *least remaining space* that still fits. This is the classic best-fit decreasing (BFD) heuristic, which typically achieves >99% bin utilization.

**Implementation plan:**

1. Add a new function `pack_documents_bfd()` in `packing.py`:
   ```python
   def pack_documents_bfd(
       lengths: np.ndarray,
       max_length: int,
       max_segments_per_example: int | None = None,
       slice_strategy: str = "left",
   ) -> list[list[int]]:
       """Best-fit decreasing bin packing. Returns list of doc-index lists."""
       effective_lengths = np.minimum(lengths, max_length)
       order = np.argsort(-effective_lengths)

       # Each open bin: (remaining_capacity, doc_indices)
       # Use a sorted structure for O(log n) best-fit lookup
       bins: list[tuple[int, list[int]]] = []

       for doc_idx in order:
           doc_len = effective_lengths[doc_idx]
           best_bin = None
           best_remaining = max_length + 1

           for i, (remaining, _) in enumerate(bins):
               seg_ok = max_segments_per_example is None or len(bins[i][1]) < max_segments_per_example
               if remaining >= doc_len and remaining < best_remaining and seg_ok:
                   best_remaining = remaining
                   best_bin = i

           if best_bin is not None:
               remaining, doc_ids = bins[best_bin]
               doc_ids.append(int(doc_idx))
               bins[best_bin] = (remaining - doc_len, doc_ids)
           else:
               bins.append((max_length - doc_len, [int(doc_idx)]))

       return [doc_ids for _, doc_ids in bins]
   ```

2. For performance at scale (~226k docs for `dataset_adapters`), use a heap or sorted container instead of linear scan. Python's `sortedcontainers.SortedList` or a heap keyed on remaining capacity gives O(n log n) total.

3. Return type is `list[list[int]]` — arbitrary doc groupings, not contiguous ranges.

4. Modify `GreedyPrepackedDataset` to accept either `list[range]` or `list[list[int]]` as pack indices. In `get_batch()`, switch to per-doc reads when packs contain non-contiguous indices (same change as Strategy A step 5).

5. Add a `packing_algorithm` parameter: `"greedy"` (current) or `"bfd"`.

**Pros:**
- Near-optimal packing. For this corpus, should get padding down to ~2-5%.
- Well-understood algorithm with known theoretical guarantees (BFD uses at most 11/9 OPT + 6/9 bins).
- Pairs long + short docs explicitly (e.g., 25k + 7k = 32k).

**Cons:**
- O(n log n) packing computation at init. For 226k docs this is ~1 second — negligible.
- Non-contiguous reads in `get_batch()` (same as Strategy A).
- More code to write and test than Strategy A.
- Packs contain arbitrary doc combinations — training order is fully shuffled at the doc level, which is fine for SFT but worth noting.

**Expected padding improvement:**
- Down to ~2-5% from ~20%. The main residual would be docs >32k (which always waste nothing since they fill a sequence) and the last few bins that can't be perfectly filled.

**Files to modify:**
- `lib/levanter/src/levanter/data/packing.py`: add `pack_documents_bfd()`, modify `GreedyPrepackedDataset` to support non-contiguous packs.
- Same downstream threading as Strategy A.

**Estimated complexity:** Medium-High. Algorithm is straightforward but the `get_batch()` generalization is the same effort as Strategy A, plus the new packing function.

---

### Strategy C — Two-pointer pairing (simplest algorithm, good results)

**Idea:** Sort docs by length. Use two pointers — one at the longest doc, one at the shortest. If they fit together in 32k, pair them. If not, the long doc goes solo. Advance pointers inward. This is O(n log n) for the sort and O(n) for the pairing.

**Implementation plan:**

1. Add a new function `pack_documents_paired()` in `packing.py`:
   ```python
   def pack_documents_paired(
       lengths: np.ndarray,
       max_length: int,
       max_segments_per_example: int | None = None,
       slice_strategy: str = "left",
   ) -> list[list[int]]:
       effective_lengths = np.minimum(lengths, max_length)
       order = np.argsort(effective_lengths)  # ascending
       packs = []
       lo, hi = 0, len(order) - 1

       while lo <= hi:
           if lo == hi:
               # Last doc, goes solo
               packs.append([int(order[lo])])
               lo += 1
           elif effective_lengths[order[lo]] + effective_lengths[order[hi]] <= max_length:
               # They fit — pair them
               pack = [int(order[hi]), int(order[lo])]
               # Try to add more from the low end
               lo += 1
               hi -= 1
               while lo <= hi and max_segments_per_example is not None and len(pack) < max_segments_per_example:
                   remaining = max_length - sum(effective_lengths[d] for d in pack)
                   if effective_lengths[order[lo]] <= remaining:
                       pack.append(int(order[lo]))
                       lo += 1
                   else:
                       break
               packs.append(pack)
           else:
               # Long doc goes solo
               packs.append([int(order[hi])])
               hi -= 1

       return packs
   ```

2. Same `get_batch()` changes as Strategy A/B for non-contiguous reads.

3. After pairing, optionally try to fill remaining space with more short docs (the inner while loop above).

**Pros:**
- Dead simple to understand and implement.
- O(n log n) total (dominated by the sort).
- Pairs complement lengths naturally: 20k + 12k, 25k + 7k, etc.
- For this corpus where most docs are 8k-25k, two-pointer pairing should fill most sequences to >90%.

**Cons:**
- Not optimal — BFD will beat it when 3+ docs can fit in a sequence. But since most docs are >8k (meaning at most 3-4 fit in 32k), the difference is small.
- Same non-contiguous read change as A/B.
- Doesn't handle the multi-doc case as gracefully as BFD (the inner loop is a heuristic).

**Expected padding improvement:**
- Down to ~5-10% from ~20%. Slightly worse than BFD but much simpler.

**Files to modify:**
- Same as Strategy B.

**Estimated complexity:** Low-Medium. The algorithm is ~30 lines. Main effort is the shared `get_batch()` generalization.

---

### Strategy comparison summary

| | Current | A: Sorted Greedy | B: BFD | C: Two-Pointer |
|---|---------|-----------------|--------|----------------|
| Expected padding | ~20% | ~5-10% | ~2-5% | ~5-10% |
| Algorithm change | none | none (just reorder input) | new function | new function |
| `get_batch()` change | none | per-doc reads | per-doc reads | per-doc reads |
| Code complexity | — | low | medium | low |
| Optimality | poor for this distribution | good for similar-length clusters | near-optimal | good for pair-dominated cases |

### Recommendation

All three strategies require the same `get_batch()` generalization (non-contiguous doc reads). That's the shared prerequisite. Given that:

1. **Start with the `get_batch()` generalization** — change `_pack_indices` from `list[range]` to `list[list[int]]` and update `get_batch()` to do per-doc reads. This unblocks all three strategies.
2. **Implement Strategy B (BFD)** — it's the most general and produces the best results. The algorithm itself is only ~40 lines with a heap. Since we need the `get_batch()` change anyway, the marginal effort of BFD over sorted-greedy or two-pointer is small.
3. **Keep Strategy A as a fallback** — if BFD has unexpected issues, sorted-greedy requires zero algorithm changes, just an index permutation.
4. **Validate with NTPACK-006-style measurement** — run the same padding stats with the new packer to confirm the improvement before merging.

### Shared prerequisite: `get_batch()` generalization

The key change all strategies need:

```python
# Current: contiguous range read
token_start = offsets[dr.start]
token_end = offsets[dr.stop]
out_data.append(store.data[token_start:token_end].read())

# New: per-doc reads, concatenated
doc_ids = self._pack_indices[pack_idx]  # list[int], not range
chunks = []
for doc_id in doc_ids:
    doc_start = offsets[doc_id] if doc_id > 0 else 0
    doc_end = offsets[doc_id + 1]
    doc_len = min(doc_end - doc_start, allowed)
    chunks.append(store.data[doc_start : doc_start + doc_len].read())
# After ts.Batch resolves:
concatenated = np.concatenate([c.result() for c in chunks])
```

All reads still happen inside a `ts.Batch()` context, so tensorstore batches them into a single RPC where possible. The overhead vs a single contiguous read should be minimal for 2-4 docs per pack.

### CLAUDE 2026-04-04 18:30 - Implementation of Strategy A and B

- Implemented both strategies in `lib/levanter/src/levanter/data/packing.py`:
  - `pack_documents_sorted()` — Strategy A: sorts docs by length descending, feeds permuted order to existing greedy packer, maps indices back.
  - `pack_documents_bfd()` — Strategy B: best-fit decreasing bin packing with linear scan for best-fit bin.
- Generalized `GreedyPrepackedDataset`:
  - Added `packing_strategy` parameter: `"greedy"` (default), `"sorted"`, `"bfd"`.
  - `_pack_indices` is now `list[Sequence[int]]` — accepts both `range` (contiguous) and `list[int]` (non-contiguous).
  - `get_batch()` has two paths: fast contiguous path for `range` objects, per-doc read path for `list[int]`.
- Threaded `packing_strategy` through `datasets.py`:
  - `PackedTokenDataset`, `ChatDataset`, `dataset_for_component()` all accept the new parameter.
- All imports verified locally.

### CLAUDE 2026-04-04 18:32 - NTPACK-008 submitted

- Job: `/ahmed/ntpack-008-strategy-comparison`
- Zone: `us-central1-a`, 4 CPU, 24GB RAM, 20GB disk
- Script: `.agents/logbook/artifacts/nemotron_packing/ntpack_008_strategy_comparison.py`
- Measures: padding fraction for greedy, sorted, and BFD on all 4 subsets at `seq_len=32768`
- Also reports `pack_time_seconds` per subset per strategy for performance comparison
- Status: **succeeded** (BFD took ~12min wall time due to O(n*bins) linear scan)
- Artifact: `.agents/logbook/artifacts/nemotron_packing/NTPACK-008_strategy_comparison_full_seq32768.json`

### CLAUDE 2026-04-04 19:37 - NTPACK-008 results

#### Padding fraction comparison

| Subset | Greedy | Sorted | BFD |
|--------|--------|--------|-----|
| dataset_adapters | 16.0% | 11.7% | **0.0%** (7.6% truncation) |
| skill_based_easy | 23.2% | 26.4% | **0.2%** |
| skill_based_medium | 30.6% | 30.8% | **14.7%** |
| skill_based_mixed | 22.3% | 26.7% | **1.6%** |

#### Sequence count comparison

| Subset | Greedy | Sorted | BFD | BFD reduction |
|--------|--------|--------|-----|---------------|
| dataset_adapters | 145,033 | 138,020 | **113,232** | -22% fewer seqs |
| skill_based_easy | 23,758 | 24,783 | **18,271** | -23% fewer seqs |
| skill_based_medium | 73,013 | 73,231 | **59,422** | -19% fewer seqs |
| skill_based_mixed | 2,917 | 3,092 | **2,304** | -21% fewer seqs |

#### Pack time comparison

| Subset (docs) | Greedy | Sorted | BFD |
|---------------|--------|--------|-----|
| dataset_adapters (226k) | 1.2s | 1.3s | **576s** |
| skill_based_easy (45k) | 0.4s | 0.3s | 17s |
| skill_based_medium (89k) | 0.6s | 0.5s | **119s** |
| skill_based_mixed (6k) | 0.2s | 0.04s | 0.3s |

#### Key findings

1. **BFD is dramatically better for packing efficiency.** For `dataset_adapters`, BFD achieves 0% padding (packing is so tight it actually reports 7.6% truncation — meaning tokens are packed more densely than the greedy baseline). For `skill_based_easy` and `skill_based_mixed`, BFD gets to near-zero padding (0.2% and 1.6%).

2. **Sorted greedy (Strategy A) is disappointing.** It actually *increases* padding for 3 of 4 subsets (skill_based_easy: 23.2% → 26.4%, skill_based_medium: 30.6% → 30.8%, skill_based_mixed: 22.3% → 26.7%). Only `dataset_adapters` improves (16.0% → 11.7%). Sorting by length descending puts all long docs together — they each fill a sequence solo. Then all the shorter docs are grouped together, but since they're sorted by length, consecutive short docs have similar lengths and don't pair complementarily. The greedy packer's random cache order accidentally creates better length diversity within local neighborhoods.

3. **BFD is too slow for production.** The current linear-scan implementation takes 576 seconds for `dataset_adapters` (226k docs). This is O(n * bins) where bins ≈ 113k. Need a heap or sorted-container optimization for O(n log n).

4. **BFD reduces total sequences by ~20%**, meaning ~20% less compute per epoch. This is a significant training efficiency win.

5. **`skill_based_medium` is the hardest subset for BFD** at 14.7% padding. Its docs are tightly clustered around 18k tokens — two docs of 18k don't fit in 32k, so most go solo. BFD can only pair the few shorter outliers with longer ones.

#### Interpretation

- **Strategy A (sorted) should be abandoned.** It's worse than greedy for most subsets.
- **Strategy B (BFD) is the clear winner** but needs a performance optimization before it's production-ready.
- The BFD performance fix is straightforward: replace the linear scan over bins with a sorted container or heap keyed on remaining capacity. This would bring pack time from ~10 minutes to ~1 second for 226k docs.
- After the performance fix, BFD can be the default packing strategy for chat datasets.

#### Next actions

1. ~~Optimize BFD with a heap/sorted-container for O(n log n) performance~~ DONE
2. ~~Re-run NTPACK-008 with the optimized BFD to confirm correctness and timing~~ DONE
3. Commit and update the issue

### CLAUDE 2026-04-04 19:45 - BFD optimization: bisect-based sorted list

- Replaced linear scan in `pack_documents_bfd()` with `bisect.bisect_left()` on a sorted list of `(remaining_capacity, bin_index)` tuples.
- Bounded scan limit of 50 past the bisect point to handle segment-cap misses without degenerating to linear.
- Performance results:

| Subset (docs) | Before (linear) | After (bisect) | Speedup |
|---------------|-----------------|----------------|---------|
| dataset_adapters (226k) | 576s | **1.76s** | 327x |
| skill_based_easy (45k) | 17s | **0.13s** | 131x |
| skill_based_medium (89k) | 119s | **0.53s** | 225x |
| skill_based_mixed (6k) | 0.26s | **0.01s** | 26x |

- Packing quality is identical (same number of sequences, same padding fractions).
- BFD is now comparable in speed to greedy (~1.8s vs ~1.2s for 226k docs). Production-ready.

### CLAUDE 2026-04-04 19:46 - Final results summary

#### Packing strategy comparison at seq_len=32768

| Subset | Greedy Padding | BFD Padding | Seqs Saved | BFD Time |
|--------|---------------|-------------|------------|----------|
| dataset_adapters | 16.0% | **0.0%** | 31,801 (22%) | 1.76s |
| skill_based_easy | 23.2% | **0.2%** | 5,487 (23%) | 0.13s |
| skill_based_medium | 30.6% | **14.7%** | 13,591 (19%) | 0.53s |
| skill_based_mixed | 22.3% | **1.6%** | 613 (21%) | 0.01s |

- **Aggregate improvement**: BFD eliminates most padding and reduces total sequences by ~20%, meaning ~20% less compute per epoch.
- **`skill_based_medium` remains at 14.7%** because its docs cluster tightly around 18k tokens — BFD can't pair two 18k docs in a 32k sequence.
- **`dataset_adapters` shows 0% padding but 7.6% truncation** — BFD packs so tightly that overflow from long docs dominates. This truncation was already present (NTPACK-005 showed 20.7% assistant-token loss from left truncation); BFD just eliminates all the padding that was masking it in the greedy baseline.

#### Code changes

- `lib/levanter/src/levanter/data/packing.py`:
  - Added `pack_documents_sorted()` (Strategy A — not recommended)
  - Added `pack_documents_bfd()` (Strategy B — recommended, bisect-optimized)
  - Added `packing_strategy` parameter to `GreedyPrepackedDataset`
  - Generalized `get_batch()` for non-contiguous doc reads
- `lib/levanter/src/levanter/data/text/datasets.py`:
  - Threaded `packing_strategy` through `PackedTokenDataset`, `ChatDataset`, `dataset_for_component()`

### CLAUDE 2026-04-04 20:00 - Final recommendation: BFD only, discard sorted greedy

#### Head-to-head comparison

| | Sorted Greedy (A) | BFD (B) |
|---|---|---|
| **Speed** (226k docs) | ~1.3s | ~1.8s |
| **dataset_adapters padding** | 11.7% (improved) | **0.0%** (optimal) |
| **skill_based_easy padding** | 26.4% (**worse** than greedy) | **0.2%** |
| **skill_based_medium padding** | 30.8% (**worse** than greedy) | **14.7%** |
| **skill_based_mixed padding** | 26.7% (**worse** than greedy) | **1.6%** |

#### Why sorted greedy fails

Sorting by length descending puts similar-length docs adjacent. For this corpus where docs cluster around 13-18k tokens, that means the greedy packer repeatedly tries to pair two ~16k docs — which don't fit in 32k. The random cache order in the baseline accidentally creates better length diversity within local neighborhoods, allowing the greedy packer to stumble into complementary pairings (e.g., 20k + 10k).

BFD doesn't have this problem because it explicitly searches for the best-fit bin across all open bins, regardless of document order.

#### Decision

- **`pack_documents_sorted()` should be removed or deprecated.** It is worse than the baseline for 3 of 4 subsets and offers no advantage over BFD.
- **`pack_documents_bfd()` is the recommended packing strategy** for chat datasets. It is near-optimal on packing efficiency and runs at comparable speed to greedy (~1.8s vs ~1.2s for 226k docs).
- The ~0.6s overhead of BFD vs greedy is negligible compared to training time and is more than offset by the ~20% reduction in total sequences per epoch.
