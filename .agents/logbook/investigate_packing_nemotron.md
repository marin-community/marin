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
