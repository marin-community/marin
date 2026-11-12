# Qwen3-130M with RMNPH

Run pure RMNPH on FineWeb-Edu with Paloma C4 validation. Hidden matrices use row-normalized momentum followed by a norm-preserving update. The output head uses AdamH; the input embedding uses ordinary row-normalized momentum; RMSNorm and other vector parameters use Adam. No Muon/MuonH switching is enabled.

## Recipe

| Setting | Value |
| --- | --- |
| Model | Qwen3-130M recipe; 6 layers, hidden size 512, intermediate size 1792, 8 heads |
| Sequence length / total batch | 4096 / 128 |
| Training updates | 4959 (2,599,944,192 tokens) |
| Embedding / vector LR | 0.012 / 0.060 |
| Hidden matrix / output head LR | 0.015 / 0.015 |
| Hyperball radius scale | 0.8 times each matrix's initialization norm; hidden matrices and head only |
| Schedule | 1000 linear LR warmup steps, then cosine decay to zero |
| Momentum / Nesterov | 0.95 / enabled |
| Adam beta1 / beta2 / epsilon | 0.8 / 0.98 / 1e-15 |
| Row normalization epsilon | 1e-10 |
| Weight decay setting / gradient clipping setting | 0.0 / 1.0; hidden/head/vector groups are clipped separately; embedding is not clipped |
| Seed / data seed | 0 / 42 |
| Precision | FP32 parameters, BF16 computation |
| Per-device microbatch | 8 by default; total batch remains 128 |

The radius is applied once to fresh hidden matrices and the output head. Full-state checkpoint restoration retains saved weights and optimizer state. Embeddings keep their original initialization scale. The head and embeddings are untied; `hybrid_norm=true` and gradient checkpointing are enabled; the normalization layout is defined in `lib/levanter/src/levanter/models/qwen.py`.

The earlier three-seed search recorded final C4 BPB values 1.175585, 1.177649, and 1.174204 (mean 1.175813). These are historical experiment records, not a completed validation of this upstream port. Hardware, code revision, data cache, and seed must accompany new results. See the [recorded search](https://github.com/Dominator-Index/Marin-RMNP/blob/3ea5c9d4a/Scripts/SWEEP_4LR_ADAPTIVE.md).

The research runs recorded `weight_decay=0.1`, but the default decay mask excluded embeddings and the other groups had no decay transform, so their effective decay was zero. This recipe states that behavior explicitly as `weight_decay=0.0`. Setting a nonzero weight decay now applies it to the already-selected embedding group; the hidden, head, and vector groups remain unchanged.

A separate eight-A100 seed-0 run of the research checkout completed with final C4 BPB **1.178139** (global step 4958, microbatch 8). It is not included in the earlier H100 three-seed mean and does not validate this upstream port. See the [A100 run](https://app.bandw.top/bxl307-university-of-birmingham/Marin-RMNP/runs/rmnph-best-r08-seed0).

## Data and environment

Use an installed Marin/Levanter environment. Set `PYTHON_BIN` to its interpreter. Prepare the complete [FineWeb-Edu pretokenized 10B cache](https://huggingface.co/datasets/marin-community/fineweb-edu-pretokenized-10B) and a tokenized Paloma C4 validation cache using `marin-community/marin-tokenizer`, EOS enforcement, and all six `c4_en/val/val-*.jsonl.gz` shards from `allenai/paloma` revision `65cd6fc` (2091 documents). Raw JSONL files are not a Levanter tokenized cache.

The training cache must contain `train/shard_ledger.json`; the evaluation cache must contain `validation/shard_ledger.json`. Both must be complete before training. Downloading the pretokenized training cache requires approximately 21.4 GB. Cache paths below are absolute paths on the training machine.

### Build the C4 evaluation cache

Run this once in the same installed environment. `EVAL_CACHE` must be an absolute output path.

```bash
export EVAL_CACHE=/path/to/paloma-c4-en-tokenized-marin
"$PYTHON_BIN" - <<'PYCODE'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig

raw = Path(snapshot_download("allenai/paloma", repo_type="dataset", revision="65cd6fc",
                            allow_patterns="c4_en/val/val-*.jsonl.gz"))
shards = sorted((raw / "c4_en/val").glob("val-*.jsonl.gz"))
assert len(shards) == 6
config = LmDataConfig(
    tokenizer="marin-community/marin-tokenizer", auto_build_caches=True, enforce_eos=True,
    components={"paloma/c4_en": DatasetComponent(
        cache_dir=os.environ["EVAL_CACHE"],
        source=UrlDatasetSourceConfig(validation_urls=[str(p) for p in shards]),
    )},
)
config.build_caches("validation")
PYCODE
```

Download the training cache with `huggingface_hub.snapshot_download` using repo ID
`marin-community/fineweb-edu-pretokenized-10B`, `repo_type="dataset"`, and `local_dir=TRAIN_CACHE`.

## Run

From the repository root:

```bash
export PYTHON_BIN=/path/to/venv/bin/python
export TRAIN_CACHE=/path/to/fineweb-edu-pretokenized-10B
export EVAL_CACHE=/path/to/paloma-c4-en-tokenized-marin
export MICRO_BATCH_SIZE=8
export SEED=0

bash Scripts/run_qwen130m.sh \
  --trainer.id qwen3-130m-rmnph-r08-seed0 \
  --trainer.checkpointer.base_path /path/to/output/rmnph-r08-seed0/checkpoints \
  --trainer.load_checkpoint false
```

Use a distinct trainer ID and checkpoint directory for each fresh run. Omit `--trainer.load_checkpoint false` when intentionally restoring a full-state checkpoint. To select W&B, append:

```bash
--trainer.tracker '[{"type":"json_logger"},{"type":"wandb","project":"Marin-RMNP","name":"qwen3-130m-rmnph-r08-seed0"}]'
```

Configure W&B authentication outside the repository. Network mirrors and proxy settings depend on the machine; credentials must not be included in commands committed to Git.

Report `eval/paloma/c4_en/bpb` at global step 4958 from the configured JSON logger or W&B tracker. Earlier intermediate evaluations do not replace the final evaluation. Repeat with `SEED=1` and `SEED=2` for a three-seed comparison.
