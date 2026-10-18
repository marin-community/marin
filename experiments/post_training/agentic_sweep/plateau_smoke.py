"""CPU smoke run of plateau dropping on a slice of the #202 sweep rows with a tiny Llama.

The rows are the real prepared rows (same tokens and assistant masks), cut down to fit a CPU:
an equal number of rows per family, a window of each row starting just before its first
supervised token, and token ids remapped to a compact vocabulary (the most frequent ids in the
slice, plus one bucket for the rest). It checks the mechanics and logs the per-epoch statistics;
it says nothing about how fast the 67B model's per-token losses plateau.

    uv run python -m experiments.post_training.agentic_sweep.plateau_smoke \
        --prebuilt train-prebuilt.jsonl.gz --index train-index.jsonl --output /tmp/plateau-smoke
"""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from levanter.checkpoint import CheckpointerConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.main.train_lm import TrainLmConfig, main
from levanter.models.llama import LlamaConfig
from levanter.optim.config import AdamConfig
from levanter.plateau import PlateauDroppingConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from experiments.post_training.agentic_sweep.callbacks import EpochTrackerConfig, SweepExportsConfig
from experiments.post_training.agentic_sweep.train import BATCH_SIZE


def select_rows(prebuilt: Path, index: Path, per_family: int) -> list[dict]:
    """The first ``per_family`` training rows of each family, in file order."""
    wanted: dict[int, str] = {}
    counts: Counter[str] = Counter()
    for line in index.read_text().splitlines():
        entry = json.loads(line)
        if entry["split"] == "train" and counts[entry["family"]] < per_family:
            counts[entry["family"]] += 1
            wanted[entry["row"]] = entry["family"]
    rows = []
    with gzip.open(prebuilt, "rt") as lines:
        for number, line in enumerate(lines):
            if number in wanted:
                rows.append({"family": wanted[number], **json.loads(line)})
            if len(rows) == len(wanted):
                break
    return rows


def window(row: dict, length: int, lead: int) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(row["loss_weights"], dtype=np.float32)
    start = max(0, int(np.flatnonzero(weights)[0]) - lead)
    start = min(start, len(weights) - length)
    return np.asarray(row["input_ids"][start : start + length]), weights[start : start + length]


def compact_vocabulary(windows: list[np.ndarray], size: int) -> dict[int, int]:
    """Map the most frequent ids to 2..size-1; 0 is padding and 1 holds every other id."""
    counts = Counter(int(token) for ids in windows for token in ids)
    return {token: rank + 2 for rank, (token, _) in enumerate(counts.most_common(size - 2))}


def run(args) -> list[dict]:
    rows = select_rows(Path(args.prebuilt), Path(args.index), args.per_family)
    windows = [window(row, args.length, args.lead) for row in rows]
    vocabulary = compact_vocabulary([ids for ids, _ in windows], args.vocab)
    examples = []
    for ids, weights in windows:
        tokens = np.array([vocabulary.get(int(token), 1) for token in ids], dtype=np.int32)
        examples.append(GrugLmExample.causal(jnp.asarray(tokens), loss_weight=jnp.asarray(weights)))
    if len(examples) % BATCH_SIZE:
        raise ValueError(f"{len(examples)} rows do not fill batches of {BATCH_SIZE}")
    steps_per_epoch = len(examples) // BATCH_SIZE
    supervised = int(sum(float(example.loss_weight.sum()) for example in examples))
    output = Path(args.output)
    tokenizer_path = output / "tokenizer"
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel(
                {"<pad>": 0, "<other>": 1, **{f"t{i}": i for i in range(2, args.vocab)}}, unk_token="<other>"
            )
        ),
        pad_token="<pad>",
        eos_token="<other>",
    ).save_pretrained(tokenizer_path)
    data = LmDataConfig(
        components={"train": DirectDatasetComponent(datasets={"train": ListAsyncDataset(examples)})},
        tokenizer=str(tokenizer_path),
        shuffle=False,
        mixture_block_size=BATCH_SIZE,
    )
    validation = LmDataConfig(
        components={"tmax": DirectDatasetComponent(datasets={"validation": ListAsyncDataset(examples[:BATCH_SIZE])})},
        tokenizer=str(tokenizer_path),
        shuffle=False,
    )
    config = TrainLmConfig(
        data=data,
        train_seq_len=args.length,
        model=LlamaConfig(
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            max_seq_len=args.length,
            hidden_dim=128,
            intermediate_dim=256,
            attn_backend=None,
            reference_checkpoint=str(tokenizer_path),
        ),
        optimizer=AdamConfig(
            learning_rate=args.learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.0,
            max_grad_norm=1.0,
            warmup=0,
            lr_schedule="constant",
        ),
        trainer=TrainerConfig(
            id="plateau-smoke",
            seed=42,
            train_batch_size=BATCH_SIZE,
            per_device_parallelism=BATCH_SIZE,
            num_train_steps=args.epochs * steps_per_epoch,
            load_checkpoint=False,
            tracker=(EpochTrackerConfig(str(output), steps_per_epoch, supervised), JsonLoggerConfig()),
            checkpointer=CheckpointerConfig(
                base_path=str(output / "checkpoints"), append_run_id_to_base_path=False, keep=[], save_interval=None
            ),
            require_accelerator=False,
            log_jaxprs=False,
            log_xla_hlo=False,
            distributed=DistributedConfig(initialize_jax_distributed=False),
        ),
        adapter=SweepExportsConfig(validation, steps_per_epoch, args.length),
        hf_save_path=str(output / "hf"),
        data_seed=42,
        plateau_dropping=PlateauDroppingConfig(),
    )
    main(config)
    epochs = [json.loads(line) for line in (output / "epoch-metrics.jsonl").read_text().splitlines()]
    summary = {
        "rows": Counter(row["family"] for row in rows),
        "window": args.length,
        "vocabulary": args.vocab,
        "in_vocabulary_share": float(
            np.mean([vocabulary.get(int(token), 1) != 1 for ids, _ in windows for token in ids])
        ),
        "supervised_tokens_per_epoch": supervised,
        "learning_rate": args.learning_rate,
        "epochs": epochs,
    }
    (output / "smoke-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return epochs


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prebuilt", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-family", type=int, default=32)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--lead", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    epochs = run(parser.parse_args())
    for row in epochs:
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "epoch",
                        "train_loss",
                        "plateau_dropped_fraction",
                        "plateau_kept_loss",
                        "plateau_dropped_loss",
                    )
                }
            )
        )


if __name__ == "__main__":
    main_cli()
