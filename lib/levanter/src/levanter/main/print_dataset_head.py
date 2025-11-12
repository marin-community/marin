# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List, Optional, Union

import jax.random as jrandom
import numpy as np

import levanter
from haliax import Axis
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfig, UrlSingleDatasetLMConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.logging import init_logging


@dataclass
class PrintDatasetHeadConfig:
    data: Union[SingleDatasetLMConfig, LMMixtureDatasetConfig] = field(default_factory=UrlSingleDatasetLMConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    z_loss_weight: float = 0.0
    log_entropy: bool = False

    seq_len: int = 4096
    seed: int = 0
    dataset_name: Optional[str] = None
    num_tokens: Optional[int] = None
    initial_batch_size: Optional[int] = None
    max_view_batches: Optional[List[int]] = None


@levanter.config.main()
def main(config: PrintDatasetHeadConfig):
    """Print the first token window for a dataset defined in a Levanter config."""
    init_logging(".", "print_dataset_head.log")

    tokenizer = config.data.the_tokenizer
    Pos = Axis("pos", config.seq_len)
    dataset_key = jrandom.PRNGKey(config.seed)
    train_sets_kwargs = dict(Pos=Pos, monitors=False, key=dataset_key)

    initial_batch_size = config.initial_batch_size
    if initial_batch_size is None:
        initial_batch_size = config.trainer.batch_schedule.batch_size_at_step(0)

    if isinstance(config.data, LMMixtureDatasetConfig):
        train_sets_kwargs["initial_batch_size"] = initial_batch_size

    datasets = config.data.train_sets(**train_sets_kwargs)

    dataset_name = config.dataset_name or getattr(config.data, "debug_print_dataset", None)
    if dataset_name is None:
        dataset_name = next(iter(datasets.keys()))

    if dataset_name not in datasets:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {sorted(datasets.keys())}")

    target_dataset = datasets[dataset_name]
    sync_dataset = target_dataset.as_sync_dataset()

    def display_sequence(sync_ds, *, description: str) -> None:
        banner = description.upper()
        print("\n\n" + "=" * 100)
        print(f"= {banner}")
        print("=" * 100)
        sequence_count = len(sync_ds)
        if sequence_count == 0:
            print("DATASET IS EMPTY (0 SEQUENCES).")
            return

        with use_cpu_device():
            example = sync_ds[0]

        if hasattr(example, "tokens"):
            token_array = np.asarray(example.tokens.array)
        else:
            token_array = np.asarray(example)

        if config.num_tokens is not None:
            token_array = token_array[: config.num_tokens]

        print("TOKEN IDS:")
        print(token_array.tolist())

        decoded = tokenizer.decode(token_array.tolist(), skip_special_tokens=False)
        print("\nDECODED TEXT (INCLUDING SPECIAL TOKENS):")
        print(decoded)
        print()

    if config.max_view_batches:
        total_sequences = len(sync_dataset)
        print(f"Dataset '{dataset_name}' contains {total_sequences} sequences before applying max_view_batches.")

        for max_batches in config.max_view_batches:
            if max_batches < 0:
                raise ValueError("max_view_batches entries must be non-negative integers.")

            num_sequences = max_batches * initial_batch_size
            if num_sequences > total_sequences:
                raise ValueError(
                    f"Requested {num_sequences} sequences (max_view_batches={max_batches}, "
                    f"initial_batch_size={initial_batch_size}) exceeds dataset size {total_sequences}."
                )

            print(f"\nSelecting {num_sequences} sequences from '{dataset_name}' with max_view_batches={max_batches}.")
            truncated_dataset = target_dataset.slice_dataset(end_index=num_sequences)
            truncated_sync = truncated_dataset.as_sync_dataset()
            display_sequence(
                truncated_sync,
                description=(f"MAX_VIEW_BATCHES={max_batches} | FIRST SEQUENCE FROM DATASET '{dataset_name}'"),
            )

        display_sequence(
            sync_dataset,
            description=f"NO MAX_VIEW_BATCHES CAP | FIRST SEQUENCE FROM DATASET '{dataset_name}'",
        )
    else:
        display_sequence(
            sync_dataset,
            description=f"DEFAULT VIEW | FIRST SEQUENCE FROM DATASET '{dataset_name}'",
        )


if __name__ == "__main__":
    main()
