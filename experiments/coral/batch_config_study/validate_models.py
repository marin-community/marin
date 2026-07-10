# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print parameter counts for the Coral batch config study models."""

from experiments.coral.batch_config_study import MODEL_SPECS
from experiments.llama import llama3_tokenizer_vocab_size


def main() -> None:
    for model_spec in MODEL_SPECS:
        params = model_spec.model.total_trainable_params(llama3_tokenizer_vocab_size)
        print(f"{model_spec.name}\t{params:,}")


if __name__ == "__main__":
    main()
