# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FineTranslations dataset definitions for the pretraining dataset CLI."""

from experiments.defaults import default_tokenize
from experiments.finetranslations.prepare_finetranslations import finetranslations_prepared, finetranslations_raw
from experiments.marin_models import marin_tokenizer
from fray.cluster import ResourceConfig

finetranslations_download = finetranslations_raw.step

finetranslations_tokenized = default_tokenize(
    "finetranslations_parallel",
    finetranslations_prepared / "**/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    worker_resources=ResourceConfig(ram="20g", disk="10g"),
)
