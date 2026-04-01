# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FineTranslations dataset definitions for the pretraining dataset CLI."""

from levanter.data.text import TextLmDatasetFormat

from experiments.finetranslations.prepare_finetranslations import finetranslations_prepared, finetranslations_raw
from experiments.marin_models import marin_tokenizer
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

finetranslations_download = finetranslations_raw.step

finetranslations_tokenized = ExecutorStep(
    name="tokenized/finetranslations_parallel",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[finetranslations_prepared / "**/*.jsonl.gz"],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(marin_tokenizer),
        format=TextLmDatasetFormat(),
        worker_resources=ResourceConfig(ram="20g", disk="10g"),
    ),
)
