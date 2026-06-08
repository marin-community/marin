# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize and tokenize the PPL circuit coverage v2 plain-text corpus."""

from __future__ import annotations

from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import ExecutorStep, executor_main

from experiments.defaults import default_tokenize
from experiments.evals.ppl_circuit_coverage_v2 import (
    PLAIN_TEXT_PRETRAINING_SOURCE,
    PLAIN_TEXT_PRETRAINING_TARGET_TOKENS,
    ppl_circuit_coverage_v2_plain_text_pretraining_executor,
)
from experiments.marin_models import marin_tokenizer

RUN_KEY = "ppl_circuit_coverage_v2_plain_text_1b_issue6070_v2"

TOKENIZED = default_tokenize(
    name=RUN_KEY,
    dataset=ppl_circuit_coverage_v2_plain_text_pretraining_executor,
    tokenizer=marin_tokenizer,
    format=TextLmDatasetFormat(text_key="text"),
    tags=[
        f"source={PLAIN_TEXT_PRETRAINING_SOURCE}",
        "dataset_bundle=ppl_circuit_coverage_v2",
        "template=compact_v1",
        f"target_tokens={PLAIN_TEXT_PRETRAINING_TARGET_TOKENS}",
        "issue:6070",
    ],
)

STEPS: list[ExecutorStep] = [TOKENIZED]


if __name__ == "__main__":
    executor_main(
        STEPS,
        description="Materialize and tokenize the ~1B-token PPL circuit coverage v2 plain-text corpus.",
    )
