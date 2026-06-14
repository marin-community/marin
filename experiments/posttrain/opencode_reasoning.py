# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenCodeReasoning-2 SFT data surfaces."""

from levanter.data.text import ChatLmDatasetFormat
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, output_path_of, this_output_path
from marin.transform.conversation.opencode_reasoning import (
    OCR2_DEFAULT_SOURCE_REVISIONS,
    OPENCODE_REASONING_2_REVISION,
    OCR2HydrationConfig,
    OCR2TransformConfig,
    OCR2View,
    hydrate_ocr2_questions,
    transform_ocr2_sft,
)

from experiments.marin_models import marin_tokenizer
from experiments.tokenization import default_tokenize

OCR2_SOURCE_REVISIONS = OCR2_DEFAULT_SOURCE_REVISIONS

opencode_reasoning_2_hydration = ExecutorStep(
    name="documents/nvidia/OpenCodeReasoning-2/hydration",
    fn=hydrate_ocr2_questions,
    config=OCR2HydrationConfig(
        output_path=this_output_path(),
        source_revisions=OCR2_SOURCE_REVISIONS,
    ),
    override_output_path=f"documents/nvidia/OpenCodeReasoning-2/hydration-{OPENCODE_REASONING_2_REVISION[:7]}",
)

opencode_reasoning_2_python_r1_generation = ExecutorStep(
    name="documents/nvidia/OpenCodeReasoning-2/python/r1-generation",
    fn=transform_ocr2_sft,
    config=OCR2TransformConfig(
        output_path=this_output_path(),
        hydration_path=output_path_of(opencode_reasoning_2_hydration),
        split="python",
        view=OCR2View.R1_GENERATION,
    ),
    override_output_path=f"documents/nvidia/OpenCodeReasoning-2/python/r1-generation-{OPENCODE_REASONING_2_REVISION[:7]}",
)

opencode_reasoning_2_cpp_r1_generation = ExecutorStep(
    name="documents/nvidia/OpenCodeReasoning-2/cpp/r1-generation",
    fn=transform_ocr2_sft,
    config=OCR2TransformConfig(
        output_path=this_output_path(),
        hydration_path=output_path_of(opencode_reasoning_2_hydration),
        split="cpp",
        view=OCR2View.R1_GENERATION,
    ),
    override_output_path=f"documents/nvidia/OpenCodeReasoning-2/cpp/r1-generation-{OPENCODE_REASONING_2_REVISION[:7]}",
)

OPENCODE_REASONING_2_SFT_STEPS = {
    "nvidia/OpenCodeReasoning-2/python/r1-generation": opencode_reasoning_2_python_r1_generation,
    "nvidia/OpenCodeReasoning-2/cpp/r1-generation": opencode_reasoning_2_cpp_r1_generation,
}

opencode_reasoning_2_python_r1_generation_tokenized = default_tokenize(
    name="opencode_reasoning_2_python_r1_generation",
    dataset=opencode_reasoning_2_python_r1_generation / "**/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=ChatLmDatasetFormat(),
)

opencode_reasoning_2_cpp_r1_generation_tokenized = default_tokenize(
    name="opencode_reasoning_2_cpp_r1_generation",
    dataset=opencode_reasoning_2_cpp_r1_generation / "**/*.jsonl.gz",
    tokenizer=marin_tokenizer,
    format=ChatLmDatasetFormat(),
)

OPENCODE_REASONING_2_TOKENIZED_STEPS = {
    "nvidia/OpenCodeReasoning-2/python/r1-generation": opencode_reasoning_2_python_r1_generation_tokenized,
    "nvidia/OpenCodeReasoning-2/cpp/r1-generation": opencode_reasoning_2_cpp_r1_generation_tokenized,
}

if __name__ == "__main__":
    executor_main(
        steps=[
            opencode_reasoning_2_python_r1_generation_tokenized,
            opencode_reasoning_2_cpp_r1_generation_tokenized,
        ]
    )
