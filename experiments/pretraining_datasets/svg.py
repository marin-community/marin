# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SVG (nyuuzyou/svgfind) dataset download, normalization, and tokenization."""

from marin.datakit.download.svgfind import svgfind_creativecommons_normalize_steps
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.marin_models import marin_tokenizer

svg_normalized = svgfind_creativecommons_normalize_steps()[-1].as_executor_step()

svg_tokenized = ExecutorStep(
    name="tokenized/svg",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[output_path_of(svg_normalized, "outputs/main/*.parquet")],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(marin_tokenizer),
    ),
)


if __name__ == "__main__":
    executor_main(steps=[svg_tokenized])
