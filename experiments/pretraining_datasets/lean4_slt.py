# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lean 4 Statistical Learning Theory corpus.

Downloads liminho123/lean4-stat-learning-theory-corpus and templates each row
into a text representation resembling a Lean 4 source file:

  -- SLT/Chaining.lean
  import Init
  import SLT.CoveringNumber
  import Mathlib.Analysis.SpecialFunctions.Pow.Real

  def dyadicScale (D : R) (k : N) : R := ...

  lemma dyadicScale_pos ...
"""

import logging
from dataclasses import dataclass

from datasets import load_dataset
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, tokenize
from zephyr import Dataset, ZephyrContext

from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

HF_DATASET_ID = "liminho123/lean4-stat-learning-theory-corpus"
HF_REVISION = "f4890ac"


def _path_to_import(path: str) -> str:
    """Convert a file path like '.lake/packages/mathlib/Mathlib/Foo/Bar.lean' to 'Mathlib.Foo.Bar'."""
    # Strip .lean extension
    if path.endswith(".lean"):
        path = path[:-5]

    # Strip .lake/packages/<pkg>/ prefix
    parts = path.split("/")
    if parts[0] == ".lake" and len(parts) > 3:
        parts = parts[3:]

    # Strip src/lean/ prefix (for Init)
    if len(parts) >= 2 and parts[0] == "src" and parts[1] == "lean":
        parts = parts[2:]

    return ".".join(parts)


def _row_to_text(row: dict) -> str:
    """Template a dataset row into Lean 4 source text."""
    lines = [f"-- {row['path']}"]

    for imp in row["imports"]:
        lines.append(f"import {_path_to_import(imp)}")

    lines.append("")

    for premise in row["premises"]:
        lines.append(premise["code"])
        lines.append("")

    return "\n".join(lines)


@dataclass(frozen=True)
class Lean4SLTConfig:
    output_path: str = ""


def build_lean4_slt(config: Lean4SLTConfig) -> None:
    ds = load_dataset(HF_DATASET_ID, "corpus", split="train", revision=HF_REVISION)
    records = [{"text": _row_to_text(row)} for row in ds]
    logger.info("Templated %d Lean 4 files", len(records))

    pipeline = Dataset.from_list(records).write_jsonl(
        f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz"
    )
    ctx = ZephyrContext(name="lean4-slt", max_workers=1)
    ctx.execute(pipeline)


lean4_slt = ExecutorStep(
    name="documents/lean4_slt",
    fn=build_lean4_slt,
    config=Lean4SLTConfig(
        output_path=this_output_path(),
    ),
    description="Template Lean 4 SLT corpus into pretraining text",
)

tokenized_lean4_slt = ExecutorStep(
    name="tokenized/lean4_slt",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[lean4_slt],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned(llama3_tokenizer),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[lean4_slt, tokenized_lean4_slt])
