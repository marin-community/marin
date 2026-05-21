# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transform `openai/gsm8k` train split into Q+A SFT JSONL with `messages` field."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import fsspec
import pyarrow.parquet as pq
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path, versioned
from marin.execution.remote import remote

from experiments.eval_datasets import gsm8k_raw

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GSM8KQATransformConfig:
    output_path: str
    gsm8k_raw_path: str
    split: str
    revision: str


def transform_gsm8k_qa(config: GSM8KQATransformConfig) -> None:
    pattern = os.path.join(config.gsm8k_raw_path, "main", f"{config.split}-*.parquet")
    fs, _ = fsspec.core.url_to_fs(pattern)
    matches = sorted(fs.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No parquet files matched {pattern}")

    output_file = os.path.join(config.output_path, "data.jsonl.gz")
    n_rows = 0
    with fsspec.open(output_file, "wt", compression="gzip") as out:
        for parquet_path in matches:
            full_uri = parquet_path if "://" in parquet_path else f"gs://{parquet_path}"
            with fsspec.open(full_uri, "rb") as raw:
                table = pq.read_table(raw, columns=["question", "answer"])
            for row in table.to_pylist():
                rec = {
                    "messages": [
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": row["answer"]},
                    ],
                    "id": n_rows,
                }
                out.write(json.dumps(rec) + "\n")
                n_rows += 1
    logger.info("Wrote %d GSM8K Q+A rows to %s", n_rows, output_file)


def build_gsm8k_qa_transform_step() -> ExecutorStep:
    return ExecutorStep(
        name="documents/downstream_scaling/sft/gsm8k_qa",
        fn=remote(transform_gsm8k_qa, resources=ResourceConfig.with_cpu(cpu=2, ram="4g")),
        config=GSM8KQATransformConfig(
            output_path=this_output_path(),
            gsm8k_raw_path=output_path_of(gsm8k_raw),
            split=versioned("train"),
            revision=versioned("e53f048"),
        ),
    )
