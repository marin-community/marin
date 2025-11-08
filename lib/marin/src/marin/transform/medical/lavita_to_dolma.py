# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transforming the Huggingface dataset lavita/medical-qa-datasets to dolma format.

Note: It may not be a good idea to use lavita's allprocessed subset since it is contaminated
with MMLU. We need to run it through a decontamination pipeline.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=1000 --cluster=us-central2 \
    lib/marin/src/marin/transform/medical/lavita_to_dolma.py \
    --input_path gs://marin-us-central2/raw/medical/lavita-medical-qa-datasets/ \
    --output_path gs://marin-data/processed/medical/lavita-v1.0/ \
    --subset pubmed-qa \
    --split train
"""

import hashlib
import os
from dataclasses import dataclass

import draccus
from zephyr import Dataset, flow_backend, load_parquet


@dataclass
class LavitaToDolmaConfig:
    """Configuration for Lavita medical QA dataset transformation."""

    input_path: str
    """Path to the lavita raw directory"""
    output_path: str
    """Path to store lavita dolma files"""
    subset: str
    """Subset type: all-processed, medmcqa, or pubmed-qa"""
    split: str
    """Dataset split to process (e.g., train, test, validation)"""


def lavita_pubmedqa_to_dolma(row):
    try:
        context_str_joined = "\n".join(row["CONTEXTS"])

        return {
            "id": (
                hashlib.sha256(
                    (context_str_joined + row["QUESTION"] + row["final_decision"] + row["LONG_ANSWER"]).encode("utf-8")
                ).hexdigest()
            ),
            "text": (
                context_str_joined
                + "\n\n"
                + row["QUESTION"]
                + "\n\n"
                + row["final_decision"]
                + "\n"
                + row["LONG_ANSWER"]
            ),
            "source": "lavita/medical-qa-datasets/pubmed-qa",
        }
    except Exception as e:
        print(e)
        return None


def lavita_allprocessed_to_dolma(row):
    try:
        return {
            "id": hashlib.sha256((row["instruction"] + row["input"] + row["output"]).encode("utf-8")).hexdigest(),
            "text": row["instruction"] + "\n\n" + "Context: \n" + row["input"] + "\n\n" + "Answer: \n" + row["output"],
            "source": "lavita/medical-qa-datasets/all-processed",
        }
    except Exception as e:
        print(e)
        return None


def lavita_medmcqa_to_dolma(row):
    try:
        answer_list = ["a", "b", "c", "d"]
        answer_str = answer_list[int(row["cop"])]

        if row["exp"] is None:
            explanation = ""
        else:
            explanation = row["exp"]

        return {
            "id": row["id"],
            "text": (
                (
                    row["question"]
                    + "\n\n"
                    + "Answer choices: \n"
                    + f"a. {row['opa']}\n"
                    + f"b. {row['opb']}\n"
                    + f"c. {row['opc']}\n"
                    + f"d. {row['opd']}\n"
                    + "Answer: "
                    + answer_str
                    + "\n"
                    + explanation
                ).strip()
            ),
            "source": "lavita/medical-qa-datasets/medmcqa",
        }
    except Exception as e:
        print(e)
        print(row)
        return None


def lavita_record_to_dolma(row: dict, subset: str):
    """Transform a single Lavita record to Dolma format.

    Args:
        row: Record from parquet file
        subset: Subset type (all-processed, medmcqa, or pubmed-qa)

    Returns:
        Transformed record in Dolma format, or None if invalid
    """
    if subset == "all-processed":
        result = lavita_allprocessed_to_dolma(row)
    elif subset == "medmcqa":
        result = lavita_medmcqa_to_dolma(row)
    elif subset == "pubmed-qa":
        result = lavita_pubmedqa_to_dolma(row)
    else:
        raise ValueError(f"Invalid subset: {subset}")

    if result and result.get("text") and len(result["text"]) > 0:
        return result
    return None


@draccus.wrap()
def convert_lavita_split_to_dolma(cfg: LavitaToDolmaConfig) -> None:
    """Transform Lavita medical QA data to Dolma format."""
    input_path = os.path.join(cfg.input_path, cfg.subset)

    backend = flow_backend()
    pipeline = (
        Dataset.from_files(input_path, f"*{cfg.split}*.parquet")
        .flat_map(load_parquet)
        .map(lambda row: lavita_record_to_dolma(row, subset=cfg.subset))
        .filter(lambda record: record is not None)
        .write_parquet(f"{cfg.output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    list(backend.execute(pipeline))


if __name__ == "__main__":
    convert_lavita_split_to_dolma()
