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
"""

import hashlib
import os
from dataclasses import dataclass

import pandas as pd
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory


@dataclass
class LavitaToDolmaConfig:
    input_path: str
    output_path: str
    subset: str
    split: str


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


@ray.remote
@cached_or_construct_output(success_suffix="success")
def lavita_file_to_dolma(input_path: str, output_path: str, subset: str):
    df = pd.read_parquet(input_path)
    if subset == "all-processed":
        result_series = df.apply(lavita_allprocessed_to_dolma, axis=1)
    elif subset == "medmcqa":
        result_series = df.apply(lavita_medmcqa_to_dolma, axis=1)
    elif subset == "pubmed-qa":
        result_series = df.apply(lavita_pubmedqa_to_dolma, axis=1)
    else:
        raise ValueError(f"Invalid subset: {subset}")

    # Convert the series of dictionaries back to a DataFrame
    df = pd.DataFrame(result_series.tolist())
    df = df[df["text"].str.len() > 0]

    df.to_parquet(output_path)


def convert_lavita_split_to_dolma(config: LavitaToDolmaConfig):
    input_path = os.path.join(config.input_path, config.subset)
    futures = map_files_in_directory(
        lavita_file_to_dolma.remote, input_path, f"*{config.split}*.parquet", config.output_path, subset=config.subset
    )

    ray.get(futures)
