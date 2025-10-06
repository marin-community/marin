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

"""
The Paloma eval sets, downloaded and tokenized

https://huggingface.co/datasets/allenai/paloma
"""

import os.path

from src.marin.download.uncheatable_eval.download import make_uncheatable_eval_step

# cyclic dependency
# from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize import TokenizeConfig
from marin.processing.tokenize.data_configs import TokenizerStep

llama3_tokenizer = "meta-llama/Meta-Llama-3.1-8B"


# The datasets in the uncheatable eval set and their paths within the uncheatable eval repository
# https://huggingface.co/datasets/allenai/paloma
UNCHEATABLE_EVAL_TO_FILE_PATTERN = {
    # "wikipedia_arabic": "wikipedia_arabic_*.jsonl.gz",
    "wikipedia_english": "wikipedia_english_*.jsonl.gz",
    # "wikipedia_french": "wikipedia_french_*.jsonl.gz",
    # "wikipedia_german": "wikipedia_german_*.jsonl.gz",
    # "wikipedia_japanese": "wikipedia_japanese_*.jsonl.gz",
    # "wikipedia_spanish": "wikipedia_spanish_*.jsonl.gz",
    "github_python": "github_python_*.jsonl.gz",
    "github_cpp": "github_cpp_*.jsonl.gz",
    "bbc_news": "bbc_news_*.jsonl.gz",
    "arxiv_physics": "arxiv_physics_*.jsonl.gz",
    "arxiv_computer_science": "arxiv_computer_science_*.jsonl.gz",
    # "ao3_chinese": "ao3_chinese_*.jsonl.gz",
    "ao3_english": "ao3_english_*.jsonl.gz",
}

uncheatable_eval = make_uncheatable_eval_step()


def uncheatable_eval_tokenized(
    *, base_path="tokenized/", tokenizer: str = llama3_tokenizer, uncheatable_eval_raw: ExecutorStep = uncheatable_eval
) -> dict[str, TokenizerStep]:
    """
    Returns a dictionary of steps to tokenize the Paloma eval sets. Keys are the subset names (with `paloma/` prefix)
    """
    # avoid cyclic dependency
    from experiments.defaults import default_tokenize

    uncheatable_eval_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for dataset, path_part in UNCHEATABLE_EVAL_TO_FILE_PATTERN.items():
        uncheatable_eval_steps[os.path.join("uncheatable_eval", dataset)] = default_tokenize(
            name=os.path.join("uncheatable_eval", dataset),
            dataset=uncheatable_eval_raw.cd(f"{path_part}"),
            tokenizer=tokenizer,
            is_validation=True,
        )

    return uncheatable_eval_steps


if __name__ == "__main__":
    executor_main(steps=[uncheatable_eval, *uncheatable_eval_tokenized().values()])
