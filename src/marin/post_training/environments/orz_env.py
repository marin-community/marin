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

import logging
from typing import ClassVar

import datasets

from .math_env import MathEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ORZEnv(MathEnv):
    """
    Open-Reasoner-Zero (ORZ), the first open source implementation of large-scale reasoning-oriented RL
    training focusing on scalability, simplicity and accessibility. Using the same base model as
    DeepSeek-R1-Zero-Qwen-32B, their implementation achieves superior performance on AIME2024, MATH500, and
    the GPQA Diamond benchmark while demonstrating remarkable efficiency—requiring only a tenth of the
    training steps, compared to DeepSeek-R1-Zero pipeline.

    Paper: https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf
    HuggingFace: https://huggingface.co/datasets/Open-Reasoner-Zero/orz_math_72k_collection_extended

    Each example consists of two dicts: one for the user prompt and one for the answer. For example:

    {
        "from": "human",
        "value": "3B. A tower made of cards requires 2 cards for one level, 7 cards for two levels, and
        15 cards for a tower of three levels (see diagram). How many cards are needed for a tower of $n$ levels?"
    }

    {
        "from": "assistant",
        "ground_truth": {
            "value": "\\frac{n(3n+1)}{2}"
        }
    }

    Note: Some of the examples have user prompts that start with the question number (e.g., "3B. "), which
    may not be necessary for the model to understand the question. We could remove this prefix if needed,
    but it is kept here for consistency with the original dataset.

    @misc{hu2025openreasonerzeroopensourceapproach,
          title={Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model},
          author={Jingcheng Hu and Yinmin Zhang and Qi Han and Daxin Jiang and Xiangyu Zhang and Heung-Yeung Shum},
          year={2025},
          eprint={2503.24290},
          archivePrefix={arXiv},
          primaryClass={cs.LG},
          url={https://arxiv.org/abs/2503.24290},
    }
    """

    HF_DATASET_NAME: ClassVar[str] = "Open-Reasoner-Zero/orz_math_72k_collection_extended"

    # Since only a train set is available, allocate 1000 examples out of the 72.4K examples for evaluation
    DEV_SET_SIZE: ClassVar[int] = 1000

    # Seed for reproducibility when splitting the dataset
    SPLIT_RANDOM_SEED: ClassVar[int] = 42

    def __init__(self, tokenizer, **kwargs):
        def process_split(split: datasets.Dataset) -> list[dict]:
            examples: list[dict] = []
            for item in split:
                user_turn: dict = item["0"]
                assert user_turn["from"] == "human"
                prompt: str = self.add_instruction(user_turn["value"])

                assistant_turn: dict = item["1"]
                assert assistant_turn["from"] == "assistant" and "ground_truth" in assistant_turn
                answer: str = assistant_turn["ground_truth"]["value"]
                examples.append({"prompt": prompt, "answer": answer})
            return examples

        self.tokenizer = tokenizer

        # Load the dataset from HuggingFace with a fixed size for the dev set
        hf_dataset = datasets.load_dataset(self.HF_DATASET_NAME, split="train", trust_remote_code=True)
        splits = hf_dataset.train_test_split(test_size=self.DEV_SET_SIZE, seed=self.SPLIT_RANDOM_SEED, shuffle=True)

        self.train_examples = process_split(splits["train"])
        self.eval_examples = process_split(splits["test"])
        logger.info(
            f"Initialized ORZEnv with {len(self.train_examples)} train examples "
            f"and {len(self.eval_examples)} eval examples"
        )
