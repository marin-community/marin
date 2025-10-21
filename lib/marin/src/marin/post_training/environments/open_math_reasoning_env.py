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
import random
from typing import ClassVar

import datasets
from tqdm.auto import tqdm

from .math_env import MathEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenMathReasoningEnv(MathEnv):
    """
    OpenMathReasoning is a large-scale math reasoning dataset for training large language models (LLMs).

    This dataset contains 306K unique mathematical problems sourced from AoPS forums with:
        * 3.2M long chain-of-thought (CoT) solutions
        * 1.7M long tool-integrated reasoning (TIR) solutions
        * 566K samples that select the most promising solution out of many candidates (GenSelect)
    There is also an additional 193K problems sourced from AoPS forums (problems only, no solutions)

    The authors used Qwen2.5-32B-Instruct to preprocess problems, and DeepSeek-R1 and
    QwQ-32B to generate solutions.

    Paper: https://arxiv.org/abs/2504.16891
    HuggingFace: https://huggingface.co/datasets/nvidia/OpenMathReasoning

    @article{moshkov2025aimo2,
      title   = {AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with
                 OpenMathReasoning dataset},
      author  = {Ivan Moshkov and Darragh Hanley and Ivan Sorokin and Shubham Toshniwal and Christof Henkel and
                 Benedikt Schifferer and Wei Du and Igor Gitman},
      year    = {2025},
      journal = {arXiv preprint arXiv:2504.16891}
    }
    """

    HF_DATASET_NAME: ClassVar[str] = "nvidia/OpenMathReasoning"

    # Seed for reproducibility when splitting the dataset
    SPLIT_RANDOM_SEED: int = 42

    VAL_SIZE: ClassVar[int] = 1000

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

        # Load the dataset from HuggingFace.
        hf_dataset = datasets.load_dataset(self.HF_DATASET_NAME, split="cot", streaming=True, trust_remote_code=True)

        all_examples: list[dict] = []
        seen_examples: set[str] = set()

        for row in tqdm(hf_dataset, desc="Processing OpenMathReasoning dataset"):
            problem: str = row["problem"]
            answer: str | int | float = row["expected_answer"]
            key: str = f"{problem}_{answer}"
            if key in seen_examples:
                continue

            seen_examples.add(key)
            prompt: str = self.add_instruction(problem)
            all_examples.append({"prompt": prompt, "answer": answer})

        # Fix random seed before splitting the dataset for reproducibility
        random.seed(self.SPLIT_RANDOM_SEED)

        # Reserve 1000 examples for evaluation, the rest for training
        random.shuffle(all_examples)
        split_index: int = len(all_examples) - self.VAL_SIZE
        self.train_examples = all_examples[:split_index]
        self.eval_examples = all_examples[split_index:]

        logger.info(
            f"Initialized OpenMathReasoningEnv with {len(self.train_examples)} train examples "
            f"and {len(self.eval_examples)} eval examples"
        )
