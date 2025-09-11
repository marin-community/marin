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
from tqdm.auto import tqdm

from .math_env import MathEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AquaRatEnv(MathEnv):
    """
    AQUA-RAT (Algebra Question Answering with Rationales) is a large (~100K) crowdsourced collection of
    multiple-choice algebraic/math word problems where each item includes the problem statement,
    five answer options (A-E), a natural-language step-by-step rationale explaining the solution,
    and the correct option.

    For this environment, we converted the multiple-choice problems to an open-ended format.

    HuggingFace: https://huggingface.co/datasets/deepmind/aqua_rat

    @article{ling2017program,
      title={Program induction by rationale generation: Learning to solve and explain algebraic word problems},
      author={Ling, Wang and Yogatama, Dani and Dyer, Chris and Blunsom, Phil},
      journal={ACL},
      year={2017}
    }
    """

    HF_DATASET_NAME: ClassVar[str] = "deepmind/aqua_rat"

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

        # Load the dataset from HuggingFace and generate splits
        hf_dataset = datasets.load_dataset(self.HF_DATASET_NAME, trust_remote_code=True)
        self.train_examples = self._process_split(hf_dataset["train"])
        self.eval_examples = self._process_split(hf_dataset["validation"])

        logger.info(
            f"Initialized AquaRatEnv with {len(self.train_examples)} train examples "
            f"and {len(self.eval_examples)} eval examples"
        )

    def _process_split(self, split) -> list[dict]:
        examples: list[dict] = []
        for item in tqdm(split):
            question: str = item["question"]
            letter_answer: str = item["correct"]
            answer_index: int = ord(letter_answer) - ord("A")
            # Options look like this ['A)21', 'B)21.5', 'C)22', 'D)22.5', 'E)23']
            raw_answer: str = item["options"][answer_index]
            assert raw_answer.startswith(f"{letter_answer})")
            # Remove the letter and closing parenthesis, e.g., 'A)21' -> '21'
            answer: str = raw_answer.replace(f"{letter_answer})", "").strip()

            # Since we converted the multiple-choice problems to an open-ended format,
            # we skip any questions where the correct answer is "none" or similar,
            # which requires the choices to be provided.
            if answer.lower().startswith("none"):
                logger.warning(f"Skipping question '{question}' with answer '{answer}'.")
                continue

            prompt: str = self.add_instruction(question)
            examples.append({"prompt": prompt, "answer": answer})
        return examples
