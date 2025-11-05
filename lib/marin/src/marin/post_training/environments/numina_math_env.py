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
from .math_utils import last_boxed_only_string, remove_boxed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NuminaMathEnv(MathEnv):
    """
    Approximately 860k math problems, where each solution is formatted in a Chain of Thought (CoT) manner.
    The sources of the dataset range from Chinese high school math exercises to US and international mathematics
    olympiad competition problems. The data were primarily collected from online exam paper PDFs and mathematics
    discussion forums. The processing steps include (a) OCR from the original PDFs, (b) segmentation into
    problem-solution pairs, (c) Translation into English, (d) realignment to produce a CoT reasoning format,
    and (e) final answer formatting.

    Source breakdown:
    Source	Number of Samples
    aops_forum	30201
    amc_aime	4072
    cn_k12	276591
    gsm8k	7345
    math	7478
    olympiads	150581
    orca_math	153334
    synthetic_amc	62111
    synthetic_math	167895

    HuggingFace: https://huggingface.co/datasets/AI-MO/NuminaMath-CoT

    @misc{numina_math_datasets,
      author = {Jia LI and Edward Beeching and Lewis Tunstall and Ben Lipkin and Roman Soletskyi
      and Shengyi Costa Huang and Kashif Rasul and Longhui Yu and Albert Jiang and Ziju Shen and Zihan Qin
      and Bin Dong and Li Zhou and Yann Fleureau and Guillaume Lample and Stanislas Polu},
      title = {NuminaMath},
      year = {2024},
      publisher = {Numina},
      journal = {Hugging Face repository},
      howpublished = {https://huggingface.co/AI-MO/NuminaMath-CoT}
      (https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf)}}
    }
    """

    HF_DATASET_NAME: ClassVar[str] = "AI-MO/NuminaMath-CoT"

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

        # Load the dataset from HuggingFace and generate splits
        hf_dataset = datasets.load_dataset(self.HF_DATASET_NAME, streaming=True, trust_remote_code=True)
        self.train_examples = self._process_split(hf_dataset["train"])
        self.eval_examples = self._process_split(hf_dataset["test"])

        logger.info(
            f"Initialized NuminaMathEnv with {len(self.train_examples)} train examples "
            f"and {len(self.eval_examples)} eval examples"
        )

    def _process_split(self, split) -> list[dict]:
        """Process a dataset split into prompt-answer pairs with boxed answers only."""
        examples: list[dict] = []
        for item in tqdm(split):
            try:
                answer: str = item["solution"]
                answer = remove_boxed(last_boxed_only_string(answer))
            except Exception as e:
                logger.error(f"Error processing item {item}: {e}")
                continue

            prompt: str = self.add_instruction(item["problem"])
            examples.append({"prompt": prompt, "answer": answer})
        return examples
