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

from typing import ClassVar

import datasets

from .math_env import MathEnv


class OlymMathEnv(MathEnv):
    """
    Environment for the OlymMATH dataset, a novel Olympiad-level mathematical benchmark, designed to rigorously
    test the complex reasoning capabilities of LLMs. OlymMATH features 200 meticulously curated problems,
    each manually verified and available in parallel English and Chinese versions. The problems are
    systematically organized into two distinct difficulty tiers: (1) AIME-level problems (easy) that
    establish a baseline for mathematical reasoning assessment, and (2) significantly more challenging problems
    (hard) designed to push the boundaries of current state-of-the-art models.

    Paper: https://arxiv.org/abs/2503.21380
    HuggingFace: https://huggingface.co/datasets/RUC-AIBOX/OlymMATH
    GitHub: https://github.com/RUCAIBox/OlymMATH

    @misc{sun2025challengingboundariesreasoningolympiadlevel,
      title={Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models},
      author={Haoxiang Sun and Yingqian Min and Zhipeng Chen and Wayne Xin Zhao and Zheng Liu and Zhongyuan Wang
      and Lei Fang and Ji-Rong Wen},
      year={2025},
      eprint={2503.21380},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.21380},
    }
    """

    HF_DATASET_NAME: ClassVar[str] = "RUC-AIBOX/OlymMATH"
    SUPPORTED_DIFFICULTIES: ClassVar[list[str]] = ["easy", "hard"]
    SUPPORTED_LANGUAGES: ClassVar[list[str]] = ["en", "zh"]

    # Seed for reproducibility when splitting the dataset
    SPLIT_RANDOM_SEED: int = 42

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

        # Determine which subset of OlymMATH to use based on the initialization arguments
        difficulty: str = kwargs.get("difficulty", "easy")
        assert (
            difficulty in self.SUPPORTED_DIFFICULTIES
        ), f"Unsupported difficulty '{difficulty}'. Supported difficulties are: {self.SUPPORTED_DIFFICULTIES}"
        language: str = kwargs.get("language", "en")
        assert (
            language in self.SUPPORTED_LANGUAGES
        ), f"Unsupported language '{language}'. Supported languages are: {self.SUPPORTED_LANGUAGES}"
        hf_subset_name: str = f"{language}-{difficulty}"

        # Load the dataset from HuggingFace.
        # Since only a test split is available, reserve 20% for evals and 80% for training.
        hf_dataset = datasets.load_dataset(self.HF_DATASET_NAME, hf_subset_name, split="test", trust_remote_code=True)
        splits = hf_dataset.train_test_split(test_size=0.2, seed=self.SPLIT_RANDOM_SEED, shuffle=True)

        self.train_examples = []
        for item in splits["train"]:
            prompt = self.add_instruction(item["problem"])
            answer = item["answer"]
            self.train_examples.append({"prompt": prompt, "answer": answer})

        self.eval_examples = []
        for item in splits["test"]:
            prompt = self.add_instruction(item["problem"])
            answer = item["answer"]
            self.eval_examples.append({"prompt": prompt, "answer": answer})

        print(
            f"Initialized OlymMathEnv with {len(self.train_examples)} train examples "
            f"and {len(self.eval_examples)} eval examples"
        )
