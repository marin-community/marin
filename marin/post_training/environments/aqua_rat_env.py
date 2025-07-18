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

    HuggingFace: https://huggingface.co/datasets/deepmind/aqua_rat

    @article{ling2017program,
      title={Program induction by rationale generation: Learning to solve and explain algebraic word problems},
      author={Ling, Wang and Yogatama, Dani and Dyer, Chris and Blunsom, Phil},
      journal={ACL},
      year={2017}
    }
    """

    HF_DATASET_NAME: ClassVar[str] = "deepmind/aqua_rat"

    INSTRUCTION: str = (
        "Show your work in <think> </think> tags. And return just the letter of the correct answer "
        "(one of A, B, C, D or E) as the final answer in <answer> </answer> "
        "tags. Assistant: Let me solve this step by step. <think>"
    )

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
            options: str = " ".join(item["options"])
            prompt: str = self.add_instruction(f"{question} {options}")
            answer: str = item["correct"]
            assert answer in ["A", "B", "C", "D", "E"]
            examples.append({"prompt": prompt, "answer": answer})
        return examples
