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


class OlympiadBenchEnv(MathEnv):
    """
    OlympiadBench is an Olympiad-level bilingual multimodal scientific benchmark, featuring 8,476 problems
    from Olympiad-level mathematics and physics competitions, including the Chinese college entrance exam.
    Each problem is detailed with expert-level annotations for step-by-step reasoning.

    In this environment, we use the text-only subsets of open-ended math and physics problems.

    An example physics problem from this dataset:

    Question:
    Consider an LC circuit with one inductor and one capacitor. The amplitude of the charge on the plates
    of the capacitor is $Q=10 \\mathrm{C}$ and the two plates are initially at a distance $d=1 \\mathrm{~cm}$
    away from each other. The plates are then slowly pushed together to a distance $0.5 \\mathrm{~cm}$ from
    each other. Find the resultant amplitude of charge on the parallel plates of the capacitor after this
    process is completed. Note that the initial current in the circuit is zero and assume that the plates
    are grounded.

    Final Answer:
    11.892

    Full Solution:
    In slow steady periodic processes (when the time for the change in parameters $\\tau$ is much less than
    the total systems frequency $f$ ), a quantity called the adiabatic invariant $I$ is conserved ${ }^{a}$.
    The adiabatic invariant corresponds to the area of a closed contour in phase space (a graph with momentum
    $p$ and position $x$ as its axes). Note the we can electrostatically map this problem to a mechanics one
    as the charge corresponds to position, while the momentum would correspond to $L I$ where $I$ is the current
    and $L$ is the inductance. Thus, in phase space, we have an elliptical contour corresponding to the equation:
    $\\frac{Q^{2}}{2 C}+\\frac{(L I)^{2}}{2 L}=C$ where $C$ is a constant in the system. As the area under the curve
    is conserved, then it can be written that $\\pi Q_{0} L I_{0}=\\pi Q_{f} L I_{f}$. It is also easy to conserve
    energy such that $L I^{2}=\\frac{Q^{2}}{C}$ which tells us $I=\\frac{Q}{\\sqrt{L C}}$. As $C \\propto 1 / x$,
    we then can write the adiabatic invariant as $x q^{4}$ which tells us $Q_{f}=\\sqrt[4]{2} Q$.\nWe can also solve
    this regularly by looking at the changes analytically. From Gauss's law, the electric field between the plates
    of the capacitators initially can be estimated as\n\n$$\nE=\\frac{Q}{2 \\varepsilon_{0} A}\n$$\n\nwhere $A$
    is the area of the plate. The plates of the capacitator is attracted to the other one with a force
    of\n\n$$\nF=Q E=\\frac{Q^{2}}{2 \\varepsilon_{0} A}\n$$\n\nThe charges of the plates as a function of time
    can be approximated as\n\n$$\nQ_{c}= \\pm Q \\sin (\\omega t+\\phi)\n$$\n\nwhere $\\omega=\\frac{1}{\\sqrt{L C}}$.
    Using this equation, we estimate the average force $\\langle F\\rangle$ applied on the plate after a period of
    oscillations to be\n\n$$\n\\langle F\\rangle=\\frac{\\left\\langle Q^{2}\\right\\rangle}{2
    \\varepsilon_{0} A}=\\frac{Q^{2}}{2 \\varepsilon_{0} A}\\left\\langle\\sin ^{2}(\\omega t+\\phi)\\right
    \\rangle=\\frac{Q^{2}}{2 \\varepsilon_{0} A} \\cdot\\left(\\frac{1}{2 \\pi} \\int_{0}^{2 \\pi} \\sin ^{2}(x)
    d x\\right)=\\frac{Q^{2}}{4 \\varepsilon_{0} A}\n$$\n\nthis means that after one period, the amount of work
    done to push the plates closer together is given by\n\n$$\nW_{F}=\\langle F\\rangle d x=\\frac{Q^{2}}{4
    \\varepsilon_{0} A} d x\n$$\n\nIn this cycle, the amount of incremental work done by the $\\mathrm{LC}$
    circuit will be given by\n\n$$\nd W_{\\mathrm{LC}}=\\Delta(F x)=\\Delta\\left(\\frac{Q^{2} x}{2
    \\varepsilon_{0} A}\\right)=\\frac{Q x}{\\varepsilon_{0} A} d Q+\\frac{Q^{2}}{2 \\varepsilon_{0} A}
    d x\n$$\n\n\n\nFrom conservation of energy, $W_{F}=W_{L C}$. Or in other words,\n\n$$\n\\frac{Q^{2}}{4
    \\varepsilon_{0} A} d x=\\frac{Q x}{\\varepsilon_{0} A} d Q+\\frac{Q^{2}}{2 \\varepsilon_{0} A} d
    x\n$$\n\nsimplifying gives us\n\n$$\n\\begin{aligned}\n\\frac{Q x}{\\varepsilon_{0} A} d Q & =-\\frac{Q^{2}}{4
    \\varepsilon_{0} A} d x \\\\\n\\frac{1}{4} \\int \\frac{d x}{x} & =-\\int \\frac{d Q}{Q} \\\\\n\\frac{1}{4}
    \\ln x+\\ln Q & =\\text { const. }\n\\end{aligned}\n$$\n\nWe now find our adiabatic invariant to be\n\n$$\nx
    Q^{4}=\\text { const. }\n$$\n\nSubstituting values into our equation, we find that\n\n$$\nd Q_{i}^{4}=\\frac{d}{2}
    Q_{f}^{4} \\Longrightarrow Q_{f}=\\sqrt[4]{2} Q=11.892 \\mathrm{C}\n$$[^0]

    Paper: https://arxiv.org/abs/2402.14008
    HuggingFace: https://huggingface.co/datasets/Hothan/OlympiadBench

    @article{he2024olympiadbench,
      title={Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual
             multimodal scientific problems},
      author={He, Chaoqun and Luo, Renjie and Bai, Yuzhuo and Hu, Shengding and Thai, Zhen Leng and Shen, Junhao
              and Hu, Jinyi and Han, Xu and Huang, Yujie and Zhang, Yuxiang and others},
      journal={arXiv preprint arXiv:2402.14008},
      year={2024}
    }
    """

    HF_DATASET_NAME: ClassVar[str] = "Hothan/OlympiadBench"

    SUPPORTED_LANGUAGES: ClassVar[list[str]] = ["en", "zh"]
    SUPPORTED_SUBJECTS: ClassVar[list[str]] = ["maths", "physics"]

    # Since only a train set is available, allocate 100 examples for evaluation
    DEV_SET_SIZE: ClassVar[int] = 100

    # Seed for reproducibility when splitting the dataset
    SPLIT_RANDOM_SEED: ClassVar[int] = 42

    def __init__(self, tokenizer, **kwargs):
        def process_split(split: datasets.Dataset) -> list[dict]:
            examples: list[dict] = []
            for item in split:
                context: str | None = item["context"]
                question: str = item["question"]

                # For some physics problems, the context is needed to answer the question
                question = f"{context}\n{question}" if context else question

                prompt: str = self.add_instruction(question)
                answer: str = item["final_answer"][0]
                examples.append({"prompt": prompt, "answer": answer})
            return examples

        self.tokenizer = tokenizer

        # Determine which subset of OlympiadBench to use based on the initialization arguments
        subject: str = kwargs.get("subject", "maths")
        assert (
            subject in self.SUPPORTED_SUBJECTS
        ), f"Unsupported subject '{subject}'. Supported difficulties are: {self.SUPPORTED_SUBJECTS}"
        language: str = kwargs.get("language", "en")
        assert (
            language in self.SUPPORTED_LANGUAGES
        ), f"Unsupported language '{language}'. Supported languages are: {self.SUPPORTED_LANGUAGES}"

        hf_subset_name: str = f"OE_TO_{subject}_{language}_COMP"

        # Load the dataset from HuggingFace with a fixed size for the dev set
        hf_dataset = datasets.load_dataset(self.HF_DATASET_NAME, hf_subset_name, split="train", trust_remote_code=True)
        splits = hf_dataset.train_test_split(test_size=self.DEV_SET_SIZE, seed=self.SPLIT_RANDOM_SEED, shuffle=True)

        self.train_examples = process_split(splits["train"])
        self.eval_examples = process_split(splits["test"])
        logger.info(
            f"Initialized OlympiadBenchEnv with {len(self.train_examples)} train examples "
            f"and {len(self.eval_examples)} eval examples"
        )
