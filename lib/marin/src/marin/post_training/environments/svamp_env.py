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

import csv
import logging
import os
import tempfile
import zipfile
from io import BytesIO

import requests
from tqdm.auto import tqdm

from .math_env import MathEnv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SVAMPEnv(MathEnv):
    """
    SVAMP test a model across different aspects of solving math word problems (MWPs).

    For this environment, we use the `cv_svamp_augmented` subset that contains MAWPS, ASDiv-A and SVAMP.
    In each fold, the test set consists of problems from only SVAMP while the train set consists of
    problems from the rest of SVAMP and complete MAWPS and ASDiv-A.

    An example:
    Prompt: bobby ate some pieces of candy . then he ate 25 more . if he ate a total of 43 pieces of
    candy how many pieces of candy had he eaten at the start ?
    Show your work in <think> </think> tags. And return the final answer in <answer> </answer>
    tags. Assistant: Let me solve this step by step. <think>
    Answer: 18

    Dataset: https://github.com/arkilpatel/SVAMP

    @inproceedings{patel-etal-2021-nlp,
        title = "Are {NLP} Models really able to Solve Simple Math Word Problems?",
        author = "Patel, Arkil  and
          Bhattamishra, Satwik  and
          Goyal, Navin",
        booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association
        for Computational Linguistics: Human Language Technologies",
        month = jun,
        year = "2021",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2021.naacl-main.168",
        doi = "10.18653/v1/2021.naacl-main.168",
        pages = "2080--2094",
        abstract = "The problem of designing NLP solvers for math word problems (MWP) has seen sustained
        research activity and steady gains in the test accuracy. Since existing solvers achieve high performance
        on the benchmark datasets for elementary level MWPs containing one-unknown arithmetic word problems,
        such problems are often considered "solved" with the bulk of research attention moving to more
        complex MWPs. In this paper, we restrict our attention to English MWPs taught in grades four and lower.
        We provide strong evidence that the existing MWP solvers rely on shallow heuristics to achieve high
        performance on the benchmark datasets. To this end, we show that MWP solvers that do not have access
        to the question asked in the MWP can still solve a large fraction of MWPs. Similarly, models that treat
        MWPs as bag-of-words can also achieve surprisingly high accuracy. Further, we introduce a challenge
        dataset, SVAMP, created by applying carefully chosen variations over examples sampled from existing datasets.
        The best accuracy achieved by state-of-the-art models is substantially lower on SVAMP, thus showing that much
        remains to be done even for the simplest of the MWPs.",
    }
    """

    NUM_FOLDS: int = 5

    # Where the whole repository lives on GitHub
    _REPO_ZIP = "https://github.com/arkilpatel/SVAMP/archive/refs/heads/main.zip"
    # Subdirectory inside that ZIP that we need
    _SUBDIR_IN_ZIP = "SVAMP-main/data/cv_svamp_augmented"

    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

        # Download and extract the SVAMP dataset to a temporary directory
        self._tmpdir = tempfile.TemporaryDirectory(prefix="svamp_")
        self.local_dataset_path = self._download_and_extract(self._tmpdir.name)

        train_examples: list[dict] = []
        eval_examples: list[dict] = []

        for fold in range(self.NUM_FOLDS):
            train_examples.extend(self._process_fold(fold, split="train"))
            eval_examples.extend(self._process_fold(fold, split="dev"))

        self.train_examples = train_examples
        self.eval_examples = eval_examples
        logger.info(
            f"Initialized SVAMPEnv with {len(self.train_examples)} train examples "
            f"and {len(self.eval_examples)} eval examples."
        )

        # Clean up the temporary directory
        self._tmpdir.cleanup()
        assert not os.path.exists(self._tmpdir.name), "Temporary directory was not cleaned up properly."

    def _download_and_extract(self, destination_dir: str) -> str:
        logger.info("Downloading the SVAMP repository...")
        response = requests.get(self._REPO_ZIP)
        response.raise_for_status()

        with zipfile.ZipFile(BytesIO(response.content)) as z:
            subdir_prefix = self._SUBDIR_IN_ZIP.rstrip("/") + "/"
            relevant_files = [f for f in z.namelist() if f.startswith(subdir_prefix)]

            logger.info(f"Extracting {len(relevant_files)} files from {subdir_prefix}")
            z.extractall(destination_dir, members=relevant_files)

        extracted_subdir = os.path.join(destination_dir, self._SUBDIR_IN_ZIP)
        assert os.path.exists(extracted_subdir), f"Expected directory not found: {extracted_subdir}"
        return extracted_subdir

    def _process_fold(self, fold: int, split: str) -> list[dict]:
        fold_folder: str = f"fold{fold}"
        split_filename: str = f"{split}.csv"
        full_annotation_path: str = os.path.join(self.local_dataset_path, fold_folder, split_filename)
        assert os.path.exists(full_annotation_path), f"Annotation does not exist at path: {full_annotation_path}"

        examples: list[dict] = []
        with open(full_annotation_path, "r") as csv_file:
            reader = csv.DictReader(csv_file)

            for item in tqdm(reader):
                # Construct the question by replacing the number keys (e.g., number0)
                # with the actual number (e.g., 25),
                # so 'bobby ate some pieces of candy . then he ate number0 more'
                # becomes 'bobby ate some pieces of candy . then he ate 25 more
                question: str = item["Question"]
                raw_numbers: list[str] = item["Numbers"].split(" ")
                for index, number in enumerate(raw_numbers):
                    key: str = f"number{index}"
                    assert key in question, f"Invalid question: {question}"
                    question = question.replace(key, number)

                prompt: str = self.add_instruction(question)
                answer: str = item["Answer"]
                examples.append({"prompt": prompt, "answer": answer})

        return examples
