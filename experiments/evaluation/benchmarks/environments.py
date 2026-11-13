# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned grader dependencies and source data archives."""

from experiments.post_training.task_curation.datasets.environments import PUNKT_TAB
from experiments.post_training.task_curation.environment import Archive, Environment

BASIC_GRADER = Environment(requirements=("tomli-w==1.2.0",))
PYTHON_GRADER = Environment(requirements=("numpy==2.3.5",))
# The vendored LiveCodeBench scorer preloads NumPy and pandas for every submission.
LIVECODEBENCH_GRADER = Environment(requirements=("numpy==2.3.5", "pandas==2.2.2"))
# Harness Unicode normalization uses NumPy's fixed-width string operations.
ALIAS_GRADER = Environment(requirements=(*BASIC_GRADER.requirements, "numpy==2.3.5"))
MATH_GRADER = Environment(
    requirements=("tomli-w==1.2.0", "math-verify==0.9.0", "sympy==1.13.3", "antlr4-python3-runtime==4.11.0")
)
MATH_JUDGE_GRADER = Environment(requirements=(*MATH_GRADER.requirements, "openai==2.24.0"))

NLTK_DATA_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
NLTK_DATA_ROOT = "/srv/nltk_data"
STOPWORDS_ENGLISH = Archive(
    f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_DATA_REVISION}/packages/corpora/stopwords.zip",
    "48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b",
    "stopwords/english",
)
ENGLISH_TAGGER = Archive(
    f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_DATA_REVISION}/"
    "packages/taggers/averaged_perceptron_tagger_eng.zip",
    "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b",
    "averaged_perceptron_tagger_eng",
)
COMMON_REQUIREMENTS = ("nltk==3.10.3", "tomli-w==1.2.0")
PUNKT_FILES = {NLTK_DATA_ROOT + "/tokenizers/punkt_tab/english": PUNKT_TAB}
IFEVAL_GRADER = Environment(
    requirements=(*COMMON_REQUIREMENTS, "langdetect==1.0.9", "immutabledict==4.3.1"),
    files=PUNKT_FILES,
    env={"NLTK_DATA": NLTK_DATA_ROOT},
)
IFBENCH_GRADER = Environment(
    requirements=(*COMMON_REQUIREMENTS, "emoji==2.16.0", "syllapy==0.8.0"),
    files={
        **PUNKT_FILES,
        NLTK_DATA_ROOT + "/corpora/stopwords/english": STOPWORDS_ENGLISH,
        NLTK_DATA_ROOT + "/taggers/averaged_perceptron_tagger_eng": ENGLISH_TAGGER,
    },
    env={"NLTK_DATA": NLTK_DATA_ROOT},
)
INSTRUCTION_ENVIRONMENTS = (IFEVAL_GRADER, IFBENCH_GRADER)
