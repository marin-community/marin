# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned instruction-checker packages and shared NLTK resource archives.

Package versions match the frozen Evalchemy oracle. NLTK data is mounted once
per environment, outside individual tasks; graders run without network access.
"""

from experiments.post_training.task_curation.datasets.environments import PUNKT_TAB
from experiments.post_training.task_curation.environment import Archive, Environment

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
