# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pinned NLTK data archive the instruction-following scorers load through NLTK.

The grader images mount ``PUNKT_TAB_ENGLISH`` from this archive under ``$NLTK_DATA/tokenizers``. The data is
not vendored because the English model is trained on LDC-licensed Penn Treebank text.
"""

PUNKT_TAB_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/4f15a3d89eefe9748ec1c05be495d91289197155"
    "/packages/tokenizers/punkt_tab.zip"
)
PUNKT_TAB_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
PUNKT_TAB_ENGLISH = "punkt_tab/english"
