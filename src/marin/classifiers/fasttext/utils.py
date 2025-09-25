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

"""
utils.py

Utility functions for training fastText models.
"""

import regex as re


def preprocess(text: str) -> str:
    """
    Preprocesses text for fastText training by stripping newline characters.
    """
    return re.sub(r"[\n\r]", " ", text)


def format_example(data: dict) -> str:
    """
    Converts example to fastText training data format.
    """
    return f'__label__{data["label"]}' + " " + preprocess(data["text"])
