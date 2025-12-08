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

import re
import string


# inspired by https://github.com/allenai/duplodocus/blob/194b50001ac3c950b086fee0df4454947b872643/src/minhash_base.rs#L466
# SlimPajama text cleaning process
def clean_text(text: str) -> str:
    """Clean text using the SlimPajama text cleaning process.

    This function performs the following steps:
    1. Converts the text to lowercase.
    2. Removes punctuation characters.
    3. Replaces multiple whitespace characters with a single space.
    4. Trims leading and trailing whitespace.

    Args:
        text: The input text string to be cleaned.

    Returns:
        The cleaned text string.
    """

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text
