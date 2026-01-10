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


# remove citations
def clean_text(text):
    # This regex pattern matches any sequence of [number]
    # get rid of references
    pattern = r"\^\([^)]+\)"
    cleaned_text = re.sub(pattern, "", text)
    # clean empty lines
    lines = cleaned_text.split("\n")
    clean_lines = []
    for line in lines:
        if not line.strip():
            clean_lines.append("\n")
        elif line.strip() == "[]":
            clean_lines.append("\n")
        else:
            clean_lines.append(line)
    cleaned_text = "\n".join(clean_lines)
    cleaned_text = re.sub("[\n]{2,}", "\n\n", cleaned_text)
    return cleaned_text


# convert html to md
