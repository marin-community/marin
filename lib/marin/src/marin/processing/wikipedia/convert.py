# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
