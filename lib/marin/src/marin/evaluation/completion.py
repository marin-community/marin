# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select scorer text without discarding the archived completion channels.

Preserves Marin Evalchemy's CompletionResponse normalization and MMLU-Pro's
content-first extraction policy. A length-limited reasoning trace is not a
complete final answer under the latter policy.
"""

import re
from collections.abc import Mapping
from typing import Any

from taskcompendium.evaluation import ResponseSelection


def completion_text(message: Mapping[str, Any], stop_reason: str, selection: ResponseSelection) -> str:
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    if selection == "final_or_complete_boxed_reasoning":
        return content or (
            reasoning if stop_reason != "length" and re.search(r"\\(?:boxed|fbox)\s*\{", reasoning) else ""
        )
    if selection == "final_or_complete_reasoning":
        return content or (reasoning if stop_reason != "length" else "")
    if selection == "combine":
        return f"{reasoning}\n\n{content}" if content and reasoning else reasoning or content
    raise ValueError(f"Unknown completion response selection: {selection}")
