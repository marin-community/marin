# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project tested one-to-one tool mappings while retaining historical results."""

import copy
import hashlib
import json
import re

from experiments.tool_normalization.kernel_setup import kernel_setup_calls
from experiments.tool_normalization.native_seed import KERNEL_CODE_ARGUMENTS


def historical_projection(original: list[dict], seed: dict) -> tuple[dict, list[dict], list[dict]]:
    """Return native calls, original results and their provenance; never fabricate a setup result."""
    candidate = copy.deepcopy(seed)
    observed = []
    provenance = []
    consumed = set()
    turns = candidate["turns"]
    for position, turn in enumerate(turns):
        index = turn["source_message_index"]
        end = turns[position + 1]["source_message_index"] if position + 1 < len(turns) else len(original)
        legacy = [
            json.loads(match[1])
            for match in re.finditer(r"<tool_call>(.*?)</tool_call>", original[index]["content"], re.DOTALL)
        ]
        calls = turn["calls"]
        if len(calls) == len(legacy) + 1 and calls[0] == kernel_setup_calls()[0]:
            if not legacy or legacy[0]["name"] not in KERNEL_CODE_ARGUMENTS or calls[1]["name"] != "bash":
                raise ValueError("Kernel setup is not attached to its original computation")
            # Setup is silent and is part of the first Bash invocation, not an invented tool-response turn.
            calls = [
                {
                    "name": "bash",
                    "arguments": {
                        "command": (
                            "set -e\n" + calls[0]["arguments"]["command"] + "\n" + calls[1]["arguments"]["command"]
                        )
                    },
                },
                *calls[2:],
            ]
            turn["calls"] = calls
        results = [(i, original[i]) for i in range(index + 1, end) if original[i]["role"] in ("tool", "ipython")]
        if (
            index == len(original) - 1
            and len(legacy) == 1
            and legacy[0]["name"] == "finish"
            and not calls
            and not results
        ):
            # The original user-facing completion is now assistant text, with no invented tool receipt.
            continue
        if index == len(original) - 1 and len(legacy) == len(calls) == 1 and not results:
            # Preserve an unanswered final action as such, without asserting it ran or succeeded.
            continue
        if len(calls) != len(legacy) or len(results) != len(legacy):
            raise ValueError("Historical projection requires one recorded result per source operation")
        for action, call, (result_index, message) in zip(legacy, calls, results, strict=True):
            match = re.fullmatch(
                r'\s*<tool_response name="([^"]+)">(.*)</tool_response>\s*', message["content"], re.DOTALL
            )
            if match is None or match[1] != action["name"]:
                raise ValueError("Historical response wrapper does not match its source tool")
            content = match[2]
            if action["name"] in KERNEL_CODE_ARGUMENTS and content.strip() == "Timed out":
                raise ValueError("Source Python timeout/reset semantics are unresolved")
            if (
                action["name"] == "str_replace_editor"
                and action["arguments"].get("command") == "view"
                and content.lstrip().startswith("ERROR:\nInvalid `view_range` parameter:")
            ):
                raise ValueError("Source editor range failure differs from native read behavior")
            call_id = f"replay_{len(observed)}"
            observed.append({"role": "tool", "name": call["name"], "tool_call_id": call_id, "content": content})
            provenance.append(
                {
                    "tool_call_id": call_id,
                    "source_assistant_index": index,
                    "source_result_index": result_index,
                    "source_tool": action["name"],
                    "target_tool": call["name"],
                    "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "origin": "historical_source_not_reexecuted",
                }
            )
            consumed.add(result_index)
    expected = {i for i, message in enumerate(original) if message["role"] in ("tool", "ipython")}
    if consumed != expected:
        raise ValueError("Unpaired historical tool results")
    return candidate, observed, provenance
