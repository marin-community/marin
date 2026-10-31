# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count a retained failed request without running inference or changing its text."""

import argparse
import json

from rigging.filesystem.buckets import filesystem_for
from transformers import AutoTokenizer

from experiments.evaluation.mini_policy_runtime.profile import MODEL, MODEL_REVISION


def main(evidence_uri: str) -> None:
    fs, path = filesystem_for(evidence_uri)
    request = json.loads(fs.cat_file(path))
    # vLLM's OpenAI chat parser decodes tool arguments before rendering Jinja.
    for message in request["messages"]:
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    options = request["options"]
    tokens = tokenizer.apply_chat_template(
        request["messages"],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        tools=options.get("tools"),
        **options.get("chat_template_kwargs", {}),
    )
    print(
        json.dumps(
            {
                "rendered_prompt_tokens": len(tokens),
                "requested_output": options.get("max_completion_tokens"),
                "total_requested_tokens": len(tokens) + options["max_completion_tokens"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-uri", required=True)
    main(parser.parse_args().evidence_uri)
