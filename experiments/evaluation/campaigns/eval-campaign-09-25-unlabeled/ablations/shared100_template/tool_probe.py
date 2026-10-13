# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue matched OpenAI tool-call probes to a running ablation endpoint."""

import argparse
import json
import urllib.request

TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Add two integers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    },
}


def request(base_url: str, payload: dict) -> dict:
    message = json.dumps(payload).encode()
    call = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=message,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(call, timeout=180) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    base = {
        "model": args.model,
        "tools": [TOOL],
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": True, "xml_tools": [TOOL]},
    }
    probes = {
        "first_call": [
            {"role": "system", "content": "Use the calculator tool for arithmetic. Do not calculate mentally."},
            {"role": "user", "content": "Call calculator to add 19 and 23."},
        ],
        "history_with_reasoning": [
            {"role": "system", "content": "Use the calculator tool for arithmetic. Do not calculate mentally."},
            {"role": "user", "content": "Call calculator to add 19 and 23."},
            {
                "role": "assistant",
                "reasoning_content": "Call the calculator for the sum.",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_probe_1",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"a":19,"b":23}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_probe_1", "name": "calculator", "content": "42"},
            {"role": "user", "content": "Now call calculator to add 42 and 8."},
        ],
    }
    results = {}
    for name, messages in probes.items():
        try:
            results[name] = request(args.base_url, {**base, "messages": messages})
        except Exception as exc:
            results[name] = {"error": str(exc)}
    print(json.dumps({"base_url": args.base_url, "model": args.model, "probes": results}, indent=2))


if __name__ == "__main__":
    main()
