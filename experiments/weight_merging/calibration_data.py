# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate independent, deterministic numeric and tool-behavior calibration data."""

import argparse
import hashlib
import json
import random
from fractions import Fraction
from pathlib import Path


def calibration_examples(seed: int, per_domain: int) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for index in range(per_domain):
        digits = (4, 8, 16, 32)[(index // 8 + index % 3) % 4]
        a = rng.randrange(10 ** (digits - 1), 10**digits)
        b = rng.randrange(10 ** (digits - 1), 10**digits)
        operation = index % 8
        if operation == 0:
            question, answer = f"Add two numbers: {a} + {b} =", str(a + b)
        elif operation == 1:
            question, answer = f"Subtract two numbers: {a} - {b} =", str(a - b)
        elif operation == 2:
            question, answer = f"Get the maximal number: {a} and {b} =", str(max(a, b))
        elif operation == 3:
            question, answer = f"Multiply two numbers: {a} * {b} =", str(a * b)
        elif operation == 4:
            question, answer = f"Compute integer division: {a} // {b} =", str(a // b)
        elif operation == 5:
            question, answer = f"Compute the remainder: {a} % {b} =", str(a % b)
        elif operation == 6:
            question, answer = f"Count the digits in {a} =", str(len(str(a)))
        else:
            question, answer = f"Divide as an irreducible fraction: {a} / {b} =", str(Fraction(a, b))
        prompt = "Directly return only the answer without any comma separator. " + question
        examples.append(_example("numeric", index, 0, prompt, answer))

        item = f"calibration_item_{rng.randrange(10**8, 10**9)}"
        amount = rng.randrange(3, 99)
        old = rng.randrange(100, 999)
        function = ("lookup_inventory", "reserve_inventory", "set_temperature", "calculate_shipping")[index % 4]
        schemas = {
            "lookup_inventory": {"sku": item, "warehouse": f"depot_{index % 7}"},
            "reserve_inventory": {"sku": item, "quantity": amount, "order_id": f"order_{seed}_{index}"},
            "set_temperature": {"device_id": item, "temperature_celsius": amount},
            "calculate_shipping": {"package_id": item, "weight_grams": old, "destination": f"zone_{index % 5}"},
        }
        arguments = schemas[function]
        schema = {
            "name": function,
            "description": "Execute the requested operation using the supplied arguments.",
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": "integer" if isinstance(value, int) else "string"} for key, value in arguments.items()
                },
                "required": list(arguments),
                "additionalProperties": False,
            },
        }
        system = (
            "You are an assistant with tools. Call the available function with exactly the requested arguments. "
            "Write a JSON function call inside <tool_call> and </tool_call>. Available function: "
            + json.dumps(schema, sort_keys=True)
        )
        prompt = f"Execute {function} with these arguments: {json.dumps(arguments, sort_keys=True)}"
        answer = "<tool_call>" + json.dumps({"name": function, "arguments": arguments}, sort_keys=True) + "</tool_call>"
        row = _example("tools", index, 2, prompt, answer)
        row["messages"].insert(0, {"role": "system", "content": system})
        examples.append(row)

        x, y, z = (rng.randrange(11, 999) for _ in range(3))
        if index % 4 == 0:
            prompt = f"A store has {x} boxes with {y} objects each and receives {z} more objects. Explain the total."
            answer = f"The boxes contain {x} * {y} = {x*y} objects. Adding {z} gives {x*y+z} objects."
        elif index % 4 == 1:
            prompt = f"Return JSON with keys sum and difference for integers {x} and {y}."
            answer = json.dumps({"sum": x + y, "difference": x - y})
        elif index % 4 == 2:
            prompt = f"Write a Python function that returns the sum of integers from {x} through {x+y}, inclusive."
            answer = f"def interval_sum():\n    return ({x} + {x+y}) * {y+1} // 2"
        else:
            prompt = f"Sort these integers in increasing order and return only the list: {x}, {y}, {z}."
            answer = json.dumps(sorted([x, y, z]))
        examples.append(_example("retention", index, 1, prompt, answer))
    return examples


def _example(domain: str, index: int, teacher: int, prompt: str, answer: str) -> dict:
    return {
        "id": f"merge-calibration-{domain}-{index:04d}",
        "domain": domain,
        "teacher": teacher,
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--per-domain", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = calibration_examples(args.seed, args.per_domain)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    args.output.write_bytes(payload)
    print(json.dumps({"rows": len(rows), "sha256": hashlib.sha256(payload).hexdigest(), "seed": args.seed}))


if __name__ == "__main__":
    main()
