# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record the production solver's advertised model and accepted request settings."""

import argparse
import json
from datetime import UTC, datetime

import httpx
from finestore.rl.difficulty import SOLVER_MODELS, Solver
from rigging.chat_templates import chat_template_messages
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from transformers import AutoTokenizer

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.task_curation.difficulty_protocol import PINS, request_parameters


def probe_endpoint(solver: Solver, endpoint: str, api_model: str, api_key: str) -> dict:
    """Verify pinned tools/thinking tokenization against authoritative serving usage."""
    pin = PINS[solver]
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=300) as client:
        models = client.get(endpoint.rstrip("/") + "/models")
        models.raise_for_status()
        parameters = request_parameters(solver)
        probe = {
            "model": api_model,
            "messages": [
                {"role": "user", "content": "Look up record A."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "lookup-1",
                            "type": "function",
                            "function": {"name": "lookup_record", "arguments": '{"record_id":"A"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "lookup-1", "content": "Record A exists."},
                {"role": "user", "content": "Reply with exactly OK."},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup_record",
                        "description": "Look up one record.",
                        "parameters": {
                            "type": "object",
                            "properties": {"record_id": {"type": "string"}},
                            "required": ["record_id"],
                        },
                    },
                }
            ],
            "max_tokens": 16384,
            **{key: value for key, value in parameters.items() if key != "extra_body"},
            **parameters.get("extra_body", {}),
        }
        tokenizer = AutoTokenizer.from_pretrained(SOLVER_MODELS[solver], revision=pin.revision)
        prompt_tokens = len(
            tokenizer.apply_chat_template(
                chat_template_messages(probe["messages"]),
                tools=probe["tools"],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **probe.get("chat_template_kwargs", {}),
            )
        )
        probe_response = client.post(endpoint.rstrip("/") + "/chat/completions", json=probe)
        probe_response.raise_for_status()
        completion = probe_response.json()
        if completion["usage"]["prompt_tokens"] != prompt_tokens:
            raise ValueError("Pinned tools/thinking rendering differs from authoritative server usage")
    return {
        "advertised_models": models.json(),
        "tools_thinking_probe": {"request": probe, "completion": completion, "hf_prompt_tokens": prompt_tokens},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relay-job", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    output = StoragePath(args.output_root)
    if output.exists():
        raise FileExistsError(f"Preserve prior serving probes: {output}")
    base_url = resolve_glm_base_url(args.relay_job)
    receipt = probe_endpoint(Solver.HOSTED, base_url, GLM_MODEL, require_secret(GLM_BULK_TOKEN_ENV))
    (output / "serving-probe.json").write_text(
        json.dumps(
            {"created_at": datetime.now(UTC).isoformat(), "relay_job": args.relay_job, "base_url": base_url, **receipt},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"receipt": str(output / "serving-probe.json")}))


if __name__ == "__main__":
    main()
