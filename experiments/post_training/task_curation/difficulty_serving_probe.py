# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record the production hosted solver's advertised model and accepted request settings."""

import argparse
import json
from datetime import UTC, datetime

import httpx
from finestore.rl.difficulty import SOLVER_MODELS, Solver
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from transformers import AutoTokenizer

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.task_curation.difficulty_protocol import PINS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relay-job", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fixture", choices=("plain", "tools-thinking"), default="plain")
    args = parser.parse_args()
    configure_coreweave_s3()
    output = StoragePath(args.output_root)
    if output.exists():
        raise FileExistsError(f"Preserve prior serving probes: {output}")
    base_url = resolve_glm_base_url(args.relay_job)
    body = {
        "model": GLM_MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
        "max_tokens": 16384,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "repetition_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "reasoning_effort": "low",
    }
    if args.fixture == "tools-thinking":
        body["tools"] = [
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
        ]
        body["chat_template_kwargs"] = {"enable_thinking": True}
    tokenizer = AutoTokenizer.from_pretrained(SOLVER_MODELS[Solver.HOSTED], revision=PINS[Solver.HOSTED].revision)
    token_ids = tokenizer.apply_chat_template(
        body["messages"],
        tools=body.get("tools"),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
        **body.get("chat_template_kwargs", {}),
    )
    with httpx.Client(headers={"Authorization": f"Bearer {require_secret(GLM_BULK_TOKEN_ENV)}"}, timeout=300) as client:
        models = client.get(f"{base_url}/models")
        models.raise_for_status()
        tokenization = client.post(
            f"{base_url.removesuffix('/v1')}/tokenize",
            json={"model": GLM_MODEL, "messages": body["messages"], "add_generation_prompt": True},
        )
        completion = client.post(f"{base_url}/chat/completions", json=body)
        receipt = {
            "created_at": datetime.now(UTC).isoformat(),
            "relay_job": args.relay_job,
            "base_url": base_url,
            "models": models.json(),
            "request": body,
            "completion_status": completion.status_code,
            "completion": completion.json(),
            "tokenization_status": tokenization.status_code,
            "tokenization": tokenization.text,
            "difficulty_measurements": 0,
            "fixture": args.fixture,
            "hf_prompt_tokens": len(token_ids),
        }
        (output / "serving-probe.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "completion_status": completion.status_code,
                    "tokenization_status": tokenization.status_code,
                    "receipt": str(output / "serving-probe.json"),
                }
            )
        )
        completion.raise_for_status()
        if completion.json()["usage"]["prompt_tokens"] != len(token_ids):
            raise ValueError("Pinned HF tools/thinking rendering differs from authoritative server usage")


if __name__ == "__main__":
    main()
