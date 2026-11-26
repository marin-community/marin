# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record the production solver's advertised model and accepted request settings."""

import argparse
import json
from datetime import UTC, datetime

import httpx
from finestore.rl.difficulty import SOLVER_MODELS, SamplingSettings, Solver
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from rolloutengine.chat import chat_template_messages
from transformers import AutoTokenizer

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV, GLM_MODEL, resolve_glm_base_url
from experiments.post_training.task_curation.difficulty.state import ActionFailure, GapReason
from experiments.post_training.task_curation.difficulty_config import DeclaredJudgePin
from experiments.post_training.task_curation.difficulty_protocol import PINS
from experiments.post_training.task_curation.difficulty_sampling import request_parameters, sampling_settings
from experiments.post_training.task_curation.harmony_probe import harmony_prompt_tokens
from experiments.post_training.task_curation.judges import JudgeTierDefinition

SOLVER_PROBE_OUTPUT_TOKENS = 512

PROBE_MESSAGES = [
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
]
PROBE_TOOLS = [
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


def probe_endpoint(solver: Solver, endpoint: str, api_model: str, api_key: str, settings: SamplingSettings) -> dict:
    """Verify pinned tools/thinking tokenization against authoritative serving usage."""
    pin = PINS[solver]
    parameters = request_parameters(settings)
    request = {
        "model": api_model,
        "messages": PROBE_MESSAGES,
        "tools": PROBE_TOOLS,
        "max_tokens": min(settings.output_tokens, SOLVER_PROBE_OUTPUT_TOKENS),
        **{key: value for key, value in parameters.items() if key != "extra_body"},
        **parameters.get("extra_body", {}),
    }
    return probe_serving_endpoint(
        endpoint,
        api_key,
        request,
        expected_prompt_tokens=hf_prompt_tokens(
            request,
            model_identity=SOLVER_MODELS[solver],
            model_revision=pin.revision,
            chat_template_kwargs=request.get("chat_template_kwargs", {}),
        ),
    )


def probe_declared_judge_endpoint(
    endpoint: str, api_model: str, api_key: str, temperature: float, pin: DeclaredJudgePin
) -> dict:
    """Verify a declared judge's text request with its pinned tokenizer."""
    request = {
        "model": api_model,
        "messages": PROBE_MESSAGES,
        "max_tokens": 512,
        "temperature": temperature,
    }
    return probe_serving_endpoint(
        endpoint,
        api_key,
        request,
        expected_prompt_tokens=hf_prompt_tokens(
            request, model_identity=api_model, model_revision=pin.revision, chat_template_kwargs={}
        ),
    )


def hf_prompt_tokens(
    request: dict,
    *,
    model_identity: str,
    model_revision: str,
    chat_template_kwargs: dict,
) -> int:
    """Prompt token count the pinned repo chat template renders for this request."""
    tokenizer = AutoTokenizer.from_pretrained(model_identity, revision=model_revision)
    return len(
        tokenizer.apply_chat_template(
            chat_template_messages(request["messages"]),
            tools=request.get("tools"),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            **chat_template_kwargs,
        )
    )


def probe_serving_endpoint(
    endpoint: str,
    api_key: str,
    request: dict,
    *,
    expected_prompt_tokens: int,
) -> dict:
    """Verify pinned tools/thinking tokenization against authoritative serving usage."""
    # A cold vLLM engine can still be compiling when the endpoint accepts connections;
    # the probe must outlast warmup instead of killing the campaign at 300 s.
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=1800) as client:
        models = client.get(endpoint.rstrip("/") + "/models")
        models.raise_for_status()
        probe_response = client.post(endpoint.rstrip("/") + "/chat/completions", json=request)
        probe_response.raise_for_status()
        completion = probe_response.json()
        served_prompt_tokens = completion["usage"]["prompt_tokens"]
        if served_prompt_tokens != expected_prompt_tokens:
            raise ActionFailure(
                GapReason.SERVING_PROBE_MISMATCH,
                "Pinned tools/thinking rendering differs from authoritative server usage: "
                f"pinned rendering is {expected_prompt_tokens} prompt tokens, "
                f"the server rendered {served_prompt_tokens}",
            )
    return {
        "advertised_models": models.json(),
        "tools_thinking_probe": {
            "request": request,
            "completion": completion,
            "pinned_prompt_tokens": expected_prompt_tokens,
        },
    }


def judge_probe_request(definition: JudgeTierDefinition, temperature: float) -> dict:
    """The thinking probe request a judge tier must answer with its own rendering.

    Judge contracts never call tools, so the probe does not advertise any: vLLM
    rejects tool-bearing chat requests when tool choice is disabled, and the
    served judge must accept exactly the request shape the grading path sends.
    """
    return {
        "model": definition.spec.model,
        "messages": PROBE_MESSAGES,
        "max_tokens": definition.spec.max_completion_tokens,
        "temperature": temperature,
        **(definition.spec.chat_template_kwargs or {}),
    }


def judge_expected_prompt_tokens(definition: JudgeTierDefinition, request: dict) -> int:
    """The pinned prompt token count for a judge tier's probe request.

    Chat-template tiers render through the pinned repo tokenizer; harmony-served
    tiers (gpt-oss families) render through the openai_harmony encoder the
    engine itself uses.
    """
    if definition.served_encoding == "harmony":
        return harmony_prompt_tokens(request)
    assert definition.model_revision is not None
    return hf_prompt_tokens(
        request,
        model_identity=definition.model_identity,
        model_revision=definition.model_revision,
        chat_template_kwargs=definition.spec.chat_template_kwargs or {},
    )


def probe_judge_endpoint(endpoint: str, definition: JudgeTierDefinition, temperature: float, api_key: str) -> dict:
    """Verify the judge tier's pinned rendering and identity against its served endpoint."""
    if definition.model_revision is None:
        raise ValueError(f"The {definition} tier has no immutable checkpoint pin to probe")
    return probe_serving_endpoint(
        endpoint,
        api_key,
        judge_probe_request(definition, temperature),
        expected_prompt_tokens=judge_expected_prompt_tokens(definition, judge_probe_request(definition, temperature)),
    )


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
    receipt = probe_endpoint(
        Solver.HOSTED, base_url, GLM_MODEL, require_secret(GLM_BULK_TOKEN_ENV), sampling_settings(Solver.HOSTED, [])
    )
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
