# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise installed-agent tool requests through the pinned serving renderer."""

import asyncio
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import skyrl_train.models.grug_moe  # noqa: F401 - register the rollout model architecture
from skyrl_train.inference_engines.vllm.utils import pop_vllm_wrapper_kwargs
from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.renderers.registry import renderer_from_config


async def check_request(inputs: dict, destination: Path) -> None:
    release = inputs["release"]
    if version("vllm") != release["version"]:
        raise ValueError("Tool request probe must use the rollout serving release")
    model = ModelConfig(
        model=inputs["model_config_directory"],
        tokenizer=inputs["tokenizer"]["repo"],
        tokenizer_revision=inputs["tokenizer"]["revision"],
        dtype="bfloat16",
        max_model_len=inputs["window"],
        hf_overrides=inputs["engine_kwargs"].get("hf_overrides", {}),
    )
    renderer = renderer_from_config(VllmConfig(model_config=model))
    cases = []
    without_tools = {
        key: value
        for key, value in inputs["engine_kwargs"].items()
        if key not in {"enable_auto_tool_choice", "enable_auto_tools", "tool_call_parser", "tool_parser"}
    }
    for name, kwargs in (("without_auto_tools", without_tools), ("submitted", dict(inputs["engine_kwargs"]))):
        wrapper = pop_vllm_wrapper_kwargs(kwargs)
        online = OnlineRenderer(
            model_config=model,
            renderer=renderer,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            default_chat_template_kwargs=inputs["chat_template_kwargs"],
            **wrapper,
        )
        request = ChatCompletionRequest.model_validate(inputs["request"])
        result = await online.render_chat(request)
        if isinstance(result, ErrorResponse):
            cases.append({"name": name, "accepted": False, "error": result.model_dump(mode="json")})
        else:
            conversation, prompts = result
            cases.append({"name": name, "accepted": True, "messages": len(conversation), "prompts": len(prompts)})
    evidence = {
        "vllm_commit": release["source_commit"],
        "vllm_version": version("vllm"),
        "cases": cases,
        "request_sha256": hashlib.sha256(json.dumps(inputs["request"], sort_keys=True).encode()).hexdigest(),
        "scope": "actual request validation and tokenization; no generation or optimizer",
    }
    destination.write_text(json.dumps(evidence, indent=2))
    if cases[0]["accepted"] or not cases[1]["accepted"]:
        raise ValueError(f"Tool request diagnosis not reproduced: {cases}")


if __name__ == "__main__":
    asyncio.run(check_request(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])))
