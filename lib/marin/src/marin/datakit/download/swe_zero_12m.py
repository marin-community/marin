# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AlienKevin/SWE-ZERO-12M-trajectories dataset download and transform.

12.29M execution-free agentic-coding trajectories generated with the
``mini-swe-agent`` v1 format. Each row contains a multi-turn conversation
(``messages``: list of ``{role, content}``) along with the rollout's
``exit_status`` (``Submitted``, ``incomplete``, etc.).
"""

import re
from textwrap import dedent

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import (
    CHAT_CONTROL_TOKEN,
    REASONING_TOKEN,
    checked_openai_chat_document,
    load_parquet_batched,
    render_role_message,
    text_document,
)
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "AlienKevin/SWE-ZERO-12M-trajectories"
HF_REVISION = "44e0280"

_BASH_ACTION = re.compile(r"\A\s*(?P<reasoning>.*?)```bash\s*\n(?P<command>.*?)\n```\s*\Z", re.DOTALL)
_THOUGHT_PREFIX = re.compile(r"\A\s*THOUGHT:\s*", re.IGNORECASE)
_OBSERVATION_PREFIX = re.compile(r"\A\s*Observation:\s*", re.IGNORECASE)
_COMPLETION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
_COMPLETION_COMMAND = re.compile(rf"\A\s*(?:echo|printf)\s+[\"']?{_COMPLETION_MARKER}(?:[\"'\\n]|\s|;|&|\||\Z)")

# Only the recorded response and completion protocol changes; retain all other instructions.
_SYSTEM_PROTOCOL_REPLACEMENTS = (
    (
        dedent(
            """\
            Every response must contain EXACTLY ONE bash code block (triple backticks) with EXACTLY ONE command.
            Before the bash block, include a THOUGHT section explaining your reasoning. Put ALL explanation in
            THOUGHT — do NOT prefix the bash command with `# comment` lines.

            Format:
            THOUGHT: <your reasoning>

            ```bash
            <one bash command>
            ```
        """
        ).strip(),
        "Use the bash tool to run one command at a time. Put reasoning before the tool call.",
    ),
    (
        dedent(
            """\
            - Do NOT prefix your bash command with a `# comment` line. Bash will run the command after the
              comment, but the comment wastes input tokens. Put explanation in THOUGHT only.
        """
        ).strip(),
        "",
    ),
    (
        dedent(
            """\
            - The FIRST LINE of the output of your bash command must be exactly
              `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. The standard way is `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
        """
        ).strip(),
        "- When the task is complete, give a final response to the user.",
    ),
    (
        "5. Submit with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.",
        "5. Finish with a final response to the user.",
    ),
)

BASH_TOOL = {
    "type": "function",
    "name": "bash",
    "description": "Run one shell command from the repository root in a fresh subshell.",
    "parameters": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}


def row_to_doc(row: dict) -> list[dict]:
    messages = row.get("messages")
    if not messages:
        counters.pipeline.update_counter("swe_zero_12m/dropped", 1)
        return []

    text = "\n\n".join(render_role_message(m) for m in messages)

    counters.pipeline.update_counter("swe_zero_12m/kept", 1)
    return [text_document(text, HF_DATASET_ID)]


def _system_prompt_for_tools(content: str) -> str:
    for original, replacement in _SYSTEM_PROTOCOL_REPLACEMENTS:
        content = content.replace(original, replacement)
    return content


def row_to_chat_doc(row: dict) -> list[dict]:
    messages = row.get("messages")
    if not messages:
        return []

    canonical: list[dict] = []
    pending_call: tuple[str, str] | None = None
    skip_completion_observation = False
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            counters.pipeline.update_counter("swe_zero_12m/chat_malformed_filtered", 1)
            return []
        if role == "system":
            if canonical:
                counters.pipeline.update_counter("swe_zero_12m/chat_malformed_filtered", 1)
                return []
            canonical.append({"role": "system", "content": _system_prompt_for_tools(content)})
            continue
        if role == "user":
            if skip_completion_observation:
                skip_completion_observation = False
                continue
            if pending_call is not None:
                call_id, tool_name = pending_call
                observation = _OBSERVATION_PREFIX.sub("", content, count=1)
                canonical.append({"role": "tool", "content": observation, "name": tool_name, "tool_call_id": call_id})
                pending_call = None
                continue
            canonical.append({"role": "user", "content": content})
            continue
        if role != "assistant" or pending_call is not None:
            counters.pipeline.update_counter("swe_zero_12m/chat_malformed_filtered", 1)
            return []

        match = _BASH_ACTION.fullmatch(content)
        if match is None:
            counters.pipeline.update_counter("swe_zero_12m/chat_malformed_filtered", 1)
            return []
        reasoning = _THOUGHT_PREFIX.sub("", match.group("reasoning"), count=1).strip()
        reasoning_content = f"<think>{reasoning}</think>" if reasoning else ""
        command = match.group("command").strip()
        if (
            CHAT_CONTROL_TOKEN.search(command)
            or REASONING_TOKEN.search(command)
            or "<think>" in command
            or "</think>" in command
        ):
            counters.pipeline.update_counter("swe_zero_12m/chat_control_token_filtered", 1)
            return []
        remaining = messages[index + 1 :]
        has_final_observation = (
            len(remaining) == 1
            and remaining[0].get("role") == "user"
            and _COMPLETION_MARKER in (remaining[0].get("content") or "")
        )
        if _COMPLETION_COMMAND.search(command) and (not remaining or has_final_observation):
            canonical.append({"role": "assistant", "content": f"{reasoning_content}\n\nTask complete.".strip()})
            skip_completion_observation = True
            continue

        call_id = f"call_bash_{index}"
        canonical.append(
            {
                "role": "assistant",
                "content": reasoning_content,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"command": command}},
                    }
                ],
            }
        )
        pending_call = (call_id, "bash")

    if not canonical or canonical[-1]["role"] != "assistant":
        counters.pipeline.update_counter("swe_zero_12m/chat_incomplete_filtered", 1)
        return []
    return checked_openai_chat_document(
        canonical,
        HF_DATASET_ID,
        counter_prefix="swe_zero_12m/chat",
        chat_template_kwargs={"tools": [BASH_TOOL]},
    )


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="swe-zero-12m-transform", resources=ResourceConfig(cpu=1, ram="32g"))
    ctx.execute(pipeline)


def transform_chat(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(row_to_chat_doc)
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"), schema=CHAT_SCHEMA, skip_existing=True
        )
    )
    ZephyrContext(name="swe-zero-12m-chat-transform", resources=ResourceConfig(cpu=1, ram="32g")).execute(pipeline)


def download_swe_zero_12m_step() -> StepSpec:
    """Download SWE-ZERO-12M-trajectories and render each rollout into a Parquet text document."""
    dl = download_hf_step(
        "raw/swe-zero-12m-trajectories",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
    )

    return StepSpec(
        name="processed/swe-zero-12m-trajectories",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def swe_zero_12m_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for swe-zero-12m."""
    processed = download_swe_zero_12m_step()
    return (
        processed,
        normalize_step(name="normalized/swe-zero-12m", download=processed),
    )


def swe_zero_12m_chat_normalize_steps() -> tuple[StepSpec, ...]:
    dl = download_hf_step("raw/swe-zero-12m-trajectories", hf_dataset_id=HF_DATASET_ID, revision=HF_REVISION)
    processed = StepSpec(
        name="processed-chat/swe-zero-12m-trajectories",
        deps=[dl],
        fn=lambda output_path: transform_chat(dl.output_path, output_path),
        hash_attrs={"version": "2026.09.10.preserve-system-instructions"},
    )
    return processed, normalize_chat_step(
        output_schema=CHAT_SCHEMA, name="normalized-chat/swe-zero-12m", download=processed
    )
