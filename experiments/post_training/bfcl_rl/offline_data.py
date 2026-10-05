# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curate verifier-correct BFCL traces through Datakit's Harmony SFT pipeline."""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

from levanter.tokenizers import load_tokenizer
from marin.datakit.chat_normalize import InvalidToolCallPolicy, normalize_chat_to_parquet
from marin.datakit.chat_render import render_marin_chat
from marin.datakit.download.rollout_transforms import (
    normalize_reasoning_delimiters,
    openai_chat_document,
    openai_chat_messages,
)
from marin.datakit.sft import SftInput, SftTokenStore, build_sft_store
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.bfcl_rl.data import PARTITION_MANIFEST_SHA256, BFCLPartition
from experiments.post_training.bfcl_rl.preferences import RolloutOutcome
from experiments.post_training.bfcl_rl.retained_preferences import CollectionIdentity, retained_rollout

STUDENT_REASONING_MODE = "/think"


@dataclass(frozen=True)
class NativeModelTrace:
    """Native message evidence joined to an audited generation-only retained record."""

    identity: CollectionIdentity
    seed: int
    retained_record: Mapping[str, Any]
    retained_uri: str
    native_trace_uri: str
    messages: list[dict]
    tools: list[dict]
    assistant_prefill: str
    initial_messages: list[dict]
    initial_tools: list[dict]
    initial_prompt_sha256: str
    model_tokenizer: str
    assistant_completion_token_ids: tuple[tuple[int, ...], ...]


def native_prompt_sha256(messages: list[dict], tools: list[dict], assistant_prefill: str) -> str:
    prompt = render_marin_chat(
        openai_chat_messages(
            messages, assistant_prefill=assistant_prefill, invalid_tool_call_policy=InvalidToolCallPolicy.RETAIN
        ),
        tools=tools,
        enable_thinking=STUDENT_REASONING_MODE,
        add_generation_prompt=True,
    )
    return hashlib.sha256(prompt.encode()).hexdigest()


def native_model_trace(
    *,
    identity: CollectionIdentity,
    seed: int,
    retained_record: Mapping[str, Any],
    retained_uri: str,
    native_trace_uri: str,
    trial_result: Mapping[str, Any],
    literal_entries: Iterable[Mapping[str, Any]],
    partition: BFCLPartition,
    assistant_prefill: str,
    model_tokenizer: str,
) -> NativeModelTrace:
    """Join a scored native branch's parsed messages to its exact model-token evidence."""
    retained = retained_rollout(retained_record, identity=identity, partition=partition, trajectory_uri=retained_uri)
    if retained.rollout.outcome is RolloutOutcome.UNSCORED:
        raise ValueError("Unscored native traces cannot supply preference text")
    if trial_result["task_name"] != retained_record["trajectory"]["instance_id"]:
        raise ValueError("Native trial and retained task differ")
    score = 1.0 if retained.rollout.outcome is RolloutOutcome.CORRECT else 0.0
    if trial_result["exception_info"] is not None or trial_result["verifier_result"]["rewards"] != {"reward": score}:
        raise ValueError("Native trial does not confirm the retained verifier outcome")
    agent = trial_result["agent_info"]
    if identity.harness != f"{agent['name']}@{agent['version']}":
        raise ValueError("Native trial and collection harness differ")
    correlation_id = trial_result["agent_result"]["metadata"]["rollout_correlation_id"]
    if not correlation_id:
        raise ValueError("Native trial lacks its literal correlation ID")
    entries = sorted(
        (
            entry
            for entry in literal_entries
            if entry["trial_id"] == correlation_id
            and entry["status_code"] == 200
            and entry["literal"] is not None
            and entry["literal"]["completion_token_ids"]
        ),
        key=lambda entry: entry["timestamp"],
    )
    selected = []
    for step in retained.steps:
        # One Harbor environment step can contain several model calls. Its exact
        # served stream includes masked generation prefixes, tool observations,
        # and optional trailing template tokens after the final sampled EOS.
        stream = [*step.prompt_token_ids, *step.response_token_ids]
        covered = [0] * len(step.response_token_ids)
        matches = []
        for entry in entries:
            prompt = entry["literal"]["prompt_token_ids"]
            completion = entry["literal"]["completion_token_ids"]
            start = len(prompt) - len(step.prompt_token_ids)
            end = start + len(completion)
            if start < 0 or end > len(covered) or stream[: len(prompt)] != prompt:
                continue
            if list(step.response_token_ids[start:end]) != completion or not all(step.loss_mask[start:end]):
                continue
            if any(covered[start:end]):
                raise ValueError("Native literal completions ambiguously cover retained trainable tokens")
            covered[start:end] = [1] * len(completion)
            matches.append(entry)
        if not matches or covered != list(step.loss_mask):
            raise ValueError("Native literal completions differ from retained trainable tokens")
        selected.extend(matches)
    if not selected or any(a["timestamp"] >= b["timestamp"] for a, b in pairwise(selected)):
        raise ValueError("Retained native steps lack an ordered literal chain")
    final = selected[-1]
    request = final["request"]
    assistant = final["literal"]["assistant_message"]
    if assistant is None or assistant["role"] != "assistant":
        raise ValueError("Native capture lacks the parsed final assistant message")
    initial = selected[0]["request"]
    # A retained continuation can start after tool-call serialization changes or
    # context compaction. Compare the original task request when the final chat
    # still preserves it exactly, with the same tool definitions.
    for entry in entries:
        candidate = entry["request"]
        messages = candidate["messages"]
        if (
            any(message["role"] == "user" for message in messages)
            and all(message["role"] in ("system", "developer", "user") for message in messages)
            and request["messages"][: len(messages)] == messages
            and (candidate.get("tools") or []) == (request.get("tools") or [])
        ):
            initial = candidate
            break
    messages = [*request["messages"], assistant]
    completions = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or (index and messages[index - 1]["role"] == "assistant"):
            continue
        candidates = [
            entry
            for entry in entries
            if entry["request"]["messages"] == messages[:index]
            and (entry["request"].get("tools") or []) == (request.get("tools") or [])
        ]
        if len(candidates) != 1:
            raise ValueError("Every native assistant turn requires one unambiguous captured completion")
        completions.append(tuple(candidates[0]["literal"]["completion_token_ids"]))
    return NativeModelTrace(
        identity,
        seed,
        retained_record,
        retained_uri,
        native_trace_uri,
        messages,
        request.get("tools") or [],
        assistant_prefill,
        initial["messages"],
        initial.get("tools") or [],
        native_prompt_sha256(initial["messages"], initial.get("tools") or [], assistant_prefill),
        model_tokenizer,
        tuple(completions),
    )


def native_chat_document(trace: NativeModelTrace) -> dict:
    """Adapt a native model branch using the shared OpenAI-to-Harmony conversion."""
    final = trace.messages[-1]
    messages = trace.messages
    tokenizer = load_tokenizer(trace.model_tokenizer)
    literals = []
    for completion in trace.assistant_completion_token_ids:
        tokens = list(completion)
        while tokens and tokens[-1] == tokenizer.eos_token_id:
            tokens.pop()
        text = tokenizer.decode(tokens)
        if (
            trace.assistant_prefill
            and re.search(r"</think>|<\|end_think\|>", text)
            and not re.search(r"<think>|<\|start_think\|>", text)
        ):
            text = trace.assistant_prefill + text
        literals.append(normalize_reasoning_delimiters(text))
    if not any(final.get(field) for field in ("content", "reasoning_content", "tool_calls", "function_call")):
        # The structural placeholder supplies a final Harmony turn; its text is
        # replaced by the captured literal during shared rendering/tokenization.
        messages = [*messages[:-1], {**final, "unparsed_content": "Captured assistant completion"}]
    document = openai_chat_document(
        messages,
        f"bfcl-complement/{trace.identity.harness}",
        source_id=f"{trace.identity.run_id}/{trace.retained_record['record_id']}",
        assistant_prefill=trace.assistant_prefill,
        invalid_tool_call_policy=InvalidToolCallPolicy.RETAIN,
        chat_template_kwargs={"tools": trace.tools, "enable_thinking": STUDENT_REASONING_MODE},
    )
    document["assistant_literals"] = literals
    return document


def verifier_selected_chat(trace: NativeModelTrace, partition: BFCLPartition) -> dict | None:
    """Validate complement provenance and hand correct native messages to the shared adapter.

    The collection reader joins messages and tool definitions from the native
    trace to its retained record. Teacher token IDs remain evidence only;
    Datakit subsequently renders and tokenizes Harmony for the student.
    """
    retained = retained_rollout(
        trace.retained_record,
        identity=trace.identity,
        partition=partition,
        trajectory_uri=trace.retained_uri,
    )
    if retained.rollout.outcome is not RolloutOutcome.CORRECT:
        return None
    return native_chat_document(trace)


def build_verified_sft_store(
    traces: Iterable[NativeModelTrace],
    *,
    partition: BFCLPartition,
    output_path: str,
    student_tokenizer: str,
    max_length: int,
    seed: int,
    num_shards: int,
    max_workers: int,
) -> SftTokenStore:
    """Reuse Harmony normalization, deduplication and masked student token-store construction.

    The selection manifest retains every accepted trace's run, seed, model,
    harness, task digest and raw evidence locations across content deduplication.
    Datakit supplies normalization quarantine and overlength accounting.
    """
    root = StoragePath(output_path)
    raw = root / "selected-chat"
    raw.mkdirs()
    dispositions: Counter[str] = Counter()
    selected = []
    seen = set()
    with (raw / "traces.jsonl").open("w") as destination:
        for trace in traces:
            record_id = trace.retained_record["record_id"]
            identity = (trace.identity.run_id, record_id)
            if identity in seen:
                raise ValueError(f"Duplicate teacher evidence: {identity}")
            seen.add(identity)
            document = verifier_selected_chat(trace, partition)
            if document is None:
                dispositions["incorrect_or_unscored"] += 1
                continue
            destination.write(json.dumps(document, ensure_ascii=False) + "\n")
            task_name = trace.retained_record["trajectory"]["instance_id"]
            task = next(task for task in partition.complement if task.name == task_name)
            selected.append(
                {
                    "source_id": document["source_id"],
                    "identity": asdict(trace.identity),
                    "seed": trace.seed,
                    "task": asdict(task),
                    "retained_uri": trace.retained_uri,
                    "native_trace_uri": trace.native_trace_uri,
                    "assistant_prefill": trace.assistant_prefill,
                }
            )
            dispositions["verifier_correct"] += 1
    report = {
        "dataset_commit": partition.dataset_commit,
        "partition_manifest_sha256": PARTITION_MANIFEST_SHA256,
        "student_tokenizer": student_tokenizer,
        "student_reasoning_mode": STUDENT_REASONING_MODE,
        "max_length": max_length,
        "seed": seed,
        "dispositions": dict(dispositions),
        "selected": selected,
    }
    (root / "selection.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    if not selected:
        raise ValueError("No verifier-correct complement traces; no offline update")
    normalized = normalize_chat_to_parquet(
        input_path=str(raw),
        output_path=prefix_join(output_path, "harmony"),
        file_extensions=(".jsonl",),
        max_workers=max_workers,
        invalid_tool_call_policy=InvalidToolCallPolicy.RETAIN,
    )
    return build_sft_store(
        (SftInput("bfcl-complement", str(normalized.main_output_dir)),),
        output_path=prefix_join(output_path, "student-store"),
        tokenizer=student_tokenizer,
        max_length=max_length,
        seed=seed,
        num_shards=num_shards,
        max_workers=max_workers,
    )
