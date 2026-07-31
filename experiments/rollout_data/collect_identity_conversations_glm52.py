# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate Marin identity conversations with GLM-5.2."""

import argparse
import dataclasses
import json
import logging
import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import requests
from iris.client import iris_ctx
from rigging.filesystem import StoragePath
from zephyr.writers import write_jsonl_file

from experiments.rollout_data.generate_identity_conversation_seeds import (
    ConversationSeed,
    HumanPhrasing,
    IdentityProfile,
    generate_seeds,
    load_human_phrasings,
    load_identity_profile,
)
from experiments.rollout_data.glm52_vllm import (
    MODEL,
    Glm52LaunchConfig,
    ServerConfig,
    submit_glm52,
    wait_for_endpoint_url,
)

logger = logging.getLogger(__name__)

VLLM_ENDPOINT = "identity-glm52-openai"
RAY_ENDPOINT = "identity-glm52-ray"
REQUEST_TIMEOUT = 3600
MAX_GENERATION_ATTEMPTS = 3
DEFAULT_TARGET_TOKENS = 10_000_000
DEFAULT_MICROBATCH_SIZE = 8
DEFAULT_CONCURRENCY = 96
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_MAX_NUM_SEQS = 64
CHUNK_PATTERN = re.compile(r"chunk-(?P<start>\d+)-(?P<end>\d+)-tokens-(?P<tokens>\d+)\.jsonl\.gz")
UNSUPPORTED_USER_DETAIL_MARKERS = (
    "checkpoint",
    "codename",
    "commit history",
    "dense transformer",
    "documentation says",
    "docs say",
    "endpoint header",
    "expert subnetworks",
    "latency profile",
    "metadata says",
    "mixture-of-experts",
    "quota tier",
    "release notes",
    "routing layer",
    "tokenization behavior",
    "training phase",
    "x-organization",
)


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float
    top_p: float
    max_tokens: int


@dataclass(frozen=True)
class CollectionConfig:
    run_id: str
    output_path: StoragePath
    shard_index: int
    num_shards: int
    target_tokens: int
    microbatch_size: int
    concurrency: int
    random_seed: int
    human_phrasings: tuple[HumanPhrasing, ...]
    identity_profile: IdentityProfile


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class CompletionRecord:
    seed: ConversationSeed
    conversation: tuple[Message, ...] | None
    raw_response: str
    accepted_tokens: int
    usage: dict[str, Any]
    finish_reason: str | None
    response_id: str | None
    error: str | None
    attempts: int
    elapsed_seconds: float


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    tokens: int


@dataclass(frozen=True)
class Progress:
    run_id: str
    profile_id: str
    shard_index: int
    num_shards: int
    state: str
    target_tokens: int
    accepted_tokens: int
    accepted_conversations: int
    rejected_conversations: int
    completed_microbatches: int
    updated_at: str


def _conversation(text: str, seed: ConversationSeed) -> tuple[Message, ...]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    payload = json.loads(cleaned)
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("response must be a JSON array with at least two messages")
    messages = tuple(Message(role=row["role"], content=row["content"].strip()) for row in payload)
    if any(message.role not in {"user", "assistant"} or not message.content.strip() for message in messages):
        raise ValueError("conversation messages must have non-empty user or assistant content")
    expected_roles = ("user", "assistant") if seed.neutral_bridge is None else ("user", "assistant", "user", "assistant")
    if tuple(message.role for message in messages) != expected_roles:
        raise ValueError("conversation roles must start with user and alternate through the final assistant answer")
    if seed.neutral_bridge is not None and messages[1].content != seed.neutral_bridge:
        raise ValueError("earlier assistant message must match the supplied neutral bridge exactly")
    if messages[-1].content != seed.canonical_answer:
        raise ValueError("final assistant message must match the supplied canonical answer exactly")
    user_text = " ".join(message.content.lower() for message in messages if message.role == "user")
    if any(marker in user_text for marker in UNSUPPORTED_USER_DETAIL_MARKERS):
        raise ValueError("user messages introduced an unsupported implementation detail")
    return messages


def _completion(vllm_url: str, seed: ConversationSeed, sampling: SamplingConfig) -> CompletionRecord:
    started = time.time()
    last_text = ""
    last_usage: dict[str, Any] = {}
    last_finish_reason = None
    last_response_id = None
    last_error = None
    for attempt in range(MAX_GENERATION_ATTEMPTS):
        response = requests.post(
            f"{vllm_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Generate synthetic training conversations exactly as requested. Do not answer as "
                            "yourself and do not add commentary outside the requested JSON."
                        ),
                    },
                    {"role": "user", "content": seed.generation_prompt},
                ],
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "max_tokens": sampling.max_tokens,
                "seed": int(seed.seed_id.rsplit("-", 1)[1]) + attempt,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=REQUEST_TIMEOUT,
        )
        if not response.ok:
            raise RuntimeError(f"vLLM returned {response.status_code}: {response.text[:2000]}")
        payload = response.json()
        choice = payload["choices"][0]
        last_text = choice["message"].get("content") or ""
        last_usage = payload.get("usage", {})
        last_finish_reason = choice.get("finish_reason")
        last_response_id = payload.get("id")
        try:
            conversation = _conversation(last_text, seed)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            last_error = str(error)
            continue
        return CompletionRecord(
            seed=seed,
            conversation=conversation,
            raw_response=last_text,
            accepted_tokens=int(last_usage.get("completion_tokens", 0)),
            usage=last_usage,
            finish_reason=last_finish_reason,
            response_id=last_response_id,
            error=None,
            attempts=attempt + 1,
            elapsed_seconds=time.time() - started,
        )
    return CompletionRecord(
        seed=seed,
        conversation=None,
        raw_response=last_text,
        accepted_tokens=0,
        usage=last_usage,
        finish_reason=last_finish_reason,
        response_id=last_response_id,
        error=last_error,
        attempts=MAX_GENERATION_ATTEMPTS,
        elapsed_seconds=time.time() - started,
    )


def _chunk_path(output_path: StoragePath, shard_index: int, chunk: Chunk) -> StoragePath:
    return (
        _shard_response_path(output_path, shard_index)
        / f"chunk-{chunk.start:09d}-{chunk.end:09d}-tokens-{chunk.tokens:07d}.jsonl.gz"
    )


def _shard_response_path(output_path: StoragePath, shard_index: int) -> StoragePath:
    return output_path / "responses" / f"shard-{shard_index:02d}"


def _existing_chunks(output_path: StoragePath, shard_index: int) -> dict[int, Chunk]:
    chunks = {}
    for path in (_shard_response_path(output_path, shard_index) / "*.jsonl.gz").glob():
        match = CHUNK_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected response filename: {path}")
        chunk = Chunk(
            start=int(match.group("start")),
            end=int(match.group("end")),
            tokens=int(match.group("tokens")),
        )
        chunks[chunk.start] = chunk
    return chunks


def _next_batch_start(existing: set[int], active: set[int], microbatch_size: int) -> int:
    start = 0
    while start in existing or start in active:
        start += microbatch_size
    return start


def _seeds_for_batch(config: CollectionConfig, start: int) -> list[ConversationSeed]:
    seeds = []
    for local_index in range(start, start + config.microbatch_size):
        global_index = config.shard_index + local_index * config.num_shards
        seeds.extend(
            generate_seeds(
                count=1,
                start_index=global_index,
                random_seed=config.random_seed,
                human_phrasings=config.human_phrasings,
                identity_profile=config.identity_profile,
            )
        )
    return seeds


def _write_chunk(config: CollectionConfig, start: int, records: list[CompletionRecord]) -> Chunk:
    tokens = sum(record.accepted_tokens for record in records)
    chunk = Chunk(start=start, end=start + len(records) - 1, tokens=tokens)
    accepted_records = [record for record in records if record.conversation is not None]
    write_jsonl_file(accepted_records, str(_chunk_path(config.output_path, config.shard_index, chunk)))
    return chunk


def _write_progress(
    config: CollectionConfig,
    state: str,
    chunks: dict[int, Chunk],
    accepted_conversations: int,
    rejected_conversations: int,
) -> None:
    progress = Progress(
        run_id=config.run_id,
        profile_id=config.identity_profile.profile_id,
        shard_index=config.shard_index,
        num_shards=config.num_shards,
        state=state,
        target_tokens=config.target_tokens,
        accepted_tokens=sum(chunk.tokens for chunk in chunks.values()),
        accepted_conversations=accepted_conversations,
        rejected_conversations=rejected_conversations,
        completed_microbatches=len(chunks),
        updated_at=datetime.now(UTC).isoformat(),
    )
    (config.output_path / "progress" / f"shard-{config.shard_index:02d}.json").write_text(
        json.dumps(dataclasses.asdict(progress), indent=2, sort_keys=True)
    )


def _run_collection(vllm_url: str, config: CollectionConfig, sampling: SamplingConfig) -> None:
    chunks = _existing_chunks(config.output_path, config.shard_index)
    accepted_tokens = sum(chunk.tokens for chunk in chunks.values())
    if accepted_tokens >= config.target_tokens:
        _write_progress(config, "complete", chunks, 0, 0)
        return

    max_active_batches = max(1, config.concurrency // config.microbatch_size)
    active_records: dict[int, list[CompletionRecord]] = {}
    active_counts: dict[int, int] = {}
    futures: dict[Future[CompletionRecord], int] = {}
    accepted_conversations = 0
    rejected_conversations = 0

    def launch_batch(executor: ThreadPoolExecutor) -> None:
        active = set(active_records)
        start = _next_batch_start(set(chunks), active, config.microbatch_size)
        active_records[start] = []
        active_counts[start] = config.microbatch_size
        for seed in _seeds_for_batch(config, start):
            futures[executor.submit(_completion, vllm_url, seed, sampling)] = start

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        for _ in range(max_active_batches):
            launch_batch(executor)

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            completed_batches = set()
            for future in done:
                start = futures.pop(future)
                record = future.result()
                active_records[start].append(record)
                active_counts[start] -= 1
                if active_counts[start] == 0:
                    completed_batches.add(start)

            for start in completed_batches:
                records = active_records.pop(start)
                del active_counts[start]
                chunk = _write_chunk(config, start, records)
                chunks[start] = chunk
                accepted_conversations += sum(record.conversation is not None for record in records)
                rejected_conversations += sum(record.conversation is None for record in records)
                accepted_tokens += chunk.tokens
                _write_progress(config, "running", chunks, accepted_conversations, rejected_conversations)
                logger.info(
                    "Shard %d saved microbatch %d with %d tokens; total %d/%d",
                    config.shard_index,
                    start,
                    chunk.tokens,
                    accepted_tokens,
                    config.target_tokens,
                )

            for _ in completed_batches:
                if accepted_tokens < config.target_tokens:
                    launch_batch(executor)

    _write_progress(config, "complete", chunks, accepted_conversations, rejected_conversations)
    if accepted_tokens < config.target_tokens:
        raise RuntimeError(f"Shard stopped at {accepted_tokens} of {config.target_tokens} target tokens")


def run(config: CollectionConfig, server: ServerConfig, sampling: SamplingConfig) -> None:
    ctx = iris_ctx()
    if ctx is None or ctx.client is None:
        raise RuntimeError("Collector must run inside an Iris job")
    endpoint_suffix = f"{config.run_id}-s{config.shard_index}"
    vllm_endpoint = f"{VLLM_ENDPOINT}-{endpoint_suffix}"
    ray_endpoint = f"{RAY_ENDPOINT}-{endpoint_suffix}"
    vllm_job = submit_glm52(ctx, Glm52LaunchConfig(vllm_endpoint, ray_endpoint, server))
    try:
        vllm_url = wait_for_endpoint_url(vllm_endpoint, vllm_job)
        _run_collection(vllm_url, config, sampling)
    finally:
        vllm_job.terminate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--microbatch-size", type=int, default=DEFAULT_MICROBATCH_SIZE)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--phrasing-input")
    parser.add_argument("--identity-profile", required=True)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--execution-mode", choices=("compile", "eager"), default="eager")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    for name in ("num_shards", "target_tokens", "microbatch_size", "concurrency", "max_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.concurrency < args.microbatch_size:
        parser.error("--concurrency must be at least --microbatch-size")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")

    human_phrasings = tuple(load_human_phrasings(StoragePath(args.phrasing_input))) if args.phrasing_input else ()
    identity_profile = load_identity_profile(StoragePath(args.identity_profile))
    run(
        CollectionConfig(
            run_id=args.run_id,
            output_path=StoragePath(args.output_path),
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            target_tokens=args.target_tokens,
            microbatch_size=args.microbatch_size,
            concurrency=args.concurrency,
            random_seed=args.random_seed,
            human_phrasings=human_phrasings,
            identity_profile=identity_profile,
        ),
        ServerConfig(
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            kv_cache_dtype=args.kv_cache_dtype,
            execution_mode=args.execution_mode,
        ),
        SamplingConfig(args.temperature, args.top_p, args.max_tokens),
    )


if __name__ == "__main__":
    main()
