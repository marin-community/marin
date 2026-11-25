# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect train-only, frozen Qwen completion groups from an existing Iris serving job."""

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import click
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from marin.evaluation.model_config import load_model_config
from openai import APIError, AsyncOpenAI
from rigging.config_discovery import find_project_root
from rigging.timing import Duration
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineSpec, PackageSpec
from taskcompendium.chat import chat_conversation
from taskcompendium.grading_result import Outcome
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import GradingAttempt, TaskSpec
from taskcompendium.runtime.grading import grade_in_sandbox
from taskcompendium.submission import conversation_messages, submission_instruction
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from verifyit.modes.extract import extract_boxed

from experiments.models import qwen3_0_6b
from experiments.post_training.task_curation import qwen_sampler_lab as lab

MODEL_URI = qwen3_0_6b.path(prefix=lab.CANONICAL_ARTIFACT_ROOT)
GROUP_SIZE = 4
TERSE_INSTRUCTION = (
    r"Only output \boxed{answer}, with no explanation. For example, if the answer is 42, output \boxed{42}."
)
SAMPLING = MappingProxyType(
    {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "min_tokens": 1,
        "repetition_penalty": 1.0,
        "n": GROUP_SIZE,
        "return_token_ids": True,
        "skip_special_tokens": True,
        "include_stop_str_in_output": True,
    }
)


@dataclass(frozen=True)
class PreparedPrompt:
    task: TaskSpec
    task_sha256: str
    family: str
    difficulty: str
    token_ids: tuple[int, ...]


def prepare_panel(
    tokenizer: PreTrainedTokenizerFast,
    *,
    prompts_per_cell: int,
    seed: int,
    enable_thinking: bool,
    style: str,
    excluded_task_ids: frozenset[str] = frozenset(),
) -> list[PreparedPrompt]:
    """Draw reproducible, distinct training prompts from each family and difficulty."""
    cells: dict[tuple[str, str], list[TaskSpec]] = {
        (family, difficulty): [] for family in lab.FAMILIES for difficulty in lab.Difficulty
    }
    for example in lab.lab_examples():
        if example.split == "train" and example.task.id not in excluded_task_ids:
            cells[example.strata.family, example.strata.difficulty].append(example.task)
    selected = []
    for (family, difficulty), tasks in sorted(cells.items()):
        if len(tasks) < prompts_per_cell:
            raise ValueError(f"only {len(tasks)} eligible training tasks in {family}/{difficulty}")
        rng = random.Random(f"{lab.NAME}/{lab.VERSION}/{seed}/{family}/{difficulty}")
        for task in rng.sample(sorted(tasks, key=lambda item: item.id), prompts_per_cell):
            instruction = TERSE_INSTRUCTION if style == "terse" else submission_instruction(task.answer_format)
            messages = [*conversation_messages(task.context.events), {"role": "user", "content": instruction}]
            tokens = cast(
                list[int],
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                ),
            )
            selected.append(
                PreparedPrompt(
                    task,
                    canonical_sha256(task.model_dump(mode="json")),
                    family,
                    difficulty,
                    tuple(tokens),
                )
            )
    return selected


def completion_seed(seed: int, task_id: str, group: int) -> int:
    """Pair the same sampling seed across caps while drawing independent groups per task."""
    digest = hashlib.sha256(f"{seed}/{task_id}/{group}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


async def verified_answer(task: TaskSpec, text: str, factory: ShellSimMachineFactory) -> bool:
    """Return whether the native sandbox grader awarded the full binary reward."""
    messages = [*conversation_messages(task.context.events), {"role": "assistant", "content": text}]
    attempt = GradingAttempt(chat_conversation(messages))
    grade = await grade_in_sandbox(task, attempt, factory, MachineSpec(PackageSpec()))
    if grade.status not in (Outcome.GRADED, Outcome.SUBMISSION_FAILURE) or grade.reward not in (0.0, 1.0):
        raise ValueError(f"Missing binary reward for {task.id}: {grade.status} ({grade.error})")
    return grade.reward == 1.0


def reasoning_split(token_ids: list[int], *, opening: int, closing: int) -> tuple[int, bool]:
    """Count the generated thinking prefix and detect a generated opening marker without closure."""
    if closing in token_ids:
        return token_ids.index(closing) + 1, False
    if opening in token_ids:
        return len(token_ids), True
    return 0, False


def api_url(cluster: str, endpoint: str) -> str:
    with connect_controller(cluster_name=cluster) as controller:
        with IrisClient.remote(controller.url, credentials=controller.credentials) as client:
            response = client.mint_endpoint_token(endpoint, ttl=Duration.from_hours(1))
    if not response.capability_url:
        raise RuntimeError("Iris did not return a capability URL for the serving endpoint")
    return response.capability_url.rstrip("/") + "/v1"


async def capture_group(
    client: AsyncOpenAI,
    tokenizer: PreTrainedTokenizerFast,
    prompt: PreparedPrompt,
    factory: ShellSimMachineFactory,
    *,
    cap: int,
    group: int,
    seed: int,
) -> list[dict[str, Any]]:
    request_seed = completion_seed(seed, prompt.task.id, group)
    completion = await client.completions.create(
        model=MODEL_URI,
        prompt=list(prompt.token_ids),
        max_tokens=cap,
        seed=request_seed,
        extra_body=dict(SAMPLING),
    )
    if len(completion.choices) != GROUP_SIZE:
        raise ValueError(f"expected {GROUP_SIZE} replies for {prompt.task.id}, got {len(completion.choices)}")
    opening = tokenizer.convert_tokens_to_ids("<think>")
    closing = tokenizer.convert_tokens_to_ids("</think>")
    rows = []
    for choice in completion.choices:
        extra = choice.model_extra or {}
        ids = extra.get("token_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"endpoint omitted response token IDs for {prompt.task.id}")
        reasoning_tokens, incomplete_thinking = reasoning_split(ids, opening=opening, closing=closing)
        final_answer = tokenizer.decode(ids[reasoning_tokens:], skip_special_tokens=True)
        verified = await verified_answer(prompt.task, choice.text, factory)
        final_verified = (
            verified if final_answer == choice.text else await verified_answer(prompt.task, final_answer, factory)
        )
        rows.append(
            {
                "task_id": prompt.task.id,
                "task_sha256": prompt.task_sha256,
                "family": prompt.family,
                "difficulty": prompt.difficulty,
                "prompt_sha256": canonical_sha256({"token_ids": prompt.token_ids}),
                "group": group,
                "max_tokens": cap,
                "request_seed": request_seed,
                "sample": choice.index,
                "text": choice.text,
                "response_token_ids": ids,
                "response_tokens": len(ids),
                "reasoning_tokens": reasoning_tokens,
                "incomplete_thinking": incomplete_thinking,
                "finish_reason": choice.finish_reason,
                "stop_reason": extra.get("stop_reason"),
                "complete_box": extract_boxed(choice.text) is not None,
                "verified": verified,
                "final_answer_verified": final_verified,
            }
        )
    return rows


async def collect(
    *,
    tokenizer: PreTrainedTokenizerFast,
    panel: list[PreparedPrompt],
    caps: tuple[int, ...],
    groups_per_prompt: int,
    seed: int,
    base_url: str,
    output: Path,
) -> tuple[int, int]:
    """Write verified groups and return the gross generated token count and number of replies."""
    factory = ShellSimMachineFactory()
    requests = ((prompt, cap, group) for cap in caps for prompt in panel for group in range(groups_per_prompt))
    output_tokens = reply_count = 0
    try:
        async with AsyncOpenAI(base_url=base_url, api_key="iris-capability", max_retries=0, timeout=600) as client:
            models = await client.models.list()
            if MODEL_URI not in {model.id for model in models.data}:
                raise ValueError("The endpoint must serve the pinned regional Qwen model URI")
            with output.open("x") as destination:
                for batch in batched(requests, 16):
                    results = await asyncio.gather(
                        *(
                            capture_group(client, tokenizer, prompt, factory, cap=cap, group=group, seed=seed)
                            for prompt, cap, group in batch
                        )
                    )
                    for rows in results:
                        for row in rows:
                            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                            output_tokens += row["response_tokens"]
                            reply_count += 1
    except APIError as error:
        raise RuntimeError(f"Iris inference failed ({type(error).__name__})") from None
    return output_tokens, reply_count


@click.command(help=__doc__)
@click.option("--cluster", default="marin", show_default=True)
@click.option("--endpoint", required=True, help="Name of the already-running Iris vLLM endpoint.")
@click.option("--output", type=click.Path(path_type=Path, exists=False), required=True)
@click.option("--cap", "caps", type=click.IntRange(min=1), multiple=True, required=True)
@click.option("--enable-thinking/--disable-thinking", default=True, show_default=True)
@click.option("--style", type=click.Choice(("task", "terse")), default="task", show_default=True)
@click.option("--prompts-per-cell", type=click.IntRange(1, lab.TRAIN_PER_CELL), default=16, show_default=True)
@click.option("--groups-per-prompt", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--seed", type=int, required=True)
@click.option(
    "--exclude-task-ids",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON list of task IDs to exclude.",
)
def main(
    cluster: str,
    endpoint: str,
    output: Path,
    caps: tuple[int, ...],
    enable_thinking: bool,
    style: str,
    prompts_per_cell: int,
    groups_per_prompt: int,
    seed: int,
    exclude_task_ids: Path | None,
) -> None:
    if len(set(caps)) != len(caps):
        raise click.BadParameter("each cap must be distinct", param_hint="--cap")
    serving_model = load_model_config(
        find_project_root(Path(__file__)) / "experiments/evaluation/serve/models/Qwen/Qwen3-0.6B.yaml"
    )
    tokenizer = AutoTokenizer.from_pretrained(serving_model.location, revision=serving_model.revision)
    assert isinstance(tokenizer, PreTrainedTokenizerFast) and isinstance(tokenizer.chat_template, str)
    excluded = [] if exclude_task_ids is None else json.loads(exclude_task_ids.read_text())
    if not isinstance(excluded, list) or not all(isinstance(task_id, str) for task_id in excluded):
        raise click.BadParameter("expected a JSON list of task IDs", param_hint="--exclude-task-ids")
    panel = prepare_panel(
        tokenizer,
        prompts_per_cell=prompts_per_cell,
        seed=seed,
        enable_thinking=enable_thinking,
        style=style,
        excluded_task_ids=frozenset(excluded),
    )
    base_url = api_url(cluster, endpoint)
    output.mkdir()
    responses = output / "responses.jsonl"
    started = time.monotonic()
    tokens, replies = asyncio.run(
        collect(
            tokenizer=tokenizer,
            panel=panel,
            caps=caps,
            groups_per_prompt=groups_per_prompt,
            seed=seed,
            base_url=base_url,
            output=responses,
        )
    )
    elapsed = time.monotonic() - started
    with responses.open("rb") as source:
        response_digest = hashlib.file_digest(source, "sha256").hexdigest()
    manifest = {
        "schema_version": 1,
        "purpose": "frozen-policy sampler replay",
        "selection": f"{lab.NAME}@{lab.VERSION}",
        "split": "train",
        "model_uri": MODEL_URI,
        "model_revision": serving_model.revision,
        "tokenizer_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "endpoint": endpoint,
        "cluster": cluster,
        "prompt_style": style,
        "enable_thinking": enable_thinking,
        "caps": caps,
        "decoding": dict(SAMPLING),
        "seed": seed,
        "excluded_task_ids_sha256": canonical_sha256({"task_ids": sorted(set(excluded))}),
        "prompts_per_cell": prompts_per_cell,
        "groups_per_prompt": groups_per_prompt,
        "reply_count": replies,
        "generated_tokens": tokens,
        "request_wall_seconds": elapsed,
        "responses_sha256": response_digest,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    click.echo(f"Saved {replies} replies and {tokens} tokens in {elapsed:.1f} seconds to {output}")


if __name__ == "__main__":
    main()
