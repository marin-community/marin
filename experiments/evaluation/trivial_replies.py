# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score replies that ignore the task on published evaluation corpora through the standard TaskSpec driver.

Each reply stands in for the model in ``run_evaluation``, so cohort selection, prompt variants, response
selection and grading are those of a model run. The run configuration is that of the model run to compare
with: its tokenizer, context window and budgets select the cohort. Runs on this host, as ``task_driver
--local`` does, and writes one FineStore archive per reply under ``--output``.
"""

import argparse
import asyncio
import json
import logging
from collections.abc import Sequence
from enum import StrEnum
from functools import partial

from evalrecords.rollout_results import benchmark_scores
from finestore.reader import ReadView
from levanter.compat.hf_checkpoints import load_tokenizer
from marin.evaluation.task_worker import DEFAULT_MAX_ATTEMPTS, execution_config
from marin.rollouts.evaluation import EvaluationInfrastructureError, EvaluationRunConfig, run_evaluation
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import retry_with_backoff
from rolloutengine.models import ModelRequest, ModelTurn
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import Backend

# Bump when a reply's text changes so a resumed archive cannot mix grades of different replies.
REPLY_REVISION = "2026.11.18.1"


class TrivialReply(StrEnum):
    """A reply that ignores the task."""

    EMPTY = "empty"
    WORD = "word"
    ZERO = "zero"
    ABSTAIN = "abstain"
    ECHO = "echo"
    """The request's last user message, repeated."""
    JSON_EMPTY = "json_empty"
    JSON_WORD = "json_word"
    JSON_ABSTAIN = "json_abstain"
    JSON_ECHO = "json_echo"


PLAIN_REPLIES = (TrivialReply.EMPTY, TrivialReply.WORD, TrivialReply.ZERO, TrivialReply.ABSTAIN, TrivialReply.ECHO)
# A JSON-value answer format parses the whole reply as one JSON value; these replies are the JSON strings
# of the plain ones. ``0`` is already a JSON value.
JSON_REPLIES = {
    TrivialReply.JSON_EMPTY: TrivialReply.EMPTY,
    TrivialReply.JSON_WORD: TrivialReply.WORD,
    TrivialReply.JSON_ABSTAIN: TrivialReply.ABSTAIN,
    TrivialReply.JSON_ECHO: TrivialReply.ECHO,
}
CONSTANT_REPLIES = {
    TrivialReply.EMPTY: "",
    TrivialReply.WORD: "hello",
    TrivialReply.ZERO: "0",
    TrivialReply.ABSTAIN: "I don't know.",
}


def reply_text(reply: TrivialReply, request: ModelRequest) -> str:
    """The text of ``reply`` to ``request``."""
    if reply in JSON_REPLIES:
        return json.dumps(reply_text(JSON_REPLIES[reply], request), ensure_ascii=False)
    if reply is not TrivialReply.ECHO:
        return CONSTANT_REPLIES[reply]
    prompts = [message["content"] for message in request.messages if message["role"] == "user"]
    if not prompts:
        raise ValueError("The echo reply repeats a user message, and this request has none")
    return prompts[-1]


async def reply_turn(request: ModelRequest, *, reply: TrivialReply) -> ModelTurn:
    """Answer ``request`` with ``reply`` as a completed turn that used no tokens."""
    return ModelTurn(
        message={"role": "assistant", "content": reply_text(reply, request)},
        prompt_token_ids=(),
        response_token_ids=(),
        logprobs=None,
        stop_reason="stop",
    )


def reply_config(config: EvaluationRunConfig, reply: TrivialReply) -> EvaluationRunConfig:
    """``config`` with the model identity of ``reply``, so its archive cannot pass for the model's."""
    model = config.model.model_copy(
        update={"name": f"trivial-reply/{reply}", "revision": REPLY_REVISION, "serving": {"backend": "constant-reply"}}
    )
    return config.model_copy(update={"model": model, "judge": None})


def evaluate_reply(datasets: tuple[str, ...], output: str, config: EvaluationRunConfig, reply: TrivialReply) -> None:
    """Evaluate ``reply`` on the corpora into the archive at ``output``, retaining completed attempts on resume.

    The driver refuses corpora scored by likelihood or graded by a judge.
    """
    configure_coreweave_s3()
    config = execution_config(reply_config(config, reply))
    tokenizer = load_tokenizer(config.model.tokenizer, revision=config.model.tokenizer_revision, trust_remote_code=False)
    model = partial(reply_turn, reply=reply)
    asyncio.run(
        run_evaluation(
            datasets,
            output,
            config,
            model,
            completion_model=model,
            machine_factories={Backend.LOCAL: LocalMachineFactory()},
            tokenizer=tokenizer,
        )
    )


def floor_table(output: str, replies: Sequence[TrivialReply]) -> str:
    """A Markdown table of each benchmark's primary metric per reply, as Evaldash reduces it.

    A pass/fail metric is followed by the trials that passed over the trials scored, and any other metric by
    the trials scored.
    """
    cells: dict[tuple[str, str], dict[TrivialReply, str]] = {}
    for reply in replies:
        for benchmark, scores in benchmark_scores(ReadView(str(StoragePath(output) / reply))).items():
            metric = scores.protocol.primary_metric
            score = scores.metrics.get(metric)
            coverage = scores.coverage
            trials = (
                str(coverage.n_scored) if coverage.n_correct is None else f"{coverage.n_correct}/{coverage.n_scored}"
            )
            cells.setdefault((benchmark, metric), {})[reply] = "-" if score is None else f"{score:.3f} ({trials})"
    header = ["Benchmark", "Metric", *replies]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for (benchmark, metric), row in cells.items():
        lines.append("| " + " | ".join([benchmark, metric, *(row.get(reply, "-") for reply in replies)]) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", action="append", required=True, help="Published corpus root; repeat for a mixed run."
    )
    parser.add_argument(
        "--output", required=True, help="Root for one FineStore archive per reply, including existing runs to resume."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="EvaluationRunConfig JSON of the model run to compare with; its model name and judge are not used.",
    )
    parser.add_argument(
        "--reply",
        action="append",
        type=TrivialReply,
        choices=tuple(TrivialReply),
        help="Reply to score; repeat for several. Defaults to the plain replies.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Total passes for grader infrastructure failures; successful attempts are retained between passes.",
    )
    arguments = parser.parse_args()
    if arguments.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    config = EvaluationRunConfig.model_validate_json(StoragePath(arguments.config).read_text())
    datasets = tuple(arguments.dataset)
    replies = tuple(arguments.reply or PLAIN_REPLIES)
    logging.basicConfig(level=logging.INFO)
    for reply in replies:
        retry_with_backoff(
            partial(evaluate_reply, datasets, str(StoragePath(arguments.output) / reply), config, reply),
            retryable=lambda error: isinstance(error, EvaluationInfrastructureError),
            max_attempts=arguments.max_attempts,
            operation=f"Trivial-reply evaluation ({reply})",
        )
    print(floor_table(arguments.output, replies))


if __name__ == "__main__":
    main()
