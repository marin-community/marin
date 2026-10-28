# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthesize the tasks of fixed capability ideas before training, to test the generate box apart from its timing.

The base policy, served here with vLLM, is the solver of the Taskforge gate, as the live policy is at the start of a
loop. Every session runs to its end, and the training instances of the accepted tasks go to ``synthesized.jsonl``;
``run.py skyrl-oracle --tasks-file`` then trains on them from the first step. Run it in an Iris GPU task through
``job.sh``, with ``GLM_BULK_TOKEN`` and ``PARALLEL_API_KEY`` set.
"""

import asyncio
import logging
from contextlib import AsyncExitStack

import click
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.post_training.rsi.evaluate import PolicyOptions, Served, policy_options, serve
from experiments.post_training.rsi.feedback import generate_tasks, read_ideas
from experiments.post_training.rsi.run import (
    FeedbackOptions,
    _pop_feedback,
    feedback_options,
    synthesis_config,
)
from experiments.post_training.rsi.streams import from_iterable

logger = logging.getLogger(__name__)

SYNTHESIZED_FILE = "synthesized.jsonl"


@click.command()
@policy_options
@feedback_options
@click.option("--output-dir", required=True)
def main(output_dir: str, **arguments) -> None:
    """Synthesize the tasks of ``--ideas-file`` with the policy of ``--model`` as the gate's solver."""
    configure_logging(logging.INFO)
    feedback = _pop_feedback(arguments)
    if feedback.ideas_file is None:
        raise click.UsageError("--ideas-file is required")
    asyncio.run(_synthesize(feedback, PolicyOptions(**arguments), StoragePath(output_dir)))


async def _synthesize(feedback: FeedbackOptions, options: PolicyOptions, output: StoragePath) -> None:
    assert feedback.ideas_file is not None
    ideas = read_ideas(feedback.ideas_file)
    path = output / SYNTHESIZED_FILE
    lines: list[str] = []
    with serve(options) as server:
        solver = Served(options, server).policy
        async with AsyncExitStack() as stack:
            synthesis = await synthesis_config(stack, feedback, solver, output)
            async for task in generate_tasks(from_iterable(ideas), synthesis):
                lines.append(task.model_dump_json() + "\n")
                await asyncio.to_thread(path.write_text, "".join(lines))
    logger.info("Synthesized %d training instances for %d ideas", len(lines), len(ideas))


if __name__ == "__main__":
    main()
