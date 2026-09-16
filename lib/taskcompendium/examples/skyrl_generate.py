# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Harbor lowerings through MarinSkyRL's trajectory consumer interface.

Run in the pinned MarinSkyRL environment with TaskCompendium's Harbor extra.
This adapter does not construct a trainer, Ray worker, or inference engine.
"""

import argparse
import asyncio
import json
from pathlib import Path

from transformers import AutoTokenizer

from taskcompendium.skyrl import TaskCompendiumTrajectoryRunner, request_batch


async def generate(args) -> None:
    rows = json.loads(args.requests.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    runner = TaskCompendiumTrajectoryRunner(tokenizer, args.output, concurrency=args.concurrency)
    await runner.startup()
    try:
        batch = await runner.run(request_batch(rows, repetitions=args.repetitions))
    finally:
        await runner.shutdown()
    # Dataclass trajectory identities need an explicit JSON wire projection.
    record = {**batch, "trajectory_ids": [vars(identity) for identity in batch["trajectory_ids"]]}
    with (args.output / "batch.json").open("x") as output:
        json.dump(record, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests", type=Path, required=True, help="JSON array of uid, task_dir, and resolved execution"
    )
    parser.add_argument(
        "--tokenizer", required=True, help="Tokenizer repository or local path; separate from served model ID"
    )
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    asyncio.run(generate(parser.parse_args()))


if __name__ == "__main__":
    main()
