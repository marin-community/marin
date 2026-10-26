# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train or serve a standalone Grug EAGLE3 draft inside an Iris GPU task."""

import argparse
import json
from dataclasses import asdict

from fray.types import ResourceConfig, create_environment
from marin.drafting import DraftTrainingConfig, draft_serving, prepare_coding_sample, resume_draft, train_draft
from marin.inference.config import IrisConfig
from marin.inference.iris import IrisServiceConfig, run_iris_service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Write a pinned deterministic coding prompt sample (CPU)")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--sample-size", type=int, default=12)
    train = commands.add_parser("train", help="Generate responses, extract states, train and export")
    for name in ("target", "tokenizer", "tokenizer-revision", "data", "output"):
        train.add_argument(f"--{name}", required=True)
    train.add_argument("--initial-draft", help="Compatible serving export; starts a new optimizer schedule")
    defaults = DraftTrainingConfig("", "", "", "", "")
    for name in ("epochs", "sequence_length", "max_new_tokens"):
        train.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    for name in ("holdout_fraction", "learning_rate", "muon_learning_rate"):
        train.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    resume = commands.add_parser("resume", help="Restore the original run, including optimizer/scheduler/progress")
    resume.add_argument("--output", required=True)
    serve = commands.add_parser("serve", help="Serve the recorded target with its exported draft through Marin")
    serve.add_argument("--output", required=True)
    serve.add_argument("--endpoint", required=True)
    serve.add_argument("--timeout-hours", type=float, default=1.0)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    if command == "prepare":
        print(prepare_coding_sample(**arguments))
    elif command == "train":
        print(json.dumps(asdict(train_draft(DraftTrainingConfig(**arguments)))))
    elif command == "resume":
        print(json.dumps(asdict(resume_draft(**arguments))))
    else:
        with draft_serving(arguments["output"]) as (model, engine):
            run_iris_service(
                IrisServiceConfig(
                    model=model,
                    engine=engine,
                    iris=IrisConfig(
                        worker_resources=ResourceConfig.with_gpu("H100", 8),
                        worker_environment=create_environment(extras=["cpu"]),
                    ),
                    endpoint_name=arguments["endpoint"],
                    timeout_hours=arguments["timeout_hours"],
                )
            )


if __name__ == "__main__":
    main()
