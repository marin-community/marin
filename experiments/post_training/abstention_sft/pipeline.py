# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the abstention SFT data: Nemotron problems, the student probe, then chat rows.

``abstention_sft_steps`` wires the three Marin steps for a cluster run on the v1 SFT model. ``main``
runs the same functions against an OpenAI-compatible endpoint on one machine, for a small prototype.
The probe splits replies on Marin's think delimiters, so the local student must use the Marin chat
template.
"""

import argparse
from pathlib import Path

from marin.evaluation.hardware import AcceleratorChoice
from marin.evaluation.model_config import ModelConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.inference.types import OpenAIEndpoint
from openai import OpenAI
from rigging.secrets import require_secret
from verifyit.modes.grade_judge import JudgeConnection

from experiments.post_training.abstention_sft.knowledge import SAMPLES
from experiments.post_training.abstention_sft.probe import probe_problems, probe_step, write_probe
from experiments.post_training.abstention_sft.problems import load_problems, problems_step
from experiments.post_training.abstention_sft.rows import build_rows, rows_step

VERSION = "2026.11.25"


def abstention_sft_steps(
    *,
    model: ModelConfig,
    accelerator: AcceleratorChoice,
    judge_endpoint: str,
    judge_api_key_env: str,
    judge_model: str,
    seed: int,
) -> ArtifactStep[Artifact]:
    """The rows step and its dependencies; the same deployment judges answers and writes abstain replies."""
    problems = problems_step()
    probe = probe_step(
        problems,
        model=model,
        accelerator=accelerator,
        judge_endpoint=judge_endpoint,
        judge_api_key_env=judge_api_key_env,
        judge_model=judge_model,
        seed=seed,
        version=VERSION,
    )
    return rows_step(
        problems,
        probe,
        writer_endpoint=judge_endpoint,
        writer_api_key_env=judge_api_key_env,
        writer_model=judge_model,
        version=VERSION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-endpoint", required=True, help="OpenAI-compatible base URL, e.g. .../v1")
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--judge-endpoint", required=True)
    parser.add_argument("--judge-api-key-env", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--limit", type=int, required=True, help="number of problems to probe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    problems = load_problems()[: arguments.limit]
    api_key = require_secret(arguments.judge_api_key_env)
    connection = JudgeConnection(arguments.judge_endpoint, api_key, default_model=arguments.judge_model)
    student = OpenAIEndpoint(base_url=arguments.student_endpoint, model=arguments.student_model)
    samples, labels = probe_problems(student, problems, connection, arguments.seed)
    write_probe(
        str(arguments.output / "probe"),
        samples,
        labels,
        {"model": arguments.student_model, "judge_model": arguments.judge_model, "samples_per_question": SAMPLES},
    )
    build_rows(
        problems,
        samples,
        labels,
        OpenAI(base_url=arguments.judge_endpoint, api_key=api_key),
        arguments.judge_model,
        str(arguments.output / "rows"),
    )


if __name__ == "__main__":
    main()
