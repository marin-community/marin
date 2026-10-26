# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Luke Bailey's logit-averaging teacher prompts for GSM8K.

Each prompt is the standard five-shot GSM8K prompt followed by the problem's
reference working, without its ``####`` answer line, and a newline:
``...Question: {question}\\nAnswer: {working}\\n``. For the first 256 test
problems this equals, byte for byte, the ``logit_average`` teacher prompt that
``experiments/query_hinted_benchmarks/make_prompts.py`` writes in
LukeBailey181/emergence-logit-mixing at 56073be1dc97163b08121edbfc2b13e696b2aea7.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import fsspec
from fray.cluster import ResourceConfig
from thalas.execution.executor import ExecutorStep, output_path_of
from thalas.execution.remote import remote
from thalas.execution.types import this_output_path

from experiments.downstream_scaling.evals.framework.schema import PROMPTS_FILENAME, prompts_file, read_prompt_rows
from experiments.downstream_scaling.evals.tasks.gsm8k import GSM8KTask, GSM8KTaskConfig
from experiments.downstream_scaling.evals.utils import version_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LukeTeacherGSM8KPromptsConfig:
    output_path: str
    source_prompts_path: str


@dataclass(frozen=True)
class LukeTeacherGSM8KTask:
    config: GSM8KTaskConfig

    make_grade_step = GSM8KTask.make_grade_step

    def make_prompts_step(self) -> ExecutorStep:
        source_prompts = GSM8KTask(config=self.config).make_prompts_step()
        return ExecutorStep(
            name="downstream_scaling/evals/prompts/gsm8k_luke_teacher",
            fn=remote(write_luke_teacher_gsm8k_prompts, resources=ResourceConfig.with_cpu(cpu=1, ram="4g")),
            config=LukeTeacherGSM8KPromptsConfig(
                output_path=this_output_path(),
                source_prompts_path=version_path(output_path_of(source_prompts) / PROMPTS_FILENAME),  # type: ignore[arg-type]
            ),
        )


def write_luke_teacher_gsm8k_prompts(config: LukeTeacherGSM8KPromptsConfig) -> None:
    """Append each problem's reference working to its standard GSM8K prompt."""
    path = prompts_file(config.output_path)
    count = 0
    with fsspec.open(path, "wt", compression="gzip") as f:
        for row in read_prompt_rows(config.source_prompts_path):
            solution = row["metadata"]["solution"]
            working = solution[: solution.rindex("####")].rstrip()
            f.write(json.dumps({**row, "prompt": f"{row['prompt']} {working}\n"}) + "\n")
            count += 1
    logger.info("Wrote %d Luke teacher GSM8K prompts to %s", count, path)
