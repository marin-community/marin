# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Python benchmarks with private test programs run as Shellbox commands.

The 3-second (CruxEval: 10-second) budget includes Python startup and compilation,
unlike the source's exec-only alarm. Backend limits replace the source's process
memory limit and reliability guard.
"""

import hashlib
import json
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.code import THREAD_ENVIRONMENT
from taskcompendium.convert.evaluation import source_evidence
from taskcompendium.convert.tasks import conversation_task
from taskcompendium.evaluation import EvaluationItem, EvaluationProtocol, Sampling, TextGeneration, TrialPolicy
from taskcompendium.grader import GraderPackage
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import ExitCodeReward, ScriptGrader, TaskSpec, TextMessage
from taskcompendium.pipeline.evaluation import chat_item
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat, required_grader_environment
from taskcompendium.pipeline.models import IntendedUse, RawRow
from taskcompendium.runtime.resources import inline_resource
from verifyit.adapters import evalchemy_code
from verifyit.adapters.evalchemy_code import HUMANEVAL_REQUEST_STOPS, function_prompt, mbpp_prompt

from experiments.post_training.task_curation.datasets.evalchemy.datasets import ACCURACY
from experiments.post_training.task_curation.datasets.evalchemy.pipeline import EvalDataSet, curate
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

ASSETS = Path(__file__).parent / "assets"
PYTHON_GRADER = Environment(requirements=("numpy==2.3.5",))
CODE_ASSETS = (
    inline_resource("prepare.py", (ASSETS / "python_prepare.py").read_bytes()),
    inline_resource("code_protocol.py", Path(evalchemy_code.__file__).read_bytes()),
    inline_resource("NOTICE.txt", (ASSETS / "NOTICE.txt").read_bytes()),
    inline_resource(
        "lm-evaluation-harness-MIT.txt",
        (Path(evalchemy_code.__file__).parent / "licenses/lm-evaluation-harness-MIT.txt").read_bytes(),
    ),
)
CODE_REVISION = (
    "2-"
    + canonical_sha256(
        {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(evalchemy_code.__file__),
                ASSETS / "python_prepare.py",
                ASSETS / "crux_input.txt",
                ASSETS / "crux_output.txt",
            )
        }
    )[:16]
)


def read_mbpp(path: StoragePath, _context: ConversionContext):
    """Keep the source's first 500 rows and its fixed examples at positions 1, 2 and 3."""
    with path.open("rb") as file:
        rows = pq.read_table(file).to_pylist()
    examples = rows[1:4]
    for row in rows[:500]:
        yield {**row, "evaluation_prompt": mbpp_prompt(row, examples)}


def read_crux(path: StoragePath, _context: ConversionContext):
    with path.open("rt") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    for direction in ("input", "output"):
        template = (ASSETS / f"crux_{direction}.txt").read_text()
        for row in rows:
            prompt = template.replace("{input}", row["input"]).replace("{output}", row["output"])
            yield {**row, "direction": direction, "evaluation_prompt": prompt.replace("{code}", row["code"])}


def convert_python(row: RawRow, context: ConversionContext, *, suite: str) -> TaskSpec:
    problem = dict(row.data)
    prompt = (
        (problem["prompt"] if suite == "humaneval" else function_prompt(problem))
        if suite in {"humaneval", "humanevalplus"}
        else problem["evaluation_prompt"]
    )
    timeout = 10 if suite == "cruxeval" else 3
    environment = required_grader_environment(context)
    environment = environment.model_copy(
        update={
            "setup_commands": (*environment.setup_commands, "python3 /tests/prepare.py"),
            "environment_variables": {**environment.environment_variables, **THREAD_ENVIRONMENT},
        }
    )
    config = {"suite": suite, "problem": problem, "direction": problem.get("direction", "")}
    package = GraderPackage(
        ScriptGrader(
            argv=("sh", "-c", "python3 /app/program.py && test -f /app/passed"),
            cwd="/app",
            environment=environment,
            timeout=timeout,
            answer_path="/app/answer.txt",
            reward=ExitCodeReward(),
        ),
        (*CODE_ASSETS, inline_resource("config.json", json.dumps(config, allow_nan=False, sort_keys=True).encode())),
    )
    task = conversation_task(row, events=(TextMessage(role="user", content=prompt),), package=package, evidence=problem)
    return task.model_copy(update={"tags": ("eval", "source:evalchemy", f"benchmark:{suite}", "code")})


def python_item(task: TaskSpec, *, completion: bool) -> EvaluationItem:
    item = chat_item(task)
    row = source_evidence(task)
    if completion:
        item = item.model_copy(update={"operation": TextGeneration(context=row["prompt"])})
    return item.model_copy(
        update={"groups": {"direction": row["direction"]} if "direction" in row else {"language": "python"}}
    )


def python_source(
    name: str,
    suite: str,
    source: HfSource,
    rows: int,
    max_tokens: int,
) -> RlDataSource[EvalDataSet]:
    completion = suite == "humaneval"
    sampled = suite == "cruxeval"
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + suite,
            title=name,
            origin="Evalchemy",
            family="python-code",
            tags=("eval", "source:evalchemy", "benchmark", "code", "single-turn"),
            count=rows,
        ),
        config=EvalDataSet(
            recipe=CurationRecipe(
                name="evalchemy-" + suite,
                source=source,
                convert=partial(convert_python, suite=suite),
                version=CODE_REVISION,
                intended_use=IntendedUse.EVAL,
                grader=PYTHON_GRADER,
            ),
            protocol=EvaluationProtocol(
                benchmark=name,
                version="source-sampling-code-v5",
                operation="text_generation" if completion else "chat_generation",
                sampling=Sampling(
                    temperature=0.2 if sampled else 0,
                    top_p=None,
                    top_k=None,
                    max_tokens=max_tokens,
                    stop=HUMANEVAL_REQUEST_STOPS if completion else (),
                    chat_template_kwargs={} if completion else {"enable_thinking": False},
                ),
                trials=TrialPolicy(count=1, seed=0 if sampled else None),
                metrics=(ACCURACY,),
                primary_metric="accuracy",
                reducer="mean_by_direction_v1" if sampled else "mean_v1",
                expected_population=rows,
            ),
            item_builder=partial(python_item, completion=completion),
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataSet]]:
    return [
        python_source(
            "humaneval",
            "humaneval",
            HfSource(
                "openai/openai_humaneval",
                "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                ("openai_humaneval/test-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            164,
            1024,
        ),
        python_source(
            "HumanEvalPlus",
            "humanevalplus",
            HfSource(
                "evalplus/humanevalplus",
                "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
                ("test.jsonl",),
                SourceFormat.JSONL,
            ),
            164,
            1024,
        ),
        python_source(
            "MBPPPlus",
            "mbppplus",
            HfSource(
                "evalplus/mbppplus",
                "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
                ("data/test-00000-of-00001-d5781c9c51e02795.parquet",),
                SourceFormat.PARQUET,
                read=read_mbpp,
            ),
            378,
            1024,
        ),
        python_source(
            "CruxEval",
            "cruxeval",
            HfSource(
                "cruxeval-org/cruxeval",
                "b96af0450242eb4da433032b90998f25588a5d0f",
                ("test.jsonl",),
                SourceFormat.JSONL,
                read=read_crux,
            ),
            1600,
            2048,
        ),
    ]
