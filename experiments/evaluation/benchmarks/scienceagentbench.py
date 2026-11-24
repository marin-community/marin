# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ScienceAgentBench Verified paired knowledge conditions and task 66 integration pilot."""

import ast
import json
import shlex
import zipfile
from functools import partial
from io import BytesIO
from pathlib import Path

from shellbox.machine import NetworkPolicy
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import AgentEpisode, EpisodeInputs, EpisodeLimits, EvaluationItem, PiAgent
from taskcompendium.importers.scienceagentbench import (
    ANNOTATION_FILE,
    ANNOTATION_REVISION,
    VERIFIED_SOURCE,
    KnowledgeCondition,
    VerifiedAnnotation,
    dataset_files,
    published_evaluator,
    read_verified_annotations,
    scienceagentbench_task,
)
from taskcompendium.importers.scienceagentbench_assets.visual_judge import MODEL, VENICE_MODEL
from taskcompendium.importers.scienceagentbench_grading import deterministic_grader, papyrus_grader, visual_grader
from taskcompendium.models import (
    CommandSemantics,
    DockerBuildContext,
    EnvironmentRequirements,
    TaskResource,
    TaskSpec,
)
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, RawRow
from taskcompendium.runtime.resources import inline_resource

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.harbor import harbor_protocol
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, RowSelector, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

NAME = "scienceagentbench-verified-66-knowledge-off"
PILOT_VERSION = "2026.11.24.1"
PANEL_VERSION = "2026.11.24.1"
UNRUN_TASK_IDS = frozenset({78})
IMAGES = Path(__file__).resolve().parents[1] / "images"
VISUAL = IMAGES / "scienceagentbench-visual"
PROMPT_FILE = "gpt4_visual_judge.py"
CARTOPY_DIRECTORY = "/setup_files/cartopy"
CARTOPY_FEATURE = {48: "land", 49: "coastline", 56: "coastline", 86: "coastline"}
BASEMAP_SOURCES = {
    "ne_110m_land": UrlSource(
        "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip",
        "1926c621afd6ac67c3f36639bb1236134a48d82226dc675d3e3df53d02d2a3de",
        "ne_110m_land.zip",
        SourceFormat.GENERATED,
    ),
    "ne_50m_land": UrlSource(
        "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_land.zip",
        "0b8e670cf80dce9cbebe2a193bc44ba5602758c22e1fa603980553646d7ff162",
        "ne_50m_land.zip",
        SourceFormat.GENERATED,
    ),
    "ne_110m_coastline": UrlSource(
        "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_coastline.zip",
        "664449b39070027e882abb295974d182afec18ca21107273d17e9e8bf6f64817",
        "ne_110m_coastline.zip",
        SourceFormat.GENERATED,
    ),
    "ne_50m_coastline": UrlSource(
        "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_coastline.zip",
        "640f805509b822f57f4840a2e18d9ff2412cf1cf6976124701c2789436166fde",
        "ne_50m_coastline.zip",
        SourceFormat.GENERATED,
    ),
}
GLACIER_FILES = json.loads((IMAGES / "scienceagentbench-pi/glacier-support.json").read_text())
GLACIER_SOURCES = {
    Path(entry["cache_path"]).name: UrlSource(
        entry["url"], entry["sha256"], Path(entry["cache_path"]).name, SourceFormat.GENERATED
    )
    for entry in GLACIER_FILES
}
OGGM_CONFIG = b"""dl_cache_dir = ~/OGGM/download_cache
dl_cache_readonly = False
tmp_dir = ~/OGGM/tmp
rgi_dir = ~/OGGM/rgi
test_dir = ~/OGGM/tests
has_internet = False
"""

AGENT = AgentEpisode(
    harness=PiAgent(version="0.87.0", setup_timeout=360),
    max_input_tokens=65536,
    limits=EpisodeLimits(
        max_turns=None,
        max_total_tokens=None,
        model_turn_timeout=600,
        total_turn_timeout=3600,
        verifier_timeout=1800,
        attempt_timeout=14400,
    ),
)


def pilot_row(data, _context: ConversionContext) -> bool:
    return data["instance_id"] == 66


def panel_row(data, _context: ConversionContext) -> bool:
    return data["instance_id"] not in UNRUN_TASK_IDS


def deterministic_row(data, context: ConversionContext) -> bool:
    return panel_row(data, context) and not published_evaluator(VerifiedAnnotation.model_validate(data)).visual


def visual_row(data, context: ConversionContext) -> bool:
    return panel_row(data, context) and published_evaluator(VerifiedAnnotation.model_validate(data)).visual


def software_image(directory: Path) -> DockerBuildContext:
    return DockerBuildContext(
        files=tuple(
            inline_resource(name, (directory / name).read_bytes()) for name in ("Dockerfile", "requirements.lock")
        )
    )


def published_judge_prompt(context: ConversionContext) -> str:
    source = (context.inputs["prompt"] / PROMPT_FILE).read_text()
    (prompt,) = [
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PROMPT_ORIGIN" for target in node.targets)
    ]
    return prompt


def cartopy_files(feature: str, context: ConversionContext) -> tuple[TaskResource, ...]:
    """Stage the public plotting library assets before the offline agent and replay run."""
    files = []
    for scale in ("110m", "50m"):
        name = f"ne_{scale}_{feature}"
        source = BASEMAP_SOURCES[name]
        data = (context.inputs[name] / source.filename).read_bytes()
        with zipfile.ZipFile(BytesIO(data)) as archive:
            for extension in ("shp", "shx", "dbf", "prj", "cpg"):
                filename = f"{name}.{extension}"
                target = f"{CARTOPY_DIRECTORY.lstrip('/')}/shapefiles/natural_earth/physical/{filename}"
                files.append(inline_resource(target, archive.read(filename)))
    return tuple(files)


def convert(
    row: RawRow, context: ConversionContext, *, condition: KnowledgeCondition, judge_model: str = MODEL
) -> TaskSpec:
    annotation = VerifiedAnnotation.model_validate(row.data)
    published = published_evaluator(annotation)
    inputs = dataset_files(annotation)
    if annotation.instance_id in (2, 101, 102):
        actor_dir = IMAGES / "scienceagentbench-materials"
    elif annotation.instance_id in (12, 13, 15, 17, 22):
        actor_dir = IMAGES / "scienceagentbench-molecules"
    else:
        actor_dir = IMAGES / "scienceagentbench-pi"
    actor = EnvironmentRequirements(
        docker_build=software_image(actor_dir),
        command_semantics=CommandSemantics.LINUX_PROCESS,
        working_directory="/testbed",
        setup_commands=tuple(
            f"printf '%s\\n' {shlex.quote(f'{file.sha256}  /testbed/{file.member}')} | sha256sum -c -" for file in inputs
        ),
    )
    worker = tuple(
        TaskResource(path=file.member.replace("benchmark/", "setup_files/", 1), source=file) for file in inputs
    )
    if annotation.instance_id == 26:
        actor = actor.model_copy(update={"environment_variables": {"PYTHONPATH": "/opt/prolif-1.1.0"}})
    if feature := CARTOPY_FEATURE.get(annotation.instance_id):
        actor = actor.model_copy(update={"environment_variables": {"CARTOPY_DATA_DIR": CARTOPY_DIRECTORY}})
        worker += cartopy_files(feature, context)
    if annotation.instance_id in (64, 65, 74):
        for entry in GLACIER_FILES:
            name = Path(entry["cache_path"]).name
            worker += (
                inline_resource(
                    "setup_files/glacier/" + entry["cache_path"], (context.inputs[name] / name).read_bytes()
                ),
            )
        worker += (inline_resource("setup_files/glacier/.oggm_config", OGGM_CONFIG),)
        actor = actor.model_copy(update={"setup_commands": (*actor.setup_commands, "cp -R /setup_files/glacier/. ~/")})
    private_files = published.resources()
    loader = b"import runpy\neval = runpy.run_path('/tests/private_evaluator.py')['eval']\n"
    if published.visual:
        grader = EnvironmentRequirements(
            docker_build=software_image(VISUAL),
            capabilities=("network",),
            working_directory="/testbed",
            command_semantics=CommandSemantics.LINUX_PROCESS,
        )
        package = visual_grader(
            annotation,
            judge_prompt=published_judge_prompt(context),
            judge_model=judge_model,
            evaluator_program=b"import gpt4_visual_judge\n" + loader,
            environment=grader,
            replay_environment=actor,
            replay_timeout=900,
            verifier_timeout=1800,
        )
    else:
        grader = actor.model_copy(
            update={
                "setup_commands": (
                    "mkdir -p /testbed/benchmark/eval_programs && "
                    "ln -s /tests/gold_results /testbed/benchmark/eval_programs/gold_results",
                )
            }
        )
        if annotation.instance_id == 22:
            _, gold = private_files
            package = papyrus_grader(
                environment=grader, replay_environment=actor, gold_rows=gold, replay_timeout=900, verifier_timeout=1800
            )
            private_files = ()
        else:
            package = deterministic_grader(
                annotation,
                environment=grader,
                replay_environment=actor,
                evaluator_program=loader,
                replay_timeout=900,
                verifier_timeout=1800,
            )
    task = scienceagentbench_task(
        annotation,
        condition,
        environment=actor,
        package=package,
        reference_program=None,
        importer_revision=row.source.importer_revision,
    )
    return task.model_copy(
        update={
            "id": row.id,
            "source": row.source,
            "tags": ("eval",),
            "labels": {
                **task.labels,
                "instance_id": str(annotation.instance_id),
                "max_judge_calls": str(3 * len(published.gold) if published.visual else 0),
            },
            "resources": task.resources.model_copy(
                update={"worker": worker, "verifier": task.resources.verifier + private_files}
            ),
        }
    )


def item(task: TaskSpec) -> EvaluationItem:
    return chat_item(task).model_copy(
        update={
            "operation": AGENT,
            "episode": EpisodeInputs(
                network=NetworkPolicy.DENY,
                cpus=2,
                memory_mb=8192,
                storage_mb=10240,
                agent_timeout=3600,
                verifier_timeout=1800,
            ),
        }
    )


def benchmark_source(
    name: str, condition: KnowledgeCondition, count: int, *, select: RowSelector = panel_row, judge_model: str = MODEL
) -> RlDataSource[EvaluationCorpusRecipe]:
    version = PILOT_VERSION if name == NAME else PANEL_VERSION
    recipe = CurationRecipe(
        name=name,
        version=version,
        intended_use=IntendedUse.EVAL,
        convert=partial(convert, condition=condition, judge_model=judge_model),
        # The largest task references 910,657,346 bytes before its software and verifier helpers.
        resource_budget_bytes=1024**3,
        source=HfSource(
            VERIFIED_SOURCE.dataset,
            ANNOTATION_REVISION,
            (ANNOTATION_FILE,),
            SourceFormat.PARQUET,
            read=read_verified_annotations,
            select=select,
        ),
        inputs={
            **({**BASEMAP_SOURCES, **GLACIER_SOURCES} if name != NAME else {}),
            "prompt": UrlSource(
                "https://raw.githubusercontent.com/OSU-NLP-Group/ScienceAgentBench/"
                "c26e151ed601ba109dc4d35e057ff8e73fec469d/" + PROMPT_FILE,
                "e9a95f36638485c6e478c7412b14e8d9b94e0f0055a8db57ff71be6092a3d61a",
                PROMPT_FILE,
                SourceFormat.GENERATED,
            ),
        },
    )
    protocol = harbor_protocol(
        name, version, count, max_output_tokens=16384, requires_judge=select is not deterministic_row, trials=1
    )
    return RlDataSource(
        info=SourceInfo(
            id=name,
            title=f"ScienceAgentBench Verified {condition.value}" + (" task 66" if name == NAME else ""),
            origin="ScienceAgentBench",
            family="scientific-programming",
            tags=("eval", "agent"),
            count=count,
        ),
        config=EvaluationCorpusRecipe(recipe, protocol, item, protocol_name="pi", agent=AGENT),
        pipeline=prepare_evaluation,
    )


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(
        [
            benchmark_source(NAME, KnowledgeCondition.OFF, 1, select=pilot_row),
            *(
                benchmark_source(f"scienceagentbench-verified-{condition.value}", condition, 102 - len(UNRUN_TASK_IDS))
                for condition in KnowledgeCondition
            ),
            *(
                benchmark_source(
                    f"scienceagentbench-verified-deterministic-{condition.value}",
                    condition,
                    37,
                    select=deterministic_row,
                )
                for condition in KnowledgeCondition
            ),
            *(
                benchmark_source(
                    f"scienceagentbench-verified-visual-{condition.value}",
                    condition,
                    64,
                    select=visual_row,
                    judge_model=VENICE_MODEL,
                )
                for condition in KnowledgeCondition
            ),
        ]
    )
