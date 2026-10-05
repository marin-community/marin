# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind pinned TaskTrove archive components to reusable family policies."""

from taskcompendium.pipeline.datasets import (
    atlas_arc_injection,
    atlas_code,
    atlas_math_qa,
    calendar_tasks,
    competitive_coding,
    executable_tasks,
    if_calendar,
    instruction_following,
    multichallenge,
    python_tasks,
    qa_tasks,
    reasoning_tasks,
    repository_tasks,
    rubric_tasks,
    structured_output,
    tasktrove_math,
)
from taskcompendium.pipeline.datasets.nemotron import structured_outputs
from taskcompendium.pipeline.datasets.raw_conversion import RawConverter
from taskcompendium.pipeline.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.inputs import hub_inputs
from taskcompendium.pipeline.models import DatasetRecipe, HFSource, IntendedUse, TaskPipeline

TASKTROVE_DATASET = "open-thoughts/TaskTrove"
TASKTROVE_REVISION = "02923004846e4e73862c20962f823a6d05100e7a"


def _recipe(name: str, version: str, config: str, pipeline: TaskPipeline) -> DatasetRecipe:
    return DatasetRecipe(
        name=name,
        version=version,
        source=HFSource(TASKTROVE_DATASET, TASKTROVE_REVISION, config, "train"),
        pipeline=pipeline,
        intended_use=IntendedUse.TRAIN,
        inputs=hub_inputs(TASKTROVE_DATASET, TASKTROVE_REVISION, tasktrove_files(config)),
    )


RECIPES: dict[str, DatasetRecipe] = {
    "advanced_calculations": _recipe(
        "tasktrove-advanced_calculations",
        "tasktrove-advanced_calculations-v1",
        "laion__nemotron-gym-math-advanced-calculations-v4",
        atlas_math_qa.pipeline("advanced_calculations"),
    ),
    "all_puzzles": _recipe(
        "tasktrove-puzzles",
        "tasktrove-puzzles-v1",
        "laion__all-puzzles-v2",
        reasoning_tasks.puzzle_pipeline(),
    ),
    "arc_inductive": _recipe(
        "tasktrove-arc_inductive",
        "tasktrove-arc_inductive-v1",
        "laion__nemotron-gym-arc-agi-python-inductive-v2",
        atlas_arc_injection.pipeline("arc_inductive"),
    ),
    "arc_transductive": _recipe(
        "tasktrove-arc_transductive",
        "tasktrove-arc_transductive-v1",
        "laion__nemotron-gym-arc-agi-transductive-v3",
        atlas_arc_injection.pipeline("arc_transductive"),
    ),
    "calendar": _recipe(
        "tasktrove-calendar",
        "tasktrove-calendar-v1",
        "laion__nemotron-gym-agent-calendar-v2",
        calendar_tasks.pipeline(),
    ),
    "codereview": _recipe(
        "tasktrove-codereview",
        "tasktrove-codereview-v1",
        "laion__stackexchange-codereview-sandboxes-verified-v2",
        rubric_tasks.pipeline(rubric_tasks.RUBRICS["codereview"]),
    ),
    "glaive_code": _recipe(
        "tasktrove-glaive_code",
        "tasktrove-glaive_code-v1",
        "laion__glaive-code-assistant-sandboxes-verified-v2",
        rubric_tasks.pipeline(rubric_tasks.RUBRICS["glaive_code"]),
    ),
    "if_calendar": _recipe(
        "tasktrove-if_calendar",
        "tasktrove-if_calendar-v1",
        "laion__nemotron-gym-instruction-following-calendar-v3",
        if_calendar.pipeline(),
    ),
    "indirect_injection": _recipe(
        "tasktrove-indirect_injection",
        "tasktrove-indirect_injection-v1",
        "laion__nemotron-gym-agentic-indirect-prompt-injection-v3",
        atlas_arc_injection.pipeline("indirect_injection"),
    ),
    "instruction_following": _recipe(
        "tasktrove-ifeval",
        "tasktrove-ifeval-v1",
        "laion__nemotron-gym-instruction-following-v3",
        instruction_following.pipeline(),
    ),
    "knowledge_mcqa": _recipe(
        "tasktrove-knowledge_mcqa",
        "tasktrove-knowledge_mcqa-v1",
        "laion__nemotron-gym-knowledge-mcqa-v2",
        atlas_math_qa.pipeline("knowledge_mcqa"),
    ),
    "knowledge_openqa": _recipe(
        "knowledge-openqa",
        "knowledge-openqa-v1",
        "laion__nemotron-gym-knowledge-openqa-v4",
        qa_tasks.pipeline(),
    ),
    "math_gym": _recipe(
        "tasktrove-math_gym",
        "tasktrove-math_gym-v1",
        "laion__nemotron-gym-math-v5",
        tasktrove_math.pipeline("math_gym"),
    ),
    "math_openreasoning": _recipe(
        "tasktrove-math_openreasoning",
        "tasktrove-math_openreasoning-v1",
        "laion__nemotron-gym-math-openmathreasoning-v2",
        atlas_math_qa.pipeline("math_openreasoning"),
    ),
    "math_oracle": _recipe(
        "tasktrove-math_oracle",
        "tasktrove-math_oracle-v1",
        "SankalpKJ__nemotron-math-oracle-filtered-v2",
        tasktrove_math.pipeline("math_oracle"),
    ),
    "math_prism": _recipe(
        "tasktrove-math_prism",
        "tasktrove-math_prism-v1",
        "laion__nemo-prism-math-v3",
        tasktrove_math.pipeline("math_prism"),
    ),
    "math_stack": _recipe(
        "tasktrove-math_stack",
        "tasktrove-math_stack-v1",
        "laion__nemotron-gym-math-stack-overflow-v3",
        tasktrove_math.pipeline("math_stack"),
    ),
    "multichallenge": _recipe(
        "tasktrove-multichallenge",
        "tasktrove-multichallenge-v1",
        "laion__nemotron-gym-multichallenge-advanced-v4",
        multichallenge.pipeline(),
    ),
    "qa_abstention": _recipe(
        "tasktrove-qa_abstention",
        "tasktrove-qa_abstention-v1",
        "laion__nemotron-gym-qa-abstention-v4",
        atlas_math_qa.pipeline("qa_abstention"),
    ),
    "reasoning_gym": _recipe(
        "tasktrove-reasoning-gym",
        "tasktrove-reasoning-gym-v1",
        "laion__nemotron-gym-reasoning-gym-v2",
        reasoning_tasks.reasoning_pipeline(),
    ),
    "safety": _recipe(
        "tasktrove-safety",
        "tasktrove-safety-v1",
        "laion__nemotron-gym-safety-v3",
        rubric_tasks.pipeline(rubric_tasks.RUBRICS["safety"]),
    ),
    "science_openqa": _recipe(
        "science-openqa",
        "science-openqa-v1",
        "laion__nemotron-gym-science-so-openq-v3",
        qa_tasks.pipeline(),
    ),
    "stack_overflow": _recipe(
        "tasktrove-stack_overflow",
        "tasktrove-stack_overflow-v1",
        "laion__stackexchange-overflow-sandboxes-verified-v2",
        rubric_tasks.pipeline(rubric_tasks.RUBRICS["stack_overflow"]),
    ),
    "structured_output": _recipe(
        "tasktrove-structured",
        "tasktrove-structured-v1",
        "laion__nemotron-gym-instruction-following-structured-v3",
        structured_output.pipeline(),
    ),
    "superuser": _recipe(
        "tasktrove-superuser",
        "tasktrove-superuser-v1",
        "laion__stackexchange-superuser-sandboxes-verified-v2",
        rubric_tasks.pipeline(rubric_tasks.RUBRICS["superuser"]),
    ),
    "swe_rebench": _recipe(
        "tasktrove-swe_rebench",
        "tasktrove-swe_rebench-v1",
        "DCAgent__swe_rebench_v2_patched_oracle-v2",
        repository_tasks.pipeline("swe_rebench"),
    ),
    "swesmith": _recipe(
        "tasktrove-swesmith",
        "tasktrove-swesmith-v1",
        "laion__swesmith-oracle-filtered-v2",
        repository_tasks.pipeline("swesmith"),
    ),
    "tezos": _recipe(
        "tasktrove-tezos",
        "tasktrove-tezos-v1",
        "laion__stackexchange-tezos-sandboxes-verified-v2",
        rubric_tasks.pipeline(rubric_tasks.RUBRICS["tezos"]),
    ),
    "unix": _recipe(
        "tasktrove-unix",
        "tasktrove-unix-v1",
        "laion__stackexchange-unix-sandboxes-verified-v2",
        rubric_tasks.pipeline(rubric_tasks.RUBRICS["unix"]),
    ),
    "web_search_mcqa": _recipe(
        "tasktrove-web_search_mcqa",
        "tasktrove-web_search_mcqa-v1",
        "laion__nemotron-gym-knowledge-web-search-mcqa-v2",
        atlas_math_qa.pipeline("web_search_mcqa"),
    ),
    "wizard_orca": _recipe(
        "tasktrove-wizard_orca",
        "tasktrove-wizard_orca-v1",
        "laion__wizardlm-orca-v4",
        rubric_tasks.pipeline(rubric_tasks.RUBRICS["wizard_orca"]),
    ),
}

EXECUTABLE_SOURCES = {
    "code_contests": "DCAgent__code-contests-noblock",
    "codeforces": "laion__codeforces-v3",
    "codenet": "laion__exp_rpt_codenet-python-v4",
    "competitive_coding": "laion__nemotron-gym-competitive-coding-v2",
    "curriculum_easy": "DCAgent__exp_rpt_curriculum-easy",
    "curriculum_medium": "DCAgent__exp_rpt_curriculum-medium-v2",
    "e2egit": "DCAgent__exp_rpt_e2egit-v2",
    "e2egit_large": "DCAgent__exp_rpt_e2egit-large",
    "multifile": "DCAgent__exp_rpt_multifile-v3",
    "nl2bash": "DCAgent2__nl2bash-tasks-cleaned-oracle-v2",
    "pymethods": "DCAgent__exp_rpt_pymethods2test-v3",
    "pymethods_large": "DCAgent__exp_rpt_pymethods2test-large-v2",
    "stack_pytest": "DCAgent__exp_rpt_stack-pytest-v2",
    "taco": "laion__exp_rpt_taco-v2",
    "unitsyn": "DCAgent__exp_rpt_unitsyn-python-v4",
    "unitsyn_large": "DCAgent__exp_rpt_unitsyn-python-large-v2",
}


def executable_recipe(
    name: str,
    image: str,
    *,
    converter: RawConverter,
    converter_revision: str,
    timeout: float,
    memory_mb: int,
) -> DatasetRecipe:
    """Bind an immutable grader image and converter to one archive component."""
    if name in atlas_code.RUBRICS:
        policy = atlas_code.pipeline(
            name, image, converter=converter, converter_revision=converter_revision, timeout=timeout, memory_mb=memory_mb
        )
    elif name in python_tasks.RUBRICS:
        policy = python_tasks.pipeline(
            image,
            rubric=python_tasks.RUBRICS[name],
            converter=converter,
            converter_revision=converter_revision,
            timeout=timeout,
            memory_mb=memory_mb,
        )
    elif name == "competitive_coding":
        policy = competitive_coding.pipeline(
            image, converter=converter, converter_revision=converter_revision, timeout=timeout, memory_mb=memory_mb
        )
    else:
        policy = executable_tasks.pipeline(
            name, image, converter=converter, converter_revision=converter_revision, timeout=timeout, memory_mb=memory_mb
        )
    recipe_name = f"tasktrove-{name}"
    return _recipe(recipe_name, recipe_name + "-v1-raw-conversion-v2", EXECUTABLE_SOURCES[name], policy)


def structured_outputs_recipe(converter: RawConverter, converter_revision: str) -> DatasetRecipe:
    """Bind the legacy structured-output converter to its pinned archive."""
    return _recipe(
        "tasktrove-structured_outputs",
        "tasktrove-structured_outputs-v2-raw-conversion-v2",
        "laion__nemotron-gym-structured-outputs-v4",
        structured_outputs.pipeline(converter=converter, converter_revision=converter_revision),
    )
