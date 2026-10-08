# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common source declaration and runtime binding helpers."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

from shellbox.machine import MachineFactory, MachineSpec
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import AssistantToolCalls, ConversationTrace, GradingAttempt, TaskSpec, TextMessage
from taskcompendium.pipeline.execution_binding import ExecutionAdapter, bind_executable
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, hub_inputs
from taskcompendium.pipeline.models import DatasetRecipe, HFSource, IntendedUse, TaskPolicy
from taskcompendium.pipeline.recipes import hf_recipe
from taskcompendium.runtime.task_grading import sandbox_grade

from experiments.post_training.task_curation.pipeline import RlDataPipeline, SourceRuntimeConfig


async def grade_final_message(
    task: TaskSpec,
    message: TextMessage | AssistantToolCalls,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    timeout: float,
) -> GradeResult:
    """Grade a control response in a fresh machine."""
    attempt = GradingAttempt(ConversationTrace(events=(*task.context.events, message)))
    return await sandbox_grade(task, attempt, factory, machine_spec, timeout=timeout)


def _fixed_recipe(recipe: DatasetRecipe, _runtime: SourceRuntimeConfig) -> DatasetRecipe:
    return recipe


def hf_pipeline(
    *,
    source_key: str,
    runtime_binding: Callable[..., DatasetRecipe] | None = None,
    name: str,
    version: str,
    hf_id: str,
    revision: str,
    config: str,
    split: str,
    files: SourceFiles | None = None,
    policy: TaskPolicy,
    intended_use: IntendedUse,
    inputs: RecipeInputs | Callable[[str, str], RecipeInputs] | None = None,
) -> RlDataPipeline:
    """Declare one Hub source and optionally bind its scorer to a selected runtime.

    The policy converts and checks rows before a grader image is selected.
    ``runtime_binding`` adds the executable scorer only when that source has an
    explicit image in the campaign manifest.
    """
    if inputs is None:
        assert files is not None
        inputs = hub_inputs(hf_id, revision, files)
    else:
        assert files is None
        if callable(inputs):
            inputs = inputs(hf_id, revision)
    recipe = hf_recipe(
        name=name,
        version=version,
        hf_id=hf_id,
        revision=revision,
        config=config,
        split=split,
        policy=policy,
        intended_use=intended_use,
        inputs=inputs,
    )
    builder = (
        partial(_fixed_recipe, recipe)
        if runtime_binding is None
        else partial(_optional_verification_recipe, recipe, runtime_binding, source_key)
    )
    return RlDataPipeline(source_key, recipe.source, intended_use, None, builder)


def _optional_verification_recipe(
    recipe: DatasetRecipe, binder: Callable[..., DatasetRecipe], name: str, runtime: SourceRuntimeConfig
) -> DatasetRecipe:
    if name not in runtime.images:
        # Campaign credentials do not select a backend for this source. Keep its
        # unbound grader; credentials reach a binder only with an explicit runtime.
        return recipe
    return _runtime_recipe(partial(binder, recipe), name, runtime)


def _runtime_recipe(builder: Callable[..., DatasetRecipe], name: str, runtime: SourceRuntimeConfig) -> DatasetRecipe:
    environment = runtime.images[name]
    return builder(
        image=environment.image,
        verification_runtime=environment.backend,
        controller_url=runtime.controller_url,
        qemu_bundle=Path(environment.qemu_bundle) if environment.qemu_bundle else None,
        worker_image=environment.worker_image,
        verifier_secret_env=runtime.verifier_secret_env or None,
    )


def executable_pipeline(
    *,
    source_key: str,
    name: str,
    version: str,
    hf_id: str,
    revision: str,
    config: str,
    split: str,
    files: SourceFiles,
    adapter: ExecutionAdapter,
    intended_use: IntendedUse,
) -> RlDataPipeline:
    """Declare a source whose converter needs an explicitly selected execution image."""
    builder = partial(
        bind_executable,
        name=name,
        version=version,
        hf_id=hf_id,
        revision=revision,
        config=config,
        split=split,
        files=files,
        adapter=adapter,
    )
    return RlDataPipeline(
        source_key,
        HFSource(hf_id, revision, config, split),
        intended_use,
        None,
        partial(_runtime_recipe, builder, source_key),
    )
