# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish source instruction constraints without exposing private grader inputs."""

import json
from functools import partial

import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from shellbox.machine import PackageSpec
from zephyr.context import ZephyrContext

from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.evaluation.spec import EvaluationProtocol, Metrics, OperationKind, Sampling, SummaryKind, TrialPolicy
from taskcompendium.importers.evalchemy.instructions import convert_instructions, instruction_item
from taskcompendium.importers.evalchemy.scorers.instruction_spec import InstructionProfile
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TextMessage
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, SourceRecipe
from taskcompendium.runtime.resources import resource_bytes


@pytest.mark.parametrize("profile", list(InstructionProfile))
def test_instruction_publication_preserves_prompt_channel_and_private_constraints(profile, tmp_path):
    prompt = "Respond with two bullet points."
    row = {
        "prompt": prompt,
        "instruction_id_list": ["detectable_format:number_bullet_lists"],
        "kwargs": [{"num_bullets": 2, "unused_checker_parameter": None}],
    }
    root = tmp_path / "source"
    root.mkdir()
    (root / "input.jsonl").write_text(json.dumps(row) + "\n")
    recipe = SourceRecipe(
        name="instruction-publication",
        version="fixture",
        source=SourceFiles("instruction-fixture", "fixture", ("input.jsonl",), SourceFormat.JSONL),
        convert=partial(convert_instructions, profile=profile),
        intended_use=IntendedUse.EVAL,
        rubric=None,
        controls=None,
        grader_environment=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            package_spec=PackageSpec(pypi_packages=("absl-py==2.3.1",)),
        ),
    )
    operation = OperationKind.TEXT_GENERATION if profile is InstructionProfile.HARNESS else OperationKind.CHAT_GENERATION
    protocol = EvaluationProtocol(
        benchmark="instructions",
        version="fixture",
        operation=operation,
        sampling=Sampling(temperature=0, top_p=None, top_k=None, max_tokens=128, stop=(), chat_template_kwargs={}),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metrics.ACCURACY,),
        primary_metric="accuracy",
        summary=SummaryKind.WEIGHTED_MEAN,
        expected_population=1,
    )
    output = str(tmp_path / "published")
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
        ):
            publish_evaluation_corpus(
                recipe, protocol, partial(instruction_item, profile=profile), context, str(root), output
            )
        (case,) = list(evaluation_cases(output))
    finally:
        client.shutdown()
    assert case.task.context.events == (TextMessage(role="user", content=prompt),)
    assert case.item.operation.kind == operation
    assert not case.task.resources.worker
    private = {resource.path: resource_bytes(resource) for resource in case.task.resources.verifier}
    assert json.loads(private["config.json"]) == {
        "constraints": [{"name": "detectable_format:number_bullet_lists", "params": {"num_bullets": 2}}],
        "profile": profile.value,
        "prompt": prompt,
    }
    assert any(
        path.startswith("source_grader/" + ("ifeval" if profile is InstructionProfile.HARNESS else "ifbench") + "/")
        for path in private
    )
