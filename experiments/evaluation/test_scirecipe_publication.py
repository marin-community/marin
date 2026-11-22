# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.importers.evalchemy.scirecipe import SYSTEM_PROMPT, scirecipe_prompt
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.runtime.resources import resource_bytes
from zephyr.context import ZephyrContext

from experiments.evaluation.benchmarks.scirecipe import benchmarks
from experiments.post_training.task_curation.pipeline import source_recipe


def test_scirecipe_publication_preserves_prompts_private_references_and_groups(tmp_path):
    [source] = benchmarks()[0].resolve()
    dataset = source.config
    document = {
        "exp_goal": "Prepare a sample.",
        "action": ["mix", "wait"],
        "key": 'Step 1: {"action": "mix", "objects": ["water"], "parameters": []}',
        "orc": "Mix water.",
        "think": "Private reasoning",
        "note": "Private note",
        "type": "Planning",
        "level": "level2",
    }
    source_root = tmp_path / "source"
    source_root.mkdir()
    pq.write_table(
        pa.Table.from_pylist([document, {**document, "type": "Scaling"}]), source_root / dataset.recipe.files[0]
    )
    environment = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="fixture.lock")
    recipe = source_recipe(dataset.recipe, {}, environment)
    protocol = dataset.protocol.model_copy(update={"expected_population": 2})
    output = tmp_path / "output"
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
        ):
            publish_evaluation_corpus(recipe, protocol, dataset.item_builder, context, str(source_root), str(output))
    finally:
        client.shutdown()
    cases = sorted(evaluation_cases(str(output)), key=lambda case: case.item.source_ordinal)
    assert [case.item.groups["type"] for case in cases] == ["Planning", "Scaling"]
    for case in cases:
        assert [event.content for event in case.task.context.events] == [SYSTEM_PROMPT, scirecipe_prompt(document)]
        config = json.loads(resource_bytes(next(r for r in case.task.resources.verifier if r.path == "config.json")))
        assert config == {"key": document["key"], "orc": document["orc"]}
        assert case.item.operation.response_selection == "final_only"
