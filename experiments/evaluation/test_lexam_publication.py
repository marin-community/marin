# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LEXam publication keeps English rows, renders the authors' prompt and binds gold letters."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
from fray.local_backend import LocalClient
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.runtime.resources import resource_bytes
from zephyr.context import ZephyrContext

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.post_training.task_curation.pipeline import source_recipe


def _row(index: int, language: str, gold: int) -> dict:
    return {
        "question": f"Frage {index}?",
        "choices": "['eins', 'zwei', 'drei', 'vier']",
        "gold": gold,
        "course": "Strafrecht",
        "language": language,
        "area": "Criminal",
        "jurisdiction": "Swiss",
        "year": 2020,
        "n_statements": 4,
        "none_as_an_option": False,
        "id": f"lexam-{index}",
        "negative_question": False,
    }


def test_publication_renders_english_lexam_prompts(tmp_path):
    [source] = evaluation_benchmarks()["lexam-mcq-4-en"].resolve()
    dataset = source.config
    rows = [_row(0, "en", 2), _row(1, "de", 1), _row(2, "en", 0)]
    source_root = tmp_path / "source"
    primary = source_root / dataset.recipe.files[0]
    primary.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), primary)
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "0" * 64
    )
    recipe = source_recipe(dataset.recipe, {}, environment)
    protocol = dataset.protocol.model_copy(update={"expected_population": 2})
    output = tmp_path / "output"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, dataset.item_builder, context, str(source_root), str(output))
    finally:
        client.shutdown()
    cases = sorted(evaluation_cases(str(output)), key=lambda case: case.item.source_ordinal)
    [event] = cases[0].task.context.events
    assert event.role == "user"
    assert event.content.startswith("You are an expert in Strafrecht and address legal issues")
    assert event.content.endswith("Question:\n Frage 0?\nA. eins\nB. zwei\nC. drei\nD. vier\n\nAnswer:")
    configs = [
        json.loads(resource_bytes(next(r for r in case.task.resources.verifier if r.path == "config.json")))
        for case in cases
    ]
    assert configs == [{"expected": "C"}, {"expected": "A"}]
