# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GenomeQA publication preserves its fixed generation budget in native selection."""

import json
from dataclasses import replace

from fray.local_backend import LocalClient
from taskcompendium.evaluation.selection import select_cases
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.genomeqa import sources
from experiments.post_training.task_curation.pipeline import source_recipe


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return list(text.encode())

    def apply_chat_template(self, conversation, **kwargs):
        return list(json.dumps(conversation).encode())


def test_published_genomeqa_excludes_reduced_budget_without_truncating_prompt(tmp_path):
    dataset = sources()[0].config
    source = tmp_path / "source"
    source.mkdir()
    (source / "rows.jsonl").write_text(
        json.dumps(
            {
                "id": 0,
                "subtask": "6_motif",
                "format": "binary",
                "question": "Synthetic DNA: AcgT",
                "options": ["alpha", "beta"],
                "answer_index": 1,
                "answer_text": "beta",
            }
        )
        + "\n"
    )
    (source / "run.py").write_text('SIMPLE_SYSTEM_PROMPT = "Synthetic protocol prompt"\n')
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "0" * 64
    )
    recipe = replace(
        source_recipe(dataset.recipe, {"prompt": str(source)}, environment),
        source=SourceFiles("synthetic", "fixture", ("rows.jsonl",), SourceFormat.JSONL),
    )
    protocol = dataset.protocol.model_copy(update={"expected_population": 1})
    output = str(tmp_path / "published")
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, dataset.item_builder, context, str(source), output)
    finally:
        client.shutdown()
    for cap, context_window, reason in [
        (8192, 32768, "output_cap"),
        (16384, 16400, "context_window"),
        (16384, 32768, None),
    ]:
        [decision] = list(
            select_cases(
                output,
                protocol,
                dataset_id="genomeqa-fixture",
                context_window=context_window,
                max_tokens=cap,
                limit=None,
                tokenizer=CharacterTokenizer(),
            )
        )
        assert decision.record.reason == reason
        if reason is None:
            assert decision.record.max_tokens == 16384
            assert decision.case.task.context.events[-1].content.startswith("Question:\nSynthetic DNA: AcgT\n")
            assert decision.case.item.groups["cell"] == "6_motif/binary"
        else:
            assert decision.case is None
