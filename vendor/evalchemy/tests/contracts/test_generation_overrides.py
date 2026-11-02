"""Sampling overrides observed by a local model through the shared runtime."""

import asyncio
from argparse import Namespace
import json

import pytest
import yaml
from lm_eval.tasks import TaskManager

from eval.contracts.eval_spec import SourceRef
from eval.execution.backend import LMBackend
from eval.execution.run_spec import RunSpec
from eval.sources.common import SourceOptions, text_specs
from eval.sources.lm_eval import LMEvalSource


class _RecordingModel:
    world_size = 1

    def __init__(self):
        self.generation = []

    def apply_chat_template(self, messages):
        return messages[0]["content"]

    def generate_until(self, requests):
        self.generation.extend(request.args[1] for request in requests)
        return ["answer"] * len(requests)


@pytest.mark.parametrize(
    ("overrides", "expected_temperature"),
    [
        ({"temperature": 0.85, "do_sample": False}, 0.0),
        ({"temperature": 0.85, "do_sample": True}, 0.85),
        ({"temperature": 0.85}, 0.85),
        ({"temperature": 0.0}, 0.0),
    ],
)
def test_generation_overrides_respect_explicit_sampling_at_model_boundary(overrides, expected_temperature):
    [spec] = text_specs(
        [{"id": "question"}], SourceOptions("sampling", SourceRef("fixture"), generation=overrides),
        prompt=lambda row: row["id"], target=lambda _row: "answer", grader="exact_match",
        generation_kwargs={"do_sample": False, "temperature": 0.7},
    )
    model = _RecordingModel()

    async def run():
        async with LMBackend(model) as backend:
            await backend.execute(spec.requests[0])

    asyncio.run(run())

    assert model.generation == [{"temperature": expected_temperature}]


@pytest.mark.parametrize("alias", ["max_tokens", "max_new_tokens", "max_gen_toks", "max_completion_tokens"])
def test_native_import_canonicalizes_caps_before_applying_explicit_overrides(tmp_path, alias):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"question": "Question", "answer": "answer"}) + "\n")
    config = {
        "task": "cap_probe", "dataset_path": "json", "dataset_kwargs": {"data_files": {"test": str(data)}},
        "test_split": "test", "output_type": "generate_until", "doc_to_text": "{{question}}",
        "doc_to_target": "{{answer}}", "num_fewshot": 0, "metric_list": [{"metric": "exact_match"}],
        "generation_kwargs": {"max_new_tokens": 17, "until": ["END"]},
    }
    (tmp_path / "cap_probe.yaml").write_text(yaml.safe_dump(config))
    task = TaskManager(include_path=str(tmp_path)).load(["cap_probe"])["tasks"]["cap_probe"]
    run = RunSpec.from_args(Namespace(gen_kwargs=f"{alias}=29"))
    source = LMEvalSource("cap_probe", task, 1, 0, run.generation, 1234)
    [spec] = source
    model = _RecordingModel()

    async def execute():
        async with LMBackend(model) as backend:
            await backend.execute(spec.requests[0])

    asyncio.run(execute())

    assert spec.requests[0].generation == {"max_gen_toks": 29, "until": ["END"]}
    assert model.generation == [{"max_gen_toks": 29, "until": ["END"]}]
