# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline Hub metadata, submitter credentials and a fixture agent-episode corpus."""

import json
from types import SimpleNamespace

import pytest
from finestore.rl.status_taxonomy import DataRestriction
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from huggingface_hub import HfApi
from shellbox.machine import NetworkPolicy
from taskcompendium.convert.answers import answer_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import (
    AgentEpisode,
    EpisodeInputs,
    EpisodeLimits,
    EvaluationItem,
    EvaluationProtocol,
    Metric,
    MetricKind,
    OperationKind,
    Sampling,
    SummaryKind,
    TerminusAgent,
    TrialPolicy,
)
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, RawRow
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from verifyit.spec import ExactSpec
from zephyr.context import ZephyrContext

from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin

AGENT = AgentEpisode(
    harness=TerminusAgent(),
    limits=EpisodeLimits(
        max_turns=None,
        max_total_tokens=None,
        model_turn_timeout=600,
        total_turn_timeout=900,
        verifier_timeout=300,
        attempt_timeout=3600,
    ),
    max_input_tokens=24576,
)


@pytest.fixture(autouse=True)
def hub_metadata(tmp_path_factory, monkeypatch):
    config = tmp_path_factory.mktemp("hub-metadata") / "config.json"
    config.write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "max_position_embeddings": 1048576,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "num_hidden_layers": 24,
            }
        )
    )
    monkeypatch.setattr(HfApi, "model_info", lambda *_args, **_kwargs: SimpleNamespace(sha="f" * 40, siblings=()))
    monkeypatch.setattr("marin.evaluation.serving_config.hf_hub_download", lambda *_args, **_kwargs: str(config))


@pytest.fixture(autouse=True)
def coreweave_credentials(monkeypatch):
    """Stand in for the submitter's storage keys that CoreWeave launches require."""
    monkeypatch.setenv("CW_KEY_ID", "local-test-id")
    monkeypatch.setenv("CW_KEY_SECRET", "local-test-secret")


def convert_answer(row: RawRow, _context: ConversionContext) -> TaskSpec:
    task = answer_task(row, prompt=row.data["prompt"], spec=ExactSpec(expected=("red",)))
    return task.model_copy(update={"tags": ("eval",)})


def episode_item(task: TaskSpec, *, agent: AgentEpisode = AGENT) -> EvaluationItem:
    return chat_item(task).model_copy(update={"operation": agent, "episode": EpisodeInputs(network=NetworkPolicy.DENY)})


@pytest.fixture
def agent_source() -> RlDataSource[EvaluationCorpusRecipe]:
    """A one-row corpus that runs Terminus-2 and reports a primary reward and a solved rate."""
    return RlDataSource(
        info=SourceInfo(
            data_restriction=DataRestriction.OPEN,
            upstream_dataset="terminal-agent",
            upstream_name="terminal-agent",
            build_path="test",
            origin=SourceOrigin.MARIN,
            family="test",
            tags=(),
            count=1,
        ),
        config=EvaluationCorpusRecipe(
            CurationRecipe(
                name="terminal-agent",
                source=HfSource("test/terminal-agent", "0" * 40, ("rows.jsonl",), SourceFormat.JSONL),
                convert=convert_answer,
                version="1",
                intended_use=IntendedUse.EVAL,
            ),
            EvaluationProtocol(
                benchmark="terminal-agent",
                version="1",
                operation=OperationKind.AGENT_EPISODE,
                sampling=Sampling(
                    temperature=None, top_p=None, top_k=None, max_tokens=4096, stop=(), chat_template_kwargs={}
                ),
                trials=TrialPolicy(count=1, seed=None),
                metrics=(
                    Metric(name="reward", kind=MetricKind.CONTINUOUS, higher_is_better=True),
                    Metric(name="solved", kind=MetricKind.BINARY, higher_is_better=True),
                ),
                primary_metric="reward",
                summary=SummaryKind.MEAN,
                expected_population=1,
            ),
            episode_item,
            protocol_name="terminus-2",
            agent=AGENT,
        ),
        pipeline=prepare_evaluation,
    )


@pytest.fixture
def local_graph(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    runtime = CampaignRuntime()
    options = PipelineOptions(SourceProcessingMode.QUICK, runtime, inputs=InputOverrides(root=str(root)))
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            yield root, options
    finally:
        client.shutdown()
