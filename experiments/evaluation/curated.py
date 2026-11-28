# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve benchmark selections to immutable corpus dependencies and run settings."""

from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from evalrecords.records import (
    DEFAULT_EVALUATION_CONCURRENCY,
    CuratedAgentRef,
    CuratedRef,
    CuratedSettings,
    EvalRef,
    EvaluationSelection,
    PromptFormat,
    ResolvedEvaluationSelection,
    TaskEvidence,
    TaskImageBuild,
    TaskMachineBackend,
    TaskMachineSettings,
    scoring_settings,
)
from huggingface_hub import HfApi
from marin.evaluation.dataset_evaluator import pi_thinking_format, run_max_tokens
from marin.evaluation.model_config import ModelConfig
from marin.evaluation.task_worker import EPISODE_HARNESS_REVISIONS
from marin.execution.lazy import ArtifactStep
from rigging.secrets import is_secret_reference
from taskcompendium.evaluation.spec import OperationKind, PiAgent
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.source_processing import SourceProcessingMode

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.source import RlDataSource

# The standard flow seeds every run alike, so runs of different models draw the same samples.
DEFAULT_RUN_SEED = 0
EVALUATIONS_KEY = "evaluations"


def pinned_model(model: ModelConfig) -> ModelConfig:
    """Freeze mutable Hub refs before they reach serving, tokenization or run identity."""
    api = HfApi()
    revision = model.revision
    if "://" not in model.location and not Path(model.location).is_dir():
        revision = api.model_info(model.location, revision=revision).sha
    tokenizer = model.tokenizer or model.location
    tokenizer_revision = (
        model.effective_tokenizer_revision
        or canonical_sha256({"tokenizer": tokenizer, "model_identity": model.identity})
        if "://" in tokenizer or Path(tokenizer).is_dir()
        else api.model_info(tokenizer, revision=model.effective_tokenizer_revision).sha
    )
    if revision is None and model.identity is None and "://" not in model.location:
        raise ValueError(f"No immutable model identity for {model.location}")
    return replace(model, revision=revision, tokenizer=tokenizer, tokenizer_revision=tokenizer_revision)


def load_evaluations(path: Path) -> tuple[EvaluationSelection, ...]:
    document = yaml.safe_load(path.read_text())
    entries = document[EVALUATIONS_KEY]
    return tuple(EvaluationSelection.model_validate({"label": name, **entry}) for name, entry in entries.items())


def selection_dataset(selection: EvaluationSelection, model_format: PromptFormat | None) -> EvaluationCorpusRecipe:
    """The catalog dataset, with its protocol, that a selection runs as for this model format."""
    catalog = evaluation_benchmarks()
    if selection.benchmark not in catalog:
        raise ValueError(f"Unknown evaluation benchmark: {selection.benchmark}")
    sources = catalog[selection.benchmark].resolve(
        selection.protocol, selection.num_fewshot, selection.prompt_format, model_format=model_format
    )
    dataset = sources[0].config
    assert dataset is not None
    return dataset


def resolve_selection(selection: EvaluationSelection, model_format: PromptFormat | None) -> ResolvedEvaluationSelection:
    dataset = selection_dataset(selection, model_format)
    form = {"chat_generation": PromptFormat.CHAT, "text_generation": PromptFormat.COMPLETION}.get(
        dataset.protocol.operation
    )
    return ResolvedEvaluationSelection(
        benchmark=selection.benchmark,
        protocol=dataset.protocol_name,
        num_fewshot=dataset.protocol.num_fewshot,
        prompt_format=form,
        partitions=selection.partitions,
    )


def selected_sources(selection: ResolvedEvaluationSelection) -> tuple[RlDataSource[EvaluationCorpusRecipe], ...]:
    benchmark = evaluation_benchmarks()[selection.benchmark]
    sources = benchmark.resolve(
        selection.protocol,
        selection.num_fewshot,
        PromptFormat(selection.prompt_format) if selection.prompt_format is not None else None,
    )
    if selection.partitions:
        available = {source.config.partition for source in sources if source.config is not None}
        unknown = set(selection.partitions) - available
        if unknown:
            raise ValueError(f"Unknown {selection.benchmark} partitions: {sorted(unknown)}")
        sources = tuple(
            source for source in sources if source.config is not None and source.config.partition in selection.partitions
        )
    return sources


def corpus_steps(selection: ResolvedEvaluationSelection, runtime: CampaignRuntime) -> tuple[ArtifactStep, ...]:
    options = PipelineOptions(SourceProcessingMode.QUICK, runtime)
    return tuple(prepare_evaluation(source, options) for source in selected_sources(selection))


def requires_judge(selection: EvaluationSelection) -> bool:
    return any(
        source.config is not None and source.config.protocol.requires_judge
        for source in evaluation_benchmarks()[selection.benchmark].resolve(selection.protocol, selection.num_fewshot)
    )


def declared_task_evidence(selection: ResolvedEvaluationSelection) -> TaskEvidence:
    """Reference when any selected corpus declares it, else embedded."""
    if any(
        source.config is not None and source.config.task_evidence is TaskEvidence.REFERENCE
        for source in selected_sources(selection)
    ):
        return TaskEvidence.REFERENCE
    return TaskEvidence.EMBEDDED


def runs_agent(selection: ResolvedEvaluationSelection) -> bool:
    return any(
        source.config is not None and source.config.protocol.operation == OperationKind.AGENT_EPISODE
        for source in selected_sources(selection)
    )


def agent_ref_for(
    selection: ResolvedEvaluationSelection, model: ModelConfig, settings: CuratedSettings
) -> CuratedAgentRef | None:
    """The harness and token budgets each attempt receives, resolved as the worker resolves them.

    The agent context limit is the served window, capped by the protocol; each reply is capped by the
    run budget and the protocol's sampling limit. Agent selections therefore require an explicit
    ``serve.max_model_len``, which the worker checks against the running server.
    """
    plans = set()
    for source in selected_sources(selection):
        dataset = source.config
        assert dataset is not None
        if dataset.agent is None:
            continue
        assert dataset.protocol.sampling is not None
        plans.add((dataset.agent.harness, dataset.agent.max_input_tokens, dataset.protocol.sampling.max_tokens))
    if not plans:
        return None
    if len(plans) != 1:
        raise ValueError(f"{selection.benchmark} corpora declare different agents or budgets")
    ((harness, input_limit, reply_limit),) = plans
    if harness.kind not in EPISODE_HARNESS_REVISIONS:
        raise ValueError(f"No evaluation harness is registered for {harness.kind!r}: {selection.benchmark}")
    if isinstance(harness, PiAgent) and pi_thinking_format(model) is None:
        raise ValueError(f"{selection.benchmark} runs Pi; set agent.agent_kwargs.thinking_format for {model.name}")
    window = model.serve.max_model_len
    if window is None:
        raise ValueError(f"{selection.benchmark} runs an agent; set serve.max_model_len, its context window")
    run = run_max_tokens(window, settings, model)
    return CuratedAgentRef(
        harness=harness.kind,
        revision=EPISODE_HARNESS_REVISIONS[harness.kind],
        max_input_tokens=min(input_limit or window, window),
        max_output_tokens=min(run, reply_limit or run),
    )


def corpus_key(step: ArtifactStep) -> str:
    """Identify the publication recipe and version independently of its storage region."""
    return f"{step.name}@{step.version}"


def settings_for(
    selection: EvaluationSelection,
    protocol: ResolvedEvaluationSelection,
    model: ModelConfig,
    limit: int | None,
    seed: int | None,
    default_seed: int | None,
    task_machine_backend: TaskMachineBackend | None,
    task_evidence: TaskEvidence | None,
) -> CuratedSettings:
    """Resolve run settings.

    The seed is the launch seed, else the selection's, else ``default_seed``. Concurrency is the smaller
    of the model's and the selection's bounds, else the default. Agent benchmarks run on the selection's
    task machines, else the default :class:`TaskMachineSettings`; a launch backend replaces either backend.
    Task evidence is the launch's choice, else the one the selection's corpora declare.
    """
    if model.effective_tokenizer_revision is None:
        raise ValueError("Curated evaluation requires an explicit tokenizer revision")
    run = selection.run
    explicit_seed = seed if seed is not None else run.seed
    if run.judge is not None:
        url = urlsplit(run.judge_endpoint or "")
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("judge_endpoint must be an HTTP URL without credentials")
        if not run.judge_api_key or any(not is_secret_reference(ref) for ref in run.judge_api_key):
            raise ValueError("A judge requires secret references in judge_api_key")
    for name, refs in run.task_secrets.items():
        if not refs or any(not is_secret_reference(ref) for ref in refs):
            raise ValueError(f"task_secrets requires secret references for {name}")
    task_machine = run.task_machine
    if task_machine is None and runs_agent(protocol):
        task_machine = TaskMachineSettings()
    if task_machine is not None and task_machine_backend is not None:
        update: dict[str, object] = {"backend": task_machine_backend}
        # How images are built is a Daytona setting; another backend builds each image its own way.
        if task_machine_backend is not TaskMachineBackend.DAYTONA:
            update["image_build"] = TaskImageBuild.SNAPSHOT
        task_machine = TaskMachineSettings.model_validate({**task_machine.model_dump(), **update})
    concurrency_bounds = [bound for bound in (model.evaluation_num_concurrent, run.concurrency) if bound is not None]
    return CuratedSettings(
        concurrency=min(concurrency_bounds, default=DEFAULT_EVALUATION_CONCURRENCY),
        max_tokens=run.max_tokens,
        source_limit=limit if limit is not None else run.limit,
        seed=explicit_seed if explicit_seed is not None else default_seed,
        judge=run.judge,
        task_machine=task_machine,
        task_evidence=task_evidence if task_evidence is not None else declared_task_evidence(protocol),
    )


def record_ref_for(
    selection: EvaluationSelection,
    protocol: ResolvedEvaluationSelection,
    settings: CuratedSettings,
    agent: CuratedAgentRef | None,
) -> EvalRef:
    return EvalRef(
        name=selection.name,
        mechanism="curated",
        family=selection.benchmark,
        source_digest="sha256:"
        + canonical_sha256(
            {
                "selection": selection.model_dump(mode="json", exclude={"run": {"concurrency"}}),
                "settings": scoring_settings(settings.model_dump(mode="json")),
            }
        ),
        curated=CuratedRef(
            datasets=tuple(corpus_key(step) for step in corpus_steps(protocol, CampaignRuntime())),
            settings=settings,
            selection=protocol,
            agent=agent,
        ),
    )
