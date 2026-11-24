# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A trial eval policy for exercising the policy launcher.

It is the closest approximation of the October 2026 policy that ``main`` runs.

Benchmarks own their prompts, thinking mode, repeats, output caps and metrics; this module only chooses
them and the context, output limit and seed each runs with. README.md lists the differences from October.
"""

from evalrecords.records import (
    EvaluationOverrides,
    EvaluationSelection,
    PromptFormat,
    TaskImageBuild,
    TaskMachineSettings,
)

from experiments.evaluation.benchmarks.agentic import (
    CAMPAIGN_MAX_INPUT_TOKENS,
    CAMPAIGN_PROTOCOL,
    LONG_MAX_INPUT_TOKENS,
)
from experiments.evaluation.evals import EVALS
from experiments.evaluation.models import models
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate, PolicyEvaluation

VERSION = "marin-2026-11-20-trial"

# October served every candidate at this window; benchmarks whose protocol declares a smaller one run
# at the protocol's window.
SERVED_CONTEXT = 73_728
# October's seed for the sampled benchmarks it seeded.
OCTOBER_SEED = 42
# Daytona machines that build each task image without creating a snapshot, for the benchmarks whose
# images would together exceed the account's snapshot quota.
DECLARATIVE_BUILDS = TaskMachineSettings(image_build=TaskImageBuild.DECLARATIVE)

_CATALOG = models()
GLIMMER = PolicyCandidate(model=_CATALOG["muse-glimmer-30b"], max_context=131_072)
NEMOTRON = PolicyCandidate(model=_CATALOG["nvidia-nemotron-3.5-super-vl-120b-a12b-bf16"], max_context=1_048_576)
QWEN = PolicyCandidate(model=_CATALOG["qwen3.6-35b-a3b"], max_context=262_144)


def _taskspec(
    benchmark: str,
    *,
    label: str | None = None,
    protocol: str | None = None,
    num_fewshot: int | None = None,
    prompt_format: PromptFormat | None = None,
    context: int = SERVED_CONTEXT,
    seed: int | None = None,
) -> PolicyEvaluation:
    selection = EvaluationSelection(
        benchmark=benchmark, label=label, protocol=protocol, num_fewshot=num_fewshot, prompt_format=prompt_format
    )
    return PolicyEvaluation(definition=selection, context=context, max_output=None, seed=seed)


def _judged(name: str, *, seed: int | None = None) -> PolicyEvaluation:
    """A benchmark graded by the shared Together ``openai/gpt-oss-120b`` judge selection, as in October."""
    return PolicyEvaluation(definition=EVALS[name], context=SERVED_CONTEXT, max_output=None, seed=seed)


def _agent(
    benchmark: str,
    *,
    label: str,
    protocol: str,
    context: int,
    concurrency: int,
    task_machine: TaskMachineSettings | None = None,
) -> PolicyEvaluation:
    """An agent benchmark as October's campaign ran it.

    ``context`` is the protocol's agent input limit; the protocol owns the per-reply limit.
    """
    selection = EvaluationSelection(
        benchmark=benchmark,
        label=label,
        protocol=protocol,
        run=EvaluationOverrides(concurrency=concurrency, task_machine=task_machine),
    )
    return PolicyEvaluation(definition=selection, context=context, max_output=None, seed=None)


EVALUATIONS: tuple[PolicyEvaluation, ...] = (
    # Single-turn TaskSpec benchmarks, as October selected them.
    _taskspec("aime24", seed=OCTOBER_SEED),
    _taskspec("boolq"),
    _taskspec("cruxeval"),
    _judged("financebench"),
    _taskspec("gpqa-diamond"),
    # October ran the thinking-off chat variants of GSM8K and TriviaQA; pin them for every candidate.
    _taskspec("gsm8k", label="gsm8k-0shot", num_fewshot=0, prompt_format=PromptFormat.CHAT),
    _taskspec("humanevalplus"),
    _taskspec("ifbench"),
    _taskspec("math500"),
    _taskspec("mbppplus"),
    _taskspec("mmlu-pro", context=65_536),
    _taskspec("mrcr", protocol="32k", context=32_704),
    _taskspec("mrcr", protocol="65k", context=65_472),
    _taskspec("mrcr"),
    _taskspec("nupa5k-loose", label="nupa", context=73_664),
    _judged("olympiadbench", seed=OCTOBER_SEED),
    _taskspec("piqa"),
    _taskspec("triviaqa", prompt_format=PromptFormat.CHAT, label="triviaqa"),
    _taskspec("truthfulqa"),
    _taskspec("winogrande"),
    # Agent benchmarks, under the names and with the concurrency of October's Harbor runs.
    _agent(
        "terminal-bench-2.1",
        label="tb2-recovery",
        protocol=CAMPAIGN_PROTOCOL,
        context=LONG_MAX_INPUT_TOKENS,
        concurrency=16,
        task_machine=DECLARATIVE_BUILDS,
    ),
    _agent(
        "ot-tblite",
        label="ot-tblite-recovery",
        protocol=CAMPAIGN_PROTOCOL,
        context=CAMPAIGN_MAX_INPUT_TOKENS,
        concurrency=16,
        task_machine=DECLARATIVE_BUILDS,
    ),
    _agent(
        "swebench-verified",
        label="swebench-verified",
        protocol="mini-swe-agent",
        context=CAMPAIGN_MAX_INPUT_TOKENS,
        concurrency=16,
        task_machine=DECLARATIVE_BUILDS,
    ),
    _agent("ds-1000", label="ds-1000-local", protocol="terminus-2", context=CAMPAIGN_MAX_INPUT_TOKENS, concurrency=32),
    _agent("bfcl-parity", label="bfclparity-pi", protocol="pi", context=CAMPAIGN_MAX_INPUT_TOKENS, concurrency=32),
)

# Not declared:
# - τ³-bench and SOTOPIA-hard: their Harbor agents were removed from main.
# - BixBench: its package grader asks a MiniMax-M3 judge served beside the candidate, which the policy
#   does not declare yet; it joins once the policy launcher can start a judge for one benchmark.

POLICY = EvalPolicy(version=VERSION, candidates=(GLIMMER, NEMOTRON, QWEN), in_distribution=EVALUATIONS)
