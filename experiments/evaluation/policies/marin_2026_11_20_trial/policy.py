# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A trial eval policy for exercising the policy launcher.

It is the closest approximation of the October 2026 policy that ``main`` runs.

Benchmarks own their prompts, thinking mode, repeats, output caps and metrics; this module only chooses
them and the context, output limit and seed each runs with. README.md lists the differences from October.
"""

from pathlib import Path

import yaml
from evalrecords.records import EvaluationSelection, PromptFormat

from experiments.evaluation.evals import EVALS, HarborDefinition
from experiments.evaluation.models import models
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate, PolicyEvaluation

VERSION = "marin-2026-11-20-trial"

# October served every candidate at this window; benchmarks whose protocol declares a smaller one run
# at the protocol's window.
SERVED_CONTEXT = 73_728
# October's seed for the sampled benchmarks it seeded.
OCTOBER_SEED = 42

_HARBOR_DIR = Path(__file__).with_name("harbor")

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


def _harbor(name: str) -> PolicyEvaluation:
    """A Harbor config of this policy, whose agent ``model_info`` states its context and output limit."""
    definition = HarborDefinition(name=name, config_path=_HARBOR_DIR / f"{name}.yaml")
    (agent,) = yaml.safe_load(definition.config_path.read_text())["agents"]
    model_info = agent["kwargs"]["model_info"]
    return PolicyEvaluation(
        definition=definition,
        context=model_info["max_input_tokens"],
        max_output=model_info["max_output_tokens"],
        seed=None,
    )


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
    # Agentic benchmarks on vendored Harbor.
    _harbor("tb2-recovery"),
    _harbor("ot-tblite-recovery"),
    _harbor("swebench-verified"),
    _harbor("ds-1000"),
)

# Not declared, because they do not run on main:
# - τ³-bench and SOTOPIA-hard: vendored Harbor no longer has their agents (the ACP relay and the
#   SOTOPIA agent).
# - BixBench and BFCL with Pi: Harbor now requires Pi tasks to declare a prepared image with Pi
#   installed (environment/shellbox-images.json), and their registry tasks install Pi at runtime.

POLICY = EvalPolicy(version=VERSION, candidates=(GLIMMER, NEMOTRON, QWEN), in_distribution=EVALUATIONS)
