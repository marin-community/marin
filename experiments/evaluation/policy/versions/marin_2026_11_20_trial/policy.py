# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A trial eval policy for exercising the policy launcher.

It is the closest approximation of the October 2026 policy that ``main`` runs.

Benchmarks own their prompts, thinking mode, repeats, output caps, metrics and any window they enforce;
this module only chooses them and the settings each runs with. README.md lists the differences from
October.
"""

from evalrecords.records import PromptFormat, TaskImageBuild, TaskMachineSettings

from experiments.evaluation.benchmarks.agentic import CAMPAIGN_PROTOCOL
from experiments.evaluation.evals import JUDGES
from experiments.evaluation.models import models
from experiments.evaluation.policy.schema import EvalPolicy, PolicyCandidate, PolicyEvaluation

# October served every candidate at this window.
SERVED_CONTEXT = 73_728
# October's seed for the sampled benchmarks it seeded.
OCTOBER_SEED = 42
# Daytona machines that build each task image without creating a snapshot, for the benchmarks whose
# images would together exceed the account's snapshot quota.
DECLARATIVE_BUILDS = TaskMachineSettings(image_build=TaskImageBuild.DECLARATIVE)
# October's judge for FinanceBench and OlympiadBench answers.
TOGETHER_GPT_OSS_120B = JUDGES["together"]

_CATALOG = models()
GLIMMER = PolicyCandidate(model=_CATALOG["muse-glimmer-30b"], max_context=131_072)
NEMOTRON = PolicyCandidate(model=_CATALOG["nvidia-nemotron-3.5-super-vl-120b-a12b-bf16"], max_context=1_048_576)
QWEN = PolicyCandidate(model=_CATALOG["qwen3.6-35b-a3b"], max_context=262_144)

EVALUATIONS: tuple[PolicyEvaluation, ...] = (
    PolicyEvaluation("aime24", seed=OCTOBER_SEED),
    PolicyEvaluation("boolq"),
    PolicyEvaluation("cruxeval"),
    PolicyEvaluation("financebench", judge=TOGETHER_GPT_OSS_120B),
    PolicyEvaluation("gpqa-diamond"),
    # October ran the thinking-off chat variants of GSM8K and TriviaQA; pin them for every candidate.
    PolicyEvaluation("gsm8k", label="gsm8k-0shot", num_fewshot=0, prompt_format=PromptFormat.CHAT),
    PolicyEvaluation("humanevalplus"),
    PolicyEvaluation("ifbench"),
    PolicyEvaluation("math500"),
    PolicyEvaluation("mbppplus"),
    PolicyEvaluation("mmlu-pro"),
    PolicyEvaluation("mrcr", protocol="32k"),
    PolicyEvaluation("mrcr", protocol="65k"),
    PolicyEvaluation("mrcr"),
    PolicyEvaluation("nupa5k-loose", label="nupa"),
    PolicyEvaluation("olympiadbench", judge=TOGETHER_GPT_OSS_120B, seed=OCTOBER_SEED),
    PolicyEvaluation("piqa"),
    PolicyEvaluation("triviaqa", label="triviaqa", prompt_format=PromptFormat.CHAT),
    PolicyEvaluation("truthfulqa"),
    PolicyEvaluation("winogrande"),
    # Agent benchmarks, under the names and with the concurrency of October's Harbor runs.
    PolicyEvaluation(
        "terminal-bench-2.1",
        label="tb2-recovery",
        protocol=CAMPAIGN_PROTOCOL,
        concurrency=16,
        task_machine=DECLARATIVE_BUILDS,
    ),
    PolicyEvaluation(
        "ot-tblite",
        label="ot-tblite-recovery",
        protocol=CAMPAIGN_PROTOCOL,
        concurrency=16,
        task_machine=DECLARATIVE_BUILDS,
    ),
    PolicyEvaluation(
        "swebench-verified",
        label="swebench-verified",
        protocol="mini-swe-agent",
        concurrency=16,
        task_machine=DECLARATIVE_BUILDS,
    ),
    PolicyEvaluation("ds-1000", label="ds-1000-local", protocol="terminus-2", concurrency=32),
    PolicyEvaluation("bfcl-parity", label="bfclparity-pi", protocol="pi", concurrency=32),
)

# Not declared:
# - τ³-bench and SOTOPIA-hard: their Harbor agents were removed from main.
# - BixBench: its package grader asks a MiniMax-M3 judge served beside the candidate, which the policy
#   does not declare yet; it joins once the policy launcher can start a judge for one benchmark.

POLICY = EvalPolicy(
    version="marin-2026-11-20-trial",
    context=SERVED_CONTEXT,
    candidates=(GLIMMER, NEMOTRON, QWEN),
    in_distribution=EVALUATIONS,
)
