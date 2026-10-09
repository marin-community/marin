# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concrete PivotRL data runs: which candidates, which frozen policy, and how to sample and select.

Add a run by instantiating :class:`PivotRLRun` and listing it in :data:`RUNS`. Runs that share
candidates, policy, and sampling share one pass-rate artifact; only their selection differs.
"""

from dataclasses import dataclass, replace

from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import ModelConfig, ResourceHint, ServeConfig
from marin.rl.nemotron_pivot import CANDIDATES_FILENAME
from marin.rl.pass_rates import PassRateSampling

from experiments.evaluation.models import SNOWBALL_VLLM_ARGS


@dataclass(frozen=True)
class NemotronPivotCandidates:
    """Long-format candidates from the shared catalog, ``experiments/datasets/nemotron_pivot.py``."""

    dataset: str
    """Key in ``nemotron_pivot_datasets()``: ``swe`` or ``terminal``."""
    task: str
    filename: str = CANDIDATES_FILENAME
    limit: int | None = None
    """Keep only the first ``limit`` rows, for smoke runs."""

    @property
    def label(self) -> str:
        return self.dataset if self.limit is None else f"{self.dataset}-head-{self.limit}"


@dataclass(frozen=True)
class PivotRLRun:
    name: str
    candidates: NemotronPivotCandidates
    model: ModelConfig
    accelerator: AcceleratorChoice
    sampling: PassRateSampling
    difficulty_threshold: float
    criterion: str = "passed"
    """Which pass count selection uses: ``passed`` (the task's reward) or ``passed_<component>``."""
    secret_env_keys: tuple[str, ...] = ()
    """Credentials the graders read, such as a judge's API key, forwarded to grading jobs."""


# Every expert turn of each trajectory's longest release row, with the teacher's earlier reasoning.
SWE_CANDIDATES = NemotronPivotCandidates(dataset="swe", task="experiments.post_training.pivotrl.task:PIVOT_TOOL_CALL")
TERMINAL_CANDIDATES = NemotronPivotCandidates(
    dataset="terminal", task="experiments.post_training.pivotrl.task:PIVOT_TERMINAL"
)

GRUG_SFT = ModelConfig(
    name="grug-67b-a2b-datakit-sft-262k-20261005",
    location="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.10.05",
    revision="8c76b22afc38f5e95360c83ca88008f00b5ae1e0",
    resource_hint=ResourceHint(gpu={"H100": 8}, memory="512g"),
    # Serve as MarinSkyRL samples during training: hermes tool calls in a 64k request window.
    serve=ServeConfig(
        tensor_parallel_size=1,
        data_parallel_size=8,
        max_model_len=65536,
        max_num_batched_tokens=16384,
        max_num_seqs=128,
        tool_call_parser="hermes",
        vllm_extra_args=SNOWBALL_VLLM_ARGS,
    ),
)

H100X8 = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8)

QWEN3_0_6B = ModelConfig(
    name="qwen3-0.6b",
    location="Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
    resource_hint=ResourceHint(gpu={"H100": 1}),
    serve=ServeConfig(max_model_len=40960, max_num_batched_tokens=16384, max_num_seqs=256, tool_call_parser="hermes"),
)

SWE_GRUG = PivotRLRun(
    name="swe-grug",
    candidates=SWE_CANDIDATES,
    model=GRUG_SFT,
    accelerator=H100X8,
    # Thinking on: the prompts keep the teacher's earlier reasoning. Grug's reasoning delimiters are
    # special tokens; keep them so graders can strip the reasoning.
    sampling=PassRateSampling(samples_per_row=8, max_tokens=4096, enable_thinking=True, skip_special_tokens=False),
    difficulty_threshold=0.5,
)

TERMINAL_GRUG = replace(SWE_GRUG, name="terminal-grug", candidates=TERMINAL_CANDIDATES)

# Exercises the whole path on one GPU in minutes: serving, grading, chunk commits, selection.
SWE_QWEN_SMOKE = PivotRLRun(
    name="swe-qwen-smoke",
    candidates=replace(SWE_CANDIDATES, limit=64),
    model=QWEN3_0_6B,
    accelerator=AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=1),
    sampling=PassRateSampling(samples_per_row=8, max_tokens=4096, enable_thinking=True, chunk_size=16),
    difficulty_threshold=0.5,
)

TERMINAL_QWEN_SMOKE = replace(
    SWE_QWEN_SMOKE, name="terminal-qwen-smoke", candidates=replace(TERMINAL_CANDIDATES, limit=64)
)


def seeded(run: PivotRLRun, seed: int) -> PivotRLRun:
    return replace(run, name=f"{run.name}-seed{seed}", sampling=replace(run.sampling, seed=seed))


# Each smoke run twice, with different sampling seeds, to see how stable pass rates and selection are.
SMOKE_RUNS = tuple(seeded(run, seed) for run in (SWE_QWEN_SMOKE, TERMINAL_QWEN_SMOKE) for seed in (1, 2))

# The same 64-candidate smoke runs with a real policy, one 8xH100 node each.
GRUG_SMOKE_RUNS = tuple(
    seeded(
        replace(
            run,
            name=run.name.replace("qwen", "grug"),
            model=GRUG_SFT,
            accelerator=H100X8,
            sampling=replace(run.sampling, skip_special_tokens=False),
        ),
        1,
    )
    for run in (SWE_QWEN_SMOKE, TERMINAL_QWEN_SMOKE)
)

RUNS = {
    run.name: run
    for run in (
        SWE_GRUG,
        TERMINAL_GRUG,
        *SMOKE_RUNS,
        *GRUG_SMOKE_RUNS,
    )
}
