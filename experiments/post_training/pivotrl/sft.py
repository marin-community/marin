# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT baselines for PivotRL: fine-tune the frozen policy on the expert action of each pivot.

Each example is a candidate's prompt, exactly as pass-rate measurement requests it, followed by the
expert's action from the candidate's source trajectory. Loss covers only that final turn (see
``sft_templates.py``). A run trains on the candidates in one band of the frozen policy's pass counts,
such as PivotRL's pivots or every graded candidate, through the general chat-SFT launcher
(``experiments/sft/launcher.py``).

Print the plan with ``uv run python -m experiments.post_training.pivotrl.sft --version dev``,
choose runs with ``--experiment``, and add ``--run`` to build them.
"""

import importlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import click
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.optim.config import AdamConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.model_config import ModelConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.experiment.checkpoints import hf_to_levanter
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.rl.pass_rates import PASS_RATES_FILENAME, PassRateTask
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from zephyr.writers import write_jsonl_file

from experiments.post_training.pivotrl.pipeline import pivot_steps, policy_key
from experiments.post_training.pivotrl.runs import (
    GRUG_SMOKE_RUNS,
    OPENHANDS_GRUG,
    SMOKE_RUNS,
    SWE_GRUG,
    PivotRLRun,
)
from experiments.post_training.pivotrl.sft_templates import GRUG_FINAL_TURN_TEMPLATE, QWEN3_FINAL_TURN_TEMPLATE
from experiments.sft.launcher import (
    LLAMA3_CHAT_EOS_TOKEN_IDS,
    ArtifactDatasetSpec,
    ConvertedCheckpointModel,
    HFModel,
    ModelSource,
    SFTSpec,
    resources_from_accelerator,
    sft_step,
)

EXAMPLES_FILENAME = "train.jsonl.gz"
QWEN3_EOS_TOKEN_IDS = (151645, 151643)  # <|im_end|> and <|endoftext|>
GRUG_TRAIN_NODES = 4
GRUG_CONVERSION_VERSION = "2026.10.09"
ROWS_PER_BATCH = 64
# Tokens an example adds beyond its prompt's served length and its action's bytes: the turn's
# header and end markers.
TURN_TOKENS = 32

# Grug's K=16 pass rates: each full run's 8 samples per candidate pooled with 8 more from a seed-1
# rerun. They were pooled outside this pipeline, so they are adopted from where they were written.
K16_PASS_RATES_ROOT = "s3://marin-us-east-02a/marin/users/dml/pivotrl"
K16_PASS_RATES_VERSION = "2026.10.09"


class ReferenceTask(PassRateTask, Protocol):
    def reference(self, row: dict[str, Any]) -> dict[str, Any]:
        """The expert's action for this row, as an assistant message."""
        ...


@dataclass(frozen=True)
class PassBand:
    """Graded candidates that passed between ``lowest`` and ``highest`` of their K samples, inclusive."""

    name: str
    lowest: int
    highest: int


def pivot_band(name: str, samples_per_row: int, difficulty_threshold: float) -> PassBand:
    """PivotRL's pivots at K samples: mixed outcomes and a pass rate below lambda (``selection.pivot_filter``)."""
    return PassBand(name, 1, math.ceil(samples_per_row * difficulty_threshold) - 1)


def band_mask(counts: pa.Table, band: PassBand) -> pa.ChunkedArray:
    """Which rows of a pass-rate table, given by its ``passed`` and ``total`` columns, are in ``band``.

    Excluded rows (``total == 0``) are never in a band. Every graded row must share one K that
    ``band`` fits, so a band written for one K never silently selects from another.
    """
    total, passed = counts["total"], counts["passed"]
    totals = pc.unique(total.filter(pc.greater(total, 0))).to_pylist()
    if len(totals) != 1 or band.highest > totals[0]:
        raise ValueError(f"band {band} does not fit rows sampled {sorted(totals)} times")
    in_band = pc.and_(pc.greater_equal(passed, band.lowest), pc.less_equal(passed, band.highest))
    return pc.and_(pc.greater(total, 0), in_band)


def band_rows(pass_rates_path: str, band: PassBand) -> Iterator[dict[str, Any]]:
    """Stream the rows of a pass-rate artifact in ``band``, a few at a time: prompts can be long."""
    path = StoragePath(pass_rates_path) / PASS_RATES_FILENAME
    with path.open("rb") as source:
        mask = band_mask(pq.read_table(source, columns=["passed", "total"]), band)
    with path.open("rb") as source:
        offset = 0
        for batch in pq.ParquetFile(source).iter_batches(batch_size=ROWS_PER_BATCH):
            yield from batch.filter(mask.slice(offset, batch.num_rows).combine_chunks()).to_pylist()
            offset += batch.num_rows


def prompt_tokens(samples_path: str) -> dict[str, int]:
    """Each row's prompt length in tokens, as the server counted it when pass rates were sampled."""
    fs, path = url_to_fs(str(StoragePath(samples_path) / "samples"))
    samples = ds.dataset(path, filesystem=fs, format="parquet").to_table(columns=["row_id", "prompt_tokens"])
    lengths = samples.group_by("row_id").aggregate([("prompt_tokens", "max")])
    return dict(zip(lengths["row_id"].to_pylist(), lengths["prompt_tokens_max"].to_pylist(), strict=True))


def example_tokens_bound(prompt_tokens: int, action: dict[str, Any]) -> int:
    """An upper bound on an example's length: a byte-level BPE token covers at least one byte."""
    return prompt_tokens + len(json.dumps(action, ensure_ascii=False).encode()) + TURN_TOKENS


def write_sft_examples(
    *,
    pass_rates_path: str,
    samples_path: str,
    band: PassBand,
    output_path: str,
    task: str,
    enable_thinking: bool,
    max_tokens: int,
) -> None:
    """Write one chat example per row: the requested prompt and tools, then the expert action.

    Rows that may not fit in ``max_tokens`` are dropped, since truncation would cut the supervised
    action; ``manifest.json`` counts them. Lengths come from the served prompt lengths recorded with
    the samples, bounded above by the action's bytes, so no tokenizer is needed.

    Args:
        pass_rates_path: A pass-rate artifact: candidate rows with their pass counts.
        samples_path: A pass-rate artifact whose samples cover the same rows, for their prompt lengths.
        band: Which candidates to train on, by pass count.
        output_path: Where ``train.jsonl.gz`` and ``manifest.json`` go.
        task: Import path of the :class:`ReferenceTask` that reads these rows.
        enable_thinking: The thinking mode pass rates sample with, so prompts render as they were served.
        max_tokens: The training sequence length.
    """
    module, attribute = task.split(":")
    reader: ReferenceTask = getattr(importlib.import_module(module), attribute)
    served = prompt_tokens(samples_path)
    bounds: list[int] = []
    too_long = 0

    def examples() -> Iterator[dict[str, Any]]:
        nonlocal too_long
        for row in band_rows(pass_rates_path, band):
            body = reader.request(row)
            action = reader.reference(row)
            bound = example_tokens_bound(served[reader.row_id(row)], action)
            if bound > max_tokens:
                too_long += 1
                continue
            bounds.append(bound)
            yield {
                "messages": [*body["messages"], action],
                # Levanter's renderer, unlike transformers', leaves an unset ``tools`` undefined.
                "chat_template_kwargs": {"tools": body.get("tools"), "enable_thinking": enable_thinking},
            }

    output = StoragePath(output_path)
    write_jsonl_file(examples(), str(output / EXAMPLES_FILENAME))
    if not bounds:
        raise ValueError(f"no example fits in {max_tokens} tokens")
    manifest = {
        "band": band.name,
        "rows": len(bounds) + too_long,
        "examples": len(bounds),
        "too_long": too_long,
        "max_tokens": max_tokens,
        "longest_example_tokens_bound": max(bounds),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


@dataclass(frozen=True)
class SFTPolicy:
    """How to fine-tune one policy: the template it is served with, and a trainer."""

    chat_template: str
    model: ModelSource
    resources: ResourceConfig
    mesh: MeshConfig | None = None


def qwen3_policy(model: ModelConfig) -> SFTPolicy:
    """A Qwen3 HF checkpoint, trained by Levanter on one GPU."""
    reference = f"{model.location}@{model.revision}"
    return SFTPolicy(
        chat_template=QWEN3_FINAL_TURN_TEMPLATE,
        model=HFModel(
            model_ref=reference, tokenizer_path=reference, model_type="qwen3", eos_token_ids=QWEN3_EOS_TOKEN_IDS
        ),
        resources=resources_from_accelerator("1xH100"),
    )


def grug_policy(model: ModelConfig) -> SFTPolicy:
    """A Grug Datakit SFT export, converted to a Levanter Snowball checkpoint and trained on 4x8 H100.

    The mesh follows the curriculum SFT trials (``experiments/post_training/curriculum_sft/trial.py``).
    """
    conversion = hf_to_levanter(
        model.location,
        model_type="snowball",
        hf_revision=model.revision,
        tokenizer=f"hf://{model.location}@{model.revision}",
        version=GRUG_CONVERSION_VERSION,
        resources=ResourceConfig.with_cpu(cpu=64, ram="512g", disk="256g"),
    )
    return SFTPolicy(
        chat_template=GRUG_FINAL_TURN_TEMPLATE,
        model=ConvertedCheckpointModel(conversion=conversion, eos_token_ids=LLAMA3_CHAT_EOS_TOKEN_IDS),
        resources=ResourceConfig.with_gpu(
            "H100", count=8, cpu=32, ram="512g", disk="256g", replicas=GRUG_TRAIN_NODES, preemptible=False
        ),
        # Snowball shards parameters over expert x (data, context); context spans the nodes.
        mesh=MeshConfig(
            axes={"data": 1, "replica": 1, "model": 1, "expert": -1},
            dcn_axes={"context": GRUG_TRAIN_NODES},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context", "vocab": "model"},
        ),
    )


@dataclass(frozen=True)
class SFTBudget:
    seq_len: int
    """Training sequence length; longer examples are dropped."""
    batch_size: int
    steps: int
    learning_rate: float
    """Constant Adam learning rate."""


@dataclass(frozen=True)
class MeasuredPassRates:
    """The pass rates the PivotRL pipeline measures for the run."""


@dataclass(frozen=True)
class AdoptedPassRates:
    """Pass rates written outside the PivotRL pipeline, adopted from where they were written."""

    label: str
    """Names them in the paths of the SFT artifacts built from them."""
    source: str
    version: str
    samples: str
    """A pass-rate artifact whose samples cover the same rows, for their served prompt lengths."""


@dataclass(frozen=True)
class PivotSFTRun:
    name: str
    pivots: PivotRLRun
    """The PivotRL run whose policy is fine-tuned and whose task reads the candidates."""
    pass_rates: MeasuredPassRates | AdoptedPassRates
    """The policy's pass rates on the run's candidates."""
    band: PassBand
    policy: SFTPolicy
    """How to fine-tune ``pivots.model``."""
    budget: SFTBudget
    seed: int
    """The trainer seed, which orders the examples."""


def grug_k16_pass_rates(run: PivotRLRun) -> AdoptedPassRates:
    """Grug's pass rates on ``run``'s candidates at K=16 (see ``K16_PASS_RATES_ROOT``)."""
    pass_rates = f"{K16_PASS_RATES_ROOT}/{run.candidates.label}/pass-rates"
    return AdoptedPassRates(
        label=f"{run.model.name}-k16",
        source=f"{pass_rates}/{run.model.name}+seed-1/{K16_PASS_RATES_VERSION}",
        version=K16_PASS_RATES_VERSION,
        # The pooled pass rates kept no samples; the unseeded run sampled the same rows.
        samples=f"{pass_rates}/{run.model.name}/{K16_PASS_RATES_VERSION}",
    )


def pass_rates_step(run: PivotSFTRun) -> tuple[str, ArtifactStep[Artifact], ArtifactStep[Artifact] | str]:
    """The label naming the run's pass rates in SFT artifact paths, the pass rates, and their samples."""
    pivots = run.pivots
    match run.pass_rates:
        case MeasuredPassRates():
            measured = pivot_steps(pivots)["pass-rates"]
            return policy_key(pivots), measured, measured
        case AdoptedPassRates(label=label, source=source, version=version, samples=samples):
            name = user_owned_name(f"pivotrl/{pivots.candidates.label}/pass-rates/{label}")
            return label, ArtifactStep.adopt(name, version, source), samples


def sft_examples_step(run: PivotSFTRun) -> ArtifactStep[Artifact]:
    """The run's chat examples: its band of candidates, each followed by the expert action."""
    pivots = run.pivots
    label, pass_rates, samples = pass_rates_step(run)
    return apply(
        user_owned_name(
            f"pivotrl/{pivots.candidates.label}/sft-examples/{label}/{run.band.name}/max-{run.budget.seq_len}"
        ),
        remote(write_sft_examples, resources=ResourceConfig.with_cpu(cpu=2, ram="16g")),
        pass_rates_path=pass_rates,
        samples_path=samples,
        band=run.band,
        output_path=OUT,
        task=pivots.candidates.task,
        enable_thinking=pivots.sampling.enable_thinking,
        max_tokens=run.budget.seq_len,
    )


def pivot_sft_step(run: PivotSFTRun) -> ArtifactStep[LevanterCheckpoint]:
    examples = sft_examples_step(run)
    budget = run.budget
    name = user_owned_name(f"checkpoints/pivotrl-sft/{run.name}")
    spec = SFTSpec(
        name=name,
        # Deferred like the data artifacts, so --version and --override reach the checkpoint too.
        version=resolve_version(name, None),
        model=run.policy.model,
        chat_template=run.policy.chat_template,
        datasets=[
            ArtifactDatasetSpec(
                slug=run.pivots.candidates.label, artifact=examples, train_glob=EXAMPLES_FILENAME, weight=1.0
            )
        ],
        optimizer=AdamConfig(
            learning_rate=budget.learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            weight_decay=0.0,
            max_grad_norm=1.0,
            lr_schedule="constant",
            warmup=0.0,
        ),
        mesh=run.policy.mesh,
        seq_len=budget.seq_len,
        # One example per sequence, as RL trains; Snowball does not read packed attention masks.
        pack=False,
        batch_size=budget.batch_size,
        num_train_steps=budget.steps,
        hf_save_dtype="bfloat16",
        wandb_project="pivotrl-sft",
        seed=run.seed,
    )
    return sft_step(spec, run.policy.resources)


def smoke_sft_runs(pivots: PivotRLRun, policy: SFTPolicy, budget: SFTBudget) -> tuple[PivotSFTRun, ...]:
    """Training on the run's pivots, and on every graded candidate, from its own pass rates."""
    k = pivots.sampling.samples_per_row
    bands = (pivot_band("pivots", k, pivots.difficulty_threshold), PassBand("all", 0, k))
    return tuple(
        PivotSFTRun(f"{pivots.name}-{band.name}", pivots, MeasuredPassRates(), band, policy, budget, seed=0)
        for band in bands
    )


def grid_bands(samples_per_row: int) -> tuple[PassBand, ...]:
    """From every graded candidate, through ever harder pivots, to the candidates no sample solved.

    RL learns nothing from an unsolved candidate, since all its rewards are equal; SFT can.
    """
    k = samples_per_row
    return (
        PassBand("all", 0, k),
        pivot_band("mixed", k, 1.0),
        pivot_band("pivots", k, 0.5),
        pivot_band("hard", k, 0.25),
        PassBand("unsolved", 0, 0),
    )


def k16_grid_runs(pivots: PivotRLRun, budget: SFTBudget) -> tuple[PivotSFTRun, ...]:
    """Each band of Grug's K=16 pass rates, trained with each seed under one budget."""
    policy = grug_policy(pivots.model)
    pass_rates = grug_k16_pass_rates(pivots)
    return tuple(
        PivotSFTRun(f"{pivots.name}-k16-{band.name}-seed{seed}", pivots, pass_rates, band, policy, budget, seed)
        for band in grid_bands(16)
        for seed in GRID_SEEDS
    )


# The MarinSkyRL reference-action baseline's budget: 32 updates of 4 examples at a constant 2e-6.
QWEN_SMOKE_BUDGET = SFTBudget(seq_len=32768, batch_size=4, steps=32, learning_rate=2e-6)
# Grug trains in the 64k window it is served and RL-trained in, so no candidate it was graded on is cut.
GRUG_BUDGET = SFTBudget(seq_len=65536, batch_size=64, steps=32, learning_rate=2e-6)
# 9,600 examples: about one epoch of SWE's hard band, the smallest, so every band trains on as many
# examples and none repeats one.
SWE_GRID_BUDGET = SFTBudget(seq_len=65536, batch_size=64, steps=150, learning_rate=2e-6)
GRID_SEEDS = (0, 1, 2)

QWEN_SMOKE_SFT_RUNS = tuple(
    sft_run
    for pivots in SMOKE_RUNS
    if pivots.sampling.seed == 1
    for sft_run in smoke_sft_runs(pivots, qwen3_policy(pivots.model), QWEN_SMOKE_BUDGET)
)

GRUG_SMOKE_SFT_RUNS = tuple(
    sft_run for pivots in GRUG_SMOKE_RUNS for sft_run in smoke_sft_runs(pivots, grug_policy(pivots.model), GRUG_BUDGET)
)

# OpenHands has 5,441 candidates, so its bands train on 2,048 examples: about two epochs of the smallest.
GRUG_K16_GRID_RUNS = (*k16_grid_runs(SWE_GRUG, SWE_GRID_BUDGET), *k16_grid_runs(OPENHANDS_GRUG, GRUG_BUDGET))

SFT_RUNS = {run.name: run for run in (*QWEN_SMOKE_SFT_RUNS, *GRUG_SMOKE_SFT_RUNS, *GRUG_K16_GRID_RUNS)}


@click.command(help=__doc__)
@click.option(
    "--experiment",
    "names",
    type=click.Choice(sorted(SFT_RUNS)),
    multiple=True,
    help="Run to build; repeatable. Default: every run.",
)
@build_options
def main(names: tuple[str, ...]) -> dict[str, ArtifactStep]:
    return {name: pivot_sft_step(SFT_RUNS[name]) for name in names or SFT_RUNS}


if __name__ == "__main__":
    main()
