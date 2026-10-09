# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT baselines for PivotRL: fine-tune the frozen policy on the expert action of each pivot.

Each example is a candidate's prompt, exactly as pass-rate measurement requests it, followed by the
expert's action from the candidate's source trajectory. Loss covers only that final turn (see
``sft_templates.py``). A run trains on either the selected pivots (the prompts PivotRL trains on) or
every candidate, through the general chat-SFT launcher (``experiments/sft/launcher.py``).

Print the plan with ``uv run python -m experiments.post_training.pivotrl.sft --version dev``,
choose runs with ``--experiment``, and add ``--run`` to build them.
"""

import importlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import click
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.optim.config import AdamConfig
from levanter.tokenizers import load_tokenizer
from levanter.utils.mesh import MeshConfig
from marin.evaluation.model_config import ModelConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.experiment.checkpoints import hf_to_levanter
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.rl.pass_rates import PassRateTask
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath
from zephyr.writers import write_jsonl_file

from experiments.post_training.pivotrl.pipeline import candidate_step, pivot_steps, selection_key
from experiments.post_training.pivotrl.runs import (
    GRUG_SMOKE_RUNS,
    OPENHANDS_GRUG,
    SMOKE_RUNS,
    SWE_GRUG,
    TERMINAL_GRUG,
    PivotRLRun,
)
from experiments.post_training.pivotrl.selection import TRAIN_FILENAME
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


class ReferenceTask(PassRateTask, Protocol):
    def reference(self, row: dict[str, Any]) -> dict[str, Any]:
        """The expert's action for this row, as an assistant message."""
        ...


def write_sft_examples(
    *,
    rows_path: str,
    rows_filename: str,
    output_path: str,
    task: str,
    tokenizer: str,
    chat_template: str,
    enable_thinking: bool,
    max_tokens: int,
) -> None:
    """Write one chat example per row: the requested prompt and tools, then the expert action.

    Rows whose rendered example is longer than ``max_tokens`` are dropped, since truncation would cut
    the supervised action; ``manifest.json`` counts them.

    Args:
        rows_path: An artifact of candidate-schema rows (candidates or selected pivots).
        rows_filename: The parquet file of rows within ``rows_path``.
        output_path: Where ``train.jsonl.gz`` and ``manifest.json`` go.
        task: Import path of the :class:`ReferenceTask` that reads these rows.
        tokenizer: The policy's tokenizer, to measure examples.
        chat_template: The SFT chat template, to measure examples and check that each supervises tokens.
        enable_thinking: The thinking mode pass rates sample with, so prompts render as they were served.
        max_tokens: The training sequence length.
    """
    module, attribute = task.split(":")
    reader: ReferenceTask = getattr(importlib.import_module(module), attribute)
    with (StoragePath(rows_path) / rows_filename).open("rb") as source:
        rows = pq.read_table(source).to_pylist()
    marin_tokenizer = load_tokenizer(tokenizer)

    examples = []
    lengths = []
    for row in rows:
        body = reader.request(row)
        example = {
            "messages": [*body["messages"], reader.reference(row)],
            # Levanter's renderer, unlike transformers', leaves an unset ``tools`` undefined.
            "chat_template_kwargs": {"tools": body.get("tools"), "enable_thinking": enable_thinking},
        }
        encoded = marin_tokenizer.apply_chat_template_with_masks(
            [example["messages"]], chat_template=chat_template, **example["chat_template_kwargs"]
        )
        if not any(encoded["assistant_masks"][0]):
            raise ValueError(f"example for row {reader.row_id(row)} supervises no tokens")
        length = len(encoded["input_ids"][0])
        if length > max_tokens:
            continue
        examples.append(example)
        lengths.append(length)
    if not examples:
        raise ValueError(f"no example fits in {max_tokens} tokens")

    output = StoragePath(output_path)
    write_jsonl_file(examples, str(output / EXAMPLES_FILENAME))
    manifest = {
        "rows": len(rows),
        "examples": len(examples),
        "too_long": len(rows) - len(examples),
        "max_tokens": max_tokens,
        "longest_example_tokens": max(lengths),
        "example_tokens": sum(lengths),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


@dataclass(frozen=True)
class SFTPolicy:
    """How to fine-tune one policy: its tokenizer, the template it is served with, and a trainer."""

    tokenizer: str
    chat_template: str
    model: ModelSource
    resources: ResourceConfig
    mesh: MeshConfig | None = None


def qwen3_policy(model: ModelConfig) -> SFTPolicy:
    """A Qwen3 HF checkpoint, trained by Levanter on one GPU."""
    reference = f"{model.location}@{model.revision}"
    return SFTPolicy(
        tokenizer=reference,
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
        tokenizer=f"{model.location}@{model.revision}",
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


class SFTData(StrEnum):
    PIVOTS = "pivots"
    """The selected pivots of the PivotRL run: the same prompts RL trains on."""
    CANDIDATES = "candidates"
    """Every candidate of the PivotRL run, whatever its pass rate."""


@dataclass(frozen=True)
class SFTBudget:
    seq_len: int
    """Training sequence length; longer examples are dropped."""
    batch_size: int
    steps: int
    learning_rate: float
    """Constant Adam learning rate."""


@dataclass(frozen=True)
class PivotSFTRun:
    name: str
    pivots: PivotRLRun
    """The PivotRL run whose policy is fine-tuned and whose candidates or pivots are the data."""
    data: SFTData
    policy: SFTPolicy
    """How to fine-tune ``pivots.model``."""
    budget: SFTBudget


def sft_examples_step(run: PivotSFTRun) -> ArtifactStep[Artifact]:
    """The run's chat examples, from its pivots or from all of its candidates."""
    pivots = run.pivots
    dataset = pivots.candidates.label
    match run.data:
        case SFTData.PIVOTS:
            source = pivot_steps(pivots)["pivots"]
            filename = TRAIN_FILENAME
            selection = selection_key(pivots)
        case SFTData.CANDIDATES:
            source = candidate_step(pivots.candidates)
            filename = pivots.candidates.filename
            selection = f"{pivots.model.name}/candidates"
    return apply(
        user_owned_name(f"pivotrl/{dataset}/sft-examples/{selection}/max-{run.budget.seq_len}"),
        remote(write_sft_examples, resources=ResourceConfig.with_cpu(cpu=4, ram="32g")),
        rows_path=source,
        rows_filename=filename,
        output_path=OUT,
        task=pivots.candidates.task,
        tokenizer=run.policy.tokenizer,
        chat_template=run.policy.chat_template,
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
    )
    return sft_step(spec, run.policy.resources)


def sft_runs(pivots: PivotRLRun, policy: SFTPolicy, budget: SFTBudget) -> tuple[PivotSFTRun, ...]:
    """A pivots run and an all-candidates run over the same PivotRL run, policy, and budget."""
    return tuple(
        PivotSFTRun(name=f"{pivots.name}-{data}", pivots=pivots, data=data, policy=policy, budget=budget)
        for data in SFTData
    )


# The MarinSkyRL reference-action baseline's budget: 32 updates of 4 examples at a constant 2e-6.
QWEN_SMOKE_BUDGET = SFTBudget(seq_len=32768, batch_size=4, steps=32, learning_rate=2e-6)
GRUG_BUDGET = SFTBudget(seq_len=32768, batch_size=64, steps=32, learning_rate=2e-6)

QWEN_SMOKE_SFT_RUNS = tuple(
    sft_run
    for pivots in SMOKE_RUNS
    if pivots.sampling.seed == 1
    for sft_run in sft_runs(pivots, qwen3_policy(pivots.model), QWEN_SMOKE_BUDGET)
)

GRUG_SFT_RUNS = tuple(
    sft_run
    for pivots in (*GRUG_SMOKE_RUNS, SWE_GRUG, TERMINAL_GRUG, OPENHANDS_GRUG)
    for sft_run in sft_runs(pivots, grug_policy(pivots.model), GRUG_BUDGET)
)

SFT_RUNS = {run.name: run for run in (*QWEN_SMOKE_SFT_RUNS, *GRUG_SFT_RUNS)}


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
