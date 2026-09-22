# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run matched Qwen curriculum/specification/dose ablations with an exact oracle."""

from __future__ import annotations

import gzip
import json
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import click
from levanter.optim.config import AdamConfig
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_owned_name
from marin.inference.config import ServedModelConfig, VllmEngineConfig, VllmLauncherType
from marin.inference.serve import local_inference
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.curriculum_rl.pool import QWEN3_MODEL, QWEN3_REVISION
from experiments.post_training.curriculum_sft.ablation.matrix import (
    SFT_SYSTEM_PROMPT,
    AblationCell,
    CurriculumCondition,
    GenerationSpec,
    SftDose,
    ablation_matrix,
    build_held_out_tasks,
    generated_payloads_to_rows,
)
from experiments.post_training.curriculum_sft.ablation.verifier import evaluate_responses, score_response
from experiments.sft.launcher import ArtifactDatasetSpec, HFModel, SFTSpec, resources_from_accelerator, sft_step

DEFAULT_GENERATION_URI = "s3://marin-us-east-02a/marin/users/power/documents/curriculum-sft/ablation/2026.09.21.4"
TRAIN_FILENAME = "train/examples.jsonl.gz"
MANIFEST_FILENAME = "manifest.json"
ORACLE_RESULT_FILENAME = "oracle-eval.json"
DEFAULT_ACCEPTED_EXAMPLES = 16
DEFAULT_HELD_OUT_COUNT = 64
DEFAULT_SEED = 17
QWEN_EOS_TOKEN_IDS = (151643, 151645)
QWEN_SFT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}<think>\n\n</think>\n\n{{ message['content'] }}<|im_end|>{% endgeneration %}\n"
    "{% else %}{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
)


class AblationDataset(Artifact):
    """Canonical messages for one curriculum x generation-specification cell."""


class OracleEvaluation(Artifact):
    """Raw generations and exact component scores for one held-out set."""


@dataclass(frozen=True)
class MaterializeDatasetConfig:
    generation_root: str
    output_path: str
    cell: AblationCell


@dataclass(frozen=True)
class OracleEvalConfig:
    model_name: str
    model_location: str
    model_revision: str | None
    output_path: str
    task_count: int
    seed: int


class OracleModelSource(Protocol):
    def deps(self) -> tuple[ArtifactStep, ...]: ...

    def resolve(self, ctx: StepContext) -> tuple[str, str | None]: ...


@dataclass(frozen=True)
class StaticOracleModel:
    location: str
    revision: str | None

    def deps(self) -> tuple[ArtifactStep, ...]:
        return ()

    def resolve(self, _ctx: StepContext) -> tuple[str, str | None]:
        return self.location, self.revision


@dataclass(frozen=True)
class ProducedOracleModel:
    step: ArtifactStep[LevanterCheckpoint]

    def deps(self) -> tuple[ArtifactStep, ...]:
        return (self.step,)

    def resolve(self, ctx: StepContext) -> tuple[str, str | None]:
        if ctx.is_fingerprint:
            return f"artifact://{self.step.name}@{self.step.version}", None
        checkpoints = discover_hf_checkpoints(ctx.artifact_path(self.step))
        if not checkpoints:
            raise FileNotFoundError(f"no HF checkpoint found under {ctx.artifact_path(self.step)}")
        return checkpoints[-1], None


def _jsonl(rows: Sequence[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()


def _generation_cell_name(cell: AblationCell) -> str:
    return AblationCell(cell.curriculum, cell.generation_spec, SftDose.LOW).name


def materialize_dataset(config: MaterializeDatasetConfig) -> AblationDataset:
    """Filter one GLM cell and write the exact canonical messages consumed by SFT."""

    generation_path = StoragePath(config.generation_root) / "generation.json"
    ledger = json.loads(generation_path.read_text())
    expected_name = _generation_cell_name(config.cell)
    matches = [entry for entry in ledger["cells"] if entry["cell"] == expected_name]
    if len(matches) != 1:
        raise ValueError(f"expected one generation ledger entry for {expected_name}, found {len(matches)}")
    entry = matches[0]
    rows = generated_payloads_to_rows(config.cell, entry["tasks"])

    output = StoragePath(config.output_path)
    output.mkdirs()
    training_path = output / TRAIN_FILENAME
    training_path.parent.mkdirs()
    with training_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(_jsonl(rows))
    manifest = {
        "cell": config.cell.name,
        "generation_cell": expected_name,
        "generation_root": config.generation_root,
        "accepted_examples": len(rows),
        "generation_quality": {
            key: entry[key]
            for key in (
                "requested",
                "accepted",
                "unique_accepted",
                "format_rate",
                "arithmetic_rate",
                "evidence_rate",
                "replicates",
            )
        },
        "generation_batch_id": ledger["batch_id"],
        "training_file": TRAIN_FILENAME,
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return AblationDataset(path=config.output_path)


def dataset_step(
    generation: ArtifactStep[Artifact],
    cell: AblationCell,
    *,
    version: str,
) -> ArtifactStep[AblationDataset]:
    """Materialize one oracle-verified training arm from the shared GLM ledger."""

    condition = f"{cell.curriculum}__{cell.generation_spec}"

    def build_config(ctx: StepContext) -> MaterializeDatasetConfig:
        return MaterializeDatasetConfig(
            generation_root=ctx.artifact_path(generation),
            output_path=ctx.output_path,
            cell=cell,
        )

    return ArtifactStep(
        name=user_owned_name(f"documents/curriculum-sft/ablation/{condition}"),
        version=version,
        artifact_type=AblationDataset,
        run=materialize_dataset,
        build_config=build_config,
        deps=(generation,),
    )


def _request_completion(base_url: str, model: str, question: str) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read())
    return payload["choices"][0]["message"]["content"]


def run_oracle_eval_on_gpu(config: OracleEvalConfig) -> None:
    """Serve one Qwen checkpoint and score a fixed held-out set without a judge."""

    configure_coreweave_s3()
    model = ServedModelConfig(
        weights=config.model_location,
        revision=config.model_revision,
        api_model=config.model_name,
        tokenizer=QWEN3_MODEL,
        dtype="bfloat16",
        max_model_len=8192,
        tensor_parallel_size=1,
    )
    engine = VllmEngineConfig(
        launcher=VllmLauncherType.CUDA,
        max_num_batched_tokens=8192,
        max_num_seqs=32,
    )
    tasks = build_held_out_tasks(task_count=config.task_count, seed=config.seed)
    with local_inference(model, engine) as session:
        with ThreadPoolExecutor(max_workers=16) as pool:
            responses = list(
                pool.map(
                    lambda task: _request_completion(
                        session.model.endpoint.base_url,
                        session.model.endpoint.model,
                        task.question,
                    ),
                    tasks,
                )
            )
        session.check_alive()

    summary = evaluate_responses(tasks, responses)
    records = []
    for task, response in zip(tasks, responses, strict=True):
        score = score_response(task, response)
        records.append(
            {
                "task_id": task.task_id,
                "response": response,
                "format_valid": score.format_valid,
                "arithmetic_valid": score.arithmetic_valid,
                "evidence_valid": score.evidence_valid,
                "exact": score.accepted,
            }
        )
    output = StoragePath(config.output_path)
    output.mkdirs()
    result = {
        "model_name": config.model_name,
        "model_location": config.model_location,
        "seed": config.seed,
        "summary": asdict(summary),
        "records": records,
    }
    (output / ORACLE_RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def run_oracle_eval(config: OracleEvalConfig) -> OracleEvaluation:
    resources = resources_from_accelerator("1xH100")
    remote(
        run_oracle_eval_on_gpu,
        name=f"curriculum-sft-oracle-{config.model_name}",
        resources=resources,
    )(config)
    return OracleEvaluation(path=config.output_path)


def oracle_eval_step(
    model: OracleModelSource,
    *,
    model_name: str,
    version: str,
    task_count: int,
    seed: int,
) -> ArtifactStep[OracleEvaluation]:
    deps = model.deps()

    def build_config(ctx: StepContext) -> OracleEvalConfig:
        location, revision = model.resolve(ctx)
        return OracleEvalConfig(
            model_name=model_name,
            model_location=location,
            model_revision=revision,
            output_path=ctx.output_path,
            task_count=task_count,
            seed=seed,
        )

    return ArtifactStep(
        name=user_owned_name(f"evals/curriculum-sft-ablation/{model_name}"),
        version=version,
        artifact_type=OracleEvaluation,
        run=run_oracle_eval,
        build_config=build_config,
        deps=deps,
    )


def qwen_sft_step(
    dataset: ArtifactStep[AblationDataset],
    cell: AblationCell,
    *,
    version: str,
) -> ArtifactStep[LevanterCheckpoint]:
    condition = f"{cell.curriculum}__{cell.generation_spec}"
    return sft_step(
        SFTSpec(
            name=user_owned_name(f"checkpoints/curriculum-sft/ablation/qwen3-0.6b/{cell.name}"),
            version=version,
            model=HFModel(
                model_ref=f"{QWEN3_MODEL}@{QWEN3_REVISION}",
                tokenizer_path=QWEN3_MODEL,
                eos_token_ids=QWEN_EOS_TOKEN_IDS,
            ),
            chat_template=QWEN_SFT_CHAT_TEMPLATE,
            datasets=(
                ArtifactDatasetSpec(
                    slug=f"curriculum-ablation-{condition}",
                    artifact=dataset,
                    relative_pattern=TRAIN_FILENAME,
                    weight=1.0,
                ),
            ),
            optimizer=AdamConfig(
                learning_rate=1e-5,
                beta1=0.9,
                beta2=0.98,
                epsilon=1e-8,
                max_grad_norm=1.0,
                weight_decay=0.0,
                lr_schedule="constant",
                warmup=0.0,
                min_lr_ratio=0.0,
            ),
            seq_len=1024,
            batch_size=2,
            num_train_epochs=cell.train_epochs,
            wandb_project="marin-curriculum-sft-ablation",
        ),
        resources_from_accelerator("1xH100"),
    )


def selected_cells(accepted_examples: int) -> tuple[AblationCell, ...]:
    """Return the complete curriculum x specification x dose factorial."""

    return ablation_matrix(accepted_examples=accepted_examples)


@click.command(help=__doc__)
@click.option("--generation-uri", default=DEFAULT_GENERATION_URI, show_default=True)
@click.option("--accepted-examples", type=click.IntRange(min=1), default=DEFAULT_ACCEPTED_EXAMPLES, show_default=True)
@click.option("--held-out-count", type=click.IntRange(min=1), default=DEFAULT_HELD_OUT_COUNT, show_default=True)
@click.option("--seed", type=int, default=DEFAULT_SEED, show_default=True)
@click.option("--evaluation-version", help="Optional eval-only version, allowing checkpoints to be rescored.")
@click.option("--stage", type=click.Choice(("base", "data", "sft", "eval", "all")), default="all", show_default=True)
@build_options
def main(
    generation_uri: str,
    accepted_examples: int,
    held_out_count: int,
    seed: int,
    evaluation_version: str | None,
    stage: str,
) -> dict[str, ArtifactStep]:
    base_name = "documents/curriculum-sft/ablation/qwen3-0.6b"
    version = resolve_version(base_name, None)
    evaluation_version = evaluation_version or version
    generation = ArtifactStep.adopt(
        user_owned_name("documents/curriculum-sft/ablation/generation"),
        version,
        source=generation_uri,
        kind=Artifact,
    )
    cells = selected_cells(accepted_examples)
    datasets: dict[tuple[CurriculumCondition, GenerationSpec], ArtifactStep[AblationDataset]] = {}
    for cell in cells:
        key = (cell.curriculum, cell.generation_spec)
        if key not in datasets:
            datasets[key] = dataset_step(generation, cell, version=version)

    trainings = {
        cell.name: qwen_sft_step(datasets[(cell.curriculum, cell.generation_spec)], cell, version=version)
        for cell in cells
    }
    evaluations: dict[str, ArtifactStep[OracleEvaluation]] = {
        "base": oracle_eval_step(
            StaticOracleModel(QWEN3_MODEL, QWEN3_REVISION),
            model_name="qwen3-0.6b-base",
            version=evaluation_version,
            task_count=held_out_count,
            seed=seed,
        )
    }
    evaluations.update(
        {
            name: oracle_eval_step(
                ProducedOracleModel(training),
                model_name=f"qwen3-0.6b-{name}",
                version=evaluation_version,
                task_count=held_out_count,
                seed=seed,
            )
            for name, training in trainings.items()
        }
    )
    if stage == "base":
        return {"base": evaluations["base"]}
    if stage == "data":
        return {f"{curriculum}__{generation_spec}": step for (curriculum, generation_spec), step in datasets.items()}
    if stage == "sft":
        return trainings
    if stage == "eval":
        return evaluations
    return evaluations


if __name__ == "__main__":
    main()
