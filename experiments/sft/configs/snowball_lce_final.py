# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Five-way Snowball LCE SFT campaign reconstructed from marin #8225 and #8977.

Each base is a pinned HF export of a native step-157000 checkpoint. The first stage loads HF
weights into the first-class Snowball model with the architecture resolved from that exact Hub
revision; later stages use weights-only native checkpoint initialization. Every stage therefore
starts with a fresh optimizer and step counter while retaining the base-specific ``qk_mult``.

Launch a one-update full-shape smoke on RNO2A before the campaign fan-out::

    source ../secrets.env
    uv run iris --config lib/iris/config/marin.yaml job run \
      --target-cluster cw-rno2a --job-name snowball-final-qk157-smoke-coord \
      --cpu 2 --memory 2G --extra cpu --priority interactive --max-retries 10 --no-wait \
      -e MARIN_PREFIX s3://marin-us-east-02a/marin \
      -e HF_TOKEN "$HF_TOKEN" -e WANDB_API_KEY "$WANDB_API_KEY" \
      -e IRIS_PORT_JAX 19302 -- \
      python -m experiments.sft.configs.snowball_lce_final \
      --base qk157 --stage smoke --version 2026.09.08.2 --run
"""

import dataclasses

import click
from fray.cluster import ResourceConfig
from levanter.models.snowball import SnowballConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.checkpoints import resolve_lm_config
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.june_tpu_67b_a2b.moe.optimizer import GrugMoeAdamHConfig
from experiments.sft.delphi_chat_template import DELPHI_V0_CHAT_TEMPLATE
from experiments.sft.launcher import DatasetSpec, HFModel, LevanterCheckpointModel, SFTSpec, sft_step

_SEQ = 32_768
_BATCH = 64
_NODES = 8
_EXPERT_PARALLEL = 8
_WANDB_PROJECT = "marin_moe_sft_snowball_final"

# These are immutable tags created only after the uploader validated all 44 source files. Add the
# three skew revisions after their export/upload jobs finish; never train from a moving ``main``.
_BASE_REVISIONS: dict[str, tuple[str, str | None]] = {
    "qk157": ("open-athena/snowball-67b-a2b-base-262k-qk157", "2b1f526273b8968b307a0098c08fb4321bb91e35"),
    "qk175": ("open-athena/snowball-67b-a2b-base-262k-qk175", "1934e71f2bb0fbeb19e5ce82372136e5297bf0a4"),
    "qk175-skew2": ("open-athena/snowball-67b-a2b-base-262k-qk175-skew2", None),
    "qk175-skew4": ("open-athena/snowball-67b-a2b-base-262k-qk175-skew4", None),
    "qk175-skew8": ("open-athena/snowball-67b-a2b-base-262k-qk175-skew8", None),
}

_CHAT_DATASET = DatasetSpec(
    slug="wildchat_386k",
    hf_dataset_id="nyu-dice-lab/wildchat50m-rewild-sft-385700",
    revision="46a5bb5",
    adapter_kwargs={"conversation_column": "conversation"},
    weight=1.0,
)
_THINKING_DATASET = DatasetSpec(
    slug="nemotron_science_think",
    hf_dataset_id="laion/llama-nemotron-science-reasoning-on-canonical-think-full",
    revision="bae881d",
    adapter_kwargs={},
    weight=1.0,
)

_TRAIN_MESH = MeshConfig(
    axes={"expert": _EXPERT_PARALLEL},
    dcn_axes={"data": -1},
    compute_mapping={"batch": ["replica_dcn", "data", "expert"]},
)


def _resources() -> ResourceConfig:
    return ResourceConfig.with_gpu(
        "H100",
        count=8,
        cpu=64,
        ram="768g",
        disk="512g",
        replicas=_NODES,
        preemptible=True,
    )


def _optimizer(learning_rate: float) -> GrugMoeAdamHConfig:
    return GrugMoeAdamHConfig(
        learning_rate=learning_rate,
        adam_lr=learning_rate,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=1.0,
        weight_decay=0.0,
        min_lr_ratio=0.1,
        warmup=0.03,
        lr_schedule="cosine",
    )


def _base_model(base: str) -> tuple[HFModel, SnowballConfig, str]:
    repo, revision = _BASE_REVISIONS[base]
    if revision is None:
        raise ValueError(f"Base {base} has not completed its validated HF upload.")
    model = resolve_lm_config("snowball", repo, revision)
    if not isinstance(model, SnowballConfig):
        raise TypeError(f"Expected SnowballConfig for {repo}@{revision}, got {type(model).__name__}.")
    model = dataclasses.replace(
        model,
        attention_implementation="gpu_fa4_cute",
        moe_implementation="ring",
    )
    ref = f"{repo}@{revision}"
    return (
        HFModel(
            model_ref=ref,
            tokenizer_path=ref,
            model_type="snowball",
            model_config=model,
            eos_token_ids=(128001, 128009),
            trainer_mesh=_TRAIN_MESH,
            use_explicit_mesh_axes=True,
        ),
        model,
        ref,
    )


def _native_model(parent: ArtifactStep[LevanterCheckpoint], model: SnowballConfig, tokenizer: str):
    return LevanterCheckpointModel(
        init_from=parent,
        model=model,
        tokenizer_path=tokenizer,
        eos_token_ids=(128001, 128009),
        trainer_mesh=_TRAIN_MESH,
        use_explicit_mesh_axes=True,
    )


def _spec(
    *,
    base: str,
    stage: str,
    version: str | None,
    model,
    dataset: DatasetSpec,
    steps: int | None = None,
    epochs: int | None = None,
) -> SFTSpec:
    step_name = f"snowball-final/{base}/{stage}"
    resolved_version = resolve_version(step_name, version)
    return SFTSpec(
        name=user_namespaced_name(step_name, resolved_version),
        version=resolved_version,
        model=model,
        chat_template=DELPHI_V0_CHAT_TEMPLATE,
        datasets=[dataset],
        optimizer=_optimizer(5e-5),
        seq_len=_SEQ,
        batch_size=_BATCH,
        num_train_steps=steps,
        num_train_epochs=epochs,
        wandb_project=_WANDB_PROJECT,
    )


def build_smoke(base: str, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    model, _, _ = _base_model(base)
    return sft_step(
        _spec(base=base, stage="hf-smoke", version=version, model=model, dataset=_CHAT_DATASET, steps=1),
        _resources(),
    )


def build_chat(
    base: str, version: str | None = None
) -> tuple[ArtifactStep[LevanterCheckpoint], SnowballConfig, str]:
    model, config, tokenizer = _base_model(base)
    chat = sft_step(
        _spec(base=base, stage="chat", version=version, model=model, dataset=_CHAT_DATASET, epochs=1),
        _resources(),
    )
    return chat, config, tokenizer


def build_thinking(base: str, version: str | None = None) -> ArtifactStep[LevanterCheckpoint]:
    chat, config, tokenizer = build_chat(base, version)
    thinking_model = _native_model(chat, config, tokenizer)
    return sft_step(
        _spec(
            base=base,
            stage="thinking",
            version=version,
            model=thinking_model,
            dataset=_THINKING_DATASET,
            epochs=1,
        ),
        _resources(),
    )


@click.command()
@click.option("--base", type=click.Choice(tuple(_BASE_REVISIONS)), required=True)
@click.option("--stage", type=click.Choice(("smoke", "chat", "thinking")), required=True)
@build_options
def main(base: str, stage: str) -> ArtifactStep[LevanterCheckpoint]:
    if stage == "smoke":
        return build_smoke(base)
    if stage == "chat":
        return build_chat(base)[0]
    return build_thinking(base)


if __name__ == "__main__":
    main()
