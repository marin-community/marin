# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Materialize an intermediate native checkpoint for a canonical Delphi run.

The helper resumes a Delphi pretraining run from an existing full-state
checkpoint, keeps the original pretraining LR schedule length, stops at a
requested intermediate step, and writes checkpoints under a new output root.
It never writes under the registered Delphi source root.

Examples:

    uv run python scripts/materialize_delphi_prefix_checkpoint.py \
        --base 3e18 \
        --source-step 20000 \
        --target-step 25900 \
        --output-root gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e18-step25900 \
        --tpu v5p-8 \
        --ram 128g \
        --dry-run

    uv run python scripts/materialize_delphi_prefix_checkpoint.py \
        --base 3e20 \
        --source-step 20000 \
        --target-step 24785 \
        --output-root gs://marin-us-east5/checkpoints/delphi-prefix-checkpoints/delphi-3e20-step24785 \
        --tpu v5p-32 \
        --ram 256g \
        --dry-run
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import urllib.parse
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

import draccus
import fsspec
from fray.cluster import ResourceConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import (
    ExecutorMainConfig,
    MirroredValue,
    executor_main,
    mirrored,
)
from marin.execution.types import ExecutorStep, this_output_path
from marin.midtraining import assert_checkpoint_complete_for_model_type
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm, temporary_checkpoint_base_path

from experiments.delphi_models import DelphiModel, get_delphi_model
from scripts.list_delphi_checkpoints import missing_checkpoint_artifacts, read_metadata_step

CHECKPOINTS_DIR = "checkpoints"
EXECUTOR_INFO_FILENAME = ".executor_info"
RESERVED_CHECKPOINT_METADATA_KEYS = frozenset({"step", "timestamp", "is_temporary"})
SOURCE_CHECKPOINT_MIRROR_BUDGET_GB = 40.0
DELPHI_MODEL_TYPE = "qwen3"


@dataclass(frozen=True)
class MaterializeRequest:
    base: str
    source_step: int
    target_step: int
    output_root: str
    tpu: str
    ram: str
    regions: tuple[str, ...] = ()
    allow_existing_destination: bool = False
    # Optional intermediate checkpoint steps the helper should ALSO save on the
    # way to ``target_step``. Each entry must be strictly inside
    # ``(source_step, target_step)`` and distinct from the others. Levanter
    # writes a permanent checkpoint at every step listed here (via per-step
    # ``CheckpointInterval`` policies) in addition to the forced save at the
    # final ``target_step``. Used by the cooldown plan when one base needs
    # multiple prefix targets that share a single at-or-before native source —
    # one training run, multiple committed outputs.
    extra_target_steps: tuple[int, ...] = ()


@dataclass(frozen=True)
class MaterializationPlan:
    request: MaterializeRequest
    model: DelphiModel
    train_config: TrainLmConfig
    source_checkpoint_path: str
    destination_checkpoint_paths: tuple[str, ...]
    """Sorted ascending. Last entry is the final ``target_step`` destination;
    any earlier entries correspond to ``request.extra_target_steps`` and are
    saved via the keep policy before training stops at ``target_step``."""
    temporary_checkpoint_path: str
    original_num_train_steps: int

    @property
    def destination_checkpoint_path(self) -> str:
        """Backward-compat single-destination shortcut == the final target.

        Existing callers (tests, the launcher's post-train check, the
        single-target launch path) keep working. Multi-target callers should
        read ``destination_checkpoint_paths`` directly.
        """
        return self.destination_checkpoint_paths[-1]

    @property
    def all_target_steps(self) -> tuple[int, ...]:
        """All steps Levanter will commit in this run, sorted ascending."""
        return tuple(sorted({*self.request.extra_target_steps, self.request.target_step}))

    @property
    def steps_to_train(self) -> int:
        return self.request.target_step - self.request.source_step

    @property
    def source_percent(self) -> float:
        return 100 * self.request.source_step / self.original_num_train_steps

    @property
    def target_percent(self) -> float:
        return 100 * self.request.target_step / self.original_num_train_steps


@dataclass(frozen=True)
class MaterializationTrainConfig:
    train_on_pod: TrainLmOnPodConfig
    destination_checkpoint_paths: tuple[str, ...]
    model_type: str
    num_layers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Verified Delphi flops key, e.g. 3e18 or 3e20.")
    parser.add_argument("--source-step", required=True, type=int, help="Existing native checkpoint step to load.")
    parser.add_argument("--target-step", required=True, type=int, help="Intermediate checkpoint step to materialize.")
    parser.add_argument("--output-root", required=True, help="New GCS run root that will receive the checkpoint.")
    parser.add_argument("--tpu", required=True, help="TPU type for the training job, e.g. v5p-8.")
    parser.add_argument("--ram", required=True, help="Coordinator/worker RAM for the TPU resource, e.g. 128g.")
    parser.add_argument(
        "--region",
        dest="regions",
        action="append",
        default=[],
        help="Restrict the child TPU training job to this Iris region. Can be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved plan without submitting training.")
    parser.add_argument(
        "--allow-existing-destination",
        action="store_true",
        help="Allow any <output-root>/checkpoints/step-<N> directory (final or extra) to exist before launch.",
    )
    parser.add_argument(
        "--also-save-step",
        dest="extra_target_steps",
        action="append",
        default=[],
        type=int,
        help=(
            "Additional intermediate step to save during the same training run. "
            "Repeatable. Each must be strictly inside (source-step, target-step). "
            "Useful when two cooldown prefixes share a single at-or-before native source."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    request = MaterializeRequest(
        base=args.base,
        source_step=args.source_step,
        target_step=args.target_step,
        output_root=args.output_root,
        tpu=args.tpu,
        ram=args.ram,
        regions=tuple(args.regions),
        allow_existing_destination=args.allow_existing_destination,
        extra_target_steps=tuple(args.extra_target_steps),
    )
    model = get_delphi_model(request.base)
    source_config = load_source_train_config(model)
    plan = build_materialization_plan(request, model, source_config)
    validate_checkpoint_io(plan)

    print_plan(plan)
    if args.dry_run:
        print()
        print("Dry-run; no executor_main call and no training job submitted.")
        return 0

    step = build_executor_step(plan)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix_for_output_root(plan.request.output_root)),
        steps=[step],
        description="Materialize an intermediate native Delphi prefix checkpoint.",
    )
    return 0


def build_materialization_plan(
    request: MaterializeRequest,
    model: DelphiModel,
    source_config: TrainLmConfig,
) -> MaterializationPlan:
    validate_static_request(request, model, source_config)
    source_checkpoint_path = model.levanter_checkpoint_path(request.source_step)
    output_root = request.output_root.rstrip("/")
    all_steps = sorted({*request.extra_target_steps, request.target_step})
    destination_checkpoint_paths = tuple(join_path(output_root, CHECKPOINTS_DIR, f"step-{step}") for step in all_steps)
    train_config = materialized_train_config(
        source_config,
        model=model,
        request=request,
        source_checkpoint_path=source_checkpoint_path,
    )
    return MaterializationPlan(
        request=request,
        model=model,
        train_config=train_config,
        source_checkpoint_path=source_checkpoint_path,
        destination_checkpoint_paths=destination_checkpoint_paths,
        temporary_checkpoint_path=temporary_checkpoint_base_path(output_root),
        original_num_train_steps=source_config.trainer.num_train_steps,
    )


def validate_static_request(
    request: MaterializeRequest,
    model: DelphiModel,
    source_config: TrainLmConfig,
) -> None:
    if request.source_step <= 0:
        raise ValueError(f"source_step must be positive, got {request.source_step}")
    if request.target_step <= request.source_step:
        raise ValueError(f"target_step must be greater than source_step: {request.target_step} <= {request.source_step}")

    original_num_train_steps = source_config.trainer.num_train_steps
    if original_num_train_steps <= 0:
        raise ValueError(f"source TrainLmConfig has invalid trainer.num_train_steps={original_num_train_steps}")
    if request.target_step >= original_num_train_steps:
        raise ValueError(
            f"target_step must be less than the original schedule length so Levanter can write "
            f"completed checkpoint step-{request.target_step}: {request.target_step} >= {original_num_train_steps}"
        )

    output_root = request.output_root.rstrip("/")
    if path_equal_or_nested(output_root, model.gcs_run_root):
        raise ValueError(
            f"Refusing to write output_root={request.output_root!r} under original Delphi root "
            f"{model.gcs_run_root!r}."
        )

    # extra_target_steps must be a set of distinct ints strictly inside
    # (source_step, target_step). Any value at or outside this open interval
    # would either never fire (below source) or collide with the forced final
    # save (at target). The Levanter Checkpointer constructor also asserts
    # ``step_policies`` sorted ascending by ``until`` (checkpoint.py:436); we
    # build keep policies from these extras in ascending order downstream.
    extras = request.extra_target_steps
    if len(extras) != len(set(extras)):
        raise ValueError(f"extra_target_steps must be distinct, got {extras!r}")
    for extra in extras:
        if extra <= request.source_step:
            raise ValueError(
                f"extra_target_step={extra} must be > source_step={request.source_step}; "
                "Levanter cannot save at a step already in the past."
            )
        if extra >= request.target_step:
            raise ValueError(
                f"extra_target_step={extra} must be < target_step={request.target_step}; "
                "use --target-step for the final save (it is forced unconditionally at training end)."
            )


def validate_checkpoint_io(plan: MaterializationPlan) -> None:
    missing = missing_checkpoint_artifacts(plan.source_checkpoint_path)
    if missing:
        raise FileNotFoundError(
            f"Source checkpoint {plan.source_checkpoint_path!r} is missing native TensorStore artifacts: "
            f"{', '.join(missing)}"
        )

    metadata_step = read_metadata_step(plan.source_checkpoint_path)
    if metadata_step != plan.request.source_step:
        raise ValueError(
            f"Source checkpoint metadata step mismatch for {plan.source_checkpoint_path!r}: "
            f"metadata={metadata_step}, expected={plan.request.source_step}"
        )

    assert_checkpoint_complete_for_model_type(
        plan.source_checkpoint_path,
        model_type=DELPHI_MODEL_TYPE,
        num_layers=plan.model.num_layers,
    )

    if not plan.request.allow_existing_destination:
        for destination in plan.destination_checkpoint_paths:
            if path_exists_or_has_children(destination):
                raise FileExistsError(
                    f"Destination checkpoint already exists: {destination}. "
                    "Pass --allow-existing-destination only if this is an intentional relaunch."
                )


def materialized_train_config(
    source_config: TrainLmConfig,
    *,
    model: DelphiModel,
    request: MaterializeRequest,
    source_checkpoint_path: str,
) -> TrainLmConfig:
    output_root = request.output_root.rstrip("/")
    source_config = regionalize_readonly_marin_uris(
        source_config,
        target_prefix=marin_prefix_for_output_root(output_root),
    )
    load_checkpoint_path = mirrored(
        marin_relative_path(source_checkpoint_path),
        budget_gb=SOURCE_CHECKPOINT_MIRROR_BUDGET_GB,
    )
    # For each requested intermediate step X, register a Levanter keep policy
    # ``{"every": X, "until": X}``. ``_get_current_step_save_interval`` picks
    # the first policy whose ``until >= step``, then triggers a permanent save
    # when ``step % every == 0``. Since X % X == 0, the policy fires at exactly
    # step X. Past step X the policy becomes inactive (``until < step``) and
    # the next-larger extra (or no policy) takes over. Sort ascending because
    # Levanter's Checkpointer constructor asserts step_policies sorted by
    # ``until`` (lib/levanter/src/levanter/checkpoint.py:436). The final
    # target_step is NOT in ``keep`` — it gets a force=True save at training
    # end (trainer.py:568,581), so it would write whether or not a keep policy
    # mentioned it.
    keep_policies = [{"every": step, "until": step} for step in sorted(request.extra_target_steps)]
    checkpointer = dataclasses.replace(
        source_config.trainer.checkpointer,
        base_path=join_path(output_root, CHECKPOINTS_DIR),
        temporary_base_path=temporary_checkpoint_base_path(output_root),
        keep=keep_policies,
        append_run_id_to_base_path=False,
        metadata=checkpoint_metadata(model=model, request=request, source_checkpoint_path=source_checkpoint_path),
    )
    trainer = dataclasses.replace(
        source_config.trainer,
        id=None,
        initialize_from=cast(str, load_checkpoint_path),
        load_checkpoint=False,
        load_checkpoint_path=None,
        stop_step=levanter_stop_step_for_checkpoint_step(request.target_step),
        checkpointer=checkpointer,
        tracker=tracker_with_prefix_tags(source_config.trainer.tracker, request),
    )
    return dataclasses.replace(
        source_config,
        trainer=trainer,
        initialize_from_checkpoint_path=None,
        initialize_from_hf=False,
        hf_save_path=None,
        hf_upload=None,
        hf_save_steps=None,
    )


def checkpoint_metadata(
    *,
    model: DelphiModel,
    request: MaterializeRequest,
    source_checkpoint_path: str,
) -> dict[str, Any]:
    # ``all_target_steps`` lists every step the run intends to commit (extras
    # via keep policy + the final via force-save at training end), sorted
    # ascending. An intermediate ckpt's per-save ``metadata.json`` carries
    # this list so consumers can disambiguate "this step is one of several
    # intentional outputs" from "this is the only intentional save". The
    # legacy ``target_step`` field remains for backward compat and always
    # equals the FINAL target_step (== max(all_target_steps)).
    all_target_steps = sorted({*request.extra_target_steps, request.target_step})
    metadata = {
        "target_step": request.target_step,
        "all_target_steps": all_target_steps,
        "source_step": request.source_step,
        "base": model.flops_key,
        "source_checkpoint_path": mirror_marin_uri(source_checkpoint_path),
        "original_delphi_root": mirror_marin_uri(model.gcs_run_root),
        "materialization": "delphi_prefix_checkpoint",
    }
    if RESERVED_CHECKPOINT_METADATA_KEYS.intersection(metadata):
        raise AssertionError("helper metadata attempted to override Levanter checkpoint metadata")
    return metadata


def levanter_stop_step_for_checkpoint_step(checkpoint_step: int) -> int:
    """Return Levanter's exclusive next-step stop boundary for a checkpoint step.

    Levanter stores ``state.step`` as the next step to execute. Checkpoint
    callbacks name checkpoints with ``StepInfo.step``, which is the completed
    step, so writing ``step-N`` requires stopping when ``state.step == N + 1``.
    """
    return checkpoint_step + 1


def tracker_with_prefix_tags(tracker: object, request: MaterializeRequest) -> object:
    extras_tag = ",".join(str(s) for s in sorted(request.extra_target_steps)) if request.extra_target_steps else "none"
    tags = (
        "delphi_prefix_checkpoint",
        f"base:{request.base}",
        f"source_step:{request.source_step}",
        f"target_step:{request.target_step}",
        f"extra_target_steps:{extras_tag}",
        "preserve_original_lr_schedule",
    )
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(
            tracker,
            id=None,
            tags=dedupe_tags([*tracker.tags, *tags]),
            replicate_path=request.output_root.rstrip("/"),
        )
    if isinstance(tracker, tuple):
        return tuple(tracker_with_prefix_tags(item, request) for item in tracker)
    return tracker


def dedupe_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def build_executor_step(plan: MaterializationPlan) -> ExecutorStep:
    levanter_stop_step = levanter_stop_step_for_checkpoint_step(plan.request.target_step)
    return ExecutorStep(
        name=os.path.join(
            "checkpoints",
            f"delphi-prefix-{plan.request.base}-step{plan.request.target_step}-stop{levanter_stop_step}",
        ),
        description=(
            f"Materialize Delphi {plan.request.base} native checkpoint step-{plan.request.target_step} "
            f"from step-{plan.request.source_step} while preserving the original "
            f"{plan.original_num_train_steps}-step LR schedule."
        ),
        fn=run_materialization_train,
        config=MaterializationTrainConfig(
            train_on_pod=TrainLmOnPodConfig(
                train_config=plan.train_config,
                resources=ResourceConfig.with_tpu(
                    plan.request.tpu,
                    ram=plan.request.ram,
                    regions=plan.request.regions or None,
                ),
                output_path=this_output_path(),
            ),
            destination_checkpoint_paths=plan.destination_checkpoint_paths,
            model_type=DELPHI_MODEL_TYPE,
            num_layers=plan.model.num_layers,
        ),
        override_output_path=plan.request.output_root.rstrip("/"),
    )


def run_materialization_train(config: MaterializationTrainConfig) -> None:
    run_levanter_train_lm(config.train_on_pod)
    for checkpoint_path in config.destination_checkpoint_paths:
        assert_checkpoint_complete_for_model_type(
            checkpoint_path,
            model_type=config.model_type,
            num_layers=config.num_layers,
        )


def load_source_train_config(model: DelphiModel) -> TrainLmConfig:
    executor_info_path = join_path(model.gcs_run_root, EXECUTOR_INFO_FILENAME)
    try:
        with fsspec.open(executor_info_path, "r", encoding="utf-8") as f:
            executor_info = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read source executor info at {executor_info_path!r}. "
            "This helper requires GCS read access to the canonical Delphi run root."
        ) from exc
    return decode_train_config_from_executor_info(executor_info, source_path=executor_info_path)


def decode_train_config_from_executor_info(executor_info: dict[str, Any], *, source_path: str) -> TrainLmConfig:
    train_config_dict = find_train_config_dict(executor_info)
    if train_config_dict is None:
        raise ValueError(f"No TrainLmConfig payload found in executor info {source_path!r}")
    normalized = normalize_executor_train_config(train_config_dict)
    train_config = draccus.decode(TrainLmConfig, normalized)
    if not isinstance(train_config.model, Qwen3Config):
        raise ValueError(
            f"Decoded source TrainLmConfig from {source_path!r} as {type(train_config.model).__name__}; "
            "Delphi prefix materialization must use Qwen3Config so q_norm/k_norm arrays and optimizer state "
            "are preserved."
        )
    return train_config


def find_train_config_dict(executor_info: dict[str, Any]) -> dict[str, Any] | None:
    config = executor_info.get("config")
    if isinstance(config, dict):
        train_config = config.get("train_config")
        if isinstance(train_config, dict):
            return train_config
        if "trainer" in config and "optimizer" in config:
            return config

    steps = executor_info.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            found = find_train_config_dict(step)
            if found is not None:
                return found
    return None


def normalize_executor_train_config(train_config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(train_config)
    normalize_trainer_dict(normalized.get("trainer"))
    normalize_model_dict(normalized.get("model"))
    normalize_optimizer_dict(normalized.get("optimizer"))
    normalize_data_dict(normalized.get("data"))
    return normalized


def regionalize_readonly_marin_uris(value: Any, *, target_prefix: str | None) -> Any:
    if target_prefix is None:
        return value
    if isinstance(value, str):
        return regionalize_marin_uri(value, target_prefix=target_prefix)
    if isinstance(value, list):
        return [regionalize_readonly_marin_uris(item, target_prefix=target_prefix) for item in value]
    if isinstance(value, tuple):
        return tuple(regionalize_readonly_marin_uris(item, target_prefix=target_prefix) for item in value)
    if isinstance(value, dict):
        return {key: regionalize_readonly_marin_uris(item, target_prefix=target_prefix) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{
                field.name: regionalize_readonly_marin_uris(getattr(value, field.name), target_prefix=target_prefix)
                for field in dataclasses.fields(value)
            },
        )
    return value


def marin_prefix_for_output_root(output_root: str) -> str | None:
    parsed = urllib.parse.urlparse(output_root)
    if parsed.scheme != "gs" or not parsed.netloc.startswith("marin-"):
        return None
    return f"gs://{parsed.netloc}"


def regionalize_marin_uri(uri: str, *, target_prefix: str) -> str:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc.startswith("marin-"):
        return uri
    return f"{target_prefix}/{parsed.path.lstrip('/')}"


def mirror_marin_uri(uri: str) -> str:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc.startswith("marin-"):
        return uri
    return f"mirror://{parsed.path.lstrip('/')}"


def marin_relative_path(uri: str) -> str:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc.startswith("marin-"):
        raise ValueError(f"Expected a Marin GCS URI, got {uri!r}")
    return parsed.path.lstrip("/")


def normalize_trainer_dict(trainer: Any) -> None:
    if not isinstance(trainer, dict):
        return
    trainer.pop("ray", None)
    mp = trainer.get("mp")
    if isinstance(mp, dict):
        trainer["mp"] = (
            f"compute={dtype_name(mp.get('compute_dtype'))},"
            f"params={dtype_name(mp.get('param_dtype'))},"
            f"output={dtype_name(mp.get('output_dtype'))}"
        )

    checkpointer = trainer.get("checkpointer")
    if isinstance(checkpointer, dict) and isinstance(checkpointer.get("save_interval"), dict):
        checkpointer["save_interval"] = timedelta_to_string(checkpointer["save_interval"])

    ensure_choice_type(trainer.get("tracker"), "wandb")


def normalize_model_dict(model: Any) -> None:
    if not isinstance(model, dict):
        return
    ensure_choice_type(model, DELPHI_MODEL_TYPE)
    if model["type"] != DELPHI_MODEL_TYPE:
        raise ValueError(
            f"Delphi prefix materialization only supports model type {DELPHI_MODEL_TYPE!r}, " f"got {model['type']!r}."
        )
    rope = model.get("rope")
    if isinstance(rope, dict) and "type" not in rope:
        if {"low_freq_factor", "high_freq_factor", "original_max_position_embeddings"}.issubset(rope):
            rope["type"] = "llama3"
        else:
            rope["type"] = "default"


def normalize_optimizer_dict(optimizer: Any) -> None:
    if not isinstance(optimizer, dict) or "type" in optimizer:
        return
    if optimizer.get("skip_bad_steps") in (False, None):
        optimizer.pop("skip_bad_steps", None)
    optimizer["type"] = "adamH" if "adam_lr" in optimizer else "adam"


def normalize_data_dict(data: Any) -> None:
    if not isinstance(data, dict):
        return
    components = data.get("components")
    if not isinstance(components, dict):
        return
    for component in components.values():
        if not isinstance(component, dict):
            continue
        ensure_choice_type(component, "cached")
        normalize_dataset_format(component.get("format"))
        source = component.get("source")
        if isinstance(source, dict):
            ensure_choice_type(source, "hf" if "id" in source else "url")
            normalize_dataset_format(source.get("format"))


def normalize_dataset_format(format_config: Any) -> None:
    if not isinstance(format_config, dict) or "type" in format_config:
        return
    if "messages_field" in format_config:
        format_config["type"] = "chat"
    elif "input_ids_key" in format_config or "loss_weights_key" in format_config:
        format_config["type"] = "prebuilt"
    else:
        format_config["type"] = "text"


def ensure_choice_type(value: Any, choice_type: str) -> None:
    if isinstance(value, dict) and "type" not in value:
        value["type"] = choice_type


def dtype_name(value: Any) -> str:
    text = str(value)
    if "bfloat16" in text:
        return "bfloat16"
    if "float16" in text:
        return "float16"
    if "float32" in text:
        return "float32"
    if "float64" in text:
        return "float64"
    raise ValueError(f"Unsupported dtype encoding in executor info: {value!r}")


def timedelta_to_string(value: dict[str, Any]) -> str:
    delta = timedelta(
        days=int(value.get("days", 0)),
        seconds=int(value.get("seconds", 0)),
        microseconds=int(value.get("microseconds", 0)),
    )
    return f"{delta.total_seconds()}s"


def print_plan(plan: MaterializationPlan) -> None:
    print("Delphi prefix checkpoint materialization")
    print(f"  base:                     {plan.model.flops_key}")
    print(f"  original schedule length: {plan.original_num_train_steps:,} steps")
    print(
        f"  source step:              {plan.request.source_step:,} " f"({plan.source_percent:.2f}% of original schedule)"
    )
    print(
        f"  target step:              {plan.request.target_step:,} " f"({plan.target_percent:.2f}% of original schedule)"
    )
    print(f"  steps to train:           {plan.steps_to_train:,}")
    if plan.request.extra_target_steps:
        extras_str = ", ".join(
            f"{s:,} ({100*s/plan.original_num_train_steps:.2f}%)" for s in sorted(plan.request.extra_target_steps)
        )
        print(f"  extra target steps:       {extras_str}")
    print(f"  source checkpoint:        {plan.source_checkpoint_path}")
    training_load_path = display_training_load_path(plan.train_config.trainer.initialize_from)
    if training_load_path != plan.source_checkpoint_path:
        print(f"  training load path:       {training_load_path}")
    mirror_budget = training_load_mirror_budget(plan.train_config.trainer.initialize_from)
    if mirror_budget is not None:
        print(f"  source mirror budget:     {mirror_budget:g} GB")
    if len(plan.destination_checkpoint_paths) == 1:
        print(f"  destination checkpoint:   {plan.destination_checkpoint_path}")
    else:
        print("  destination checkpoints:")
        for dest in plan.destination_checkpoint_paths:
            print(f"                            {dest}")
    print(f"  temporary checkpoint dir: {plan.temporary_checkpoint_path}")
    print(f"  TPU:                      {plan.request.tpu}")
    print(f"  RAM:                      {plan.request.ram}")
    if plan.request.regions:
        print(f"  regions:                  {', '.join(plan.request.regions)}")
    print(f"  original Delphi root:     {plan.model.gcs_run_root}")
    print("  original Delphi root will not be modified")


def display_training_load_path(value: object) -> str:
    if isinstance(value, MirroredValue):
        inner = value.value
        if isinstance(inner, str):
            if inner.startswith("mirror://"):
                return inner
            return f"mirror://{inner}"
    return str(value)


def training_load_mirror_budget(value: object) -> float | None:
    if isinstance(value, MirroredValue):
        return value.budget_gb
    return None


def path_equal_or_nested(path: str, root: str) -> bool:
    normalized_path = path.rstrip("/")
    normalized_root = root.rstrip("/")
    return normalized_path == normalized_root or normalized_path.startswith(f"{normalized_root}/")


def path_exists_or_has_children(uri: str) -> bool:
    fs, path = fsspec.core.url_to_fs(uri)
    if fs.exists(path):
        return True
    try:
        return bool(fs.ls(path, detail=False))
    except (FileNotFoundError, OSError):
        return False


def join_path(base: str, *parts: str) -> str:
    return "/".join((base.rstrip("/"), *(part.strip("/") for part in parts)))


def executor_prefix_for_output_root(output_root: str) -> str:
    parsed = urllib.parse.urlparse(output_root)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return os.path.dirname(output_root.rstrip("/")) or "."


if __name__ == "__main__":
    raise SystemExit(main())
