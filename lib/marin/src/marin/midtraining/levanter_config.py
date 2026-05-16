# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render a complete Levanter ``TrainLmConfig`` YAML from a resolved spec.

Mode-specific init-section assembly is delegated to the mode object.
Eval cadence is auto-picked from the mode: CPT eval cadence denominates
against the CPT step count, cooldown denominates against the remaining
steps.
"""

from typing import Any

from marin.midtraining.data_manifest import DataCacheManifest
from marin.midtraining.modes import (
    CooldownMode,
    CptMode,
)
from marin.midtraining.spec import MidtrainSpec, ResolvedMidtrainSpec

PERMANENT_CHECKPOINTS_SUBDIR = "checkpoints"
HF_EXPORT_SUBDIR = "hf"


def render_train_lm_config(resolved: ResolvedMidtrainSpec) -> dict[str, Any]:
    """Produce the dict that serializes to a Levanter ``TrainLmConfig`` YAML."""
    spec = resolved.spec
    data_section = _render_data_section(resolved)
    rendered: dict[str, Any] = {
        "data": data_section,
        "trainer": _build_trainer_section(resolved),
        "model": dict(spec.model_config),
        "optimizer": dict(spec.optimizer_config),
        "train_seq_len": spec.seq_len,
        "hf_save_path": f"{spec.run.output_path}/{HF_EXPORT_SUBDIR}",
        "hf_save_steps": _steps_per_export(resolved),
    }
    rendered.update(spec.mode.render_init_section())
    return rendered


def _render_data_section(resolved: ResolvedMidtrainSpec) -> dict[str, Any]:
    """Emit the ``data:`` block verbatim if an override is set, else build it."""
    spec = resolved.spec
    if spec.data_section_override is not None:
        # Verbatim passthrough — preserves the val carve-out byte-for-byte.
        # The cell author is responsible for sourcing the dict from a known-
        # good reference (see experiments/midtrain_specs/data_sections/).
        return _deep_copy_jsonable(spec.data_section_override)
    assert resolved.data_manifest is not None
    return _build_data_section(spec, resolved.data_manifest)


def _deep_copy_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _deep_copy_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_deep_copy_jsonable(v) for v in value]
    return value


def render_train_lm_yaml(resolved: ResolvedMidtrainSpec) -> str:
    """Serialize the rendered config to YAML for ``--config <path>``.

    Coerces enum values to their underlying strings before ``yaml.safe_dump``
    so Levanter-side dataclass enums (e.g. ``ActivationFunctionEnum.silu``)
    that landed in ``model_config`` via ``dataclasses.asdict`` round-trip
    cleanly. Levanter parses the YAML back through draccus, which accepts
    the string form.
    """
    import yaml

    return yaml.safe_dump(
        _enum_safe(render_train_lm_config(resolved)),
        sort_keys=False,
        default_flow_style=False,
    )


def _enum_safe(value: Any) -> Any:
    import enum

    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _enum_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_safe(v) for v in value]
    return value


def _build_trainer_section(resolved: ResolvedMidtrainSpec) -> dict[str, Any]:
    spec = resolved.spec
    run = spec.run
    return {
        "id": run.run_id,
        "train_batch_size": spec.compute.batch_size,
        "num_train_steps": resolved.num_train_steps,
        "per_device_parallelism": spec.compute.per_device_parallelism,
        "steps_per_eval": _steps_per_eval(resolved),
        "require_accelerator": True,
        "mesh": {"batch_axis_name": "batch"},
        "checkpointer": {
            "base_path": f"{run.output_path}/{PERMANENT_CHECKPOINTS_SUBDIR}",
            "temporary_base_path": _temporary_checkpoint_base_path(run.output_path, run.output_region),
            "save_interval": _parse_save_interval(spec.temp_save_interval),
            "append_run_id_to_base_path": False,
            "keep": [{"every": _steps_per_export(resolved)}],
        },
        "tracker": {
            "type": "wandb",
            "project": run.wandb_project,
            "entity": run.wandb_entity,
            "id": run.run_id,
            "resume": "allow",
            "tags": list(_wandb_tags(resolved)),
        },
    }


def _parse_save_interval(value: str) -> dict[str, int]:
    """Parse ``"10m"`` / ``"600s"`` / ``"1h"`` into the dict form draccus expects for ``timedelta``."""
    value = value.strip().lower()
    if value.endswith("h"):
        return {"seconds": int(value[:-1]) * 3600}
    if value.endswith("m"):
        return {"seconds": int(value[:-1]) * 60}
    if value.endswith("s"):
        return {"seconds": int(value[:-1])}
    raise ValueError(f"Unrecognized save_interval format {value!r}; use Ns / Nm / Nh")


def _build_data_section(spec: MidtrainSpec, manifest: DataCacheManifest) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {}
    for component in manifest.components:
        components[component.logical_name] = {
            "type": "lm_cached",
            "cache_dir": component.cache_path,
            "tokenizer": component.tokenizer.hf_repo,
        }
    return {
        "type": "mixture",
        "configs": components,
        "train_weights": dict(manifest.weights),
        "stop_strategy": "restart",
        "tokenizer": spec.tokenizer.hf_repo,
        "auto_build_caches": False,
        "shuffle_before_trainval_split": manifest.shuffle_before_trainval_split,
    }


def _steps_per_eval(resolved: ResolvedMidtrainSpec) -> int:
    """Eval cadence: ~``eval_target_points`` evals over the basis the mode picks."""
    spec = resolved.spec
    basis = _eval_basis_steps(resolved)
    target_steps = max(1, basis // max(1, spec.eval_target_points))
    eval_steps = min(spec.eval_max_steps, max(spec.eval_min_steps, target_steps))
    return min(eval_steps, resolved.num_train_steps)


def _eval_basis_steps(resolved: ResolvedMidtrainSpec) -> int:
    """CPT denominates against its own step count; cooldown against the remaining tail."""
    spec = resolved.spec
    if isinstance(spec.mode, CptMode):
        return resolved.num_train_steps
    assert isinstance(spec.mode, CooldownMode)
    return spec.base.num_train_steps - spec.mode.resume.resume_step


def _steps_per_export(resolved: ResolvedMidtrainSpec) -> int:
    spec = resolved.spec
    interval = int(resolved.num_train_steps * spec.permanent_fraction)
    return max(spec.min_permanent_steps, min(resolved.num_train_steps, interval))


def _temporary_checkpoint_base_path(output_path: str, region: str) -> str:
    sanitized_output = output_path.removeprefix("gs://").strip("/").replace("/", "_")
    return f"gs://marin-{region}/tmp/ttl=14d/checkpoints-temp/{sanitized_output}/{PERMANENT_CHECKPOINTS_SUBDIR}"


def _wandb_tags(resolved: ResolvedMidtrainSpec) -> tuple[str, ...]:
    spec = resolved.spec
    tags: list[str] = [
        f"mode:{spec.mode.kind}",
        f"base:{spec.base.flops_key}",
        f"tpu:{spec.compute.tpu_type}",
        f"region:{spec.run.output_region}",
        f"attempt:{spec.run.attempt:03d}",
    ]
    if resolved.resolved_budget is not None:
        budget = resolved.resolved_budget
        tags.extend(
            (
                f"budget_kind:{budget.policy.kind.value}",
                f"requested_tokens:{budget.requested_tokens}",
                f"actual_tokens:{budget.actual_tokens}",
                f"pretrain_fraction_actual:{budget.pretrain_fraction_actual:.6f}",
                f"num_train_steps:{budget.num_train_steps}",
                f"batch_size:{budget.batch_size}",
                f"seq_len:{budget.seq_len}",
                f"budget_label:{budget.label}",
            )
        )
    if isinstance(spec.mode, CooldownMode):
        tags.append(f"resume_step:{spec.mode.resume.resume_step}")
        if spec.mode.stop_step_override is not None:
            tags.append(f"stop_step:{spec.mode.stop_step_override}")
    if isinstance(spec.mode, CptMode) and spec.mode.init.checkpoint_override is not None:
        tags.append(f"checkpoint_override:{spec.mode.init.checkpoint_override.run_name_suffix.lstrip('-')}")
    tags.extend(spec.extra_tags)
    return tuple(tags)
