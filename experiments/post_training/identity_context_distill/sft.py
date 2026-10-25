# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task 310 configuration only: native-format SFT, with no submission side effects."""

import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import asdict, replace

import jax
import jax.numpy as jnp
import jmp
from levanter.adaptor import FrozenParameterAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.models.snowball import (
    GRUG_MOE_BANNED_CONFIG_ALIASES,
    GRUG_MOE_CANONICAL_CONFIG_FIELDS,
    GrugMoeHfConfig,
    SnowballConfig,
)
from levanter.optim.config import AdamConfig
from levanter.trainer import TrainerConfig
from levanter.trainer_state import FrozenParameterStorage, TrainerState
from levanter.utils.mesh import MeshConfig

BASE_MODEL = "s3://marin-rno2a/marin/models/marin-snowball-67b-a2b-v0.1-adc0078a/"
NATIVE_TEMPLATE_SHA256 = "b3f20bc21498407f2815dd1a6c26150192926ccf04f6324ca40a1209bff7d1de"
SEQUENCE_LENGTH = 8192
GLOBAL_BATCH_SIZE = 8
logger = logging.getLogger(__name__)

EXPORT_CONFIG_ALIASES = {
    attribute: published for published, attribute in GRUG_MOE_CANONICAL_CONFIG_FIELDS if attribute != published
} | {
    "attention_head_dim": "head_dim",
    "intermediate_size": "moe_intermediate_size",
    "num_local_experts": "num_experts",
}


def source_export_overrides(source_config: dict) -> dict:
    """Preserve every source field, using only the declared single-name aliases."""
    source_model_config(source_config)
    overrides = {}
    for field, value in source_config.items():
        published = EXPORT_CONFIG_ALIASES.get(field, field)
        if published in overrides and overrides[published] != value:
            raise ValueError(f"Conflicting export aliases for {published}")
        overrides[published] = deepcopy(value)
    return overrides


def check_export_metadata(source_config: dict, exported_config: dict) -> None:
    """Check parsed config.json read from the written export, not predicted metadata.

    Additional writer fields are allowed. Every source field must survive unchanged
    under its declared canonical spelling; legacy aliases in the export fail.
    Tensor dtypes, tokenizer files and generation_config.json need separate checks.
    """
    banned = GRUG_MOE_BANNED_CONFIG_ALIASES.intersection(exported_config)
    if banned:
        raise ValueError(f"Export contains banned aliases: {sorted(banned)}")
    for field, value in source_export_overrides(source_config).items():
        if field not in exported_config:
            raise ValueError(f"Export missing source field: {field}")
        if exported_config[field] != value:
            raise ValueError(f"Export changed source field: {field}")


def source_model_config(source_config: dict) -> SnowballConfig:
    """Derive the model from explicit native metadata, rejecting ambiguous aliases."""
    for published, attribute in GRUG_MOE_CANONICAL_CONFIG_FIELDS:
        if published in source_config and attribute in source_config:
            if source_config[published] != source_config[attribute]:
                raise ValueError(f"Conflicting source config aliases: {published}, {attribute}")
    for field in ("qk_mult", "rope_theta", "vocab_size"):
        if field not in source_config:
            raise ValueError(f"Native source must explicitly specify {field}")
    if not {"max_seq_len", "max_position_embeddings"} & source_config.keys():
        raise ValueError("Native source must explicitly specify its model context")
    for field, supported in (
        ("global_every", 4),
        ("qk_mult_long_scale", 1.0),
        ("rope_fused", False),
        ("rope_scaling", None),
        ("global_kv_heads", None),
        ("local_kv_heads", None),
    ):
        if field in source_config and source_config[field] != supported:
            raise ValueError(f"Unsupported source {field}; native Snowball implementation differs")
    model = SnowballConfig.from_hf_config(GrugMoeHfConfig(**source_config))
    if model.max_seq_len < SEQUENCE_LENGTH or model.num_layers != 26:
        raise ValueError("Unexpected source context or router-layer count")
    return replace(model, reference_checkpoint=BASE_MODEL, tokenizer=BASE_MODEL)


def source_generation_eos(source_config: dict, generation_config: dict) -> list[int]:
    """Require the approved source EOS semantics; never invent or silently repair them."""
    if source_config.get("eos_token_id") != 128001 or source_config.get("pad_token_id") != 128001:
        raise ValueError("Source model EOS/PAD changed; review required")
    if source_config.get("bos_token_id") != 128000 or generation_config.get("bos_token_id") != 128000:
        raise ValueError("Source BOS changed; review required")
    if generation_config.get("eos_token_id") != [128001, 128009]:
        raise ValueError("Source generation EOS changed; review required")
    return list(generation_config["eos_token_id"])


def source_metadata_receipt(config_bytes: bytes, generation_bytes: bytes) -> dict:
    """Validate metadata and receipt exact bytes; the worker must recheck regional bytes."""
    source = json.loads(config_bytes)
    generation = json.loads(generation_bytes)
    model = source_model_config(source)
    eos = source_generation_eos(source, generation)
    return {
        "schema_version": 2,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "generation_config_sha256": hashlib.sha256(generation_bytes).hexdigest(),
        "model_config": asdict(model),
        "training_sequence_length": SEQUENCE_LENGTH,
        "generation_eos_token_ids": eos,
    }


def check_source_metadata_receipt(expected: dict, config_bytes: bytes, generation_bytes: bytes) -> None:
    """Fail closed if the worker's regional metadata differs from the approved receipt."""
    if source_metadata_receipt(config_bytes, generation_bytes) != expected:
        raise ValueError("Regional source metadata differs from the approved receipt")


def check_runtime_model(model: SnowballConfig, source_config: dict) -> None:
    """Allow only the runtime router-dot exception to differ from the source model.

    This checks architecture, not authorization of the precision exception. The
    pre-update routing gate must separately verify the measured trigger.
    """
    expected = source_model_config(source_config)
    actual = replace(
        model,
        reference_checkpoint=BASE_MODEL,
        tokenizer=BASE_MODEL,
        router_dot_fp32=expected.router_dot_fp32,
    )
    if actual != expected:
        raise ValueError("Runtime model architecture differs from the native source")


def check_runtime_recipe(config: TrainLmConfig, *, source_config: dict, source_generation_config: dict) -> None:
    """Check the recipe after distributed JAX initialization, before loading weights."""
    trainer = config.trainer
    assert jax.device_count() == 32
    assert trainer.num_slices == 4
    assert trainer.mesh.dcn_axes == {"replica_dcn": 1, "context": 4}
    assert trainer.train_batch_size == trainer.microbatch_size == GLOBAL_BATCH_SIZE
    assert trainer.compute_axis_mapping["batch"] == ("replica_dcn", "data", "expert")
    assert trainer.compute_axis_mapping["position"] == "context"
    assert isinstance(config.model, SnowballConfig)
    assert config.train_seq_len == SEQUENCE_LENGTH
    check_runtime_model(config.model, source_config)
    if config.model.router_dot_fp32:
        assert trainer.compute_dtype_overrides == {"router_bias": "float32"}
    assert config.use_hf_model_config is True
    assert config.hf_generation_eos_token_ids == source_generation_eos(source_config, source_generation_config)
    assert trainer.frozen_parameter_storage == FrozenParameterStorage.NATIVE
    assert trainer.mp.param_dtype == jnp.float32 and trainer.mp.compute_dtype == jnp.bfloat16
    assert config.initialize_from_hf == config.data.tokenizer == BASE_MODEL
    assert isinstance(config.adapter, FrozenParameterAdaptorConfig)
    assert config.adapter.frozen_field_names == ("router_bias",)
    assert config.adapter.hf_config_overrides == source_export_overrides(source_config)
    assert config.adapter.hf_save_dtype_overrides == {
        f"model.layers.{layer}.mlp.router.bias": "float32" for layer in range(26)
    }
    assert config.hf_save_dtype == "bfloat16"
    component = config.data.components["identity"]
    assert isinstance(component, DatasetComponent)
    assert isinstance(component.format, ChatLmDatasetFormat)
    assert component.pack is False and component.format.pack is False
    assert component.format.slice_strategy == "raise" and component.format.mask_user_turns
    logger.info(
        "Identity SFT source=%s template_sha256=%s native_restore=%s storage=native "
        "frozen_router_biases=26 master=f32 compute=bf16 export=bf16+router_f32 "
        "global_batch=8 microbatch=8 context=8192 pack=false slice=raise mesh=%s",
        BASE_MODEL,
        NATIVE_TEMPLATE_SHA256,
        trainer.load_checkpoint_path,
        trainer.mesh,
    )


def check_restored_progress(config: TrainLmConfig, state: TrainerState, *, completed_updates: int) -> None:
    """Check loaded progress before constructing the resumed iterator, not an HF reset."""
    assert int(state.step) == completed_updates
    counts = [
        int(value)
        for path, value in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]
        if path and isinstance(path[-1], jax.tree_util.GetAttrKey) and path[-1].name == "count"
    ]
    assert counts and all(count == completed_updates for count in counts)
    offset = config.trainer.batch_schedule.global_data_offset_by_step(int(state.step))
    assert offset == completed_updates * GLOBAL_BATCH_SIZE
    logger.info("Restored completed_updates=%d optimizer_counts=%s next_data_offset=%d", state.step, counts, offset)


def identity_sft_config(
    *,
    dataset_uri: str,
    cache_uri: str,
    output_uri: str,
    run_id: str,
    seed: int,
    hosts: int,
    source_config: dict,
    source_generation_config: dict,
    stop_after_updates: int = 16,
    resume_checkpoint: str | None = None,
) -> TrainLmConfig:
    """Bind the identity recipe without reading data, weights, or launching a job.

    Args:
        dataset_uri: QC-approved native messages JSONL, including per-row template kwargs.
        cache_uri: Separately prepared cache for this exact dataset and native tokenizer.
        output_uri: Fresh run root, or the same root for an interrupted run.
        run_id: Stable tracking identity.
        seed: Fixed model/data seed, unchanged across continuation.
        hosts: Physical host count: four H100 hosts, or one slice for CPU tests.
        source_config: Parsed config.json from the pinned native source, not Snowball defaults.
        source_generation_config: Parsed source generation_config.json with the approved EOS list.
        stop_after_updates: Absolute training horizon, initially 16, then 64 after the gate.
        resume_checkpoint: Committed native step directory. Required for continuation.

    Returns:
        Standard TrainLmConfig. Continuation restores full state with load_checkpoint_path,
        leaving the pinned HF source available to reconstruct omitted frozen parameters.
        The literal trainer.initialize_from option cannot coexist with initialize_from_hf.
        Four GPU hosts place context across DCN slices, with data and expert within each host.
        Physical cross-node placement and the BF16-forward gate still require validation.
    """
    if type(hosts) is not int or hosts not in (1, 4):
        raise ValueError("The precision recipe requires hosts=4, or hosts=1 for CPU tests")
    if stop_after_updates not in (16, 64):
        raise ValueError("The gated recipe supports absolute horizons of 16 or 64 updates")
    if stop_after_updates == 64 and resume_checkpoint is None:
        raise ValueError("64 updates requires a committed native checkpoint; no HF reset")
    if not all((dataset_uri, cache_uri, output_uri, run_id)):
        raise ValueError("Dataset, cache, output, and run identity must be explicit")

    output_uri = output_uri.rstrip("/")
    mesh = MeshConfig(
        axes={"data": 2, "expert": 4, "replica": 1, "model": 1} | ({"context": 4} if hosts == 1 else {}),
        dcn_axes={"replica_dcn": 1} | ({"context": 4} if hosts == 4 else {}),
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context", "vocab": "model"},
        param_mapping={"embed": ["data", "context"]},
    )
    return TrainLmConfig(
        data=LmDataConfig(
            tokenizer=BASE_MODEL,
            cache_dir=cache_uri,
            auto_build_caches=False,
            components={
                "identity": DatasetComponent(
                    source=UrlDatasetSourceConfig(train_urls=[dataset_uri]),
                    cache_dir=cache_uri,
                    format=ChatLmDatasetFormat(pack=False, slice_strategy="raise", mask_user_turns=True),
                    pack=False,
                )
            },
            train_weights={"identity": 1.0},
        ),
        data_seed=seed,
        model=source_model_config(source_config),
        train_seq_len=SEQUENCE_LENGTH,
        initialize_from_hf=BASE_MODEL,
        use_hf_model_config=True,
        adapter=FrozenParameterAdaptorConfig(
            frozen_field_names=("router_bias",),
            hf_save_dtype_overrides={f"model.layers.{layer}.mlp.router.bias": "float32" for layer in range(26)},
            hf_config_overrides=source_export_overrides(source_config),
        ),
        trainer=TrainerConfig(
            id=run_id,
            seed=seed,
            mesh=mesh,
            use_explicit_mesh_axes=True,
            mp=jmp.get_policy("p=f32,c=bfloat16,o=f32"),
            frozen_parameter_storage=FrozenParameterStorage.NATIVE,
            train_batch_size=GLOBAL_BATCH_SIZE,
            per_device_parallelism=1,
            num_train_steps=stop_after_updates,
            load_checkpoint=True if resume_checkpoint is not None else None,
            load_checkpoint_path=resume_checkpoint,
            checkpointer=CheckpointerConfig(
                base_path=f"{output_uri}/checkpoints",
                append_run_id_to_base_path=False,
                save_interval=None,
                keep=None,
                delete_old_temp_checkpoints=False,
            ),
        ),
        optimizer=AdamConfig(
            learning_rate=2e-6,
            lr_schedule="constant",
            warmup=2,
            weight_decay=0.0,
            max_grad_norm=0.5,
        ),
        hf_save_path=f"{output_uri}/hf",
        hf_save_steps=stop_after_updates,
        hf_save_dtype="bfloat16",
        hf_generation_eos_token_ids=source_generation_eos(source_config, source_generation_config),
    )
