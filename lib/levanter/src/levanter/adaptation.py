# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import logging
import os
from dataclasses import dataclass

import draccus
import jax.numpy as jnp
from haliax.partitioning import named_jit

from levanter.compat.hf_checkpoints import (
    GenerationConfigDict,
    HFCheckpointConverter,
    RepoRef,
    save_hf_checkpoint_callback,
)
from levanter.lora import (
    LoraConfig,
    lora_trainable_params_filter,
    loraize,
    save_merged_hf_checkpoint_callback,
    save_peft_checkpoint_callback,
    unwrap_lora_modules,
)
from levanter.models.lm_model import LmHeadModel
from levanter.utils.types import FilterTree


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptationExportConfig:
    hf_save_path: str | None = None
    hf_upload: bool | str | RepoRef = False
    hf_save_steps: int | None = None
    hf_save_dtype: str | None = None
    generation_config: GenerationConfigDict | None = None

    peft_save_path: str | None = None
    peft_hf_upload: bool | str | RepoRef = False
    merged_hf_save_path: str | None = None
    merged_hf_upload: str | RepoRef | None = None


class AdaptationConfig(abc.ABC, draccus.ChoiceRegistry):
    @classmethod
    def default_choice_name(cls) -> str | None:
        return "none"

    @abc.abstractmethod
    def apply(self, model, *, key, axis_mapping=None):
        raise NotImplementedError

    @abc.abstractmethod
    def trainable_filter(self, policy_model) -> FilterTree:
        raise NotImplementedError

    @abc.abstractmethod
    def base_model_view(self, policy_model) -> LmHeadModel | None:
        raise NotImplementedError

    @abc.abstractmethod
    def install_export_hooks(
        self,
        *,
        trainer,
        converter: HFCheckpointConverter | None,
        tokenizer,
        export: AdaptationExportConfig,
    ) -> None:
        raise NotImplementedError


def _expanded_export_path(base_path: str, trainer) -> str:
    if trainer.config.checkpointer is not None and trainer.config.checkpointer.append_run_id_to_base_path:
        return os.path.join(base_path, trainer.run_id)
    return base_path


def _parse_hf_save_dtype(hf_save_dtype: str | None) -> jnp.dtype | None:
    if hf_save_dtype is None:
        return None

    try:
        return jnp.dtype(hf_save_dtype)
    except TypeError:
        logger.warning("Invalid hf_save_dtype: %s. Defaulting to None.", hf_save_dtype)
        return None


@AdaptationConfig.register_subclass("none")
@dataclass(frozen=True)
class NoAdaptationConfig(AdaptationConfig):
    def apply(self, model, *, key, axis_mapping=None):
        del key, axis_mapping
        return model

    def trainable_filter(self, policy_model) -> FilterTree:
        del policy_model
        return True

    def base_model_view(self, policy_model) -> LmHeadModel | None:
        del policy_model
        return None

    def install_export_hooks(
        self,
        *,
        trainer,
        converter: HFCheckpointConverter | None,
        tokenizer,
        export: AdaptationExportConfig,
    ) -> None:
        del tokenizer

        if export.peft_save_path is not None or export.merged_hf_save_path is not None:
            raise ValueError("peft_save_path and merged_hf_save_path require adapter.type: lora.")

        if export.hf_save_path is None or export.hf_save_steps is None:
            return

        if converter is None:
            raise ValueError("hf_save_path requires a HF-compatible model configuration.")

        full_save_path = _expanded_export_path(export.hf_save_path, trainer)
        save_dtype = _parse_hf_save_dtype(export.hf_save_dtype)
        trainer.add_hook(
            save_hf_checkpoint_callback(
                full_save_path,
                converter,
                upload_to_hf=export.hf_upload,
                save_dtype=save_dtype,
                generation_config=export.generation_config,
            ),
            every=export.hf_save_steps,
        )


@AdaptationConfig.register_subclass("lora")
@dataclass(frozen=True)
class LoraAdaptationConfig(LoraConfig, AdaptationConfig):
    def apply(self, model, *, key, axis_mapping=None):
        if axis_mapping is None:
            return loraize(model, self, key=key)

        @named_jit(axis_resources=axis_mapping, donate_args=(True,))
        def loraize_model(inner_model):
            return loraize(inner_model, self, key=key)

        return loraize_model(model)

    def trainable_filter(self, policy_model) -> FilterTree:
        return lora_trainable_params_filter(policy_model)

    def base_model_view(self, policy_model) -> LmHeadModel | None:
        return unwrap_lora_modules(policy_model)

    def install_export_hooks(
        self,
        *,
        trainer,
        converter: HFCheckpointConverter | None,
        tokenizer,
        export: AdaptationExportConfig,
    ) -> None:
        if export.hf_save_path is not None:
            raise ValueError("adapter.type: lora does not support hf_save_path. Use merged_hf_save_path instead.")
        if export.hf_save_dtype is not None:
            raise ValueError("adapter.type: lora does not support hf_save_dtype. Use merged_hf_save_path instead.")

        if export.peft_save_path is None and export.merged_hf_save_path is None:
            return

        if converter is None:
            raise ValueError("LoRA checkpoint export requires a HF-compatible model configuration.")
        if export.hf_save_steps is None:
            raise ValueError("LoRA checkpoint export requires hf_save_steps to be set.")

        if export.peft_save_path is not None:
            full_save_path = _expanded_export_path(export.peft_save_path, trainer)
            trainer.add_hook(
                save_peft_checkpoint_callback(
                    full_save_path,
                    self,
                    converter.reference_checkpoint,
                    tokenizer,
                    export.peft_hf_upload,
                ),
                every=export.hf_save_steps,
            )

        if export.merged_hf_save_path is not None:
            full_save_path = _expanded_export_path(export.merged_hf_save_path, trainer)
            trainer.add_hook(
                save_merged_hf_checkpoint_callback(
                    full_save_path,
                    converter,
                    export.merged_hf_upload,
                    generation_config=export.generation_config,
                ),
                every=export.hf_save_steps,
            )
