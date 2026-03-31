# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Compatibility wrapper for legacy LoRA-DPO configs.

The canonical DPO entrypoint is now levanter.main.train_dpo. This module keeps
the old LoRA-DPO config shape working by translating it into the canonical
TrainDpoConfig and forwarding into train_dpo.main.
"""

from dataclasses import dataclass, field
from typing import Optional

from levanter.adaptation import LoraAdaptationConfig
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.data.text import PreferenceLmDataConfig
from levanter.dpo import ReferenceEvalCacheConfig
from levanter.lora import LoraConfig
from levanter.main.train_dpo import AdapterBaseReferenceConfig, TrainDpoConfig, main as train_dpo_main
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.trainer import TrainerConfig


@dataclass
class LoraDpoConfig:
    data: PreferenceLmDataConfig = field(default_factory=PreferenceLmDataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)

    initialize_from_hf: str = ""
    beta: float = 0.1
    train_seq_len: int | None = None
    use_hf_model_config: bool = False
    trust_remote_code: bool = False
    validation_split_fraction: float | None = 0.1
    reference_eval_cache: ReferenceEvalCacheConfig = field(default_factory=ReferenceEvalCacheConfig)

    peft_save_path: Optional[str] = None
    peft_hf_upload: bool | str = False
    merged_hf_save_path: Optional[str] = None
    merged_hf_upload: Optional[str] = None
    hf_save_steps: int | None = 10000

    data_seed: Optional[int] = None


def _translate_legacy_lora_dpo_config(config: LoraDpoConfig) -> TrainDpoConfig:
    if not config.initialize_from_hf:
        raise ValueError("initialize_from_hf must be provided for LoRA-DPO training.")

    converter = HFCheckpointConverter.from_hf(config.initialize_from_hf, trust_remote_code=config.trust_remote_code)
    model_config = converter.default_config
    if config.use_hf_model_config:
        model_config = converter.config_from_hf_config(converter.default_hf_config)

    adapter = LoraAdaptationConfig(
        target_modules=config.lora.target_modules,
        r=config.lora.r,
        alpha=config.lora.alpha,
        dropout=config.lora.dropout,
        zero_init_b=config.lora.zero_init_b,
        exclude_modules=config.lora.exclude_modules,
    )

    return TrainDpoConfig(
        data=config.data,
        trainer=config.trainer,
        model=model_config,
        train_seq_len=config.train_seq_len,
        optimizer=config.optimizer,
        reference=AdapterBaseReferenceConfig(),
        adapter=adapter,
        beta=config.beta,
        initialize_from_hf=config.initialize_from_hf,
        use_hf_model_config=config.use_hf_model_config,
        validation_split_fraction=config.validation_split_fraction,
        reference_eval_cache=config.reference_eval_cache,
        hf_save_steps=config.hf_save_steps,
        peft_save_path=config.peft_save_path,
        peft_hf_upload=config.peft_hf_upload,
        merged_hf_save_path=config.merged_hf_save_path,
        merged_hf_upload=config.merged_hf_upload,
        data_seed=config.data_seed,
    )


def main(config: LoraDpoConfig):
    train_dpo_main(_translate_legacy_lora_dpo_config(config))


if __name__ == "__main__":
    import levanter

    levanter.config.main(main)()
