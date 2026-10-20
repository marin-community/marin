from omegaconf import DictConfig

from skyrl_train.callbacks.types import CHECKPOINT_CALLBACK_TYPE, HF_MODEL_SAVE_CALLBACK_TYPE


def has_explicit_callbacks(cfg: DictConfig) -> bool:
    callbacks = cfg.trainer.get("callbacks")
    return callbacks is not None and len(callbacks) > 0


def interval_hf_export_enabled(cfg: DictConfig) -> bool:
    """Return whether legacy interval settings explicitly enable HF export."""
    destination_configured = bool(cfg.trainer.get("export_hf_artifact")) or bool(cfg.trainer.get("hf_hub_repo_id"))
    return destination_configured and int(cfg.trainer.get("hf_save_interval", -1)) > 0


def validate_hf_export_config(cfg: DictConfig) -> None:
    """Validate checkpoint and model-save callback interval alignment."""
    callbacks = cfg.trainer.get("callbacks")
    if has_explicit_callbacks(cfg):

        def callback_intervals(callback_type: str) -> list[int]:
            return [
                int(callback.get("save_steps", -1))
                for callback in callbacks
                if callback.get("type") == callback_type and int(callback.get("save_steps", -1)) > 0
            ]

        checkpoint_intervals = callback_intervals(CHECKPOINT_CALLBACK_TYPE)
        export_intervals = callback_intervals(HF_MODEL_SAVE_CALLBACK_TYPE)
    else:
        checkpoint_interval = int(cfg.trainer.get("ckpt_interval", -1))
        checkpoint_intervals = [checkpoint_interval] if checkpoint_interval > 0 else []
        export_intervals = [int(cfg.trainer.hf_save_interval)] if interval_hf_export_enabled(cfg) else []

    if cfg.trainer.get("export_hf_artifact", False) and not export_intervals:
        raise ValueError("Terminal HF artifact export requires an enabled HF model-save callback")

    for export_interval in export_intervals:
        if not checkpoint_intervals or not any(
            export_interval % checkpoint_interval == 0 for checkpoint_interval in checkpoint_intervals
        ):
            raise ValueError(
                f"HF export interval {export_interval} must be a multiple of trainer.ckpt_interval "
                f"or an explicit checkpoint callback interval; found {checkpoint_intervals or 'none'}"
            )
