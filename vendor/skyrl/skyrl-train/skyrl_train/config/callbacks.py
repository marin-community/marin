from omegaconf import DictConfig

from skyrl_train.callbacks.types import CHECKPOINT_CALLBACK_TYPE, HF_MODEL_SAVE_CALLBACK_TYPE


def has_explicit_callbacks(cfg: DictConfig) -> bool:
    callbacks = cfg.trainer.get("callbacks")
    return callbacks is not None and len(callbacks) > 0


def hf_export_destination_configured(cfg: DictConfig) -> bool:
    """Return whether the run publishes an HF artifact or Hub model."""
    return bool(cfg.trainer.get("export_hf_artifact")) or bool(cfg.trainer.get("hf_hub_repo_id"))


def validate_hf_export_config(cfg: DictConfig) -> None:
    """Require every requested HF export to have a saved checkpoint to convert.

    An HF model-save callback always requests the terminal export. A positive
    interval also requests exports at intermediate checkpoints, which pruning keeps
    until each export completes.
    """
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
        export_requested = any(callback.get("type") == HF_MODEL_SAVE_CALLBACK_TYPE for callback in callbacks)
    else:
        checkpoint_interval = int(cfg.trainer.get("ckpt_interval", -1))
        checkpoint_intervals = [checkpoint_interval] if checkpoint_interval > 0 else []
        export_requested = hf_export_destination_configured(cfg)
        export_interval = int(cfg.trainer.get("hf_save_interval", -1))
        export_intervals = [export_interval] if export_requested and export_interval > 0 else []

    if cfg.trainer.get("export_hf_artifact", False) and not export_requested:
        raise ValueError("Terminal HF artifact export requires an HF model-save callback")
    if export_requested and not checkpoint_intervals:
        raise ValueError("HF export requires checkpoint saving to produce the terminal checkpoint")

    for export_interval in export_intervals:
        if not any(export_interval % checkpoint_interval == 0 for checkpoint_interval in checkpoint_intervals):
            raise ValueError(
                f"HF export interval {export_interval} must be a multiple of trainer.ckpt_interval "
                f"or an explicit checkpoint callback interval; found {checkpoint_intervals}"
            )
