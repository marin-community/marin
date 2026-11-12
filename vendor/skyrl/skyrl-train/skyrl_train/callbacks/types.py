from enum import StrEnum


class CallbackErrorBehavior(StrEnum):
    RAISE = "raise"
    WARN = "warn"
    IGNORE = "ignore"


CHECKPOINT_CALLBACK_TYPE = "checkpoint"
HF_MODEL_SAVE_CALLBACK_TYPE = "hf_model_save"
