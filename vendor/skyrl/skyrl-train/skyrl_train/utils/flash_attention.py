"""Optional FlashAttention bindings shared by model implementations."""

from typing import Never

try:
    from flash_attn import flash_attn_func
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis as flash_index_first_axis
    from flash_attn.bert_padding import pad_input as flash_pad_input
    from flash_attn.bert_padding import unpad_input as flash_unpad_input
except ImportError as error:
    FLASH_ATTN_IMPORT_ERROR: ImportError | None = error

    def _flash_attn_missing(*_args: object, **_kwargs: object) -> Never:
        raise ImportError(
            "flash-attn is not installed but a FlashAttention-only code path was invoked"
        ) from FLASH_ATTN_IMPORT_ERROR

    flash_attn_func = _flash_attn_missing
    flash_attn_varlen_func = _flash_attn_missing
    flash_index_first_axis = _flash_attn_missing
    flash_pad_input = _flash_attn_missing
    flash_unpad_input = _flash_attn_missing
else:
    FLASH_ATTN_IMPORT_ERROR = None
