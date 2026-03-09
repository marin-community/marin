# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import hashlib
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn, idctn


class JpegTokenizerFamily(StrEnum):
    """Supported tokenizer families for the initial comparison ladder."""

    BYTES = "bytes"
    SYMBOLS = "symbols"
    COEFFS = "coeffs"


class CoefficientTokenSource(StrEnum):
    """Available implementations for coefficient extraction."""

    REFERENCE = "reference"
    LIBJPEG = "libjpeg"


@dataclass(frozen=True)
class CanonicalJpegConfig:
    """Frozen V0 defaults for canonical JPEG preprocessing."""

    resolution: int = 256
    color_mode: str = "Y"
    resize_method: str = "bicubic_center_crop"
    jpeg_quality: int = 95
    progressive: bool = False
    strip_metadata: bool = True
    version: str = "v0"


@dataclass(frozen=True)
class CoefficientTokenizerConfig:
    """Frozen V0 defaults for the coefficient-stream baseline."""

    zigzag_coefficients: int = 4
    block_size: int = 8
    coefficient_bound: int = 2047
    quality: int = 95
    source: CoefficientTokenSource = CoefficientTokenSource.REFERENCE
    version: str = "v0"


V0_CANONICAL_JPEG_CONFIG = CanonicalJpegConfig()
V0_COEFFICIENT_CONFIG = CoefficientTokenizerConfig()


@dataclass(frozen=True)
class ByteWindowTokenizerConfig:
    """Frozen V0 defaults for fixed-length canonical JPEG byte windows."""

    window_size: int = 8192
    pad_token_id: int = 256
    append_eos: bool = True
    stride: int = 8192
    version: str = "v0"


V0_BYTE_WINDOW_CONFIG = ByteWindowTokenizerConfig()


@dataclass(frozen=True)
class WholeImageByteTokenizerConfig:
    """Frozen V0 defaults for whole-image canonical JPEG byte sequences."""

    eos_token_id: int = 256
    pad_token_id: int = 257
    append_eos: bool = True
    version: str = "v0"


V0_WHOLE_IMAGE_BYTE_CONFIG = WholeImageByteTokenizerConfig()


@dataclass(frozen=True)
class SymbolTokenizerConfig:
    """Frozen V0 defaults for the symbol-stream baseline."""

    dc_bound: int = 2047
    ac_bound: int = 1023
    quality: int = 95
    version: str = "v0"


@dataclass(frozen=True)
class CanonicalJpegRepresentation:
    """Canonical JPEG bytes plus the aligned luma plane used for reference tokenizers."""

    jpeg_bytes: bytes
    luma_plane: np.ndarray
    checksum: str


V0_SYMBOL_CONFIG = SymbolTokenizerConfig()

_JPEG_LUMA_QUANT_TABLE = np.asarray(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.int32,
)

_ZIGZAG_ORDER = np.asarray(
    [
        0,
        1,
        8,
        16,
        9,
        2,
        3,
        10,
        17,
        24,
        32,
        25,
        18,
        11,
        4,
        5,
        12,
        19,
        26,
        33,
        40,
        48,
        41,
        34,
        27,
        20,
        13,
        6,
        7,
        14,
        21,
        28,
        35,
        42,
        49,
        56,
        57,
        50,
        43,
        36,
        29,
        22,
        15,
        23,
        30,
        37,
        44,
        51,
        58,
        59,
        52,
        45,
        38,
        31,
        39,
        46,
        53,
        60,
        61,
        54,
        47,
        55,
        62,
        63,
    ],
    dtype=np.int32,
)


def stable_byte_checksum(data: bytes) -> str:
    """Return a stable checksum for canonical byte streams."""

    return hashlib.sha256(data).hexdigest()


def coefficient_vocab_size(config: CoefficientTokenizerConfig = V0_COEFFICIENT_CONFIG) -> int:
    """Return the explicit bounded coefficient vocabulary size."""

    return 2 * config.coefficient_bound + 1


def byte_window_vocab_size(config: ByteWindowTokenizerConfig = V0_BYTE_WINDOW_CONFIG) -> int:
    """Return the vocabulary size for byte windows with a reserved EOS/PAD token."""

    return max(config.pad_token_id + 1, 257)


def whole_image_byte_vocab_size(config: WholeImageByteTokenizerConfig = V0_WHOLE_IMAGE_BYTE_CONFIG) -> int:
    """Return the vocabulary size for whole-image bytes with distinct EOS and PAD ids."""

    return max(config.eos_token_id, config.pad_token_id) + 1


def symbol_vocab_size(config: SymbolTokenizerConfig = V0_SYMBOL_CONFIG) -> int:
    """Return the explicit bounded symbol vocabulary size."""

    dc_vocab = 2 * config.dc_bound + 1
    max_category = config.ac_bound.bit_length()
    per_run_vocab = sum(1 << size for size in range(1, max_category + 1))
    return 2 + dc_vocab + (16 * per_run_vocab)


def canonicalize_image(
    image: Image.Image,
    *,
    config: CanonicalJpegConfig = V0_CANONICAL_JPEG_CONFIG,
) -> CanonicalJpegRepresentation:
    """Resize, crop, convert, and re-encode an image into the frozen canonical JPEG form."""

    if config.color_mode != "Y":
        raise ValueError(f"Unsupported color_mode {config.color_mode!r}; only 'Y' is implemented in V0")
    if config.resize_method != "bicubic_center_crop":
        raise ValueError(
            f"Unsupported resize_method {config.resize_method!r}; only 'bicubic_center_crop' is implemented in V0"
        )

    source = image.copy()
    luma_image = ImageOps.fit(
        source.convert("L"),
        (config.resolution, config.resolution),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    if config.strip_metadata:
        luma_image = Image.fromarray(np.asarray(luma_image, dtype=np.uint8), mode="L")

    buffer = io.BytesIO()
    luma_image.save(
        buffer,
        format="JPEG",
        quality=config.jpeg_quality,
        progressive=config.progressive,
        optimize=False,
    )
    jpeg_bytes = buffer.getvalue()
    luma_plane = np.asarray(luma_image, dtype=np.uint8).copy()
    return CanonicalJpegRepresentation(
        jpeg_bytes=jpeg_bytes,
        luma_plane=luma_plane,
        checksum=stable_byte_checksum(jpeg_bytes),
    )


def encode_jpeg_bytes(canonical_jpeg: CanonicalJpegRepresentation | bytes) -> np.ndarray:
    """Encode canonical JPEG bytes as passthrough LM token ids."""

    if isinstance(canonical_jpeg, CanonicalJpegRepresentation):
        canonical_jpeg = canonical_jpeg.jpeg_bytes
    return np.frombuffer(canonical_jpeg, dtype=np.uint8).astype(np.int32, copy=False)


def window_byte_tokens(
    byte_tokens: np.ndarray,
    *,
    config: ByteWindowTokenizerConfig = V0_BYTE_WINDOW_CONFIG,
) -> list[np.ndarray]:
    """Split canonical JPEG bytes into fixed-length windows with EOS/PAD filling."""

    if config.window_size <= 0:
        raise ValueError(f"window_size must be positive, got {config.window_size}")
    if config.stride <= 0:
        raise ValueError(f"stride must be positive, got {config.stride}")
    if config.pad_token_id < 256:
        raise ValueError(f"pad_token_id must reserve a non-byte token id, got {config.pad_token_id}")

    byte_tokens = np.asarray(byte_tokens, dtype=np.int32)
    if byte_tokens.ndim != 1:
        raise ValueError(f"Expected rank-1 byte token array, got shape {byte_tokens.shape}")

    payload = byte_tokens
    if config.append_eos:
        payload = np.concatenate([payload, np.asarray([config.pad_token_id], dtype=np.int32)])

    windows: list[np.ndarray] = []
    for start in range(0, max(len(payload), 1), config.stride):
        chunk = payload[start : start + config.window_size]
        if len(chunk) < config.window_size:
            padded = np.full(config.window_size, config.pad_token_id, dtype=np.int32)
            padded[: len(chunk)] = chunk
            chunk = padded
        windows.append(np.asarray(chunk, dtype=np.int32))
        if start + config.window_size >= len(payload):
            break

    return windows


def whole_image_byte_length(
    byte_tokens: np.ndarray,
    *,
    config: WholeImageByteTokenizerConfig = V0_WHOLE_IMAGE_BYTE_CONFIG,
) -> int:
    """Return the whole-image byte length including EOS if configured."""

    return len(byte_tokens) + (1 if config.append_eos else 0)


def pad_whole_image_byte_tokens(
    byte_tokens: np.ndarray,
    *,
    seq_len: int,
    config: WholeImageByteTokenizerConfig = V0_WHOLE_IMAGE_BYTE_CONFIG,
) -> np.ndarray:
    """Right-pad one canonical JPEG byte stream to a fixed whole-image sequence length."""

    payload = np.asarray(byte_tokens, dtype=np.int32)
    if config.append_eos:
        payload = np.concatenate([payload, np.asarray([config.eos_token_id], dtype=np.int32)])
    if len(payload) > seq_len:
        raise ValueError(f"Byte payload length {len(payload)} exceeds seq_len {seq_len}")
    padded = np.full(seq_len, config.pad_token_id, dtype=np.int32)
    padded[: len(payload)] = payload
    return padded


def encode_jpeg_symbols(
    canonical_jpeg: CanonicalJpegRepresentation,
    *,
    config: SymbolTokenizerConfig = V0_SYMBOL_CONFIG,
) -> np.ndarray:
    """Encode a deterministic reference JPEG symbol stream derived from quantized luma blocks."""

    dc_offset = 2
    dc_vocab = 2 * config.dc_bound + 1
    ac_offset = dc_offset + dc_vocab
    max_category = config.ac_bound.bit_length()
    ac_offsets = _ac_symbol_offsets(max_category)
    eob_token = 0
    zrl_token = 1

    blocks = _quantized_luma_blocks(canonical_jpeg.luma_plane, quality=config.quality)
    zigzag = blocks.reshape(-1, 64)[:, _ZIGZAG_ORDER]
    tokens: list[int] = []
    previous_dc = 0

    for block in zigzag:
        dc_value = int(block[0])
        dc_delta = dc_value - previous_dc
        previous_dc = dc_value
        tokens.append(dc_offset + _encode_bounded_value(dc_delta, config.dc_bound, "dc_delta"))

        zero_run = 0
        for coeff in block[1:]:
            coeff_value = int(coeff)
            if coeff_value == 0:
                zero_run += 1
                if zero_run == 16:
                    tokens.append(zrl_token)
                    zero_run = 0
                continue

            magnitude_category = abs(coeff_value).bit_length()
            if magnitude_category == 0 or magnitude_category > max_category:
                raise ValueError(f"AC coefficient magnitude category {magnitude_category} exceeds configured bounds")

            token = (
                ac_offset
                + ac_offsets[(zero_run, magnitude_category)]
                + _encode_category_value(
                    coeff_value,
                    magnitude_category,
                    "ac_coefficient",
                )
            )
            tokens.append(token)
            zero_run = 0

        if zero_run > 0:
            tokens.append(eob_token)

    return np.asarray(tokens, dtype=np.int32)


def encode_dct_coeffs(
    canonical_jpeg: CanonicalJpegRepresentation,
    *,
    config: CoefficientTokenizerConfig = V0_COEFFICIENT_CONFIG,
) -> np.ndarray:
    """Encode bounded low-frequency coefficients from the configured block source."""

    blocks = quantized_luma_blocks(canonical_jpeg, config=config)
    zigzag = blocks.reshape(-1, 64)[:, _ZIGZAG_ORDER[: config.zigzag_coefficients]]
    encoded = [
        _encode_bounded_value(int(value), config.coefficient_bound, "coefficient") for value in zigzag.reshape(-1)
    ]
    return np.asarray(encoded, dtype=np.int32)


def quantized_luma_blocks(
    canonical_jpeg: CanonicalJpegRepresentation,
    *,
    config: CoefficientTokenizerConfig = V0_COEFFICIENT_CONFIG,
) -> np.ndarray:
    """Return raster-ordered quantized luma DCT blocks from the selected extractor."""

    if config.source == CoefficientTokenSource.REFERENCE:
        return _quantized_luma_blocks(canonical_jpeg.luma_plane, quality=config.quality)
    if config.source == CoefficientTokenSource.LIBJPEG:
        return _libjpeg_quantized_luma_blocks(canonical_jpeg)
    raise ValueError(f"Unsupported coefficient source {config.source!r}")


def reconstruct_luma_from_coeff_tokens(
    token_ids: np.ndarray,
    image_shape: tuple[int, int],
    *,
    config: CoefficientTokenizerConfig = V0_COEFFICIENT_CONFIG,
) -> np.ndarray:
    """Reconstruct a luma image from the bounded low-frequency coefficient tokens."""

    height, width = image_shape
    if height % config.block_size != 0 or width % config.block_size != 0:
        raise ValueError(f"image_shape={image_shape} is not divisible by block_size={config.block_size}")

    blocks_per_image = (height // config.block_size) * (width // config.block_size)
    expected_tokens = blocks_per_image * config.zigzag_coefficients
    if len(token_ids) != expected_tokens:
        raise ValueError(f"Expected {expected_tokens} coeff tokens, got {len(token_ids)}")

    coefficients = np.asarray(token_ids, dtype=np.int32) - config.coefficient_bound
    zigzag = np.zeros((blocks_per_image, 64), dtype=np.int32)
    zigzag[:, : config.zigzag_coefficients] = coefficients.reshape(blocks_per_image, config.zigzag_coefficients)

    quantized = np.zeros((blocks_per_image, 64), dtype=np.int32)
    quantized[:, _ZIGZAG_ORDER] = zigzag
    quantized_blocks = quantized.reshape(blocks_per_image, 8, 8).astype(np.float32)

    quant_table = _scaled_luma_quant_table(config.quality).astype(np.float32)
    dequantized = quantized_blocks * quant_table[None, :, :]
    spatial_blocks = idctn(dequantized, type=2, axes=(-2, -1), norm="ortho") + 128.0

    block_rows = height // config.block_size
    block_cols = width // config.block_size
    image = spatial_blocks.reshape(block_rows, block_cols, 8, 8).transpose(0, 2, 1, 3).reshape(height, width)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _encode_bounded_value(value: int, bound: int, name: str) -> int:
    if abs(value) > bound:
        raise ValueError(f"{name}={value} exceeds configured bound +/-{bound}")
    return value + bound


def _encode_category_value(value: int, category: int, name: str) -> int:
    if category <= 0:
        raise ValueError(f"{name} category must be positive, got {category}")
    if value == 0 or abs(value).bit_length() != category:
        raise ValueError(f"{name}={value} is not valid for category={category}")
    if value < 0:
        return value + (1 << category) - 1
    return (1 << (category - 1)) + value - (1 << (category - 1))


def _ac_symbol_offsets(max_category: int) -> dict[tuple[int, int], int]:
    offsets: dict[tuple[int, int], int] = {}
    cursor = 0
    for run in range(16):
        for category in range(1, max_category + 1):
            offsets[(run, category)] = cursor
            cursor += 1 << category
    return offsets


def _quantized_luma_blocks(luma_plane: np.ndarray, *, quality: int) -> np.ndarray:
    """Return raster-ordered quantized 8x8 DCT blocks for the canonical luma image.

    This is a deterministic reference implementation aligned to the frozen
    quantization table. It is intended for Phase 0 comparison work and is not a
    claim of bit-exact equivalence to libjpeg internals.
    """

    if luma_plane.ndim != 2:
        raise ValueError(f"Expected a 2D luma plane, got shape {luma_plane.shape}")
    height, width = luma_plane.shape
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"Luma plane shape {luma_plane.shape} is not divisible by the 8x8 JPEG block size")

    blocks = luma_plane.astype(np.float32).reshape(height // 8, 8, width // 8, 8).transpose(0, 2, 1, 3)
    centered = blocks - 128.0
    dct_blocks = dctn(centered, type=2, axes=(-2, -1), norm="ortho")
    quant_table = _scaled_luma_quant_table(quality).astype(np.float32)
    quantized = np.rint(dct_blocks / quant_table[None, None, :, :]).astype(np.int32)
    return quantized.reshape(-1, 8, 8)


def _libjpeg_quantized_luma_blocks(canonical_jpeg: CanonicalJpegRepresentation) -> np.ndarray:
    """Return libjpeg's exact quantized luma blocks for canonical JPEG bytes."""

    import jpeglib

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
        handle.write(canonical_jpeg.jpeg_bytes)
        temp_path = Path(handle.name)

    try:
        dct = jpeglib.read_dct(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)

    y_blocks = np.asarray(dct.Y, dtype=np.int32)
    if y_blocks.ndim != 4 or y_blocks.shape[-2:] != (8, 8):
        raise ValueError(f"Expected libjpeg Y blocks with shape [rows, cols, 8, 8], got {y_blocks.shape}")
    return y_blocks.reshape(-1, 8, 8)


def _scaled_luma_quant_table(quality: int) -> np.ndarray:
    if quality <= 0 or quality > 100:
        raise ValueError(f"quality must be in [1, 100], got {quality}")
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    table = (_JPEG_LUMA_QUANT_TABLE * scale + 50) // 100
    return np.clip(table, 1, 255).astype(np.int32)


__all__ = sorted(
    [
        "ByteWindowTokenizerConfig",
        "CanonicalJpegConfig",
        "CanonicalJpegRepresentation",
        "CoefficientTokenSource",
        "CoefficientTokenizerConfig",
        "JpegTokenizerFamily",
        "SymbolTokenizerConfig",
        "V0_BYTE_WINDOW_CONFIG",
        "V0_CANONICAL_JPEG_CONFIG",
        "V0_COEFFICIENT_CONFIG",
        "V0_SYMBOL_CONFIG",
        "V0_WHOLE_IMAGE_BYTE_CONFIG",
        "WholeImageByteTokenizerConfig",
        "byte_window_vocab_size",
        "canonicalize_image",
        "coefficient_vocab_size",
        "encode_dct_coeffs",
        "encode_jpeg_bytes",
        "encode_jpeg_symbols",
        "pad_whole_image_byte_tokens",
        "quantized_luma_blocks",
        "reconstruct_luma_from_coeff_tokens",
        "stable_byte_checksum",
        "symbol_vocab_size",
        "whole_image_byte_length",
        "whole_image_byte_vocab_size",
        "window_byte_tokens",
    ]
)
