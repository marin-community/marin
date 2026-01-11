# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Base classes for Vision-Language Models (VLMs) and Vision Encoders.

This module provides abstract base classes for:
- VisionEncoderConfig: Configuration for vision encoders (e.g., SigLIP, Siglip2)
- VlmConfig: Configuration for vision-language models that combine vision encoders with LLMs
"""

import abc
from dataclasses import dataclass
from typing import Generic, Optional, Type, TypeVar

import draccus
from haliax import Axis

from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.models.lm_model import LmConfig


# =====================
# Vision Encoder Config
# =====================

VisionEncoderT = TypeVar("VisionEncoderT", bound="VisionEncoderModel")


# TODO: for some reason, mypy doesn't like the discover_packages_path argument?
@dataclass(frozen=True)
class VisionEncoderConfig(draccus.PluginRegistry, abc.ABC, Generic[VisionEncoderT], discover_packages_path="levanter.models"):  # type: ignore
    """
    Abstract base class for vision encoder configurations.

    All vision encoders (e.g., SigLIP, Siglip2) should inherit from this class
    and register themselves using the @VisionEncoderConfig.register_subclass decorator.

    Example:
        @VisionEncoderConfig.register_subclass("siglip")
        @dataclass(frozen=True)
        class SiglipVisionConfig(VisionEncoderConfig):
            ...
    """

    @property
    @abc.abstractmethod
    def model_type(cls) -> Type[VisionEncoderT]:
        """Return the corresponding model class."""
        pass

    @property
    @abc.abstractmethod
    def Embed(self) -> Axis:
        """The embedding dimension axis."""
        pass

    @property
    @abc.abstractmethod
    def NumPatches(self) -> Axis:
        """The number of patches axis."""
        pass

    @abc.abstractmethod
    def hf_checkpoint_converter(self, ref_checkpoint: Optional[str] = None) -> HFCheckpointConverter:
        """Create a HuggingFace checkpoint converter for this config."""
        pass


# =====================
# Vision Encoder Model
# =====================

class VisionEncoderModel(abc.ABC):
    """
    Abstract base class for vision encoder models.

    This is a placeholder for type hints. Concrete implementations
    should inherit from both this class and equinox.Module.
    """
    pass


# =====================
# VLM Config
# =====================

VlmT = TypeVar("VlmT", bound="VlmModel")


class VlmConfig(LmConfig[VlmT], abc.ABC, Generic[VlmT]):
    """
    Abstract base class / interface for Vision-Language Model configurations.

    Defines the interface for VLM configs that combine a vision encoder with a language model.

    IMPORTANT: Due to Python dataclass inheritance rules (fields with defaults cannot be
    followed by fields without defaults), concrete VLM configs should NOT directly inherit
    from VlmConfig. Instead, they should:
    1. Be standalone dataclasses
    2. Implement the VlmConfig interface via duck typing (define vision_config, text_config,
       Embed, max_Pos, KeyPos, VisionEmbed, etc.)
    3. Register with @LmConfig.register_subclass()

    Example:
        @LmConfig.register_subclass("llava")
        @dataclass(frozen=True)
        class LlavaConfig:  # Note: does NOT inherit from VlmConfig
            vision_config: SiglipVisionConfig
            text_config: QwenConfig
            # ... implements VlmConfig interface via properties ...
    """

    # Subclasses must define the following as dataclass fields:
    #   - vision_config: VisionEncoderConfig - The vision encoder configuration
    #   - text_config: LmConfig - The language model configuration
    #
    # We don't define them here as properties or annotations because that would
    # conflict with dataclass field assignment in frozen dataclasses.

    # Delegate to text_config for LmConfig properties
    @property
    def Embed(self) -> Axis:
        """Embedding dimension, delegated to text_config."""
        return self.text_config.Embed

    @property
    def max_Pos(self) -> Axis:
        """Maximum position axis, delegated to text_config."""
        return self.text_config.max_Pos

    @property
    def KeyPos(self) -> Axis:
        """Key position axis, delegated to text_config."""
        return self.text_config.KeyPos

    # Vision-related properties
    @property
    def VisionEmbed(self) -> Axis:
        """Vision embedding dimension from vision_config."""
        return self.vision_config.Embed

    @property
    def NumPatches(self) -> Axis:
        """Number of patches from vision_config."""
        return self.vision_config.NumPatches


# =====================
# VLM Model
# =====================

class VlmModel(abc.ABC):
    """
    Abstract base class for Vision-Language Models.

    This is a placeholder for type hints. Concrete implementations
    should inherit from both this class and equinox.Module.
    """
    pass
