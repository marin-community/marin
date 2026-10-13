# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0


"""Shared configuration registry for Levanter vision encoders."""

import abc
from dataclasses import dataclass
from typing import Generic, Optional, Type, TypeVar

import equinox as eqx
import draccus
from haliax import Axis

from levanter.compat.hf_checkpoints import HFCheckpointConverter


VisionEncoderT = TypeVar("VisionEncoderT", bound=eqx.Module)


@dataclass(frozen=True)
class VisionEncoderConfig(
    draccus.PluginRegistry, abc.ABC, Generic[VisionEncoderT], discover_packages_path="levanter.models"
):
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
