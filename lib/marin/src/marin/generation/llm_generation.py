# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Wrapper around vLLM generation.

User -> Preset Prompt, Engine Kwargs, File -> Generate Text
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar
import os

# Disable multiprocessing since running on Ray, easier to clean
# resources if not using a multiprocess.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    from vllm.inputs.data import TokensPrompt
except ImportError:
    logger.warning("vLLM is not installed, so we will not be able to generate text.")
    TokensPrompt = Any


class BaseLLMProvider(ABC):
    @abstractmethod
    def __init__(self, model_name: str):
        pass

    @abstractmethod
    def generate(self, prompts: list[str]) -> list[str]:
        """The input is a list of prompts and the output is a list of generated text."""
        raise NotImplementedError


class vLLMProvider(BaseLLMProvider):
    DEFAULT_ENGINE_KWARGS: ClassVar[dict[str, Any]] = {
        "tensor_parallel_size": 1,
        "enforce_eager": False,
    }

    DEFAULT_GENERATION_KWARGS: ClassVar[dict[str, Any]] = {
        "temperature": 0.1,
        "max_tokens": 1024,
    }

    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(model_name)

        self.model_name = model_name
        self.engine_kwargs = {**vLLMProvider.DEFAULT_ENGINE_KWARGS, **engine_kwargs}
        self.generation_kwargs = {**vLLMProvider.DEFAULT_GENERATION_KWARGS, **generation_kwargs}

        # On multi-host TPU slices (v5p-16, v5p-32, etc.), JAX cannot initialize
        # properly with a single process, causing AttributeError on device.coords.
        # Fail fast so Ray Data retries on a single-host node (e.g. v5p-8).
        self._check_tpu_single_host()

        self.llm = LLM(model=self.model_name, **self.engine_kwargs)
        self.sampling_params = SamplingParams(**self.generation_kwargs)

    @staticmethod
    def _check_tpu_single_host():
        """Raise if this node is a multi-host TPU slice (incompatible with single-process vLLM)."""
        try:
            import urllib.request

            req = urllib.request.Request(
                "http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type",
                headers={"Metadata-Flavor": "Google"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                tpu_type = resp.read().decode().strip()

            # Parse chip count from tpu_type like "v5p-8" -> 8
            parts = tpu_type.rsplit("-", 1)
            if len(parts) == 2:
                try:
                    slice_chips = int(parts[1])
                except ValueError:
                    return
                # v5p-8 = 4 chips on 1 host (single-host), v5p-16+ = multi-host
                if slice_chips > 8:
                    # Use os._exit instead of raising so the actor process dies
                    # and Ray restarts it (max_restarts=-1) on a potentially
                    # different node. Raising in __init__ counts against the
                    # limited init-retry budget and exhausts it quickly.
                    import os as _os

                    logger.error(
                        "This node is a multi-host TPU slice (%s) which is "
                        "incompatible with single-process vLLM inference. "
                        "Killing actor so Ray restarts on a different node.",
                        tpu_type,
                    )
                    _os._exit(1)
        except Exception as e:
            logger.debug("Could not check TPU type from metadata: %s", e)

    def generate(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        generated_text: list[str] = []
        for output in outputs:
            generated_text.append(" ".join([o.text for o in output.outputs]))
        return generated_text
