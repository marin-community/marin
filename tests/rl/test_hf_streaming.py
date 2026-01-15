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

"""Tests for HuggingFace streaming model loading.

These tests verify that models can be loaded from HuggingFace Hub using
the hf:// fsspec streaming protocol, avoiding local cache usage.
"""

import os
import tempfile

import pytest

from marin.rl.model_utils import is_hf_checkpoint


class TestIsHfCheckpoint:
    """Tests for is_hf_checkpoint function."""

    def test_hf_model_ids_are_detected(self):
        """Plain HuggingFace model IDs should be recognized as HF checkpoints."""
        assert is_hf_checkpoint("meta-llama/Llama-2-7b") is True
        assert is_hf_checkpoint("gpt2") is True
        assert is_hf_checkpoint("mistralai/Mistral-7B-v0.1") is True
        assert is_hf_checkpoint("hf-internal-testing/tiny-random-gpt2") is True

    def test_hf_urls_are_detected(self):
        """hf:// URLs should be recognized as HF checkpoints."""
        assert is_hf_checkpoint("hf://meta-llama/Llama-2-7b") is True
        assert is_hf_checkpoint("hf://gpt2") is True
        assert is_hf_checkpoint("hf://hf-internal-testing/tiny-random-gpt2") is True
        assert is_hf_checkpoint("hf://meta-llama/Llama-2-7b@main") is True

    def test_other_urls_are_not_hf(self):
        """Other URL schemes should not be recognized as HF checkpoints."""
        assert is_hf_checkpoint("gs://bucket/path/model") is False
        assert is_hf_checkpoint("s3://bucket/path/model") is False
        assert is_hf_checkpoint("http://example.com/model") is False

    def test_local_paths_are_not_hf(self):
        """Local file paths should not be recognized as HF checkpoints."""
        assert is_hf_checkpoint("/absolute/path/to/model") is False
        assert is_hf_checkpoint("./relative/path") is False
        assert is_hf_checkpoint("../parent/path") is False


class TestHfStreamingHelpers:
    """Tests for HF streaming helper functions in hf_checkpoints.py."""

    def test_convert_to_hf_url(self):
        """Test conversion of model IDs to hf:// URLs."""
        from levanter.compat.hf_checkpoints import _convert_to_hf_url

        assert _convert_to_hf_url("meta-llama/Llama-2-7b") == "hf://meta-llama/Llama-2-7b"
        assert _convert_to_hf_url("gpt2", "main") == "hf://gpt2@main"
        assert _convert_to_hf_url("gpt2", None) == "hf://gpt2"

    def test_is_hf_model_id(self):
        """Test detection of HF model IDs vs other paths."""
        from levanter.compat.hf_checkpoints import _is_hf_model_id

        assert _is_hf_model_id("meta-llama/Llama-2-7b") is True
        assert _is_hf_model_id("gpt2") is True
        assert _is_hf_model_id("hf://gpt2") is False  # URL, not model ID
        assert _is_hf_model_id("gs://bucket/model") is False
        assert _is_hf_model_id("/local/path") is False

    def test_should_use_hf_streaming(self):
        """Test environment variable detection for streaming mode."""
        from levanter.compat.hf_checkpoints import HF_STREAMING_ENV_VAR, _should_use_hf_streaming

        original = os.environ.get(HF_STREAMING_ENV_VAR)
        try:
            os.environ[HF_STREAMING_ENV_VAR] = "1"
            assert _should_use_hf_streaming() is True

            os.environ[HF_STREAMING_ENV_VAR] = "0"
            assert _should_use_hf_streaming() is False

            del os.environ[HF_STREAMING_ENV_VAR]
            assert _should_use_hf_streaming() is False
        finally:
            if original is not None:
                os.environ[HF_STREAMING_ENV_VAR] = original
            elif HF_STREAMING_ENV_VAR in os.environ:
                del os.environ[HF_STREAMING_ENV_VAR]


@pytest.mark.network
class TestHfStreamingIntegration:
    """Integration tests that require network access to HuggingFace Hub."""

    def test_load_small_model_with_streaming(self):
        """Test loading a small model from HF Hub with streaming enabled.

        This test verifies that:
        1. The model loads successfully via hf:// streaming
        2. The local HF cache is not populated
        """
        from levanter.compat.hf_checkpoints import HF_STREAMING_ENV_VAR
        from levanter.models.gpt2 import Gpt2Config

        # Use a very small model for testing
        model_id = "hf-internal-testing/tiny-random-gpt2"

        original_env = os.environ.get(HF_STREAMING_ENV_VAR)
        original_hf_home = os.environ.get("HF_HOME")

        try:
            # Set up a temporary HF cache directory
            with tempfile.TemporaryDirectory() as tmp_hf_cache:
                os.environ["HF_HOME"] = tmp_hf_cache
                os.environ[HF_STREAMING_ENV_VAR] = "1"

                # Create a config and converter for the model
                gpt2_config = Gpt2Config(
                    num_layers=2,
                    num_heads=2,
                    hidden_dim=32,
                    use_flash_attention=False,
                )
                converter = gpt2_config.hf_checkpoint_converter()

                # Load the state dict - this should use streaming
                state_dict = converter.load_state_dict(model_id)

                # Verify we got a state dict with content
                assert isinstance(state_dict, dict)
                assert len(state_dict) > 0

                # Verify no model files were downloaded to the cache
                hub_cache = os.path.join(tmp_hf_cache, "hub")
                if os.path.exists(hub_cache):
                    cached_files = []
                    for root, dirs, files in os.walk(hub_cache):
                        cached_files.extend(files)
                    model_files = [f for f in cached_files if f.endswith((".safetensors", ".bin"))]
                    assert len(model_files) == 0, f"Expected no model files in cache, found: {model_files}"

        finally:
            # Restore original values
            if original_env is not None:
                os.environ[HF_STREAMING_ENV_VAR] = original_env
            elif HF_STREAMING_ENV_VAR in os.environ:
                del os.environ[HF_STREAMING_ENV_VAR]

            if original_hf_home is not None:
                os.environ["HF_HOME"] = original_hf_home
            elif "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]

    def test_load_model_without_streaming(self):
        """Test that loading without streaming uses the cache as expected."""
        from levanter.compat.hf_checkpoints import HF_STREAMING_ENV_VAR
        from levanter.models.gpt2 import Gpt2Config

        model_id = "hf-internal-testing/tiny-random-gpt2"

        original_env = os.environ.get(HF_STREAMING_ENV_VAR)
        original_hf_home = os.environ.get("HF_HOME")

        try:
            with tempfile.TemporaryDirectory() as tmp_hf_cache:
                os.environ["HF_HOME"] = tmp_hf_cache
                # Explicitly disable streaming
                os.environ[HF_STREAMING_ENV_VAR] = "0"

                gpt2_config = Gpt2Config(
                    num_layers=2,
                    num_heads=2,
                    hidden_dim=32,
                    use_flash_attention=False,
                )
                converter = gpt2_config.hf_checkpoint_converter()

                # Load the state dict - this should NOT use streaming
                state_dict = converter.load_state_dict(model_id)

                # Verify we got a state dict
                assert isinstance(state_dict, dict)
                assert len(state_dict) > 0

                # With streaming disabled, the cache should be populated
                # (This verifies the non-streaming path still works)

        finally:
            if original_env is not None:
                os.environ[HF_STREAMING_ENV_VAR] = original_env
            elif HF_STREAMING_ENV_VAR in os.environ:
                del os.environ[HF_STREAMING_ENV_VAR]

            if original_hf_home is not None:
                os.environ["HF_HOME"] = original_hf_home
            elif "HF_HOME" in os.environ:
                del os.environ["HF_HOME"]
