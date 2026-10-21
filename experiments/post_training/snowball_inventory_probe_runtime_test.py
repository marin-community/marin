# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest


@pytest.mark.manual
def test_training_admission_rechecks_checkpoint_content_at_unchanged_uri(tmp_path):
    from marinskyrl.model_manifest import (  # noqa: PLC0415 - isolated optional runtime
        MODEL_MANIFEST_FILENAME,
        ModelManifestFile,
        build_model_manifest,
    )

    from experiments.post_training.snowball_inventory_probe import verify_initializer_model  # noqa: PLC0415

    def manifest(content_hash):
        return build_model_manifest(
            tuple(
                ModelManifestFile(path=name, size=1, sha256=content_hash)
                for name in ("config.json", "model.safetensors.index.json", "model-00001-of-00001.safetensors")
            ),
            model_id="example/cold-sft",
            revision="b" * 40,
            tokenizer_mode="policy",
        )

    original = manifest("a" * 64)
    marker = tmp_path / MODEL_MANIFEST_FILENAME
    marker.write_text(original.model_dump_json())
    assert verify_initializer_model(str(tmp_path), "example/cold-sft", "b" * 40, original.identity) == original
    # A warm baseline cache cannot admit a different checkpoint at the same URI,
    # even if its named source revision has not changed.
    marker.write_text(manifest("c" * 64).model_dump_json())
    with pytest.raises(ValueError, match="content differs"):
        verify_initializer_model(str(tmp_path), "example/cold-sft", "b" * 40, original.identity)


@pytest.mark.manual
def test_training_admission_rechecks_active_template_with_same_tokenizer_identity():
    from tokenizers import Tokenizer  # noqa: PLC0415 - isolated optional runtime
    from tokenizers.models import WordLevel  # noqa: PLC0415
    from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

    from experiments.post_training.snowball_inventory_probe import verify_tokenizer_template  # noqa: PLC0415

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")))
    tokenizer.chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    expected = hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()
    assert verify_tokenizer_template(tokenizer, expected) == expected
    tokenizer.chat_template += "{{ eos_token }}"
    with pytest.raises(ValueError, match="template differs"):
        verify_tokenizer_template(tokenizer, expected)
