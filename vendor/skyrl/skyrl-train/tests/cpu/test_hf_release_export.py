import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizerFast

from marin_hf_release import PrecisionMode
from marin_hf_release.validation import read_safetensors_header_specs
from skyrl_train import hf_model_io
from skyrl_train.hf_model_io import local_hf_model_dir, verify_hf_model_export
from skyrl_train.hf_release_export import ReleaseExportOptions, finalize_hf_release
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM


def _tiny_config():
    return GrugMoeConfig(
        vocab_size=5,
        hidden_size=8,
        intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_local_experts=2,
        num_experts_per_tok=1,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=16,
        sliding_window=8,
    )


def _tokenizer():
    raw = Tokenizer(WordLevel({"[BOS]": 0, "[EOS]": 1, "[EOT]": 2, "hello": 3, "[UNK]": 4}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = "{{ bos_token }}{{ messages[0]['content'] }}{{ eos_token }}"
    return tokenizer


@pytest.mark.parametrize("precision", [PrecisionMode.BF16, PrecisionMode.FP32])
def test_grug_bridge_release_preserves_router_bias_and_logits(tmp_path, precision):
    torch.manual_seed(7)
    config = _tiny_config()
    model = GrugMoeForCausalLM(config).eval()
    model.model.layers[0].mlp.router.bias.copy_(torch.tensor([0.25, -0.5]))
    tokenizer = _tokenizer()
    options = ReleaseExportOptions(
        precision=precision,
        source_checkpoint="checkpoint/global_step_12/policy",
        source_step=12,
        checkpoint_identity="sha256:checkpoint",
        source_model=None,
        source_model_commit=None,
        eos_token_ids=[1, 2],
        generation_controls={},
        chat_template=None,
    )
    with local_hf_model_dir(str(tmp_path)) as directory:
        save_file(model.state_dict(), Path(directory) / "model.safetensors")
        finalize_hf_release(Path(directory), config, tokenizer, options)

    verify_hf_model_export(str(tmp_path), require_release_manifest=True)
    headers = read_safetensors_header_specs(tmp_path / "model.safetensors")
    expected = "BF16" if precision is PrecisionMode.BF16 else "F32"
    assert headers["model.embed_tokens.weight"].source_dtype == expected
    assert headers["model.layers.0.mlp.router.bias"].source_dtype == "F32"
    exported = load_file(tmp_path / "model.safetensors")
    torch.testing.assert_close(exported["model.layers.0.mlp.router.bias"], torch.tensor([0.25, -0.5]), rtol=0, atol=0)
    assert json.loads((tmp_path / "config.json").read_text())["dtype"] == (
        "bfloat16" if precision is PrecisionMode.BF16 else "float32"
    )
    generation = GenerationConfig.from_pretrained(tmp_path)
    assert generation.eos_token_id == [1, 2]
    released_tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    assert released_tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False) == (
        "[BOS]hello[EOS]"
    )

    restored = GrugMoeForCausalLM(config).eval()
    restored.load_state_dict(exported)
    inputs = torch.tensor([[0, 3]])
    with torch.no_grad():
        original_logits = model(inputs).logits
        released_logits = restored(inputs).logits
    if precision is PrecisionMode.FP32:
        torch.testing.assert_close(released_logits, original_logits, rtol=0, atol=0)
    else:
        # BF16 storage quantizes each weight to roughly 0.4%; this tiny two-token
        # forward is evaluated in FP32 after loading the quantized weights.
        torch.testing.assert_close(released_logits, original_logits, rtol=0.03, atol=0.03)


def test_interrupted_cloud_publication_has_no_release_marker(tmp_path, monkeypatch):
    save_file({"weight": torch.ones(2)}, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / ".marinskyrl-model-manifest.json").write_text("{}")
    (tmp_path / "hf_release_manifest.json").write_text("{}")
    published = []

    def upload_file(source: str, destination: str) -> None:
        if source.endswith("config.json"):
            raise OSError("interrupted upload")
        published.append(Path(destination).name)

    monkeypatch.setattr(hf_model_io.io, "upload_file", upload_file)
    with pytest.raises(OSError, match="interrupted upload"):
        hf_model_io._upload_hf_model_directory(str(tmp_path), "s3://bucket/release")
    assert "model.safetensors" in published
    assert "hf_release_manifest.json" not in published
