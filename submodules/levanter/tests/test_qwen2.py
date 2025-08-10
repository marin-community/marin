import json
import tempfile

import numpy as np
from jax import random

import haliax as hax

from levanter.layers.attention import AttentionMask
from levanter.models.qwen import QwenConfig, QwenLMHeadModel
from test_utils import skip_if_no_torch


def get_config(vocab_size=1000):
    from transformers import Qwen2Config

    qwen_cfg = json.loads(
        """
        {
          "architectures": ["QWenLMHeadModel"],
          "attn_dropout_prob": 0.0,
          "bf16": false,
          "emb_dropout_prob": 0.0,
          "fp16": false,
          "fp32": false,
          "hidden_size": 4096,
          "intermediate_size": 22016,
          "initializer_range": 0.02,
          "kv_channels": 128,
          "layer_norm_epsilon": 1e-06,
          "max_position_embeddings": 32768,
          "model_type": "qwen",
          "no_bias": true,
          "num_attention_heads": 32,
          "num_hidden_layers": 32,
          "onnx_safe": null,
          "rotary_emb_base": 10000,
          "rotary_pct": 1.0,
          "scale_attn_weights": true,
          "seq_length": 8192,
          "tie_word_embeddings": false,
          "tokenizer_class": "QWenTokenizer",
          "transformers_version": "4.32.0",
          "use_cache": true,
          "use_dynamic_ntk": true,
          "use_flash_attn": "auto",
          "use_logn_attn": true,
          "vocab_size": 151936
        }
    """
    )
    qwen_config: Qwen2Config = Qwen2Config.from_dict(qwen_cfg)
    qwen_config.hidden_size = 16
    qwen_config.intermediate_size = 64
    qwen_config.num_attention_heads = 4
    qwen_config.head_dim = 4
    qwen_config.num_hidden_layers = 4
    qwen_config.num_key_value_heads = 2
    qwen_config.max_position_embeddings = 128
    qwen_config.vocab_size = vocab_size
    return qwen_config


@skip_if_no_torch
def test_qwen_roundtrip():
    import torch
    from transformers import Qwen2ForCausalLM

    Vocab = hax.Axis("vocab", 1000)
    hf_config = get_config(Vocab.size)

    converter = QwenConfig().hf_checkpoint_converter()

    config = QwenConfig.from_hf_config(hf_config)

    # Make input and attn_mask
    input = hax.random.randint(random.PRNGKey(0), config.Pos, 0, Vocab.size)
    attn_mask = AttentionMask.causal()
    input_torch = torch.from_numpy(np.array(input.array)).to(torch.int32).unsqueeze(0)

    torch.random.manual_seed(0)

    torch_model = Qwen2ForCausalLM(hf_config)
    torch_model.eval()

    torch_out = torch_model(input_torch)
    torch_out = torch_out.logits[0].detach().cpu().numpy()
    # torch_out = jax.nn.softmax(torch_out, axis=-1)

    with tempfile.TemporaryDirectory() as tmpdir:
        torch_model.save_pretrained(f"{tmpdir}/torch_model")

        model = converter.load_pretrained(
            QwenLMHeadModel, ref=f"{tmpdir}/torch_model", resize_vocab_to_match_tokenizer=False
        )

        @hax.named_jit
        def compute(model, input):
            model_output = model(input, attn_mask=attn_mask)
            return model_output

        jax_out = compute(model, input).array

        assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
        assert np.isclose(torch_out, np.array(jax_out), rtol=1e-4, atol=1e-4).all(), f"{torch_out} != {jax_out}"

        # now we're going to magnify the model parameters enough that differences should actualy show up
        jax_out = compute(model, input).array

        converter.save_pretrained(model, f"{tmpdir}/lev_model", save_reference_code=False)
        torch_model2 = Qwen2ForCausalLM.from_pretrained(f"{tmpdir}/lev_model")
        torch_model2.eval()

        torch_out2 = torch_model2(input_torch)
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        assert torch_out2.shape == jax_out.shape, f"{torch_out2.shape} != {jax_out.shape}"
        np.testing.assert_allclose(torch_out2, jax_out, rtol=1e-5, atol=1e-5)
