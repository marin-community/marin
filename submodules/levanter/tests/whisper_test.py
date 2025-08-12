import tempfile
from typing import cast

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from datasets import load_dataset
from jax.random import PRNGKey
from transformers import WhisperConfig as HfWhisperConfig
from transformers import WhisperForConditionalGeneration as HfWhisperModel
from transformers import WhisperProcessor

import haliax as hax
from haliax import Axis

from levanter.compat.hf_checkpoints import RepoRef
from levanter.data.audio import AudioTextExample
from levanter.layers.attention import AttentionMask
from levanter.models.whisper import WhisperASRModel, WhisperConfig, WhisperModel
from levanter.utils.tree_utils import inference_mode
from test_utils import skip_if_no_soundlibs, skip_if_no_torch


@pytest.mark.skip
@skip_if_no_soundlibs
def test_whisper_loss():
    c = HfWhisperConfig.from_pretrained("openai/whisper-tiny")
    conf = WhisperConfig.from_hf_config(c)
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model: WhisperASRModel = conf.build_asr(conf.Vocab, key=PRNGKey(42))
    ds = load_dataset("WillHeld/test_librispeech_parquet", split="validation")
    audio_sample = ds[3]
    speech_data = audio_sample["audio"]["array"]
    inputs = processor.feature_extractor(speech_data, sampling_rate=16_000, return_tensors="np")

    na = hax.NamedArray(
        inputs["input_features"].squeeze(),
        axes=(conf.Mels, Axis(name="position", size=3000)),
    )
    tokenized = processor.tokenizer("This is a test", max_length=6, padding="max_length", truncation=True)
    inp = hax.NamedArray(
        jnp.array(tokenized["input_ids"]),
        axes=(Axis("position", size=6),),
    )
    mask = AttentionMask.explicit(
        hax.NamedArray(
            jnp.array(tokenized["attention_mask"]),
            axes=(Axis("key_position", size=6),),
        ).broadcast_axis(Axis("position", size=6))
    )
    model.compute_loss(AudioTextExample.init(na, inp, attn_mask=mask))


@pytest.mark.skip
@skip_if_no_soundlibs
def test_basic_forward_whisper():
    c = HfWhisperConfig.from_pretrained("openai/whisper-tiny")
    conf = WhisperConfig.from_hf_config(c)
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperModel.init(conf.Vocab, conf, key=PRNGKey(42))
    ds = load_dataset("WillHeld/test_librispeech_parquet", split="validation")
    audio_sample = ds[3]
    speech_data = audio_sample["audio"]["array"]
    inputs = processor.feature_extractor(speech_data, sampling_rate=16_000, return_tensors="np")

    na = hax.NamedArray(
        inputs["input_features"],
        axes=(Axis(name="batch", size=1), conf.Mels, Axis(name="position", size=3000)),
    )
    inp = hax.NamedArray(
        jnp.array([processor.get_decoder_prompt_ids()])[:, :, 1],
        axes=(
            Axis("batch", size=1),
            Axis("position", size=1),
        ),
    )
    model(na, inp)


@pytest.mark.skip
@skip_if_no_soundlibs
def test_mask_forward_whisper():
    c = HfWhisperConfig.from_pretrained("openai/whisper-tiny")
    conf = WhisperConfig.from_hf_config(c)
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperModel.init(conf.Vocab, conf, key=PRNGKey(42))
    ds = load_dataset("WillHeld/test_librispeech_parquet", split="validation")
    audio_sample = ds[3]
    speech_data = audio_sample["audio"]["array"]
    inputs = processor.feature_extractor(speech_data, sampling_rate=16_000, return_tensors="np")

    na = hax.NamedArray(
        inputs["input_features"],
        axes=(Axis(name="batch", size=1), conf.Mels, Axis(name="position", size=3000)),
    )
    inp = hax.NamedArray(
        jnp.array([processor.get_decoder_prompt_ids()])[:, :, 1],
        axes=(
            Axis("batch", size=1),
            Axis("position", size=1),
        ),
    )
    model(na, inp, attn_mask=AttentionMask.causal())


@pytest.mark.skip
@skip_if_no_soundlibs
def test_namedarray_mask_forward_whisper():
    c = HfWhisperConfig.from_pretrained("openai/whisper-tiny")
    conf = WhisperConfig.from_hf_config(c)
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperModel.init(conf.Vocab, conf, key=PRNGKey(42))
    ds = load_dataset("WillHeld/test_librispeech_parquet", split="validation")
    audio_sample = ds[3]
    speech_data = audio_sample["audio"]["array"]
    inputs = processor.feature_extractor(speech_data, sampling_rate=16_000, return_tensors="np")

    na = hax.NamedArray(
        inputs["input_features"],
        axes=(Axis(name="batch", size=1), conf.Mels, Axis(name="position", size=3000)),
    )
    inp = hax.NamedArray(
        jnp.array([processor.get_decoder_prompt_ids()])[:, :, 1],
        axes=(
            Axis("batch", size=1),
            Axis("position", size=1),
        ),
    )
    model(na, inp, attn_mask=AttentionMask.causal().explicit_mask)


@pytest.mark.skip
@skip_if_no_soundlibs
@skip_if_no_torch
def test_hf_roundtrip():
    model_id = "openai/whisper-tiny"
    converter = WhisperConfig().hf_checkpoint_converter()
    c = HfWhisperConfig.from_pretrained(model_id)
    config = WhisperConfig.from_hf_config(c)
    processor = WhisperProcessor.from_pretrained(model_id)

    torch_model: HfWhisperModel = HfWhisperModel.from_pretrained(model_id)
    torch_model.eval()

    model: WhisperModel = cast(WhisperModel, converter.load_pretrained(config.model_type, RepoRef(model_id), config))
    model = inference_mode(model, True)

    ds = load_dataset("WillHeld/test_librispeech_parquet", split="validation")
    inputs = processor.feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features
    tokenized = processor.tokenizer(
        ds[0]["text"], max_length=6, padding="max_length", truncation=True, return_tensors="pt"
    )
    decoder_input_ids = tokenized["input_ids"]
    # we compare softmaxes because the numerics are wonky and we usually just care about the softmax
    torch_out = torch_model(input_features, decoder_input_ids=decoder_input_ids)
    torch_out = torch_out.logits[0].detach().cpu().numpy()
    torch_out = jax.nn.softmax(torch_out, axis=-1)

    na = hax.NamedArray(
        input_features.cpu().numpy(),
        axes=(Axis(name="batch", size=1), config.Mels, Axis(name="position", size=3000)),
    )
    inp = hax.NamedArray(
        decoder_input_ids.cpu().numpy(),
        axes=(
            Axis("batch", size=1),
            Axis("position", size=6),
        ),
    )

    def compute(na, inp):
        return hax.nn.softmax(
            model(na, inp),
            axis=model.Vocab,
        )

    compute = jax.jit(compute)
    jax_out = compute(na, inp).array[0]

    assert torch_out.shape == jax_out.shape, f"{torch_out.shape} != {jax_out.shape}"
    assert onp.isclose(torch_out, onp.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out} != {jax_out}"

    with tempfile.TemporaryDirectory() as tmpdir:
        converter.save_pretrained(model, tmpdir)

        torch_model2: HfWhisperModel = HfWhisperModel.from_pretrained(tmpdir)
        torch_model2.eval()

        torch_out2 = torch_model2(input_features, decoder_input_ids=decoder_input_ids)
        torch_out2 = torch_out2.logits[0].detach().cpu().numpy()
        torch_out2 = jax.nn.softmax(torch_out2, axis=-1)
        assert onp.isclose(torch_out2, onp.array(jax_out), rtol=1e-2, atol=1e-2).all(), f"{torch_out2} != {jax_out}"
