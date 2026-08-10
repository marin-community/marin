# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import tempfile

import pytest
from datasets import load_dataset

from levanter.tokenizers import load_tokenizer
from test_utils import skip_if_hf_model_not_accessible, skip_if_no_soundlibs
from transformers import AutoProcessor

from levanter.data.audio import AudioDatasetSourceConfig, AudioIODatasetConfig, BatchAudioProcessor
from levanter.store.cache import SerialCacheWriter


pytestmark = pytest.mark.torch


@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_whisper_batch_processor():
    try:
        processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
        tokenizer = load_tokenizer("openai/whisper-tiny")
        ds = load_dataset("WillHeld/test_librispeech_parquet", split="validation").select_columns(["audio", "text"])
        batch_processor = BatchAudioProcessor(processor, tokenizer)
        inputs = [
            (audio["array"], audio["sampling_rate"], text) for audio, text in zip(ds[:4]["audio"], ds[:4]["text"])
        ]
        batch_processor(inputs)
    except FileNotFoundError:
        pytest.skip("No whisper model found. Probably HF is being flaky.")


@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_hf_audio_loading():
    # Use the Real Librispeech Valudation. Testing one doesn't support streaming.
    ac = AudioDatasetSourceConfig(id="WillHeld/test_librispeech_parquet", text_key="text")
    audio_iterator = ac.doc_iterator("validation")
    for i in range(10):
        audio, sample, text = next(audio_iterator)


@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_hf_audio_loading_source():
    # Use the Real Librispeech Valudation. Testing one doesn't support streaming.
    ac = AudioDatasetSourceConfig(id="WillHeld/test_librispeech_parquet", text_key="text")
    audio_iterator = iter(ac.get_shard_source("validation"))
    for i in range(10):
        audio, sample, text = next(audio_iterator)


@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_hf_audio_serial_cache():
    # Use the Real Librispeech Valudation. Testing one doesn't support streaming.
    ac = AudioIODatasetConfig(id="WillHeld/test_librispeech_parquet", text_key="text")

    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    tokenizer = load_tokenizer("openai/whisper-tiny")
    batch_processor = BatchAudioProcessor(processor, tokenizer, max_length=1024)

    with tempfile.TemporaryDirectory() as tmpdir:
        with SerialCacheWriter(tmpdir, batch_processor.output_exemplar) as writer:
            for i, ex in enumerate(ac.get_shard_source("validation")):
                writer.write_batch(batch_processor([ex]))
                if i > 10:
                    break

        cache = writer.result()

        for ex in cache.get_batch_sync(list(range(10))):
            assert ex["input_features"].shape == (80, 3000), ex["input_features"].shape
            assert ex["input_ids"].shape == (1024,), ex["input_ids"].shape
            assert ex["attention_mask"].shape == (1024,), ex["attention_mask"].shape


@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_metadata_works():
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    tokenizer = load_tokenizer("openai/whisper-tiny")
    batch_processor = BatchAudioProcessor(processor, tokenizer)
    # test this doesn't throw
    assert len(batch_processor.metadata)
