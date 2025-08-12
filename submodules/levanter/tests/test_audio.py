import tempfile

import pytest
import ray
from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer

from levanter.data.audio import AudioDatasetSourceConfig, AudioIODatasetConfig, BatchAudioProcessor
from levanter.store.cache import SerialCacheWriter
from levanter.utils.py_utils import logical_cpu_core_count
from test_utils import skip_if_hf_model_not_accessible, skip_if_no_soundlibs


def setup_module(module):
    ray.init(
        "local", num_cpus=max(2 * logical_cpu_core_count(), 8), ignore_reinit_error=True
    )  # 2x cpu count is faster on my m1


def teardown_module(module):
    ray.shutdown()


@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_whisper_batch_processor():
    try:
        processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
        tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")
        ds = load_dataset("WillHeld/test_librispeech_parquet", split="validation").select_columns(["audio", "text"])
        batch_processor = BatchAudioProcessor(processor, tokenizer)
        inputs = [
            (audio["array"], audio["sampling_rate"], text) for audio, text in zip(ds[:16]["audio"], ds[:16]["text"])
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


@pytest.mark.skip("Ray randomly OSErrors.")
@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
@pytest.mark.ray
@pytest.mark.asyncio
async def test_hf_audio_ray_pipeline():
    # Use the Real Librispeech Valudation. Testing one doesn't support streaming.
    with tempfile.TemporaryDirectory() as tmpdir:
        ac = AudioIODatasetConfig(
            cache_dir=str(tmpdir), id="WillHeld/test_librispeech_parquet", text_key="text", max_length=1024
        )
        validation = ac.validation_set()
        for i in range(10):
            t = (await validation.get_batch([i]))[0]
            assert t["input_features"].shape == (80, 3000), t["input_features"].shape
            assert t["input_ids"].shape == (1024,), t["input_ids"].shape
            assert t["attention_mask"].shape == (1024,), t["attention_mask"].shape


@skip_if_no_soundlibs
@skip_if_hf_model_not_accessible("openai/whisper-tiny")
def test_hf_audio_serial_cache():
    # Use the Real Librispeech Valudation. Testing one doesn't support streaming.
    ac = AudioIODatasetConfig(id="WillHeld/test_librispeech_parquet", text_key="text")

    processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
    tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")
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
    tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")
    batch_processor = BatchAudioProcessor(processor, tokenizer)
    # test this doesn't throw
    assert len(batch_processor.metadata)
