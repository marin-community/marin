import os
import tempfile

import jax
import pytest

import levanter.main.train_asr as train_asr
import tiny_test_corpus
from levanter.distributed import RayConfig
from levanter.tracker import NoopConfig
from test_utils import skip_if_no_soundlibs


@pytest.mark.skip
@pytest.mark.entry
@skip_if_no_soundlibs
def test_train_asr():
    # just testing if train_asr has a pulse
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config = tiny_test_corpus.tiny_asr_corpus_config(tmpdir)
        try:
            config = train_asr.TrainASRConfig(
                data=data_config,
                model=train_asr.WhisperConfig(
                    d_model=32,
                ),
                trainer=train_asr.TrainerConfig(
                    num_train_steps=2,
                    train_batch_size=len(jax.devices()),
                    max_eval_batches=1,
                    tracker=NoopConfig(),
                    require_accelerator=False,
                    ray=RayConfig(auto_start_cluster=False),
                ),
                hf_save_path=f"{tmpdir}/hf_asr_output",
            )
            train_asr.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass
