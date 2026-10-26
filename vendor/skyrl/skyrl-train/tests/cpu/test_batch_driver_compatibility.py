from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from skyrl_gym.verification import VerificationResult
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.utils import validate_cfg
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


class RecordedRows:
    def set_trajectory_sink(self, sink):
        pass


@pytest.fixture
def config():
    directory = str(Path(__file__).resolve().parents[2] / "skyrl_train/config")
    with initialize_config_dir(config_dir=directory, version_base=None):
        return compose(
            config_name="ppo_base_config",
            overrides=[
                "+algorithm_recipe=ftpo",
                "trainer.use_sample_packing=false",
                "generator.sampling_params.logprobs=4",
                "generator.sampling_params.temperature=0",
                "trainer.logger=console",
                "trainer.training_metrics=true",
                "trainer.train_batch_size=4",
                "trainer.policy_mini_batch_size=4",
                "generator.n_samples_per_prompt=1",
                "generator.trajectory_retention.enabled=false",
            ],
        )


@pytest.mark.parametrize("lease", [None, 2])
def test_verified_closure_trainer_preserves_rollouts_and_trains_verified_repairs(config, lease):
    config.trainer.algorithm.ftpo = {
        "boundary": "verified_closure_and_continuation",
        "chosen_balance_strength": 0,
        "repair_sft_weight": 0.7,
        "auxiliary_weight": None,
    }
    validate_cfg(config)
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ['assistant [{"a":1}]'],
        trainers.BpeTrainer(
            vocab_size=258, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=["[PAD]", "[EOS]"]
        ),
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]")
    trainer = RayPPOTrainer(
        cfg=config,
        tracker=None,
        tokenizer=tokenizer,
        train_dataset=range(4),
        inference_engine_client=None,
        trajectory_runner=RecordedRows(),
        context=None,
        callbacks=[],
    )
    trainer.global_step = 3
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=2))])
    prefix = '[{"a":1'
    prompt = tokenizer.encode("assistant " + prefix, add_special_tokens=False)
    responses = [tokenizer.encode(text, add_special_tokens=False) for text in (" ]", " ", " ]", " ]")]
    original_responses = deepcopy(responses)
    candidates = [tokenizer.encode(text, add_special_tokens=False)[0] for text in ("]", "}", "x", "\n")]
    verification = VerificationResult.verified(
        0,
        diagnostics={
            "stop_reason": "length",
            "closure": {
                "format": "json",
                "family": "json_nested",
                "control": "close_required",
                "prefix": prefix,
                "expected": '[{"a":1}]',
            },
        },
    )
    trajectory = {
        "prompt_token_ids": [prompt] * 4,
        "response_ids": responses,
        "loss_masks": [[1] * len(row) for row in responses],
        "rollout_policy_steps": [np.arange(1, len(row) + 1, dtype=np.int32) for row in responses],
        "rewards": [[0.0] * len(row) for row in responses],
        "student_topk_indices": [np.tile(candidates, (len(row), 1)) for row in responses],
        "behavior_topk_logprobs": [
            np.tile(np.log([0.5, 0.3, 0.15, 0.05]), (len(row), 1)).astype(np.float32) for row in responses
        ],
        "verification_results": [
            verification,
            verification,
            VerificationResult.unavailable("transport failed"),
            verification,
        ],
        "is_last_step": [True] * 4,
        "exclude_from_baseline": [False, False, False, True],
    }
    uids = ["a", "b", "c", "d"]
    batch = trainer.convert_to_training_input(
        trajectory, uids, rollout_staleness=None if lease is None else [lease] * 4
    )
    assert batch.metadata["response_length"] == batch["response_mask"].shape[1] == 3
    assert max(map(len, original_responses)) == 2
    assert trainer.all_metrics["async/staleness_tokens_mean"] == pytest.approx(0.0 if lease is None else 11 / 7)
    assert trainer.all_metrics["async/staleness_tokens_stamped"] == (0.0 if lease is None else 1.0)
    assert trainer.all_metrics["async/staleness_tokens_spanning"] == (0.0 if lease is None else 0.75)
    target_weights = batch["loss_mask"]
    assert target_weights.nonzero().tolist() == [[0, 1], [1, 1], [1, 2]]
    assert batch["ftpo_repair_mask"].nonzero().tolist() == [[1, 1], [1, 2]]
    assert trajectory["response_ids"] == original_responses
    repaired = batch["sequences"][1][batch["attention_mask"][1].bool()].tolist()
    assert tokenizer.decode(repaired, clean_up_tokenization_spaces=False) == 'assistant [{"a":1 }]'
    assert batch["rollout_logprobs"] is None
    assert batch["student_topk_indices"][batch["ftpo_chosen_mask"]].tolist() == [candidates[1]]
    assert trainer.all_metrics["ftpo/selected_boundaries"] == 1
