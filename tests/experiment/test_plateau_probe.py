import tempfile

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from levanter.models.snowball import SnowballConfig
from levanter.testing.cpu_devices import run_on_cpu_devices
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from experiments.post_training.agentic_sweep.plateau_probe import (
    PlateauProbeConfig,
    load_model,
    plateau_share,
    scorer,
    summarize,
)


def test_plateau_share_needs_both_changes_below_the_threshold():
    first = np.array([1.0, 1.0, 1.0, 0.040, 2.0])
    second = np.array([1.0, 0.95, 1.0, 0.038, 2.0])
    third = np.array([1.0, 0.95, 1.5, 0.036, 1.79])
    np.testing.assert_array_equal(plateau_share([first, second, third], 0.1), [1, 1, 0, 1, 0])
    np.testing.assert_array_equal(plateau_share([first, second, third], 0.05), [1, 0, 0, 1, 0])


def test_summary_groups_rows_and_splits_learned_from_stuck():
    rows = [{"group": "train-tmax"}, {"group": "heldout-tmax"}]
    flat = np.array([0.05, 3.0])
    losses = {label: [flat, flat] for label in ("v0", "step-547", "step-1095")}
    summary = summarize(rows, losses)
    assert summary["all-train"]["rows"] == 1 and summary["all-heldout"]["rows"] == 1
    assert summary["train-tmax"]["dropped_0.1"] == 1.0
    assert summary["train-tmax"]["dropped_learned_0.1"] == 0.5
    assert summary["train-tmax"]["dropped_stuck_0.1"] == 0.5


def test_hf_model_loads_and_scores_on_the_offline_mesh():
    run_on_cpu_devices(
        "import runpy; runpy.run_path('tests/experiment/test_plateau_probe.py')['_check_scorer']()",
        device_count=8,
    )


def _check_scorer():
    trainer = PlateauProbeConfig(output_path="unused").trainer
    # The H100 flash-attention kernel rejects float32 queries.
    assert trainer.mp.compute_dtype == jnp.bfloat16
    batch, position, vocab = Axis("batch", 8), Axis("position", 16), Axis("vocab", 32)
    with tempfile.TemporaryDirectory() as hf_path:
        PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                models.WordLevel({"<pad>": 0, "<eos>": 1, **{f"t{i}": i for i in range(2, 32)}}, unk_token="<pad>")
            ),
            pad_token="<pad>",
            eos_token="<eos>",
        ).save_pretrained(hf_path)
        config = SnowballConfig(
            vocab_size=32,
            hidden_dim=16,
            intermediate_dim=32,
            shared_expert_intermediate_dim=32,
            num_experts=8,
            num_experts_per_token=1,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            head_dim=8,
            max_seq_len=16,
            sliding_window=4,
            attention_implementation="reference",
            tokenizer=hf_path,
        )
        converter = config.hf_checkpoint_converter(hf_path)
        with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
            built = config.build(vocab, key=jax.random.PRNGKey(0))
            converter.save_pretrained(built, hf_path, save_tokenizer=False, save_reference_code=False)
            model = load_model("hf", hf_path, converter, config, trainer, vocab)
            tokens = hax.shard(
                hax.named(jnp.broadcast_to(jnp.arange(16) % 32, (8, 16)).astype(jnp.int32), (batch, position))
            )
            losses = np.asarray(jax.device_get(scorer(trainer)(model, tokens).array))
            reference = np.asarray(jax.device_get(scorer(trainer)(built, tokens).array))
    assert losses.shape == (8, 16)
    assert np.isfinite(losses).all() and (losses[:, :-1] > 0).all() and (losses[:, -1] == 0).all()
    np.testing.assert_allclose(losses, reference, rtol=2e-2, atol=2e-2)
