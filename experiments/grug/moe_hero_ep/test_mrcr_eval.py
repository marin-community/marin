# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from haliax import Axis
from haliax.partitioning import set_mesh
from levanter.grug.attention import fa4_cute_segment_bounds
from levanter.grug.sharding import compact_grug_mesh
from levanter.store.cache import write_levanter_cache
from levanter.testing.devices import use_test_devices
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from experiments.datasets.mrcr import MrcrCondition
from experiments.grug.moe_hero_ep import model
from experiments.grug.moe_hero_ep.mrcr_eval import (
    MRCR_COMPONENT,
    MRCR_METRIC_PREFIX,
    MrcrConditionLoss,
    MrcrEvalConfig,
    MrcrExampleLoss,
    derive_mrcr_metrics,
    mrcr_data_config,
    pair_condition_losses,
    score_condition_rows,
)


def _source(index: int, *, band: str = "distance_0_32768", needles: int = 2, tokens: int = 100, gains=(1.0, 0.8)):
    """A source whose isolated and distracted needle gains are ``gains`` nats per token."""
    isolated, distracted = gains
    return MrcrExampleLoss(
        source_id=f"source-{index}",
        n_needles=needles,
        distance_band=band,
        evidence_distance_tokens=10,
        scored_tokens=tokens,
        scored_bytes=tokens * 4,
        full_context_loss_sum=tokens * (1.5 - distracted),
        query_only_loss_sum=tokens * 1.5,
        needle_only_loss_sum=tokens * (1.5 - isolated),
        distractor_only_loss_sum=tokens * 1.5,
    )


def _condition_rows(source: str = "s", **overrides):
    fields = dict(
        source_id=source,
        n_needles=2,
        distance_band="distance_0_32768",
        evidence_distance_tokens=5,
        scored_tokens=10,
        loss_sum=4.0,
        scored_bytes=40,
    )
    rows = {condition: MrcrConditionLoss(condition=condition.value, **fields) for condition in MrcrCondition}
    for condition, change in overrides.items():
        rows[MrcrCondition(condition)] = MrcrConditionLoss(condition=condition, **(fields | change))
    return list(rows.values())


def test_recovery_is_the_token_micro_ratio_of_aggregate_contrasts():
    rows = [_source(0, tokens=100, gains=(1.0, 0.9)), _source(1, tokens=300, gains=(1.0, 0.5))]
    metrics = derive_mrcr_metrics(rows, bootstrap_samples=200)
    # (100 * 0.9 + 300 * 0.5) / (100 * 1.0 + 300 * 1.0) is token-micro; the macro ratio weighs the sources equally.
    np.testing.assert_allclose(metrics[f"{MRCR_METRIC_PREFIX}/micro_needle_recovery_ratio"], 0.6)
    np.testing.assert_allclose(metrics[f"{MRCR_METRIC_PREFIX}/macro_needle_recovery_ratio"], 0.7)


def test_intervals_cover_the_point_estimates_and_shrink_with_more_sources():
    def rows(count):
        rng = np.random.default_rng(0)
        return [_source(i, gains=(1.0, float(rng.uniform(0.4, 1.0)))) for i in range(count)]

    few = derive_mrcr_metrics(rows(10), bootstrap_samples=2000)
    many = derive_mrcr_metrics(rows(1000), bootstrap_samples=2000)
    base = MRCR_METRIC_PREFIX
    for stem in (
        "micro_needle_recovery_ratio",
        "micro_real_gain_nll",
        "micro_oracle_gain_nll",
        "full_context/micro_loss",
        "query_only/micro_loss",
        "needle_only/micro_loss",
        "distractor_only/micro_loss",
    ):
        for metrics in (few, many):
            low, value, high = (metrics[f"{base}/{stem}{suffix}"] for suffix in ("_ci95_low", "", "_ci95_high"))
            assert low - 1e-9 <= value <= high + 1e-9, stem
    width = lambda m, s: m[f"{base}/{s}_ci95_high"] - m[f"{base}/{s}_ci95_low"]  # noqa: E731
    assert width(many, "micro_real_gain_nll") < width(few, "micro_real_gain_nll")
    assert width(many, "full_context/micro_loss") < width(few, "full_context/micro_loss")


def test_the_bootstrap_resamples_within_strata():
    # Each stratum is constant, so resampling within strata cannot move the aggregate; a pooled
    # resample of these two very different strata would.
    rows = [
        *[_source(i, band="distance_0_32768", gains=(1.0, 0.2)) for i in range(20)],
        *[_source(20 + i, band="distance_32769_65536", gains=(1.0, 0.9)) for i in range(20)],
    ]
    metrics = derive_mrcr_metrics(rows, bootstrap_samples=500)
    base = f"{MRCR_METRIC_PREFIX}/micro_real_gain_nll"
    np.testing.assert_allclose(metrics[f"{base}_ci95_low"], metrics[f"{base}_ci95_high"])


def test_an_unidentifiable_oracle_reports_no_recovery():
    metrics = derive_mrcr_metrics([_source(i, gains=(0.05, 0.05)) for i in range(5)], bootstrap_samples=100)
    assert metrics[f"{MRCR_METRIC_PREFIX}/micro_recovery_identifiable"] == 0.0
    assert f"{MRCR_METRIC_PREFIX}/micro_needle_recovery_ratio" not in metrics


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"query_only": {"scored_tokens": 11}}, "unequal or zero scored counts"),
        ({"needle_only": {"loss_sum": float("nan")}}, "non-finite loss"),
        ({"distractor_only": {"n_needles": 4}}, "paired metadata differs"),
    ],
)
def test_pairing_refuses_conditions_that_do_not_score_the_same_response(overrides, message):
    with pytest.raises(ValueError, match=message):
        pair_condition_losses(_condition_rows(**overrides))


def test_pairing_refuses_a_missing_condition():
    with pytest.raises(ValueError, match="incomplete condition set"):
        pair_condition_losses(_condition_rows()[:3])


SEQ_LEN = 16
TARGET = [11, 12, 13]
CONFIG = model.GrugModelConfig(
    vocab_size=64,
    hidden_dim=32,
    intermediate_dim=16,
    shared_expert_intermediate_dim=16,
    num_shared_experts=1,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    local_kv_heads=2,
    global_kv_heads=2,
    head_dim=8,
    max_seq_len=SEQ_LEN,
    sliding_window=4,
    global_every=2,
    initializer_std=0.1,
    qk_mult=1.3,
    capacity_factor=8.0,
    sconv=True,
    rope_fused=True,
    attention_implementation="reference",
    moe_implementation="fixed_all_to_all",
)
# Context lengths for two sources in cache order; the first document fills the sequence and the rest are padded.
CONTEXT_LENGTHS = [12, 2, 4, 8, 11, 3, 5, 9]


def _rows():
    rows = []
    for index, context_length in enumerate(CONTEXT_LENGTHS):
        context = np.random.default_rng(index).integers(20, 60, size=context_length).tolist()
        ids = np.asarray([*context, *TARGET, 1], dtype=np.int32)
        mask = np.zeros_like(ids)
        mask[context_length : context_length + len(TARGET)] = 1
        rows.append({"input_ids": ids, "assistant_masks": mask})
    return rows


def _write_fixture(directory, rows):
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, **{str(i): i for i in range(1, 64)}}, unk_token="<unk>"))
    tokenizer.save(f"{directory}/tokenizer.json")
    write_levanter_cache(iter(rows), f"{directory}/data/validation", metadata={})
    manifest = [
        {
            "source_id": f"source-{index // 4}",
            "condition": list(MrcrCondition)[index % 4].value,
            "n_needles": 2,
            "distance_band": "distance_0_32768",
            "evidence_distance_tokens": 3,
            "scored_tokens": len(TARGET),
            "length_tokens": len(rows[index]["input_ids"]),
        }
        for index in range(len(rows))
    ]
    (directory / "data" / "manifest.jsonl").write_text("".join(json.dumps(record) + "\n" for record in manifest))
    return manifest


def _reference_loss(weights, tokens, *, padding=0):
    """Score one document alone in its batch, its padding a chosen value, without the dataset or the loader."""
    length = len(tokens)
    segments = np.full((4, SEQ_LEN), -1, dtype=np.int32)
    segments[0, :length] = 0
    ids = np.full((4, SEQ_LEN), padding, dtype=np.int32)
    ids[0, :length] = tokens
    weight = np.zeros((4, SEQ_LEN), dtype=np.float32)
    # Positions predicting the response: the last context token through the second response token.
    weight[0, length - len(TARGET) - 2 : length - 2] = 1.0
    mask = model.AttentionMask.causal().with_segment_ids(jnp.asarray(segments))
    expert_mlp = weights.stacked_blocks.stacked.mlp.expert_mlp
    local = eqx.tree_at(
        lambda m: m.stacked_blocks.stacked.mlp.expert_mlp,
        weights,
        dataclasses.replace(expert_mlp, implementation="scatter", expert_chunks=1),
    )
    losses = local.next_token_loss(
        jnp.asarray(ids), jnp.asarray(weight), mask=mask, reduction="none", logsumexp_weight=None
    )
    return float(np.sum(np.asarray(losses)[0] * weight[0]))


@pytest.mark.timeout(300)
def test_scoring_binds_each_row_to_its_manifest_line_and_matches_single_document_references(tmp_path):
    rows = _rows()
    manifest = _write_fixture(tmp_path, rows)
    with use_test_devices(8):
        mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, context_axis_size=2)
        assert mesh.shape["data"] == 4
        with set_mesh(mesh):
            params = model.Transformer.init(CONFIG, key=jax.random.key(3))
            scored = score_condition_rows(
                params,
                MrcrEvalConfig(
                    data_path=str(tmp_path / "data"),
                    output_path=str(tmp_path),
                    tokenizer=str(tmp_path),
                    batch_size=4,
                    seq_len=SEQ_LEN,
                ),
                mesh=mesh,
                mp=jmp.get_policy("float32"),
                implementation="scatter",
            )
            assert [(s.source_id, s.condition) for s in scored] == [(m["source_id"], m["condition"]) for m in manifest]
            for result, source in zip(scored, rows, strict=True):
                tokens = source["input_ids"]
                assert result.scored_tokens == len(TARGET)
                assert result.scored_bytes > 0
                assert np.isfinite(result.loss_sum)
                np.testing.assert_allclose(result.loss_sum, _reference_loss(params, tokens), rtol=2e-4)
                # Padding reaches neither the attention of real tokens nor the loss.
                np.testing.assert_allclose(result.loss_sum, _reference_loss(params, tokens, padding=7), rtol=2e-4)
            # The context is attended: the same scored tokens lose differently under another context.
            assert abs(scored[0].loss_sum - scored[1].loss_sum) > 1e-3


def _padded_mask(segments):
    return model.AttentionMask.causal().with_segment_ids(jnp.asarray(segments))


def _logits(cfg, tokens, segments, *, context):
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, context_axis_size=context)
    with set_mesh(mesh):
        transformer = model.Transformer.init(cfg, key=jax.random.key(5))
        return np.asarray(eqx.filter_jit(transformer.logits)(jnp.asarray(tokens), _padded_mask(segments)))


@pytest.mark.timeout(300)
def test_log_scaling_n_from_fa4_bounds_counts_keys_in_each_padded_document(tmp_path):
    rows = _rows()
    _write_fixture(tmp_path, rows)
    with use_test_devices(8):
        data = mrcr_data_config(str(tmp_path / "data"), str(tmp_path))
        dataset = data.build_token_datasets(
            data.build_caches("validation"), Axis("position", SEQ_LEN), split="validation"
        )[MRCR_COMPONENT].as_sync_dataset()
        examples = [dataset[index] for index in range(len(rows))]
        tokens = np.stack([np.asarray(example.tokens) for example in examples])
        segments = np.stack([np.asarray(example.attn_mask.segment_ids[0]) for example in examples])
        doc_lengths = [len(row["input_ids"]) for row in rows]

        # Scored positions predict exactly the response; every context, EOS and padding position has weight 0.
        for example, source in zip(examples, rows, strict=True):
            expected = np.zeros(SEQ_LEN)
            expected[np.flatnonzero(source["assistant_masks"]) - 1] = 1
            np.testing.assert_array_equal(np.asarray(example.loss_weight), expected)

        for context in (1, 2, 4):
            with set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, context_axis_size=context)):
                bounds, valid = fa4_cute_segment_bounds(
                    _padded_mask(segments), batch_size=len(rows), seq_len=SEQ_LEN, sliding_window=None
                )
                visible = np.asarray(model._visible_full_causal_keys(bounds, valid))
            for line, length in zip(visible, doc_lengths, strict=True):
                np.testing.assert_array_equal(line[:length], np.arange(1, length + 1))
                np.testing.assert_array_equal(line[length:], 1)

        # The factor reaches the logits under the document's own n.
        floor = 4
        scaled_config = dataclasses.replace(CONFIG, log_scaling_n_floor=floor)
        off = _logits(CONFIG, tokens, segments, context=1)
        scaled = _logits(scaled_config, tokens, segments, context=1)
        for line, length in enumerate(doc_lengths):
            np.testing.assert_allclose(scaled[line, :floor], off[line, :floor], rtol=1e-4, atol=1e-4)
            factor = 1 + 0.1 * math.log(length / floor)
            uniform = _logits(dataclasses.replace(CONFIG, qk_mult_long_scale=factor), tokens, segments, context=1)
            np.testing.assert_allclose(scaled[line, length - 1], uniform[line, length - 1], rtol=1e-4, atol=1e-4)
        for context in (2, 4):
            np.testing.assert_allclose(
                _logits(scaled_config, tokens, segments, context=context), scaled, rtol=1e-4, atol=1e-4
            )
