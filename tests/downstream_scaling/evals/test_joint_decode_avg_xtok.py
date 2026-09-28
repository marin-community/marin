# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import math
import random
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fray.cluster import ResourceConfig
from thalas.execution.context import executor_context
from thalas.execution.executor import compute_output_path

from experiments.downstream_scaling.evals import algorithms
from experiments.downstream_scaling.evals.algorithms import joint_decode_avg_v2, xtok_selection
from experiments.downstream_scaling.evals.algorithms.joint_decode_avg_xtok import (
    JointDecodeConfig,
    JointDecodeExecutionConfig,
    JointDecodeLocalWorkerConfig,
    JointDecodeModelConfig,
    JointDecodePlacement,
    JointDecodePoolConfig,
    JointDecodeSamplingConfig,
    XtokChunkSpec,
    XtokPathStep,
    XtokSelectionRule,
    _child_config_from_file,
    _make_select_token,
    _SweepState,
    _write_child_config,
    joint_decode_pool_configs,
    make_joint_decode_completion_step,
    sweep_chunk_specs,
    write_sweep_chunk,
)
from experiments.downstream_scaling.evals.framework.xregion import ledger
from experiments.downstream_scaling.evals.framework.xregion.pool import EnginePlacement, WorkerPoolConfig

PATH_CONTRACT_PREFIX = "/xregion-tp-path-contract"
XTOK_TP1_PATH = f"{PATH_CONTRACT_PREFIX}/test/joint-decode-avg-xtok-14c664"


class FakeDecoder:
    def generate(self, prompts_a, prompts_b):
        assert prompts_a == prompts_b
        return [SimpleNamespace(text=f"completion for {prompt}", finish_reason="stop") for prompt in prompts_a]


class PromptRecordingDecoder:
    def __init__(self):
        self.batches = []

    def generate(self, prompts_a, prompts_b):
        self.batches.append((prompts_a, prompts_b))
        return [SimpleNamespace(text=f"completion for {prompt}", finish_reason="stop") for prompt in prompts_a]


class ScriptedRandom:
    def __init__(self, tokens, draws=()):
        self.tokens = iter(tokens)
        self.draws = iter(draws)
        self.probabilities = []

    def choices(self, population, weights, *, k):
        total = sum(weights)
        probabilities = {token: weight / total for token, weight in zip(population, weights, strict=True)}
        self.probabilities.append(probabilities)
        chosen = [next(self.tokens) for _ in range(k)]
        assert all(probabilities[token] > 0.0 for token in chosen)
        return chosen

    def random(self):
        return next(self.draws)


def make_vocab(pieces: dict[int, bytes], eos_id: int) -> xtok_selection.Vocab:
    token_bytes: list[bytes | None] = [None] * (max([eos_id, *pieces]) + 1)
    for token_id, piece in pieces.items():
        token_bytes[token_id] = piece
    return xtok_selection.Vocab(
        token_bytes=tuple(token_bytes),
        piece_ids={piece: token_id for token_id, piece in pieces.items()},
        max_piece_len=max(len(piece) for piece in pieces.values()),
        eos_id=eos_id,
    )


def make_sampling(rule: XtokSelectionRule) -> JointDecodeSamplingConfig:
    return JointDecodeSamplingConfig(
        n_samples=2,
        max_tokens=8,
        advisor_max_tokens=16,
        top_k_a=4,
        top_k_b=4,
        seed=0,
        selection_rule=rule,
        advisor_weights=(0.5,),
        temperature=0.0,
    )


def make_chunk(tmp_path) -> XtokChunkSpec:
    return XtokChunkSpec(
        chunk_id=3,
        advisor_weight=0.5,
        weight_index=1,
        chunk_start=2,
        chunk_end=6,
        output_path=str(tmp_path / "chunk-000003.jsonl.gz"),
        token_paths_path=str(tmp_path / "token-paths-000003.jsonl.gz"),
    )


def entry(token_id: int, logit: float) -> dict[str, float]:
    return {"token_id": token_id, "logit": logit}


def read_jsonl_gz(path: str) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def worker_pool(chips_per_vm: int = 4) -> WorkerPoolConfig:
    return WorkerPoolConfig(
        pool_id="test",
        num_workers=1,
        worker_resources=ResourceConfig.with_cpu(),
        vm_count=1,
        chips_per_vm=chips_per_vm,
    )


def test_tp1_completion_path_matches_pre_placement_contract():
    pool = worker_pool()
    pool_config = joint_decode_pool_configs("model", (pool,), {})[0]
    config = JointDecodeConfig(
        sampling=JointDecodeSamplingConfig(
            n_samples=2,
            max_tokens=64,
            advisor_max_tokens=128,
            top_k_a=16,
            top_k_b=16,
            seed=7,
            selection_rule=XtokSelectionRule.BYTES_UNION,
            advisor_weights=(0.0, 0.5, 1.0),
            temperature=0.4,
            stop=("stop-a", "stop-b"),
        ),
        advisor_model_path="/advisor",
        decoder_model=JointDecodeModelConfig(max_model_len=8192),
        advisor_model=JointDecodeModelConfig(max_model_len=8192),
        execution=JointDecodeExecutionConfig(worker_pools=(pool_config,)),
    )
    with executor_context():
        step = make_joint_decode_completion_step(
            name="test/joint-decode-avg-xtok",
            model_path="/model",
            prompts_path="/prompts",
            config=config,
        )

    assert compute_output_path(step.name, step.config, prefix=PATH_CONTRACT_PREFIX) == XTOK_TP1_PATH

    multichip = JointDecodePoolConfig(
        pool=worker_pool(chips_per_vm=8),
        placements=(
            JointDecodePlacement(
                decoder=EnginePlacement((0, 1, 2, 3), (2, 2, 1), 2),
                advisor=EnginePlacement((4,), (1, 1, 1), 1),
            ),
        ),
    )
    multichip_config = JointDecodeConfig(
        sampling=config.sampling,
        advisor_model_path=config.advisor_model_path,
        decoder_model=config.decoder_model,
        advisor_model=config.advisor_model,
        execution=JointDecodeExecutionConfig(worker_pools=(multichip,)),
    )
    with executor_context():
        multichip_step = make_joint_decode_completion_step(
            name="test/joint-decode-avg-xtok",
            model_path="/model",
            prompts_path="/prompts",
            config=multichip_config,
        )
    assert compute_output_path(multichip_step.name, multichip_step.config, prefix=PATH_CONTRACT_PREFIX) == XTOK_TP1_PATH


def test_pool_configs_use_tp1_default_and_exact_override():
    pool = worker_pool()
    override = (
        JointDecodePlacement(
            decoder=EnginePlacement((0, 1, 2), (3, 1, 1), 2),
            advisor=EnginePlacement((3,), (1, 1, 1), 1),
        ),
    )

    default = joint_decode_pool_configs("small", (pool,), {})[0]
    configured = joint_decode_pool_configs("large", (pool,), {("large", "test"): override})[0]

    assert default.placements == (
        JointDecodePlacement(
            decoder=EnginePlacement((0,), (1, 1, 1), 1),
            advisor=EnginePlacement((1,), (1, 1, 1), 1),
        ),
        JointDecodePlacement(
            decoder=EnginePlacement((2,), (1, 1, 1), 1),
            advisor=EnginePlacement((3,), (1, 1, 1), 1),
        ),
    )
    assert configured.placements == override


def test_child_config_json_round_trip(tmp_path):
    config = JointDecodeLocalWorkerConfig(
        decoder_model_path="/decoder",
        advisor_model_path="/advisor",
        prompts_path="/prompts",
        sampling=make_sampling(XtokSelectionRule.BYTES_UNION),
        decoder_model=JointDecodeModelConfig(max_model_len=16384, apply_rpa_block_size_patch=True),
        advisor_model=JointDecodeModelConfig(max_model_len=8192),
        ledger_path="/ledger",
        poll_backoff=0.5,
        microbatch_size=4,
        barrier_timeout_s=60,
        owner="worker-0",
        placement=JointDecodePlacement(
            decoder=EnginePlacement((0, 1, 2, 3), (2, 2, 1), 2),
            advisor=EnginePlacement((4,), (1, 1, 1), 1),
        ),
        max_num_batched_tokens=16388,
    )

    path = _write_child_config(tmp_path, config)

    assert _child_config_from_file(str(path)) == config


@pytest.mark.parametrize(
    ("rule", "advisor_weights", "temperature"),
    [
        (joint_decode_avg_v2.XtokSelectionRule.BYTES_UNION, (0.5,), 0.0),
        (joint_decode_avg_v2.XtokSelectionRule.UNNORMALIZED_ADD_BYTES_UNION, (0.0, 1.4, 16.0), 0.4),
        (joint_decode_avg_v2.XtokSelectionRule.PROB_CAP, (0.0, 1.0, 3.0), 0.4),
        (joint_decode_avg_v2.XtokSelectionRule.HARMONIC_AVG, (0.0, 0.5, 1.0), 0.4),
        (joint_decode_avg_v2.XtokSelectionRule("power_avg=-0.5"), (0.0, 0.5, 1.0), 0.4),
        (joint_decode_avg_v2.XtokSelectionRule("grad_step_size=0.5"), (0.0, 1.0, 64.0), 0.4),
    ],
)
def test_v2_child_config_json_round_trip_preserves_advisor_prompts_path(tmp_path, rule, advisor_weights, temperature):
    config = joint_decode_avg_v2.JointDecodeLocalWorkerConfig(
        decoder_model_path="/decoder",
        advisor_model_path="/advisor",
        prompts_path="/decoder-prompts",
        advisor_prompts_path="/advisor-prompts",
        sampling=joint_decode_avg_v2.JointDecodeSamplingConfig(
            n_samples=2,
            max_tokens=8,
            advisor_max_tokens=16,
            top_k_a=4,
            top_k_b=4,
            seed=0,
            selection_rule=rule,
            advisor_weights=advisor_weights,
            temperature=temperature,
        ),
        decoder_model=joint_decode_avg_v2.JointDecodeModelConfig(max_model_len=16384),
        advisor_model=joint_decode_avg_v2.JointDecodeModelConfig(max_model_len=8192),
        ledger_path="/ledger",
        poll_backoff=0.5,
        microbatch_size=4,
        barrier_timeout_s=60,
        owner="worker-0",
        placement=joint_decode_avg_v2.JointDecodePlacement(
            decoder=EnginePlacement((0,), (1, 1, 1), 1),
            advisor=EnginePlacement((1,), (1, 1, 1), 1),
        ),
    )

    path = joint_decode_avg_v2._write_child_config(tmp_path, config)

    assert joint_decode_avg_v2._child_config_from_file(str(path)) == config


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, 2.0, 16.0])
@pytest.mark.parametrize("student_weight", [0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
def test_v2_unnormalized_add_bytes_union_matches_floor_filled_probabilities(alpha, student_weight):
    sampling = joint_decode_avg_v2.JointDecodeSamplingConfig(
        n_samples=1,
        max_tokens=8,
        advisor_max_tokens=8,
        top_k_a=2,
        top_k_b=2,
        seed=0,
        selection_rule=joint_decode_avg_v2.XtokSelectionRule.UNNORMALIZED_ADD_BYTES_UNION,
        advisor_weights=(alpha,),
        temperature=0.4,
    )
    vocab_a = make_vocab({1: b"a", 2: b"b", 3: b"c"}, eos_id=9)
    vocab_b = make_vocab({11: b"a", 12: b"b", 13: b"c"}, eos_id=8)
    state = joint_decode_avg_v2._SweepState(advisor_weight=alpha, student_weight=student_weight, token_paths={})
    select_token = joint_decode_avg_v2._make_select_token(sampling, vocab_a, vocab_b, state)
    rng = ScriptedRandom([b"a"])

    tokens_a, tokens_b = select_token(
        [entry(1, 0.4 * math.log(2)), entry(2, 0.0)],
        [entry(12, 0.4 * math.log(3) - 1.0), entry(13, -1.0)],
        rng=rng,
        request_index=5,
    )

    # Floor-filled exp((beta * s + alpha * t) / 0.4) is proportional to (2**beta, 3**alpha, 1).
    total = 2.0**student_weight + 3.0**alpha + 1.0
    assert rng.probabilities[0] == pytest.approx(
        {b"a": 2.0**student_weight / total, b"b": 3.0**alpha / total, b"c": 1.0 / total}, rel=1e-12, abs=1e-12
    )
    assert (tokens_a, tokens_b) == ([1], [11])
    assert state.token_paths == {
        5: [joint_decode_avg_v2.XtokPathStep(bytes_hex=b"a".hex(), tokens_a=[1], tokens_b=[11])]
    }


@pytest.mark.parametrize(
    ("student_weight", "alpha", "temperature"),
    [(1.0, 2.0, 0.0), (1.0, 2.0, 0.4), (0.0, 2.0, 0.4), (2.0, 0.0, 0.4), (0.0, 0.0, 0.4)],
)
def test_v2_unnormalized_add_bytes_union_records_distinct_eos(student_weight, alpha, temperature):
    sampling = joint_decode_avg_v2.JointDecodeSamplingConfig(
        n_samples=1,
        max_tokens=8,
        advisor_max_tokens=8,
        top_k_a=2,
        top_k_b=1,
        seed=0,
        selection_rule=joint_decode_avg_v2.XtokSelectionRule.UNNORMALIZED_ADD_BYTES_UNION,
        advisor_weights=(alpha,),
        temperature=temperature,
    )
    vocab_a = make_vocab({1: b"a"}, eos_id=9)
    vocab_b = make_vocab({11: b"a"}, eos_id=8)
    state = joint_decode_avg_v2._SweepState(advisor_weight=alpha, student_weight=student_weight, token_paths={})
    select_token = joint_decode_avg_v2._make_select_token(sampling, vocab_a, vocab_b, state)

    tokens_a, tokens_b = select_token(
        [entry(9, 5.0), entry(1, 0.0)],
        [entry(8, 1.0)],
        rng=ScriptedRandom([xtok_selection.EOS_KEY]),
        request_index=0,
    )

    assert (tokens_a, tokens_b) == ([9], [8])
    assert state.token_paths == {0: [joint_decode_avg_v2.XtokPathStep(bytes_hex="", tokens_a=[9], tokens_b=[8])]}


@pytest.mark.parametrize("step_size", [0.5, 1.5])
@pytest.mark.parametrize("temperature", [0.4, 1.0])
def test_v2_grad_steps_matches_reference_and_records_paths(step_size, temperature):
    sampling = joint_decode_avg_v2.JointDecodeSamplingConfig(
        n_samples=1,
        max_tokens=8,
        advisor_max_tokens=8,
        top_k_a=2,
        top_k_b=2,
        seed=0,
        selection_rule=joint_decode_avg_v2.XtokSelectionRule(f"grad_step_size={step_size}"),
        advisor_weights=(0.0, 1.0, 4.0),
        temperature=temperature,
    )
    vocab_a = make_vocab({1: b"a", 2: b"b", 3: b"c"}, eos_id=9)
    vocab_b = make_vocab({11: b"a", 12: b"b", 13: b"c"}, eos_id=8)
    state = joint_decode_avg_v2._SweepState(advisor_weight=0.0, token_paths={})
    select_token = joint_decode_avg_v2._make_select_token(sampling, vocab_a, vocab_b, state)
    a_topk = [entry(1, math.log(2)), entry(2, 0.0)]
    b_topk = [entry(12, math.log(3) - 1.0), entry(13, -1.0)]

    for request_index, steps in enumerate(sampling.advisor_weights):
        state.advisor_weight = steps
        # Floor-filled logits on (a, b, c); the advisor's raw-logit softmax is (1, 3, 1) / 5.
        logits = np.array([math.log(2), 0.0, 0.0])
        advisor_probs = np.array([1.0, 3.0, 1.0]) / 5.0
        for _ in range(int(steps)):
            student_probs = np.exp(logits) / np.exp(logits).sum()
            logits -= step_size * (student_probs - advisor_probs)
        expected = np.exp(logits / temperature)
        expected /= expected.sum()
        rng = ScriptedRandom([b"a", b"c"])
        for token_a, token_b in ((1, 11), (3, 13)):
            assert select_token(a_topk, b_topk, rng=rng, request_index=request_index) == ([token_a], [token_b])
            assert rng.probabilities[-1] == pytest.approx(
                dict(zip((b"a", b"b", b"c"), expected, strict=True)), rel=1e-12, abs=1e-12
            )
        assert state.token_paths[request_index] == [
            joint_decode_avg_v2.XtokPathStep(bytes_hex=b"a".hex(), tokens_a=[1], tokens_b=[11]),
            joint_decode_avg_v2.XtokPathStep(bytes_hex=b"c".hex(), tokens_a=[3], tokens_b=[13]),
        ]


@pytest.mark.parametrize("temperature", [0.0, 0.4])
def test_v2_grad_steps_records_distinct_eos(temperature):
    sampling = joint_decode_avg_v2.JointDecodeSamplingConfig(
        n_samples=1,
        max_tokens=8,
        advisor_max_tokens=8,
        top_k_a=2,
        top_k_b=1,
        seed=0,
        selection_rule=joint_decode_avg_v2.XtokSelectionRule("grad_step_size=0.5"),
        advisor_weights=(4.0,),
        temperature=temperature,
    )
    vocab_a = make_vocab({1: b"a"}, eos_id=9)
    vocab_b = make_vocab({11: b"a"}, eos_id=8)
    state = joint_decode_avg_v2._SweepState(advisor_weight=4.0, token_paths={})
    select_token = joint_decode_avg_v2._make_select_token(sampling, vocab_a, vocab_b, state)

    tokens_a, tokens_b = select_token(
        [entry(9, 5.0), entry(1, 0.0)],
        [entry(8, 1.0)],
        rng=ScriptedRandom([xtok_selection.EOS_KEY]),
        request_index=0,
    )

    assert (tokens_a, tokens_b) == ([9], [8])
    assert state.token_paths == {0: [joint_decode_avg_v2.XtokPathStep(bytes_hex="", tokens_a=[9], tokens_b=[8])]}


@pytest.mark.parametrize(("steps", "reference_weight"), [(0.0, 0.0), (1024.0, 1.0)])
def test_v2_grad_steps_matches_student_and_converges_to_advisor(steps, reference_weight):
    sampling = joint_decode_avg_v2.JointDecodeSamplingConfig(
        n_samples=1,
        max_tokens=8,
        advisor_max_tokens=8,
        top_k_a=2,
        top_k_b=2,
        seed=0,
        selection_rule=joint_decode_avg_v2.XtokSelectionRule("grad_step_size=0.5"),
        advisor_weights=(steps,),
        temperature=0.4,
    )
    vocab_a = make_vocab({1: b"a", 2: b"b", 3: b"c"}, eos_id=9)
    vocab_b = make_vocab({11: b"a", 12: b"b", 13: b"c"}, eos_id=8)
    state = joint_decode_avg_v2._SweepState(advisor_weight=steps, token_paths={})
    select_token = joint_decode_avg_v2._make_select_token(sampling, vocab_a, vocab_b, state)
    a_topk = [entry(1, math.log(2)), entry(2, 0.0)]
    b_topk = [entry(12, math.log(3) - 1.0), entry(13, -1.0)]
    rng = ScriptedRandom([b"a"])
    reference_rng = ScriptedRandom([b"a"])

    actual = select_token(a_topk, b_topk, rng=rng, request_index=0)
    expected = xtok_selection.select_avg_bytes_union(
        a_topk,
        b_topk,
        advisor_weight=reference_weight,
        temperature=sampling.temperature,
        rng=reference_rng,
        vocab_a=vocab_a,
        vocab_b=vocab_b,
    )

    assert actual == expected
    assert rng.probabilities[0] == pytest.approx(reference_rng.probabilities[0], rel=1e-12, abs=1e-12)


def test_grad_steps_rule_normalizes_numeric_spelling():
    rule = joint_decode_avg_v2.XtokSelectionRule("grad_step_size=5e-1")
    assert rule is joint_decode_avg_v2.XtokSelectionRule("grad_step_size=0.5")
    assert rule.value == "grad_step_size=0.5"
    assert rule.grad_step_size == 0.5
    assert rule.power is None
    assert joint_decode_avg_v2.XtokSelectionRule.BYTES_UNION.grad_step_size is None


@pytest.mark.parametrize("value", ["", "nan", "inf", "-inf", "0", "-0.5"])
def test_grad_steps_rule_rejects_invalid_step_size(value):
    with pytest.raises(ValueError):
        joint_decode_avg_v2.XtokSelectionRule(f"grad_step_size={value}")


@pytest.mark.parametrize("steps", [-1.0, 0.5, math.inf, math.nan])
def test_grad_steps_rejects_invalid_step_count(steps):
    with pytest.raises(ValueError, match="step counts"):
        joint_decode_avg_v2.JointDecodeSamplingConfig(
            n_samples=1,
            max_tokens=8,
            advisor_max_tokens=8,
            top_k_a=2,
            top_k_b=2,
            seed=0,
            selection_rule=joint_decode_avg_v2.XtokSelectionRule("grad_step_size=0.5"),
            advisor_weights=(steps,),
            temperature=0.4,
        )


@pytest.mark.parametrize(
    ("strength", "expected"),
    [
        (0.0, {1: 0.2, 2: 0.2, 3: 0.6}),
        (1.0, {1: 0.4, 2: 0.4, 3: 0.2}),
        (2.0, {1: 0.48, 2: 0.48, 3: 0.04}),
    ],
)
@pytest.mark.parametrize("shifts", [(0.0, 0.0), (1000.0, -1000.0)])
def test_one_sided_logprob_avg_strengths_match_expected_probabilities(strength, expected, shifts):
    a_topk = [
        entry(token_id, math.log(probability) + shifts[0]) for token_id, probability in enumerate((0.2, 0.2, 0.6), 1)
    ]
    b_topk = [
        entry(token_id, math.log(probability) + shifts[1]) for token_id, probability in enumerate((0.3, 0.6, 0.1), 1)
    ]

    tokens, weights = joint_decode_avg_v2._one_sided_logprob_weights(
        a_topk, b_topk, advisor_weight=strength, temperature=1.0
    )

    total = sum(weights)
    probabilities = {token_id: weight / total for token_id, weight in zip(tokens, weights, strict=True)}
    assert probabilities == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("strength", "expected"),
    [(0.0, {1: 0.5, 2: 0.25, 3: 0.25}), (1.0, {1: 1 / 3, 2: 1 / 3, 3: 1 / 3})],
)
def test_one_sided_logprob_avg_partial_overlap_uses_floors_and_temperature(strength, expected):
    tokens, weights = joint_decode_avg_v2._one_sided_logprob_weights(
        [entry(1, 2 * math.log(2)), entry(2, 0.0)],
        [entry(2, 2 * math.log(2)), entry(3, 0.0)],
        advisor_weight=strength,
        temperature=2.0,
    )

    total = sum(weights)
    probabilities = {token_id: weight / total for token_id, weight in zip(tokens, weights, strict=True)}
    assert probabilities == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("rule", "alpha", "expected"),
    [
        (joint_decode_avg_v2.XtokSelectionRule.PROB_CAP, 0.0, {1: 0.5, 2: 0.25, 3: 0.25}),
        (joint_decode_avg_v2.XtokSelectionRule.PROB_CAP, 1.0, {1: 4 / 13, 2: 5 / 13, 3: 4 / 13}),
        (joint_decode_avg_v2.XtokSelectionRule.PROB_CAP, 2.0, {1: 2 / 9, 2: 5 / 9, 3: 2 / 9}),
        (joint_decode_avg_v2.XtokSelectionRule.PROB_CAP, 3.0, {1: 0.2, 2: 0.6, 3: 0.2}),
        (joint_decode_avg_v2.XtokSelectionRule.PROB_CAP, 1e14, {1: 0.2, 2: 0.6, 3: 0.2}),
        (joint_decode_avg_v2.XtokSelectionRule.HARMONIC_AVG, 0.0, {1: 0.5, 2: 0.25, 3: 0.25}),
        (joint_decode_avg_v2.XtokSelectionRule.HARMONIC_AVG, 0.5, {1: 153 / 461, 2: 189 / 461, 3: 119 / 461}),
        (joint_decode_avg_v2.XtokSelectionRule.HARMONIC_AVG, 2 / 3, {1: 77 / 269, 2: 126 / 269, 3: 66 / 269}),
        (joint_decode_avg_v2.XtokSelectionRule.HARMONIC_AVG, 1.0, {1: 0.2, 2: 0.6, 3: 0.2}),
    ],
)
@pytest.mark.parametrize("shifts", [(0.0, 0.0), (1000.0, -1000.0)])
def test_probability_caps_match_floor_filled_probabilities(rule, alpha, expected, shifts):
    # At T=0.4, the floor-filled union has p_a=(1/2, 1/4, 1/4), p_b=(1/5, 3/5, 1/5).
    # max(p_b / p_a)=2.4: alpha=3 caps every token.
    a_topk = [entry(1, 0.4 * math.log(2) + shifts[0]), entry(2, shifts[0])]
    b_topk = [entry(2, 0.4 * math.log(3) - 1.0 + shifts[1]), entry(3, -1.0 + shifts[1])]
    tokens, weights = joint_decode_avg_v2._capped_logprob_weights(
        a_topk, b_topk, selection_rule=rule, advisor_weight=alpha, temperature=0.4
    )

    total = sum(weights)
    probabilities = {token_id: weight / total for token_id, weight in zip(tokens, weights, strict=True)}
    assert probabilities == pytest.approx(expected, rel=1e-12, abs=1e-12)

    if rule is joint_decode_avg_v2.XtokSelectionRule.PROB_CAP and alpha == 1.0:
        old_tokens, old_weights = joint_decode_avg_v2._one_sided_logprob_weights(
            a_topk, b_topk, advisor_weight=1.0, temperature=0.4
        )
        old_total = sum(old_weights)
        old_probabilities = {
            token_id: weight / old_total for token_id, weight in zip(old_tokens, old_weights, strict=True)
        }
        assert probabilities == pytest.approx(old_probabilities, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("rule", "advisor_weight"),
    [(joint_decode_avg_v2.XtokSelectionRule.PROB_CAP, 2.0), (joint_decode_avg_v2.XtokSelectionRule.HARMONIC_AVG, 2 / 3)],
)
def test_probability_caps_remain_finite_for_large_logprob_differences(rule, advisor_weight):
    tokens, weights = joint_decode_avg_v2._capped_logprob_weights(
        [entry(1, 0.0), entry(2, -1000.0)],
        [entry(1, -1000.0), entry(2, 0.0)],
        selection_rule=rule,
        advisor_weight=advisor_weight,
        temperature=0.4,
    )

    total = sum(weights)
    probabilities = {token_id: weight / total for token_id, weight in zip(tokens, weights, strict=True)}
    assert probabilities == pytest.approx({1: 1 / 3, 2: 2 / 3}, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("rule", "advisor_weight"),
    [(joint_decode_avg_v2.XtokSelectionRule.PROB_CAP, 2.0), (joint_decode_avg_v2.XtokSelectionRule.HARMONIC_AVG, 0.5)],
)
@pytest.mark.parametrize("token_id", [1, 9])
def test_probability_caps_record_same_token_and_terminal_steps(rule, advisor_weight, token_id):
    sampling = joint_decode_avg_v2.JointDecodeSamplingConfig(
        n_samples=1,
        max_tokens=8,
        advisor_max_tokens=8,
        top_k_a=1,
        top_k_b=1,
        seed=0,
        selection_rule=rule,
        advisor_weights=(advisor_weight,),
        temperature=0.4,
    )
    vocab = make_vocab({1: b"a"}, eos_id=9)
    state = joint_decode_avg_v2._SweepState(advisor_weight=advisor_weight, token_paths={})
    select_token = joint_decode_avg_v2._make_select_token(sampling, vocab, vocab, state)

    tokens_a, tokens_b = select_token(
        [entry(token_id, 0.0)], [entry(token_id, 0.0)], rng=random.Random(0), request_index=5
    )

    assert (tokens_a, tokens_b) == ([token_id], [token_id])
    assert state.token_paths == {
        5: [
            joint_decode_avg_v2.XtokPathStep(
                bytes_hex=b"a".hex() if token_id == 1 else "", tokens_a=[token_id], tokens_b=[token_id]
            )
        ]
    }


@pytest.mark.parametrize(
    "rule", [joint_decode_avg_v2.XtokSelectionRule.PROB_CAP, joint_decode_avg_v2.XtokSelectionRule.HARMONIC_AVG]
)
@pytest.mark.parametrize(
    ("alpha", "temperature"),
    [(-0.1, 0.4), (math.inf, 0.4), (math.nan, 0.4), (1.0, 0.0), (1.0, math.inf), (1.0, math.nan)],
)
def test_probability_caps_reject_invalid_sampling_parameters(rule, alpha, temperature):
    with pytest.raises(ValueError):
        joint_decode_avg_v2.JointDecodeSamplingConfig(
            n_samples=1,
            max_tokens=8,
            advisor_max_tokens=8,
            top_k_a=1,
            top_k_b=1,
            seed=0,
            selection_rule=rule,
            advisor_weights=(alpha,),
            temperature=temperature,
        )


@pytest.mark.parametrize("power", [-2.0, -1.0, -0.5, 0.0, 1.0, 2.0])
@pytest.mark.parametrize("shifts", [(0.0, 0.0), (1000.0, -1000.0)])
def test_power_avg_selector_matches_direct_mean_and_records_paths(power, shifts):
    sampling = joint_decode_avg_v2.JointDecodeSamplingConfig(
        n_samples=1,
        max_tokens=8,
        advisor_max_tokens=8,
        top_k_a=2,
        top_k_b=2,
        seed=0,
        selection_rule=joint_decode_avg_v2.XtokSelectionRule(f"power_avg={power}"),
        advisor_weights=(0.0, 0.0001, 0.4, 1.0),
        temperature=0.4,
    )
    vocab = make_vocab({1: b"a", 2: b"b"}, eos_id=9)
    state = joint_decode_avg_v2._SweepState(advisor_weight=0.0, token_paths={})
    select_token = joint_decode_avg_v2._make_select_token(sampling, vocab, vocab, state)
    a_topk = [entry(1, 0.4 * math.log(2) + shifts[0]), entry(2, shifts[0])]
    b_topk = [entry(2, 0.4 * math.log(3) - 1.0 + shifts[1]), entry(9, -1.0 + shifts[1])]
    # These are the normalized, floor-filled union probabilities at T=0.4.
    student = {1: 0.5, 2: 0.25, 9: 0.25}
    teacher = {1: 0.2, 2: 0.6, 9: 0.2}

    for request_index, weight in enumerate(sampling.advisor_weights):
        state.advisor_weight = weight
        expected = {
            token: (
                student[token] ** (1.0 - weight) * teacher[token] ** weight
                if power == 0.0
                else ((1.0 - weight) * student[token] ** power + weight * teacher[token] ** power) ** (1.0 / power)
            )
            for token in student
        }
        total = sum(expected.values())
        expected = {token: value / total for token, value in expected.items()}
        rng = ScriptedRandom([1, 9])
        for token in (1, 9):
            assert select_token(a_topk, b_topk, rng=rng, request_index=request_index) == ([token], [token])
            assert rng.probabilities[-1] == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert state.token_paths[request_index] == [
            joint_decode_avg_v2.XtokPathStep(bytes_hex=b"a".hex(), tokens_a=[1], tokens_b=[1]),
            joint_decode_avg_v2.XtokPathStep(bytes_hex="", tokens_a=[9], tokens_b=[9]),
        ]


@pytest.mark.parametrize(
    ("power", "rule"),
    [
        (-1.0, joint_decode_avg_v2.XtokSelectionRule.HARMONIC_AVG),
        (0.0, joint_decode_avg_v2.XtokSelectionRule.AVG_LOGITS),
        (1.0, joint_decode_avg_v2.XtokSelectionRule.AVG_PROBS),
    ],
)
@pytest.mark.parametrize("weight", [0.0, 0.4, 1.0])
def test_power_avg_matches_existing_averaging_rules(power, rule, weight):
    sampling = joint_decode_avg_v2.JointDecodeSamplingConfig(
        n_samples=1,
        max_tokens=8,
        advisor_max_tokens=8,
        top_k_a=2,
        top_k_b=2,
        seed=0,
        selection_rule=rule,
        advisor_weights=(weight,),
        temperature=0.4,
    )
    vocab = make_vocab({1: b"a", 2: b"b", 3: b"c"}, eos_id=9)
    state = joint_decode_avg_v2._SweepState(advisor_weight=weight, token_paths={})
    a_topk = [entry(1, 0.4 * math.log(2)), entry(2, 0.0)]
    b_topk = [entry(2, 0.4 * math.log(3) - 1.0), entry(3, -1.0)]
    reference_rng = ScriptedRandom([1])
    reference = joint_decode_avg_v2._make_select_token(sampling, vocab, vocab, state)
    reference(a_topk, b_topk, rng=reference_rng, request_index=0)

    power_sampling = replace(sampling, selection_rule=joint_decode_avg_v2.XtokSelectionRule(f"power_avg={power}"))
    select_token = joint_decode_avg_v2._make_select_token(power_sampling, vocab, vocab, state)
    rng = ScriptedRandom([1])
    select_token(a_topk, b_topk, rng=rng, request_index=1)

    assert rng.probabilities[0] == pytest.approx(reference_rng.probabilities[0], rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("power", [-1e308, -2.0, -0.5, 0.0, 2.0, 1e308])
def test_power_avg_large_logit_gaps_and_powers_remain_normalizable(power):
    tokens, weights = joint_decode_avg_v2._power_avg_weights(
        [entry(1, 0.0), entry(2, -1000.0), entry(3, -1000.0)],
        [entry(1, -1000.0), entry(2, 0.0), entry(3, -1000.0)],
        advisor_weight=0.5,
        temperature=0.4,
        power=power,
    )
    total = sum(weights)
    probabilities = {token: weight / total for token, weight in zip(tokens, weights, strict=True)}
    if power < 0.0:
        # For the first two tokens, the smaller probability dominates the mean.
        ratio = 0.5 ** (1.0 / power)
        expected = {1: ratio / (2 * ratio + 1), 2: ratio / (2 * ratio + 1), 3: 1 / (2 * ratio + 1)}
    else:
        expected = {1: 0.5, 2: 0.5, 3: 0.0}
    assert probabilities == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_power_avg_rule_normalizes_numeric_spelling():
    rule = joint_decode_avg_v2.XtokSelectionRule("power_avg=-2")
    assert rule is joint_decode_avg_v2.XtokSelectionRule("power_avg=-2.0")
    assert rule.value == "power_avg=-2.0"
    assert rule.power == -2.0
    assert joint_decode_avg_v2.XtokSelectionRule.HARMONIC_AVG.power is None


@pytest.mark.parametrize("value", ["power_avg", "power_avg=", "power_avg=nan", "power_avg=inf", "unknown"])
def test_power_avg_rule_rejects_missing_nonfinite_or_unknown_values(value):
    with pytest.raises(ValueError):
        joint_decode_avg_v2.XtokSelectionRule(value)


@pytest.mark.parametrize("temperature", [0.0, math.inf, math.nan])
def test_power_avg_rejects_nonpositive_or_nonfinite_temperature(temperature):
    with pytest.raises(ValueError):
        joint_decode_avg_v2.JointDecodeSamplingConfig(
            n_samples=1,
            max_tokens=8,
            advisor_max_tokens=8,
            top_k_a=2,
            top_k_b=2,
            seed=0,
            selection_rule=joint_decode_avg_v2.XtokSelectionRule("power_avg=-0.5"),
            advisor_weights=(0.5,),
            temperature=temperature,
        )


def test_harmonic_avg_rejects_weight_above_one():
    with pytest.raises(ValueError):
        joint_decode_avg_v2.JointDecodeSamplingConfig(
            n_samples=1,
            max_tokens=8,
            advisor_max_tokens=8,
            top_k_a=1,
            top_k_b=1,
            seed=0,
            selection_rule=joint_decode_avg_v2.XtokSelectionRule.HARMONIC_AVG,
            advisor_weights=(1.1,),
            temperature=0.4,
        )


@pytest.mark.parametrize(
    "rule",
    [joint_decode_avg_v2.XtokSelectionRule.TAU_FILTER_STUDENT, joint_decode_avg_v2.XtokSelectionRule.TAU_FILTER_ADVISOR],
)
@pytest.mark.parametrize("tau", [0.0, 0.01, 10.0])
@pytest.mark.parametrize("shifts", [(0.0, 0.0), (1000.0, -1000.0)])
def test_tau_filter_matches_floor_filled_probabilities(rule, tau, shifts):
    # At T=0.4, the floor-filled union has S=(1/2, 1/4, 1/4), T=(1/5, 3/5, 1/5).
    a_topk = [entry(1, 0.4 * math.log(2) + shifts[0]), entry(2, shifts[0])]
    b_topk = [entry(2, 0.4 * math.log(3) - 1.0 + shifts[1]), entry(3, -1.0 + shifts[1])]
    base = {1: 0.5, 2: 0.25, 3: 0.25}
    other = {1: 0.2, 2: 0.6, 3: 0.2}
    if rule is joint_decode_avg_v2.XtokSelectionRule.TAU_FILTER_ADVISOR:
        base, other = other, base
    raw = {token_id: probability * other[token_id] / (other[token_id] + tau) for token_id, probability in base.items()}
    expected = {token_id: weight / sum(raw.values()) for token_id, weight in raw.items()}

    tokens, weights = joint_decode_avg_v2._tau_filter_weights(
        a_topk, b_topk, selection_rule=rule, tau=tau, temperature=0.4
    )

    total = sum(weights)
    probabilities = {token_id: weight / total for token_id, weight in zip(tokens, weights, strict=True)}
    assert probabilities == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        (joint_decode_avg_v2.XtokSelectionRule.TAU_FILTER_STUDENT, {1: 0.75, 2: 0.25}),
        (joint_decode_avg_v2.XtokSelectionRule.TAU_FILTER_ADVISOR, {1: 0.25, 2: 0.75}),
    ],
)
def test_tau_filter_remains_finite_for_large_logprob_differences(rule, expected):
    tokens, weights = joint_decode_avg_v2._tau_filter_weights(
        [entry(1, 0.0), entry(2, -1000.0)],
        [entry(1, -1000.0), entry(2, 0.0)],
        selection_rule=rule,
        tau=0.5,
        temperature=0.4,
    )

    assert all(math.isfinite(weight) for weight in weights)
    total = sum(weights)
    assert total > 0.0
    probabilities = {token_id: weight / total for token_id, weight in zip(tokens, weights, strict=True)}
    assert probabilities == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "rule",
    [joint_decode_avg_v2.XtokSelectionRule.TAU_FILTER_STUDENT, joint_decode_avg_v2.XtokSelectionRule.TAU_FILTER_ADVISOR],
)
def test_tau_filter_large_tau_approaches_normalized_product(rule):
    tokens, weights = joint_decode_avg_v2._tau_filter_weights(
        [entry(1, 0.4 * math.log(2)), entry(2, 0.0)],
        [entry(2, 0.4 * math.log(3) - 1.0), entry(3, -1.0)],
        selection_rule=rule,
        tau=1e14,
        temperature=0.4,
    )

    total = sum(weights)
    probabilities = {token_id: weight / total for token_id, weight in zip(tokens, weights, strict=True)}
    assert probabilities == pytest.approx({1: 1 / 3, 2: 1 / 2, 3: 1 / 6}, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    "rule",
    [joint_decode_avg_v2.XtokSelectionRule.TAU_FILTER_STUDENT, joint_decode_avg_v2.XtokSelectionRule.TAU_FILTER_ADVISOR],
)
@pytest.mark.parametrize("token_id", [1, 9])
def test_tau_filter_records_same_token_and_terminal_steps(rule, token_id):
    sampling = joint_decode_avg_v2.JointDecodeSamplingConfig(
        n_samples=1,
        max_tokens=8,
        advisor_max_tokens=8,
        top_k_a=1,
        top_k_b=1,
        seed=0,
        selection_rule=rule,
        advisor_weights=(10.0,),
        temperature=0.4,
    )
    vocab = make_vocab({1: b"a"}, eos_id=9)
    state = joint_decode_avg_v2._SweepState(advisor_weight=10.0, token_paths={})
    select_token = joint_decode_avg_v2._make_select_token(sampling, vocab, vocab, state)

    tokens_a, tokens_b = select_token(
        [entry(token_id, 0.0)], [entry(token_id, 0.0)], rng=random.Random(0), request_index=5
    )

    assert (tokens_a, tokens_b) == ([token_id], [token_id])
    assert state.token_paths == {
        5: [
            joint_decode_avg_v2.XtokPathStep(
                bytes_hex=b"a".hex() if token_id == 1 else "", tokens_a=[token_id], tokens_b=[token_id]
            )
        ]
    }


@pytest.fixture
def token_dropout_selector():
    sampling = joint_decode_avg_v2.JointDecodeSamplingConfig(
        n_samples=1,
        max_tokens=8,
        advisor_max_tokens=8,
        top_k_a=3,
        top_k_b=3,
        seed=0,
        selection_rule=joint_decode_avg_v2.XtokSelectionRule.TOKEN_DROPOUT,
        advisor_weights=(0.0, 0.25, 0.5, 0.75, 1.0),
        temperature=0.4,
    )
    vocab = make_vocab({1: b"a", 2: b"b"}, eos_id=9)
    state = joint_decode_avg_v2._SweepState(advisor_weight=0.5, token_paths={})
    return sampling, state, joint_decode_avg_v2._make_select_token(sampling, vocab, vocab, state)


@pytest.mark.parametrize(
    ("dropout", "expected"),
    [(0.0, {1: 0.5, 2: 0.25, 9: 0.25}), (1.0, {1: 0.2, 2: 0.6, 9: 0.2})],
)
def test_token_dropout_endpoints_use_union_floors_and_temperature(token_dropout_selector, dropout, expected):
    _, state, select_token = token_dropout_selector
    state.advisor_weight = dropout
    rng = ScriptedRandom(tokens=(9,))

    tokens = select_token(
        [entry(1, 0.4 * math.log(2)), entry(2, 0.0)],
        [entry(2, 0.4 * math.log(3) - 1.0), entry(9, -1.0)],
        rng=rng,
        request_index=5,
    )

    assert rng.probabilities == [pytest.approx(expected, rel=1e-12, abs=1e-12)]
    assert tokens == ([9], [9])
    assert state.token_paths == {5: [joint_decode_avg_v2.XtokPathStep(bytes_hex="", tokens_a=[9], tokens_b=[9])]}


@pytest.mark.parametrize(
    ("dropout", "draws", "expected_student"),
    [
        (0.25, (0.1, 0.1), {9: 1.0}),
        (0.5, (0.9, 0.9), {1: 4 / 7, 2: 2 / 7, 9: 1 / 7}),
        (0.75, (0.5, 0.9), {2: 2 / 3, 9: 1 / 3}),
    ],
)
def test_token_dropout_preserves_teacher_and_conditions_student(
    token_dropout_selector, dropout, draws, expected_student
):
    _, state, select_token = token_dropout_selector
    state.advisor_weight = dropout
    output_token = min(expected_student)
    rng = ScriptedRandom(tokens=(9, output_token), draws=draws)

    tokens = select_token(
        [entry(1, 0.4 * math.log(4)), entry(2, 0.4 * math.log(2)), entry(9, 0.0)],
        [entry(1, 0.4 * math.log(3)), entry(2, 0.0), entry(9, 0.0)],
        rng=rng,
        request_index=5,
    )

    assert rng.probabilities == [
        pytest.approx({1: 0.6, 2: 0.2, 9: 0.2}, rel=1e-12, abs=1e-12),
        pytest.approx(expected_student, rel=1e-12, abs=1e-12),
    ]
    assert tokens == ([output_token], [output_token])
    assert state.token_paths == {
        5: [
            joint_decode_avg_v2.XtokPathStep(
                bytes_hex={1: "61", 2: "62", 9: ""}[output_token], tokens_a=[output_token], tokens_b=[output_token]
            )
        ]
    }


def test_token_dropout_recenters_student_logits_after_masking(token_dropout_selector):
    _, _, select_token = token_dropout_selector
    rng = ScriptedRandom(tokens=(9, 2), draws=(0.1, 0.9))

    tokens = select_token(
        [entry(1, 0.0), entry(2, -1000.0), entry(9, -1000.0 - 0.4 * math.log(2))],
        [entry(9, 0.0)],
        rng=rng,
        request_index=5,
    )

    assert rng.probabilities[-1] == pytest.approx({2: 2 / 3, 9: 1 / 3}, rel=1e-12, abs=1e-12)
    assert tokens == ([2], [2])


@pytest.mark.parametrize(
    ("dropout", "temperature"),
    [(-0.1, 0.4), (1.1, 0.4), (math.nan, 0.4), (math.inf, 0.4), (0.5, 0.0), (0.5, math.nan), (0.5, math.inf)],
)
def test_token_dropout_rejects_invalid_sampling_parameters(token_dropout_selector, dropout, temperature):
    sampling, _, _ = token_dropout_selector
    with pytest.raises(ValueError):
        replace(sampling, advisor_weights=(dropout,), temperature=temperature)


def test_sweep_chunk_specs_tile_each_weight_exactly_once():
    # 3 prompts x 2 samples = 6 requests per weight, chunked by 4.
    specs = sweep_chunk_specs("chunks", num_prompts=3, n_samples=2, advisor_weights=(0.0, 0.5), chunk_size=4)

    assert [spec.chunk_id for spec in specs] == list(range(len(specs)))
    assert len({spec.output_path for spec in specs}) == len(specs)
    assert len({spec.token_paths_path for spec in specs}) == len(specs)
    ranges_by_weight = {}
    for spec in specs:
        assert spec.advisor_weight == (0.0, 0.5)[spec.weight_index]
        ranges_by_weight.setdefault(spec.advisor_weight, []).append((spec.chunk_start, spec.chunk_end))
    assert ranges_by_weight == {0.0: [(0, 4), (4, 6)], 0.5: [(0, 4), (4, 6)]}


@pytest.mark.parametrize("rule", [XtokSelectionRule.BYTES_UNION, XtokSelectionRule.ANCHORED_PREFIX_MASS])
def test_selector_records_committed_bytes_and_both_sides_tokens(rule):
    # A merges " the"; B segments it as " th" + "e". Both rules pick " the"
    # at these logits and must record the identical committed step.
    vocab_a = make_vocab({1: b" the", 5: b"x"}, eos_id=9)
    vocab_b = make_vocab({2: b" th", 3: b"e", 4: b" a"}, eos_id=8)
    state = _SweepState(advisor_weight=0.5, token_paths={})
    select_token = _make_select_token(make_sampling(rule), vocab_a, vocab_b, state)

    tokens_a, tokens_b = select_token(
        [entry(1, 6.0), entry(5, 0.0)],
        [entry(4, 1.0)],
        rng=random.Random(0),
        request_index=5,
    )

    assert (tokens_a, tokens_b) == ([1], [2, 3])
    assert state.token_paths == {5: [XtokPathStep(bytes_hex=b" the".hex(), tokens_a=[1], tokens_b=[2, 3])]}


def test_selector_records_eos_step_with_empty_bytes():
    vocab_a = make_vocab({1: b" the"}, eos_id=9)
    vocab_b = make_vocab({4: b" a"}, eos_id=8)
    state = _SweepState(advisor_weight=0.5, token_paths={})
    select_token = _make_select_token(make_sampling(XtokSelectionRule.BYTES_UNION), vocab_a, vocab_b, state)

    tokens_a, tokens_b = select_token(
        [entry(9, 5.0), entry(1, 0.0)],
        [entry(8, 1.0)],
        rng=random.Random(0),
        request_index=0,
    )

    assert (tokens_a, tokens_b) == ([9], [8])
    assert state.token_paths == {0: [XtokPathStep(bytes_hex="", tokens_a=[9], tokens_b=[8])]}


def test_selector_keeps_interleaved_requests_separate():
    vocab_a = make_vocab({1: b" the", 5: b"x"}, eos_id=9)
    vocab_b = make_vocab({2: b" th", 3: b"e"}, eos_id=8)
    state = _SweepState(advisor_weight=0.5, token_paths={})
    select_token = _make_select_token(make_sampling(XtokSelectionRule.BYTES_UNION), vocab_a, vocab_b, state)

    for request_index in (0, 1, 0):
        select_token([entry(1, 6.0), entry(5, 0.0)], [entry(2, 1.0)], rng=random.Random(0), request_index=request_index)

    assert {index: len(steps) for index, steps in state.token_paths.items()} == {0: 2, 1: 1}


def test_write_sweep_chunk_offsets_completion_indices_and_writes_sidecar(tmp_path):
    # weight_index=1 with n_samples=2 puts this weight's samples at
    # completion_index 2..3; range [2, 6) covers prompts 1 and 2.
    chunk = make_chunk(tmp_path)
    step = XtokPathStep(bytes_hex=b" 42".hex(), tokens_a=[10], tokens_b=[11, 12])
    write_sweep_chunk(
        chunk,
        decoder=FakeDecoder(),
        prompt_ids=["p0", "p1", "p2"],
        prompts=["q0", "q1", "q2"],
        n_samples=2,
        token_paths={batch_index: [step] for batch_index in range(4)},
    )

    records = read_jsonl_gz(chunk.output_path)
    keys = [(record["id"], record["completion_index"]) for record in records]
    assert keys == [("p1", 2), ("p1", 3), ("p2", 2), ("p2", 3)]
    assert records[0]["completion"] == {
        "text": "completion for q1",
        "metadata": {"finish_reason": "stop", "advisor_weight": 0.5},
    }

    path_records = read_jsonl_gz(chunk.token_paths_path)
    assert [(record["id"], record["completion_index"]) for record in path_records] == keys
    assert path_records[0] == {
        "id": "p1",
        "completion_index": 2,
        "advisor_weight": 0.5,
        "weight_index": 1,
        "steps": [{"bytes_hex": b" 42".hex(), "tokens_a": [10], "tokens_b": [11, 12]}],
    }


@pytest.mark.parametrize(
    ("advisor_prompts", "expected_advisor_batch"),
    [
        (None, ["q1", "q1", "q2", "q2"]),
        (["b0", "b1", "b2"], ["b1", "b1", "b2", "b2"]),
    ],
)
def test_v2_write_sweep_chunk_routes_decoder_and_advisor_prompts(
    tmp_path,
    advisor_prompts,
    expected_advisor_batch,
):
    chunk = joint_decode_avg_v2.XtokChunkSpec(
        chunk_id=3,
        advisor_weight=0.5,
        weight_index=1,
        chunk_start=2,
        chunk_end=6,
        output_path=str(tmp_path / "chunk-000003.jsonl.gz"),
        token_paths_path=str(tmp_path / "token-paths-000003.jsonl.gz"),
    )
    step = joint_decode_avg_v2.XtokPathStep(bytes_hex=b" 42".hex(), tokens_a=[10], tokens_b=[11, 12])
    decoder = PromptRecordingDecoder()

    joint_decode_avg_v2.write_sweep_chunk(
        chunk,
        decoder=decoder,
        prompt_ids=["p0", "p1", "p2"],
        prompts=["q0", "q1", "q2"],
        advisor_prompts=advisor_prompts,
        n_samples=2,
        token_paths={batch_index: [step] for batch_index in range(4)},
    )

    assert decoder.batches == [(["q1", "q1", "q2", "q2"], expected_advisor_batch)]


def test_v2_student_sweep_reuses_engines_and_records_each_pair(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts.jsonl.gz"
    with gzip.open(prompts_path, "wt") as f:
        for i in range(3):
            f.write(json.dumps({"id": f"p{i}", "prompt": f"q{i}"}) + "\n")
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    student_weights = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    sampling = joint_decode_avg_v2.JointDecodeSamplingConfig(
        n_samples=2,
        max_tokens=8,
        advisor_max_tokens=8,
        top_k_a=2,
        top_k_b=2,
        seed=0,
        selection_rule=joint_decode_avg_v2.XtokSelectionRule.UNNORMALIZED_ADD_BYTES_UNION,
        advisor_weights=(0.0, 2.0),
        temperature=0.4,
    )
    config = joint_decode_avg_v2.JointDecodeLocalWorkerConfig(
        decoder_model_path="/student",
        advisor_model_path="/advisor",
        prompts_path=str(prompts_path),
        advisor_prompts_path=None,
        sampling=sampling,
        decoder_model=joint_decode_avg_v2.JointDecodeModelConfig(),
        advisor_model=joint_decode_avg_v2.JointDecodeModelConfig(),
        ledger_path=str(tmp_path / "ledger"),
        poll_backoff=0.0,
        microbatch_size=4,
        barrier_timeout_s=60,
        owner="test-worker",
        placement=joint_decode_avg_v2.JointDecodePlacement(
            decoder=EnginePlacement((0,), (1, 1, 1), 1),
            advisor=EnginePlacement((1,), (1, 1, 1), 1),
        ),
        student_weights=student_weights,
    )
    config_path = joint_decode_avg_v2._write_child_config(tmp_path, config)
    loaded_config = joint_decode_avg_v2._child_config_from_file(str(config_path))
    assert loaded_config == config
    chunks = joint_decode_avg_v2.sweep_chunk_specs(
        str(chunks_dir), 3, sampling.n_samples, sampling.advisor_weights, 4, student_weights=student_weights
    )
    ledger.ensure_manifest(config.ledger_path, chunks)

    probabilities = []
    engine_events = []

    @contextmanager
    def open_joint_decoder(**kwargs):
        engine_events.append("open")

        def generate(prompts_a, prompts_b):
            assert prompts_a == prompts_b
            rng = ScriptedRandom([b"a"] * len(prompts_a))
            for request_index in range(len(prompts_a)):
                kwargs["select_token"](
                    [entry(1, 0.4 * math.log(2)), entry(2, 0.0)],
                    [entry(12, 0.4 * math.log(3) - 1.0), entry(13, -1.0)],
                    rng=rng,
                    request_index=request_index,
                )
            probabilities.extend(rng.probabilities)
            return [SimpleNamespace(text="a", finish_reason="stop") for _ in prompts_a]

        yield SimpleNamespace(generate=generate)
        engine_events.append("close")

    # Replace model/tokenizer I/O; retain the real worker, selector, ledger, and record writing.
    monkeypatch.setattr(
        algorithms,
        "joint_decode_backend",
        SimpleNamespace(EngineModelParams=SimpleNamespace, open_joint_decoder=open_joint_decoder),
        raising=False,
    )
    vocabs = {
        "/student": make_vocab({1: b"a", 2: b"b", 3: b"c"}, eos_id=9),
        "/advisor": make_vocab({11: b"a", 12: b"b", 13: b"c"}, eos_id=8),
    }
    monkeypatch.setattr(joint_decode_avg_v2, "_load_vocab", vocabs.__getitem__)
    monkeypatch.setenv("TPU_VISIBLE_CHIPS", "0,1")
    joint_decode_avg_v2._run_joint_decode_local_worker(loaded_config)

    assert engine_events == ["open", "close"]
    assert ledger.done_chunk_ids(config.ledger_path) == list(range(len(chunks)))
    pairs = [(alpha, beta) for alpha in sampling.advisor_weights for beta in student_weights]
    records = [record for chunk in chunks for record in read_jsonl_gz(chunk.output_path)]
    paths = [record for chunk in chunks for record in read_jsonl_gz(chunk.token_paths_path)]
    assert len(records) == 3 * sampling.n_samples * len(pairs)
    for problem_id in ("p0", "p1", "p2"):
        assert [row["completion_index"] for row in records if row["id"] == problem_id] == list(
            range(sampling.n_samples * len(pairs))
        )
    for record, path, observed in zip(records, paths, probabilities, strict=True):
        pair_index = record["completion_index"] // sampling.n_samples
        alpha, beta = pairs[pair_index]
        assert record["completion"]["metadata"] == {
            "finish_reason": "stop",
            "advisor_weight": alpha,
            "student_weight": beta,
        }
        assert path == {
            "id": record["id"],
            "completion_index": record["completion_index"],
            "advisor_weight": alpha,
            "student_weight": beta,
            "weight_index": pair_index,
            "steps": [{"bytes_hex": b"a".hex(), "tokens_a": [1], "tokens_b": [11]}],
        }
        total = 2.0**beta + 3.0**alpha + 1.0
        assert observed == pytest.approx(
            {b"a": 2.0**beta / total, b"b": 3.0**alpha / total, b"c": 1.0 / total}, rel=1e-12, abs=1e-12
        )


def test_v2_student_grid_changes_step_identity():
    config = joint_decode_avg_v2.JointDecodeConfig(
        sampling=joint_decode_avg_v2.JointDecodeSamplingConfig(
            n_samples=2,
            max_tokens=8,
            advisor_max_tokens=8,
            top_k_a=2,
            top_k_b=2,
            seed=0,
            selection_rule=joint_decode_avg_v2.XtokSelectionRule.UNNORMALIZED_ADD_BYTES_UNION,
            advisor_weights=(0.0, 1.0),
            temperature=0.4,
        ),
        advisor_model_path="/advisor",
        decoder_model=joint_decode_avg_v2.JointDecodeModelConfig(),
        advisor_model=joint_decode_avg_v2.JointDecodeModelConfig(),
        execution=joint_decode_avg_v2.JointDecodeExecutionConfig(worker_pools=()),
    )
    paths = []
    for grid in (None, (1.0,), (0.0, 1.0), (1.0, 0.0)):
        with executor_context():
            step = joint_decode_avg_v2.make_joint_decode_completion_step(
                name="student-sweep",
                model_path="/student",
                prompts_path="/prompts",
                config=replace(config, student_weights=grid),
            )
        paths.append(compute_output_path(step.name, step.config, prefix=PATH_CONTRACT_PREFIX))
    assert len(set(paths)) == len(paths)


def test_write_sweep_chunk_missing_trace_for_nonempty_text_raises(tmp_path):
    step = XtokPathStep(bytes_hex=b"a".hex(), tokens_a=[10], tokens_b=[11])
    with pytest.raises(RuntimeError, match="no token path"):
        write_sweep_chunk(
            make_chunk(tmp_path),
            decoder=FakeDecoder(),
            prompt_ids=["p0", "p1", "p2"],
            prompts=["q0", "q1", "q2"],
            n_samples=2,
            token_paths={batch_index: [step] for batch_index in (0, 1, 3)},  # 2 missing
        )


def test_write_sweep_chunk_leftover_trace_raises(tmp_path):
    step = XtokPathStep(bytes_hex=b"a".hex(), tokens_a=[10], tokens_b=[11])
    with pytest.raises(RuntimeError, match="unknown batch indices"):
        write_sweep_chunk(
            make_chunk(tmp_path),
            decoder=FakeDecoder(),
            prompt_ids=["p0", "p1", "p2"],
            prompts=["q0", "q1", "q2"],
            n_samples=2,
            token_paths={batch_index: [step] for batch_index in (0, 1, 2, 3, 99)},
        )
