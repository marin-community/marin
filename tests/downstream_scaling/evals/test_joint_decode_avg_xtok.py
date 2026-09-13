# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import math
import random
from types import SimpleNamespace

import pytest
from fray.cluster import ResourceConfig
from thalas.execution.context import executor_context
from thalas.execution.executor import compute_output_path

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


def test_v2_child_config_json_round_trip_preserves_advisor_prompts_path(tmp_path):
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
            selection_rule=joint_decode_avg_v2.XtokSelectionRule.BYTES_UNION,
            advisor_weights=(0.5,),
            temperature=0.0,
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
