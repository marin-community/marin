# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fray.cluster import ResourceConfig
from thalas.execution.context import executor_context
from thalas.execution.executor import collect_dependencies_and_version, compute_output_path

from experiments.downstream_scaling.evals.algorithms import xtok_selection
from experiments.downstream_scaling.evals.algorithms.joint_decode_entropy_xtok import (
    EntropySource,
    GateDirection,
    JointDecodeConfig,
    JointDecodeExecutionConfig,
    JointDecodeLocalWorkerConfig,
    JointDecodeModelConfig,
    JointDecodePlacement,
    JointDecodePoolConfig,
    JointDecodeSamplingConfig,
    XtokChunkSpec,
    XtokPathStep,
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
ENTROPY_XTOK_TP1_PATH = f"{PATH_CONTRACT_PREFIX}/test/joint-decode-entropy-xtok-ca17ac"


class FakeDecoder:
    def generate(self, prompts_a, prompts_b):
        assert prompts_a == prompts_b
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


def make_sampling(entropy_thresholds: tuple[float, ...] = (10.0,)) -> JointDecodeSamplingConfig:
    return JointDecodeSamplingConfig(
        n_samples=2,
        max_tokens=8,
        advisor_max_tokens=16,
        top_k_a=4,
        top_k_b=4,
        seed=0,
        entropy_thresholds=entropy_thresholds,
        temperature=0.0,
    )


def make_chunk(tmp_path) -> XtokChunkSpec:
    return XtokChunkSpec(
        chunk_id=3,
        entropy_threshold=0.5,
        threshold_index=1,
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


@pytest.mark.parametrize(
    "entropy_thresholds",
    [(), (0.5, 0.5), (-0.1,), (float("inf"),), (float("nan"),)],
)
def test_sampling_config_rejects_bad_thresholds(entropy_thresholds):
    with pytest.raises(ValueError, match="entropy_thresholds"):
        make_sampling(entropy_thresholds)


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
            entropy_thresholds=(0.0, 1.5, 1e9),
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
            name="test/joint-decode-entropy-xtok",
            model_path="/model",
            prompts_path="/prompts",
            config=config,
        )

    assert compute_output_path(step.name, step.config, prefix=PATH_CONTRACT_PREFIX) == ENTROPY_XTOK_TP1_PATH

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
            name="test/joint-decode-entropy-xtok",
            model_path="/model",
            prompts_path="/prompts",
            config=multichip_config,
        )
    assert (
        compute_output_path(multichip_step.name, multichip_step.config, prefix=PATH_CONTRACT_PREFIX)
        == ENTROPY_XTOK_TP1_PATH
    )


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
        sampling=make_sampling((0.0, 1.5, 1e9)),
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
        # Non-default: json writes the StrEnum as its value, so the round trip
        # only holds if the reader reconstructs the member.
        gate=GateDirection.ADVISOR_BELOW,
        entropy_source=EntropySource.DECODER,
        max_num_batched_tokens=16388,
    )

    path = _write_child_config(tmp_path, config)

    assert _child_config_from_file(str(path)) == config


def test_sweep_chunk_specs_tile_each_threshold_exactly_once():
    # 3 prompts x 2 samples = 6 requests per threshold, chunked by 4.
    specs = sweep_chunk_specs("chunks", num_prompts=3, n_samples=2, entropy_thresholds=(0.0, 1.5), chunk_size=4)

    assert [spec.chunk_id for spec in specs] == list(range(len(specs)))
    assert len({spec.output_path for spec in specs}) == len(specs)
    assert len({spec.token_paths_path for spec in specs}) == len(specs)
    ranges_by_threshold = {}
    for spec in specs:
        assert spec.entropy_threshold == (0.0, 1.5)[spec.threshold_index]
        ranges_by_threshold.setdefault(spec.entropy_threshold, []).append((spec.chunk_start, spec.chunk_end))
    assert ranges_by_threshold == {0.0: [(0, 4), (4, 6)], 1.5: [(0, 4), (4, 6)]}


def test_selector_records_committed_bytes_both_sides_tokens_and_entropy():
    # A merges " the"; B segments it as " th" + "e". B's near-uniform pair
    # has entropy ~ 0.69 < the 1.0 threshold, so the decoder's argmax
    # commits.
    vocab_a = make_vocab({1: b" the", 5: b"x", 6: b" ", 7: b"a"}, eos_id=9)
    vocab_b = make_vocab({2: b" th", 3: b"e", 4: b" a"}, eos_id=8)
    a_topk = [entry(1, 6.0), entry(5, 0.0)]
    b_topk = [entry(4, 1.1), entry(2, 1.0)]
    state = _SweepState(entropy_threshold=1.0, token_paths={})
    select_token = _make_select_token(
        make_sampling(), vocab_a, vocab_b, state, GateDirection.ADVISOR_AT_OR_ABOVE, EntropySource.ADVISOR
    )

    tokens_a, tokens_b = select_token(a_topk, b_topk, rng=random.Random(0), request_index=5)

    expected_entropy = xtok_selection.topk_entropy(xtok_selection.candidates(vocab_b, b_topk))
    assert (tokens_a, tokens_b) == ([1], [2, 3])
    assert state.token_paths == {
        5: [XtokPathStep(bytes_hex=b" the".hex(), tokens_a=[1], tokens_b=[2, 3], entropy=expected_entropy)]
    }


def test_selector_gates_to_advisor_at_or_above_threshold():
    # Same inputs as above but the threshold sits below the ~ 0.69 entropy:
    # the advisor's " a" commits, segmented byte-exact on A's vocab.
    vocab_a = make_vocab({1: b" the", 5: b"x", 6: b" ", 7: b"a"}, eos_id=9)
    vocab_b = make_vocab({2: b" th", 3: b"e", 4: b" a"}, eos_id=8)
    state = _SweepState(entropy_threshold=0.5, token_paths={})
    select_token = _make_select_token(
        make_sampling(), vocab_a, vocab_b, state, GateDirection.ADVISOR_AT_OR_ABOVE, EntropySource.ADVISOR
    )

    tokens_a, tokens_b = select_token(
        [entry(1, 6.0), entry(5, 0.0)], [entry(4, 1.1), entry(2, 1.0)], rng=random.Random(0), request_index=0
    )

    assert (tokens_a, tokens_b) == ([6, 7], [4])
    steps = state.token_paths[0]
    assert len(steps) == 1
    assert steps[0].bytes_hex == b" a".hex()
    assert steps[0].entropy >= state.entropy_threshold


def test_selector_gates_to_advisor_below_threshold_under_advisor_below():
    # Identical inputs and threshold to
    # test_selector_records_committed_bytes_both_sides_tokens_and_entropy,
    # where the ~ 0.69 entropy is below the 1.0 cutoff and the decoder
    # commits. Flipping only the configured direction flips the winner, so
    # the module is passing its gate through to the selector.
    vocab_a = make_vocab({1: b" the", 5: b"x", 6: b" ", 7: b"a"}, eos_id=9)
    vocab_b = make_vocab({2: b" th", 3: b"e", 4: b" a"}, eos_id=8)
    state = _SweepState(entropy_threshold=1.0, token_paths={})
    select_token = _make_select_token(
        make_sampling(), vocab_a, vocab_b, state, GateDirection.ADVISOR_BELOW, EntropySource.ADVISOR
    )

    tokens_a, tokens_b = select_token(
        [entry(1, 6.0), entry(5, 0.0)], [entry(4, 1.1), entry(2, 1.0)], rng=random.Random(0), request_index=0
    )

    assert (tokens_a, tokens_b) == ([6, 7], [4])
    steps = state.token_paths[0]
    assert len(steps) == 1
    assert steps[0].bytes_hex == b" a".hex()
    assert steps[0].entropy < state.entropy_threshold


def test_gate_direction_enters_the_version_only_when_it_is_not_the_default():
    # The completed sweeps hashed before `gate` existed: versioning it
    # unconditionally would re-key every one of their steps, and leaving it
    # unversioned would let two directions collide at one output path.
    pool_config = joint_decode_pool_configs("model", (worker_pool(),), {})[0]
    config = JointDecodeConfig(
        sampling=make_sampling(),
        advisor_model_path="/advisor",
        decoder_model=JointDecodeModelConfig(),
        advisor_model=JointDecodeModelConfig(),
        execution=JointDecodeExecutionConfig(worker_pools=(pool_config,)),
    )
    assert config.gate is GateDirection.ADVISOR_AT_OR_ABOVE

    with executor_context():
        default_step = make_joint_decode_completion_step(
            name="test/gate", model_path="/model", prompts_path="/prompts", config=config
        )
        low_step = make_joint_decode_completion_step(
            name="test/gate",
            model_path="/model",
            prompts_path="/prompts",
            config=replace(config, gate=GateDirection.ADVISOR_BELOW),
        )

    assert "gate" not in collect_dependencies_and_version(default_step.config).version
    assert collect_dependencies_and_version(low_step.config).version["gate"] == "advisor_below"


def test_entropy_source_enters_the_version_only_when_it_is_not_the_default():
    pool_config = joint_decode_pool_configs("model", (worker_pool(),), {})[0]
    config = JointDecodeConfig(
        sampling=make_sampling(),
        advisor_model_path="/advisor",
        decoder_model=JointDecodeModelConfig(),
        advisor_model=JointDecodeModelConfig(),
        execution=JointDecodeExecutionConfig(worker_pools=(pool_config,)),
    )
    assert config.entropy_source is EntropySource.ADVISOR

    with executor_context():
        default_step = make_joint_decode_completion_step(
            name="test/entropy-source", model_path="/model", prompts_path="/prompts", config=config
        )
        decoder_step = make_joint_decode_completion_step(
            name="test/entropy-source",
            model_path="/model",
            prompts_path="/prompts",
            config=replace(config, entropy_source=EntropySource.DECODER),
        )

    assert "entropy_source" not in collect_dependencies_and_version(default_step.config).version
    assert collect_dependencies_and_version(decoder_step.config).version["entropy_source"] == "decoder"


def test_selector_records_eos_step_with_empty_bytes():
    vocab_a = make_vocab({1: b" the"}, eos_id=9)
    vocab_b = make_vocab({4: b" a"}, eos_id=8)
    # B is peaked on " a" (entropy ~ 0.04 < 0.5), so the decoder's EOS
    # argmax commits.
    state = _SweepState(entropy_threshold=0.5, token_paths={})
    select_token = _make_select_token(
        make_sampling(), vocab_a, vocab_b, state, GateDirection.ADVISOR_AT_OR_ABOVE, EntropySource.ADVISOR
    )

    tokens_a, tokens_b = select_token(
        [entry(9, 5.0), entry(1, 0.0)],
        [entry(4, 5.0), entry(8, 0.0)],
        rng=random.Random(0),
        request_index=0,
    )

    assert (tokens_a, tokens_b) == ([9], [8])
    steps = state.token_paths[0]
    assert len(steps) == 1
    assert (steps[0].bytes_hex, steps[0].tokens_a, steps[0].tokens_b) == ("", [9], [8])


def test_selector_keeps_interleaved_requests_separate():
    vocab_a = make_vocab({1: b" the", 5: b"x"}, eos_id=9)
    vocab_b = make_vocab({2: b" th", 3: b"e"}, eos_id=8)
    state = _SweepState(entropy_threshold=1.0, token_paths={})
    select_token = _make_select_token(
        make_sampling(), vocab_a, vocab_b, state, GateDirection.ADVISOR_AT_OR_ABOVE, EntropySource.ADVISOR
    )

    for request_index in (0, 1, 0):
        select_token(
            [entry(1, 6.0), entry(5, 0.0)],
            [entry(2, 1.0), entry(3, 0.5)],
            rng=random.Random(0),
            request_index=request_index,
        )

    assert {index: len(steps) for index, steps in state.token_paths.items()} == {0: 2, 1: 1}


def test_write_sweep_chunk_offsets_completion_indices_and_writes_sidecar(tmp_path):
    # threshold_index=1 with n_samples=2 puts this threshold's samples at
    # completion_index 2..3; range [2, 6) covers prompts 1 and 2.
    chunk = make_chunk(tmp_path)
    step = XtokPathStep(bytes_hex=b" 42".hex(), tokens_a=[10], tokens_b=[11, 12], entropy=0.25)
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
        "metadata": {"finish_reason": "stop", "entropy_threshold": 0.5},
    }

    path_records = read_jsonl_gz(chunk.token_paths_path)
    assert [(record["id"], record["completion_index"]) for record in path_records] == keys
    assert path_records[0] == {
        "id": "p1",
        "completion_index": 2,
        "entropy_threshold": 0.5,
        "threshold_index": 1,
        "steps": [{"bytes_hex": b" 42".hex(), "tokens_a": [10], "tokens_b": [11, 12], "entropy": 0.25}],
    }


def test_write_sweep_chunk_missing_trace_for_nonempty_text_raises(tmp_path):
    step = XtokPathStep(bytes_hex=b"a".hex(), tokens_a=[10], tokens_b=[11], entropy=0.0)
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
    step = XtokPathStep(bytes_hex=b"a".hex(), tokens_a=[10], tokens_b=[11], entropy=0.0)
    with pytest.raises(RuntimeError, match="unknown batch indices"):
        write_sweep_chunk(
            make_chunk(tmp_path),
            decoder=FakeDecoder(),
            prompt_ids=["p0", "p1", "p2"],
            prompts=["q0", "q1", "q2"],
            n_samples=2,
            token_paths={batch_index: [step] for batch_index in (0, 1, 2, 3, 99)},
        )
