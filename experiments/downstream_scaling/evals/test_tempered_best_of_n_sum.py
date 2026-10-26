# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from itertools import product
from types import SimpleNamespace

import pytest

from experiments.downstream_scaling.evals import algorithms
from experiments.downstream_scaling.evals.algorithms import joint_decode_avg_v2 as joint
from experiments.downstream_scaling.evals.algorithms import xtok_selection
from experiments.downstream_scaling.evals.framework.xregion import ledger
from experiments.downstream_scaling.evals.framework.xregion.pool import EnginePlacement


class RecordingRandom:
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        self.probabilities = []

    def random(self):
        # Equal Gumbels preserve strict advisor-logit order regardless of key iteration order.
        return 0.5

    def choices(self, population, weights, *, k):
        total = sum(weights)
        self.probabilities.append({key: weight / total for key, weight in zip(population, weights, strict=True)})
        return [next(self.tokens) for _ in range(k)]


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


def entry(token_id, logit):
    return {"token_id": token_id, "logit": logit}


def softmax(scores, temperature):
    maximum = max(scores.values())
    weights = {key: math.exp((score - maximum) / temperature) for key, score in scores.items()}
    return {key: weight / sum(weights.values()) for key, weight in weights.items()}


VOCAB_A = make_vocab({1: b"a", 2: b"b", 3: b"c"}, eos_id=9)
VOCAB_B = make_vocab({11: b"a", 12: b"b", 13: b"c"}, eos_id=8)
# Each side misses one byte key, so both minimum-logit floors are exercised.
A_TOPK = [entry(1, 0.3), entry(2, -0.4)]
B_TOPK = [entry(12, 0.9), entry(13, -0.2)]


def summed_probabilities(*, effective_n, beta, temperature, a_topk=A_TOPK, b_topk=B_TOPK):
    rng = RecordingRandom([b"b"])
    xtok_selection.select_summed_best_of_n_bytes_union(
        a_topk,
        b_topk,
        advisor_weight=effective_n,
        temperature=0.7,
        teacher_temperature=0.5,
        beta=beta,
        final_temperature=temperature,
        rng=rng,
        vocab_a=VOCAB_A,
        vocab_b=VOCAB_B,
    )
    return rng.probabilities[0]


def avg_bytes_union_probabilities(*, advisor_weight, temperature):
    rng = RecordingRandom([b"b"])
    xtok_selection.select_avg_bytes_union(
        A_TOPK,
        B_TOPK,
        advisor_weight=advisor_weight,
        temperature=temperature,
        rng=rng,
        vocab_a=VOCAB_A,
        vocab_b=VOCAB_B,
    )
    return rng.probabilities[0]


@pytest.mark.parametrize("effective_n", [1.0, 4.0, 64.0])
def test_zero_beta_samples_delphi(effective_n):
    expected = avg_bytes_union_probabilities(advisor_weight=1.0, temperature=0.4)
    actual = summed_probabilities(effective_n=effective_n, beta=0.0, temperature=0.4)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("beta", [0.3, 1.0, 4.0])
def test_one_n_matches_unnormalized_add_of_llama_into_delphi(beta):
    # Unnormalized add with student weight beta and alpha 1 scores beta * z_A + z_B.
    expected = avg_bytes_union_probabilities(advisor_weight=1.0 / (1.0 + beta), temperature=0.4 / (1.0 + beta))
    actual = summed_probabilities(effective_n=1.0, beta=beta, temperature=0.4)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_integer_n_matches_enumerated_proposals():
    effective_n, beta, proposal_temperature, temperature = 3, 0.6, 0.7, 0.4
    a_topk = [entry(1, 0.2), entry(2, -0.3), entry(3, -1.1)]
    b_topk = [entry(11, -0.5), entry(12, 0.8), entry(13, 0.1)]
    z_a = {b"a": 0.2, b"b": -0.3, b"c": -1.1}
    z_b = {b"a": -0.5, b"b": 0.8, b"c": 0.1}
    proposal = softmax(z_a, proposal_temperature)
    # Equal Gumbels rank by Delphi logits.
    ranking = sorted(z_b, key=z_b.__getitem__, reverse=True)
    win = dict.fromkeys(z_a, 0.0)
    for draws in product(z_a, repeat=effective_n):
        winner = next(key for key in ranking if key in draws)
        win[winner] += math.prod(proposal[key] for key in draws)
    scores = {key: z_b[key] + beta * proposal_temperature * math.log(win[key]) for key in z_a}

    rng = RecordingRandom([b"b"])
    xtok_selection.select_summed_best_of_n_bytes_union(
        a_topk,
        b_topk,
        advisor_weight=float(effective_n),
        temperature=proposal_temperature,
        teacher_temperature=0.5,
        beta=beta,
        final_temperature=temperature,
        rng=rng,
        vocab_a=VOCAB_A,
        vocab_b=VOCAB_B,
    )
    assert rng.probabilities[0] == pytest.approx(softmax(scores, temperature), rel=1e-12, abs=1e-12)


def test_sum_sweep_roundtrip_reuses_engines_and_records_each_beta(tmp_path, monkeypatch):
    prompts_path = tmp_path / "prompts.jsonl.gz"
    with gzip.open(prompts_path, "wt") as f:
        f.write(json.dumps({"id": "p0", "prompt": "q0"}) + "\n")
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    sampling = joint.JointDecodeSamplingConfig(
        n_samples=2,
        max_tokens=8,
        advisor_max_tokens=8,
        top_k_a=2,
        top_k_b=2,
        seed=0,
        selection_rule=joint.XtokSelectionRule.TEMPERED_BEST_OF_N_SUM,
        advisor_weights=((2.0, 0.8, 0.7, 0.0), (2.0, 0.8, 0.7, 0.5), (2.0, 0.8, 0.7, 2.0)),
        temperature=0.4,
    )
    config = joint.JointDecodeLocalWorkerConfig(
        decoder_model_path="/student",
        advisor_model_path="/advisor",
        prompts_path=str(prompts_path),
        advisor_prompts_path=None,
        sampling=sampling,
        decoder_model=joint.JointDecodeModelConfig(),
        advisor_model=joint.JointDecodeModelConfig(),
        ledger_path=str(tmp_path / "ledger"),
        poll_backoff=0.0,
        microbatch_size=2,
        barrier_timeout_s=60,
        owner="test-worker",
        placement=joint.JointDecodePlacement(
            decoder=EnginePlacement((0,), (1, 1, 1), 1),
            advisor=EnginePlacement((1,), (1, 1, 1), 1),
        ),
    )
    config = joint._child_config_from_file(str(joint._write_child_config(tmp_path, config)))
    chunks = joint.sweep_chunk_specs(str(chunks_dir), 1, 2, config.sampling.advisor_weights, 2)
    ledger.ensure_manifest(config.ledger_path, [json.loads(json.dumps(asdict(chunk))) for chunk in chunks])
    probabilities = []
    engine_events = []

    @contextmanager
    def open_joint_decoder(**kwargs):
        engine_events.append("open")

        def generate(prompts_a, prompts_b):
            assert prompts_a == prompts_b
            rng = RecordingRandom([b"a", xtok_selection.EOS_KEY] * len(prompts_a))
            for request_index in range(len(prompts_a)):
                for _ in range(2):
                    kwargs["select_token"](
                        [entry(1, 0.0), entry(9, -1.0)],
                        [entry(11, 0.0), entry(8, 1.0)],
                        rng=rng,
                        request_index=request_index,
                    )
            probabilities.extend(rng.probabilities[::2])
            return [SimpleNamespace(text="a", finish_reason="stop") for _ in prompts_a]

        yield SimpleNamespace(generate=generate)
        engine_events.append("close")

    # Replace model/tokenizer I/O; run the real worker, selector, ledger, and output writer.
    monkeypatch.setattr(
        algorithms,
        "joint_decode_backend",
        SimpleNamespace(EngineModelParams=SimpleNamespace, open_joint_decoder=open_joint_decoder),
        raising=False,
    )
    vocabs = {
        "/student": make_vocab({1: b"a"}, eos_id=9),
        "/advisor": make_vocab({11: b"a"}, eos_id=8),
    }
    monkeypatch.setattr(joint, "_load_vocab", vocabs.__getitem__)
    monkeypatch.setenv("TPU_VISIBLE_CHIPS", "0,1")
    joint._run_joint_decode_local_worker(config)

    assert engine_events == ["open", "close"]
    assert ledger.done_chunk_ids(config.ledger_path) == [0, 1, 2]
    records, paths = [], []
    for chunk in chunks:
        with gzip.open(chunk.output_path, "rt") as f:
            records.extend(json.loads(line) for line in f)
        with gzip.open(chunk.token_paths_path, "rt") as f:
            paths.extend(json.loads(line) for line in f)
    assert [record["completion_index"] for record in records] == list(range(6))
    for record, path, observed in zip(records, paths, probabilities, strict=True):
        point_index = record["completion_index"] // sampling.n_samples
        n, proposal_temperature, advisor_temperature, beta = sampling.advisor_weights[point_index]
        metadata = {
            "advisor_weight": n,
            "proposal_temperature": proposal_temperature,
            "advisor_temperature": advisor_temperature,
            "beta": beta,
        }
        assert record["completion"]["metadata"] == {"finish_reason": "stop", **metadata}
        assert path == {
            "id": "p0",
            "completion_index": record["completion_index"],
            "weight_index": point_index,
            **metadata,
            "steps": [
                {"bytes_hex": b"a".hex(), "tokens_a": [1], "tokens_b": [11]},
                {"bytes_hex": "", "tokens_a": [9], "tokens_b": [8]},
            ],
        }
        student_a = 1 / (1 + math.exp(-1 / proposal_temperature))
        # EOS ranks first, so A wins only when every proposal is A.
        win_a = student_a**n
        scores = {
            b"a": 0.0 + beta * proposal_temperature * math.log(win_a),
            xtok_selection.EOS_KEY: 1.0 + beta * proposal_temperature * math.log(1 - win_a),
        }
        assert observed == pytest.approx(softmax(scores, sampling.temperature), rel=1e-12, abs=1e-12)
