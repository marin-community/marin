# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import math
import random
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from decimal import Decimal, localcontext
from itertools import permutations, product
from types import SimpleNamespace

import pytest

from experiments.downstream_scaling.evals import algorithms
from experiments.downstream_scaling.evals.algorithms import joint_decode_avg_v2 as joint
from experiments.downstream_scaling.evals.algorithms import xtok_selection
from experiments.downstream_scaling.evals.framework.xregion import ledger
from experiments.downstream_scaling.evals.framework.xregion.pool import EnginePlacement

DRAWS = 50_000
# Hoeffding + union bound: at most 1e-9 false rejection probability across three tokens.
ABS_TOLERANCE = math.sqrt(math.log(6 / 1e-9) / (2 * DRAWS))


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


def tempered_probabilities(probabilities, temperature):
    if temperature == 0.0:
        winner = max(probabilities, key=probabilities.__getitem__)
        return {key: float(key == winner) for key in probabilities}
    weights = {key: probability ** (1 / temperature) for key, probability in probabilities.items()}
    return {key: weight / sum(weights.values()) for key, weight in weights.items()}


def mixture_frequencies(
    student, advisor, *, effective_n, proposal_temperature, advisor_temperature, final_temperature, tau
):
    vocab_a = make_vocab({1: b"a", 2: b"b", 3: b"c"}, eos_id=9)
    vocab_b = make_vocab({11: b"a", 12: b"b", 13: b"c"}, eos_id=8)
    a_topk = [entry(key, proposal_temperature * math.log(p)) for key, p in student.items()]
    b_topk = [entry(key + 10, proposal_temperature * math.log(p)) for key, p in advisor.items()]
    rng = random.Random(42)
    counts = Counter(
        xtok_selection.select_mixed_best_of_n_bytes_union(
            a_topk,
            b_topk,
            advisor_weight=effective_n,
            temperature=proposal_temperature,
            teacher_temperature=advisor_temperature,
            final_temperature=final_temperature,
            tau=tau,
            rng=rng,
            vocab_a=vocab_a,
            vocab_b=vocab_b,
        )[0][0]
        for _ in range(DRAWS)
    )
    return {key: counts[key] / DRAWS for key in student}


@pytest.mark.parametrize(
    ("effective_n", "advisor_temperature", "final_temperature", "tau"),
    [(1, 0.9, 0.6, 0.4), (2, 0.9, 0.6, 0.4), (3, 0.0, 1.0, 0.05), (2, 0.9, 0.0, 0.4)],
)
def test_mixture_matches_enumerated_proposals(effective_n, advisor_temperature, final_temperature, tau):
    student = {1: 0.5, 2: 0.3, 3: 0.2}
    advisor = {1: 0.1, 2: 0.25, 3: 0.65}
    proposal_temperature = 0.7
    proposal = {key: (student[key] + tau * advisor[key]) / (1 + tau) for key in student}
    rankings = (
        [tuple(sorted(advisor, key=advisor.__getitem__, reverse=True))]
        if advisor_temperature == 0.0
        else permutations(advisor)
    )
    expected = dict.fromkeys(student, 0.0)
    for ranking in rankings:
        rank_probability = 1.0
        if advisor_temperature > 0.0:
            scores = {key: p ** (proposal_temperature / advisor_temperature) for key, p in advisor.items()}
            rank_probability = math.prod(
                scores[key] / sum(scores[other] for other in ranking[position:]) for position, key in enumerate(ranking)
            )
        conditional = dict.fromkeys(student, 0.0)
        for draws in product(student, repeat=effective_n):
            winner = next(key for key in ranking if key in draws)
            conditional[winner] += math.prod(proposal[key] for key in draws)
        for key, probability in tempered_probabilities(conditional, final_temperature).items():
            expected[key] += rank_probability * probability

    actual = mixture_frequencies(
        student,
        advisor,
        effective_n=effective_n,
        proposal_temperature=proposal_temperature,
        advisor_temperature=advisor_temperature,
        final_temperature=final_temperature,
        tau=tau,
    )
    assert actual == pytest.approx(expected, rel=0.0, abs=ABS_TOLERANCE)


def test_mixture_fractional_n_matches_two_token_marginal():
    student = {1: 0.75, 2: 0.25}
    advisor = {1: 0.2, 2: 0.8}
    tau = 0.3
    effective_n = 1.5
    final_temperature = 0.6
    q_a = (student[1] + tau * advisor[1]) / (1 + tau)
    expected_a = 0.0
    for rank_probability, win_a in ((0.2, 1 - (1 - q_a) ** effective_n), (0.8, q_a**effective_n)):
        expected_a += rank_probability * tempered_probabilities({1: win_a, 2: 1 - win_a}, final_temperature)[1]
    actual = mixture_frequencies(
        student,
        advisor,
        effective_n=effective_n,
        proposal_temperature=0.7,
        advisor_temperature=0.7,
        final_temperature=final_temperature,
        tau=tau,
    )
    assert actual == pytest.approx({1: expected_a, 2: 1 - expected_a}, rel=0.0, abs=ABS_TOLERANCE)


@pytest.mark.parametrize(("effective_n", "advisor_temperature", "final_temperature"), [(1.0, 0.0, 0.4), (2.5, 0.7, 1.2)])
def test_zero_tau_matches_existing_tempered_selector(effective_n, advisor_temperature, final_temperature):
    kwargs = dict(
        a_topk=[entry(1, 0.0), entry(2, -1.0)],
        b_topk=[entry(12, 1.0), entry(13, -1.0)],
        advisor_weight=effective_n,
        temperature=0.8,
        teacher_temperature=advisor_temperature,
        final_temperature=final_temperature,
        vocab_a=make_vocab({1: b"a", 2: b"b", 3: b"c"}, eos_id=9),
        vocab_b=make_vocab({11: b"a", 12: b"b", 13: b"c"}, eos_id=8),
    )
    rng = random.Random(42)
    reference_rng = random.Random(42)
    actual = [xtok_selection.select_mixed_best_of_n_bytes_union(**kwargs, tau=0.0, rng=rng) for _ in range(100)]
    expected = [xtok_selection.select_best_of_n_bytes_union(**kwargs, rng=reference_rng) for _ in range(100)]
    assert actual == expected


@pytest.mark.parametrize(
    ("student_logit", "advisor_logit", "tau", "final_temperature"),
    [(math.log(1e-20), 1.0, 1e-20, 10.0), (-1000.0, -990.0, 0.05, 400.0)],
)
def test_mixture_preserves_tiny_probabilities_before_heating(student_logit, advisor_logit, tau, final_temperature):
    rng = RecordingRandom([b"b"])
    xtok_selection.select_mixed_best_of_n_bytes_union(
        [entry(1, 0.0), entry(2, student_logit)],
        [entry(11, 0.0), entry(12, advisor_logit)],
        advisor_weight=2.0,
        temperature=1.0,
        teacher_temperature=0.0,
        final_temperature=final_temperature,
        tau=tau,
        rng=rng,
        vocab_a=make_vocab({1: b"a", 2: b"b"}, eos_id=9),
        vocab_b=make_vocab({11: b"a", 12: b"b"}, eos_id=8),
    )
    with localcontext() as context:
        context.prec = 80
        s = Decimal(student_logit).exp()
        a = Decimal(advisor_logit).exp()
        q = (s / (1 + s) + Decimal(tau) * a / (1 + a)) / (1 + Decimal(tau))
        win_b = 1 - (1 - q) ** 2 if advisor_logit > 0 else q**2
        exponent = 1 / Decimal(final_temperature)
        expected_b = float(win_b**exponent / (win_b**exponent + (1 - win_b) ** exponent))
    assert rng.probabilities[0] == pytest.approx({b"a": 1 - expected_b, b"b": expected_b}, rel=1e-12, abs=1e-12)


def test_mixture_sweep_roundtrip_reuses_engines_and_records_each_tau(tmp_path, monkeypatch):
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
        selection_rule=joint.XtokSelectionRule.TEMPERED_BEST_OF_N_MIXTURE,
        advisor_weights=((2.0, 0.8, 0.7, 0.0), (2.0, 0.8, 0.7, 0.05), (2.0, 0.8, 0.7, 1.0)),
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
        n, proposal_temperature, advisor_temperature, tau = sampling.advisor_weights[point_index]
        final_temperature = sampling.temperature / proposal_temperature
        metadata = {
            "advisor_weight": n,
            "proposal_temperature": proposal_temperature,
            "advisor_temperature": advisor_temperature,
            "final_temperature": final_temperature,
            "teacher_distribution_boost": tau,
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
        advisor_a = 1 - student_a
        # EOS ranks first, so A wins only when both mixed proposals are A.
        win_a = ((student_a + tau * advisor_a) / (1 + tau)) ** 2
        expected = tempered_probabilities({b"a": win_a, xtok_selection.EOS_KEY: 1 - win_a}, final_temperature)
        assert observed == pytest.approx(expected, rel=1e-12, abs=1e-12)
