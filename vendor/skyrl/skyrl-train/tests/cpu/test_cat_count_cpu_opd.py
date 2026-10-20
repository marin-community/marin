"""CatCount on-policy distillation from a programmed, noisy teacher on the production trainer's CPU harness."""

import math
import time
from pathlib import Path

import pytest
from examples.cat_count.synthetic_teacher import TeacherNoise

from tests.cpu.test_cat_count_cpu_canary import RUN_TIMEOUT_SECONDS, cat_count_policy, runs  # noqa: F401
from tests.cpu.tiny_training.cat_count_opd import evaluation_scores, run_cat_count_opd
from tests.cpu.tiny_training.experiment import read_metrics

pytestmark = pytest.mark.slow

GATE_STEPS = 20
MOPD_STEPS = 30
WORDS = ("cat", "dog")
NOISY = TeacherNoise(jitter=0.5)


def distill(
    runs,  # noqa: F811
    root: Path,
    model: Path,
    noise: TeacherNoise,
    *,
    steps: int,
    seed: int = 0,
    words: tuple[str, ...] = ("cat",),
    swap_routes: bool = False,
) -> list[dict]:
    start = time.perf_counter()
    process = runs.Process(
        target=run_cat_count_opd,
        args=(root, model, noise),
        kwargs={"steps": steps, "seed": seed, "words": words, "swap_routes": swap_routes},
    )
    process.start()
    process.join(RUN_TIMEOUT_SECONDS)
    if process.exitcode is None:
        process.kill()
        process.join()
        pytest.fail(f"the run did not finish within {RUN_TIMEOUT_SECONDS} seconds")
    assert process.exitcode == 0
    print(f"CAT_COUNT_CPU_OPD run={root.name} seconds={time.perf_counter() - start:.3f}")
    return read_metrics(root)


def endpoints(records: list[dict]) -> tuple[dict, dict]:
    evaluations = evaluation_scores(records)
    return evaluations[0], evaluations[-1]


def test_cat_count_cpu_opd_learns_from_a_noisy_teacher(tmp_path, cat_count_policy, runs):  # noqa: F811
    records = distill(runs, tmp_path / "opd", cat_count_policy, NOISY, steps=GATE_STEPS)
    before, after = endpoints(records)
    print(f"CAT_COUNT_CPU_OPD seed=0 before={before} after={after}")
    assert before["step"] == 0 and after["step"] == GATE_STEPS
    assert before["eval/train/avg_score"] < 0.5 and before["eval/heldout/avg_score"] < 0.5
    assert after["eval/train/avg_score"] >= 0.9
    assert after["eval/heldout/avg_score"] >= 0.9

    training = [row for row in records if "policy/raw_grad_norm" in row]
    assert len(training) == GATE_STEPS
    for row in training:
        assert row["distillation/teacher_count"] == 1
        assert row["distillation/route_rows/default"] > 0
        assert row["distillation/valid_tokens"] > 0
        assert math.isfinite(row["policy/policy_loss"])
        assert row["policy/raw_grad_norm"] > 0


@pytest.mark.nightly
@pytest.mark.parametrize("seed", [0, 1])
def test_cat_count_cpu_opd_flipped_teacher_unlearns(tmp_path, cat_count_policy, runs, seed):  # noqa: F811
    flipped = TeacherNoise(jitter=NOISY.jitter, flipped=True)
    before, after = endpoints(distill(runs, tmp_path / "flipped", cat_count_policy, flipped, steps=10, seed=seed))
    print(f"CAT_COUNT_CPU_OPD seed={seed} flipped before={before} after={after}")
    assert after["eval/train/avg_score"] < before["eval/train/avg_score"]
    assert after["eval/heldout/avg_score"] < before["eval/heldout/avg_score"]
    assert after["eval/train/environment/cat_count/exact"] == 0


def test_cat_count_cpu_mopd_routes_each_word_to_its_expert(tmp_path, cat_count_policy, runs):  # noqa: F811
    records = distill(runs, tmp_path / "mopd", cat_count_policy, NOISY, steps=MOPD_STEPS, words=WORDS)
    evaluations = evaluation_scores(records, WORDS)
    before, after = evaluations[0], evaluations[-1]
    print(f"CAT_COUNT_CPU_MOPD seed=0 before={before} after={after}")
    assert after["step"] == MOPD_STEPS
    # The policy counts dogs from pretraining but not cats, so only the cat route has to learn.
    assert before["eval/cat_train/avg_score"] < 0.5
    for word in WORDS:
        assert after[f"eval/{word}_train/avg_score"] >= 0.9
        assert after[f"eval/{word}_heldout/avg_score"] >= 0.9

    training = [row for row in records if "policy/raw_grad_norm" in row]
    assert len(training) == MOPD_STEPS
    for row in training:
        assert row["distillation/teacher_count"] == len(WORDS)
        assert all(row[f"distillation/route_rows/{word}"] > 0 for word in WORDS)
        assert row["distillation/valid_tokens"] > 0
        assert math.isfinite(row["policy/policy_loss"])


@pytest.mark.nightly
def test_cat_count_cpu_mopd_swapped_routes_teach_the_wrong_word(tmp_path, cat_count_policy, runs):  # noqa: F811
    records = distill(runs, tmp_path / "swapped", cat_count_policy, NOISY, steps=20, words=WORDS, swap_routes=True)
    evaluations = evaluation_scores(records, WORDS)
    before, after = evaluations[0], evaluations[-1]
    print(f"CAT_COUNT_CPU_MOPD seed=0 swapped before={before} after={after}")
    assert before["eval/dog_train/avg_score"] >= 0.9
    assert after["eval/dog_train/avg_score"] < 0.5
    assert after["eval/cat_train/avg_score"] < 0.5
    assert after["eval/dog_train/environment/cat_count/exact"] < 0.25
