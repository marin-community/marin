"""CatCount on-policy distillation from a programmed teacher on the production trainer's CPU harness.

The student is the CatCount CPU policy, which counts other words but writes ``cat`` correctly only for N=1. The
teacher is ``examples/cat_count/synthetic_teacher.py``, served over HTTP as an ``openai_compatible`` teacher.
Training uses only the teacher's signal (``reward_mode=replace`` with the uniform advantage estimator); the
environment reward is still computed for evaluation, so a rise in the CatCount score is caused by the teacher.

Usage, with a policy directory written by ``examples/cat_count/cpu_canary.py``::

    uv run --frozen --no-sync python -m tests.cpu.tiny_training.cat_count_opd \\
        --model /tmp/cat-count-policy --root /tmp/cat-count-opd --steps 10 --jitter 0.5 --error-rate 0.1
"""

import argparse
import asyncio
import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

from aiohttp import web
from examples.cat_count.synthetic_teacher import CatCountTeacher, TeacherNoise, application
from omegaconf import DictConfig, OmegaConf
from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint
from skyrl_train.tokenizer import create_tokenizer

from tests.cpu.tiny_training.cat_count import FAST_STEPS, cat_count_config
from tests.cpu.tiny_training.experiment import read_metrics, run_tiny_training

TEACHER_ID = "cat"
# At 1e-4 the per-token teacher credit overshoots and scores collapse after peaking; 3e-5 converged on seeds 0 to 2.
LEARNING_RATE = 3e-5


@contextmanager
def serve_teacher(model: Path, noise: TeacherNoise) -> Iterator[str]:
    """Serve the programmed teacher on a free local port and yield its OpenAI base URL."""
    teacher = CatCountTeacher(create_tokenizer(str(model), disable_fast_tokenizer=False), noise)
    loop = asyncio.new_event_loop()
    runner = web.AppRunner(application(teacher))
    loop.run_until_complete(runner.setup())
    listener = socket.create_server(("127.0.0.1", 0))
    loop.run_until_complete(web.SockSite(runner, listener).start())
    port = listener.getsockname()[1]
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(30)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(30)
        loop.close()


def cat_count_opd_config(
    root: Path, model: Path, teacher_url: str, *, steps: int = FAST_STEPS, seed: int = 0, lr: float = LEARNING_RATE
) -> DictConfig:
    """Return the CatCount CPU config with single-teacher OPD replacing the environment reward."""
    cfg = cat_count_config(root, model, steps=steps, seed=seed)
    cfg.trainer.policy.optimizer_config.lr = lr
    OmegaConf.set_struct(cfg, False)
    fingerprint = tokenizer_vocabulary_fingerprint(create_tokenizer(str(model), disable_fast_tokenizer=False))
    return OmegaConf.merge(
        cfg,
        {
            "trainer": {
                "algorithm": {
                    "advantage_estimator": "uniform",
                    "distillation": {
                        "objective": "sampled_reverse_kl",
                        "routing_plan": "opd",
                        "coefficient": 1.0,
                        "reward_mode": "replace",
                    },
                }
            },
            "teachers": {
                TEACHER_ID: {
                    "source": "openai_compatible",
                    "placement": "external",
                    "model": {"path": "synthetic/cat-count", "revision": "v1"},
                    "endpoints": [{"url": teacher_url, "max_concurrency": 8}],
                    "tokenizer_fingerprint": fingerprint,
                    "max_sequence_length": 256,
                    "request_timeout_seconds": 60,
                    "evidence": "chosen_token",
                }
            },
            "teacher_routing": {
                "opd": {"revision": "cat-count-v1", "routes": {"default": {"teacher": TEACHER_ID, "weight": 1.0}}}
            },
        },
    )


def run_cat_count_opd(
    root: Path, model: Path, noise: TeacherNoise, *, steps: int = FAST_STEPS, seed: int = 0, lr: float = LEARNING_RATE
) -> None:
    """Serve the teacher in this process and train the student against it."""
    with serve_teacher(model, noise) as url:
        run_tiny_training(cat_count_opd_config(root, model, url, steps=steps, seed=seed, lr=lr))


def evaluation_scores(records: list[dict]) -> list[dict]:
    keys = ("eval/train/avg_score", "eval/heldout/avg_score", "eval/train/environment/cat_count/exact")
    return [
        {"step": int(row["step"]), **{key: row[key] for key in keys if key in row}}
        for row in records
        if "eval/train/avg_score" in row
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="the CatCount CPU policy directory")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=FAST_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    defaults = TeacherNoise()
    noise_fields = [field for field in fields(TeacherNoise) if field.name != "seed"]
    for field in noise_fields:
        name = "--" + field.name.replace("_", "-")
        if field.type == "bool":
            parser.add_argument(name, action="store_true")
        else:
            parser.add_argument(name, type=type(getattr(defaults, field.name)), default=getattr(defaults, field.name))
    args = parser.parse_args()
    noise = TeacherNoise(seed=args.seed, **{field.name: getattr(args, field.name) for field in noise_fields})
    run_cat_count_opd(args.root, args.model, noise, steps=args.steps, seed=args.seed, lr=args.lr)
    for row in evaluation_scores(read_metrics(args.root)):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
