"""CatCount on-policy distillation from programmed teachers on the production trainer's CPU harness.

The student is the CatCount CPU policy, which counts other words but writes ``cat`` correctly only for N=1. Each
teacher is ``examples/cat_count/synthetic_teacher.py`` for one word, served over HTTP as an ``openai_compatible``
teacher. Training uses only the teachers' signal (``reward_mode=replace`` with the uniform advantage estimator);
the environment reward is still computed for evaluation, so a rise in the CatCount score is caused by the teachers.

With one word this is single-teacher OPD. With several (MOPD), every row asks for one word and carries that word
as its ``teacher_route``, each route goes to that word's expert, and evaluation reports each word separately.
``swap_routes`` sends every route to the next word's expert, so a working router makes each word learn the wrong
answer.

Usage, with a policy directory written by ``examples/cat_count/cpu_canary.py``::

    uv run --frozen --no-sync python -m tests.cpu.tiny_training.cat_count_opd \\
        --model /tmp/cat-count-policy --root /tmp/cat-count-opd --steps 10 --jitter 0.5 --error-rate 0.1
    uv run --frozen --no-sync python -m tests.cpu.tiny_training.cat_count_opd \
        --model /tmp/cat-count-policy --root /tmp/cat-count-mopd --steps 20 --jitter 0.5 --words cat dog
"""

import argparse
import asyncio
import json
import random
import socket
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from pathlib import Path

from aiohttp import web
from examples.cat_count.cpu_canary import HELD_OUT_N, TEMPLATE, TRAIN_N
from examples.cat_count.synthetic_teacher import SyntheticTeacher, TeacherNoise, WordCountTarget, application
from omegaconf import DictConfig, OmegaConf
from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint
from skyrl_train.tokenizer import create_tokenizer

from tests.cpu.tiny_training.cat_count import FAST_STEPS, cat_count_config
from tests.cpu.tiny_training.experiment import read_metrics, run_tiny_training

# At 1e-4 the per-token teacher credit overshoots and scores collapse after peaking; 3e-5 converged on seeds 0 to 2.
LEARNING_RATE = 3e-5


@contextmanager
def serve_teacher(model: Path, noise: TeacherNoise, word: str) -> Iterator[str]:
    """Serve one word's programmed teacher on a free local port and yield its OpenAI base URL."""
    teacher = SyntheticTeacher(create_tokenizer(str(model), disable_fast_tokenizer=False), noise, WordCountTarget(word))
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


def write_routed_rows(
    path: Path, words: tuple[str, ...], ns: list[int], split: str, repeats: int = 1, seed: int = 0
) -> Path:
    """Write one row per word for each N, keeping each N's rows adjacent so every batch holds every route.

    Training reads these rows unshuffled; the N order is shuffled here instead.
    """
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for _ in range(repeats):
            order = list(ns)
            rng.shuffle(order)
            for n in order:
                for word in words:
                    source = f"{word}_{split}"
                    row = {
                        "prompt": [{"role": "user", "content": TEMPLATE.format(W=word, N=n)}],
                        "env_class": "cat_count",
                        "data_source": source,
                        "teacher_route": word,
                        "extra_info": {"n": n, "word": word, "data_source": source},
                    }
                    handle.write(json.dumps(row) + "\n")
    return path


def cat_count_opd_config(
    root: Path,
    model: Path,
    teacher_urls: dict[str, str],
    *,
    steps: int = FAST_STEPS,
    seed: int = 0,
    lr: float = LEARNING_RATE,
    swap_routes: bool = False,
) -> DictConfig:
    """Return the CatCount CPU config with OPD from one expert per word replacing the environment reward."""
    cfg = cat_count_config(root, model, steps=steps, seed=seed)
    cfg.trainer.policy.optimizer_config.lr = lr
    OmegaConf.set_struct(cfg, False)
    fingerprint = tokenizer_vocabulary_fingerprint(create_tokenizer(str(model), disable_fast_tokenizer=False))
    words = tuple(teacher_urls)
    teachers = {
        word: {
            "source": "openai_compatible",
            "placement": "external",
            "model": {"path": f"synthetic/{word}-count", "revision": "v1"},
            "endpoints": [{"url": url, "max_concurrency": 8}],
            "tokenizer_fingerprint": fingerprint,
            "max_sequence_length": 256,
            "request_timeout_seconds": 60,
            "evidence": "chosen_token",
        }
        for word, url in teacher_urls.items()
    }
    overrides: dict = {
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
        "teachers": teachers,
    }
    if len(words) == 1:
        if swap_routes:
            raise ValueError("swapping routes needs at least two words")
        if words != ("cat",):
            # Single-teacher OPD reads the CatCount rows, which all ask for cat.
            raise ValueError("single-teacher OPD teaches cat; pass two or more words for MOPD")
        routes = {"default": {"teacher": words[0], "weight": 1.0}}
    else:
        experts = words[1:] + words[:1] if swap_routes else words
        routes = {word: {"teacher": expert, "weight": 1.0} for word, expert in zip(words, experts, strict=True)}
        overrides["data"] = {
            "shuffle": False,
            "train_data": [str(write_routed_rows(root / "data/train.jsonl", words, TRAIN_N, "train", 50, seed))],
            "val_data": [
                str(write_routed_rows(root / "data/eval_train.jsonl", words, TRAIN_N, "train")),
                str(write_routed_rows(root / "data/eval_heldout.jsonl", words, HELD_OUT_N, "heldout")),
            ],
        }
    overrides["teacher_routing"] = {"opd": {"revision": "cat-count-v1", "routes": routes}}
    return OmegaConf.merge(cfg, overrides)


def run_cat_count_opd(
    root: Path,
    model: Path,
    noise: TeacherNoise,
    *,
    words: tuple[str, ...] = ("cat",),
    steps: int = FAST_STEPS,
    seed: int = 0,
    lr: float = LEARNING_RATE,
    swap_routes: bool = False,
) -> None:
    """Serve one teacher per word in this process and train the student against them."""
    with ExitStack() as stack:
        urls = {word: stack.enter_context(serve_teacher(model, noise, word)) for word in words}
        run_tiny_training(
            cat_count_opd_config(root, model, urls, steps=steps, seed=seed, lr=lr, swap_routes=swap_routes)
        )


def evaluation_scores(records: list[dict], words: tuple[str, ...] = ("cat",)) -> list[dict]:
    """Return each evaluation's greedy scores; MOPD reports every word's splits separately."""
    if len(words) == 1:
        sources = ("train", "heldout")
    else:
        sources = tuple(f"{word}_{split}" for word in words for split in ("train", "heldout"))
    keys = (
        *(f"eval/{source}/avg_score" for source in sources),
        *(f"eval/{source}/environment/cat_count/exact" for source in sources if source.endswith("train")),
    )
    return [
        {"step": int(row["step"]), **{key: row[key] for key in keys if key in row}} for row in records if keys[0] in row
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="the CatCount CPU policy directory")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=FAST_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--words", nargs="+", default=["cat"], help="one expert per word; two or more run MOPD")
    parser.add_argument("--swap-routes", action="store_true", help="send each word's rows to the next word's expert")
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
    words = tuple(args.words)
    run_cat_count_opd(
        args.root,
        args.model,
        noise,
        words=words,
        steps=args.steps,
        seed=args.seed,
        lr=args.lr,
        swap_routes=args.swap_routes,
    )
    for row in evaluation_scores(read_metrics(args.root), words):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
