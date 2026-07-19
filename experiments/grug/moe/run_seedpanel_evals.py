# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post-hoc lm-eval logprob pass for the mve seed panel / transect checkpoints.

One process per RUN: loads the final Grug checkpoint once, then iterates the
60-task readout suite (``seedpanel_eval_tasks.json``, reconstructed from the
swarm's own per-task eval configs), writing one ``results.json`` per task in
the swarm's layout:

    s3://marin-us-east-02a/marin/evaluation/grug_logprob/<run>/<task>/results.json

Differences from the swarm's one-job-per-(run,task) ExecutorStep fan-out: the
model is loaded once per run (60x fewer checkpoint loads), the tracker is the
JSON logger (results.json is the artifact of record), and completed tasks are
skipped on retry. Eval semantics match the swarm: capacity 8.0 (applied
post-load via ``_with_eval_capacity``), batch 8, max_cont_len 256, no
instance caps.

Requires the lm-eval fork overlay (the repo venv has no lm-eval):

    uv run --with "lm-eval[math,api,ifeval]@git+https://github.com/\\
        stanford-crfm/lm-evaluation-harness@d5e3391f22cde186c827674d5c3ec7c5f4fe0cab" \\
        python experiments/grug/moe/run_seedpanel_evals.py --run-name rav_mve_seedpanel_h100_00
"""

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

import equinox as eqx
import jax
import jmp
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.distributed import DistributedConfig
from levanter.tokenizers import load_tokenizer
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from rigging.filesystem import StoragePath, marin_prefix, prefix_join

from experiments.grug.moe import launch_mve_seedpanel_b200 as b200_panel
from experiments.grug.moe.eval_logprob import (
    _apply_num_fewshot,
    _fill_missing_task_names,
    _lm_eval_spec,
    _logprob_gsm8k_task,
    _logprob_humaneval_task,
    _make_grug_lm,
    _with_eval_capacity,
)
from experiments.grug.moe.launch_mve_twobucket_h100 import _family_model
from experiments.grug.moe.model import Transformer
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).resolve().parent / "seedpanel_eval_tasks.json"

# Tasks backed by gated HF datasets: the cluster has no HF auth so lm-eval raises
# DatasetNotFoundError. Skip them rather than fail the whole job (and the readout
# excludes them). gpqa_0shot -> Idavidrein/gpqa (gated).
KNOWN_GATED_TASKS = frozenset({"gpqa_0shot"})

NAT_TO_BIT = 1 / 0.6931471805599453


def process_lambada_results(doc, results):
    """Replicates the swarm's custom lambada process_results (bpb/nll/perplexity/acc)."""
    log_likelihood, is_greedy = results[0]
    target_text = " " + doc["text"].split(" ")[-1]
    n_bytes = max(1, len(target_text.encode("utf-8")))
    return {
        "bpb": (-log_likelihood / n_bytes) * NAT_TO_BIT,
        "nll": -log_likelihood,
        "perplexity": log_likelihood,
        "acc": int(is_greedy),
    }


def _lambada_task() -> EvalTaskConfig:
    """The swarm's lambada_openai task, reconstructed from its results.json config dump."""
    metric_list = [
        {"metric": "bpb", "aggregation": "mean", "higher_is_better": False},
        {"metric": "nll", "aggregation": "mean", "higher_is_better": False},
        {"metric": "perplexity", "aggregation": "perplexity", "higher_is_better": False},
        {"metric": "acc", "aggregation": "mean", "higher_is_better": True},
    ]
    return EvalTaskConfig(
        name="lambada_openai",
        num_fewshot=0,
        task_alias="lambada_0shot",
        task_kwargs={
            "tag": ["lambada"],
            "dataset_path": "EleutherAI/lambada_openai",
            "dataset_name": "default",
            "output_type": "loglikelihood",
            "test_split": "test",
            "doc_to_text": "{{text.split(' ')[:-1]|join(' ')}}",
            "doc_to_target": "{{' '+text.split(' ')[-1]}}",
            "process_results": process_lambada_results,
            "metric_list": metric_list,
            "metadata": {"version": 1.0},
        },
    )


BUILDERS = {"gsm8k": _logprob_gsm8k_task, "humaneval": _logprob_humaneval_task, "lambada": _lambada_task}


def _add_bpb_metric(task_dict) -> None:
    """Append the fork's registered ``bpb`` metric to every task lacking it.

    Replicates the swarm eval pipeline's metric injection (their per-task
    ``configs`` all carry ``bpb`` in ``metric_list``; the base fork task YAMLs
    often do not). MC tasks with a distributional gold (e.g. truthfulqa_mc2)
    accept the metric but never emit it — same as the swarm.
    """
    from lm_eval.api.registry import (  # noqa: PLC0415 — overlay-only dep
        get_metric,
        get_metric_aggregation,
        is_higher_better,
    )

    for value in task_dict.values():
        if isinstance(value, dict):
            _add_bpb_metric(value)
            continue
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
            _add_bpb_metric(value[1])
            continue
        fn_list = getattr(value, "_metric_fn_list", None)
        if fn_list is None or "bpb" in fn_list:
            continue
        value._metric_fn_list["bpb"] = get_metric("bpb")
        value._aggregation_list["bpb"] = get_metric_aggregation("bpb")
        value._higher_is_better["bpb"] = is_higher_better("bpb")
        value._metric_fn_kwargs["bpb"] = {}


def load_task_manifest() -> tuple[dict, dict[str, EvalTaskConfig]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    tasks: dict[str, EvalTaskConfig] = {}
    for alias, spec in manifest["tasks"].items():
        if "builder" in spec:
            tasks[alias] = BUILDERS[spec["builder"]]()
        else:
            tasks[alias] = EvalTaskConfig(
                name=spec["name"],
                num_fewshot=spec["num_fewshot"],
                task_alias=spec["task_alias"],
                task_kwargs=spec["task_kwargs"],
            )
    return manifest, tasks


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="e.g. rav_mve_seedpanel_h100_00 or rav_mve_transect_e8")
    parser.add_argument("--checkpoint-path", default=None, help="default users/rav/grug/<run>/dev/checkpoints")
    parser.add_argument("--output-prefix", default=None, help="default evaluation/grug_logprob/<run>")
    parser.add_argument("--list-tasks", action="store_true", help="print resolved tasks and exit (no GPU needed)")
    parser.add_argument("--redo-tasks", default="", help="comma-separated aliases to recompute even if results exist")
    parser.add_argument(
        "--only-tasks",
        default="",
        help="comma-separated aliases to run exclusively (default: all manifest tasks)",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        default=512,
        choices=(512, 256),
        help="eval model hidden dim; pass 256 for the twobucket axis-4 d256 checkpoints (default 512, the swarm shape)",
    )
    args = parser.parse_args()

    manifest, tasks = load_task_manifest()
    only = {t for t in args.only_tasks.split(",") if t}
    if only:
        unknown = only - set(tasks)
        if unknown:
            raise SystemExit(f"--only-tasks unknown aliases: {sorted(unknown)} (valid: {sorted(tasks)})")
        tasks = {alias: task for alias, task in tasks.items() if alias in only}
    logger.info(
        "resolved %d tasks (capacity %s, batch %s)", len(tasks), manifest["eval_capacity_factor"], manifest["batch_size"]
    )
    if args.list_tasks:
        for alias, task in sorted(tasks.items()):
            print(alias, "->", task.name, task.num_fewshot)
        return

    checkpoint_base = args.checkpoint_path or prefix_join(
        marin_prefix(), f"users/rav/grug/{args.run_name}/dev/checkpoints"
    )
    output_prefix = args.output_prefix or prefix_join(marin_prefix(), f"evaluation/grug_logprob/{args.run_name}")

    trainer_config = TrainerConfig(
        tracker=JsonLoggerConfig(),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        per_device_eval_parallelism=1,
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        # Single-process single-GPU eval: never start a jax coordination service.
        # Multiple 1-GPU eval pods can share a hostNetwork node; per-pod coordinators
        # on the fixed port collide ("connected with a different incarnation").
        distributed=DistributedConfig(initialize_jax_distributed=False),
    )
    trainer_config.initialize()
    # d512 (default) is the swarm shape (B200_MODEL); d256 rebuilds the axis-4
    # twobucket model exactly (the launcher asserts _family_model(512)==B200_MODEL).
    eval_model = b200_panel.B200_MODEL if args.model_dim == 512 else _family_model(args.model_dim)
    max_seq_len = eval_model.max_seq_len

    with trainer_config.use_device_mesh():
        key = jax.random.PRNGKey(0)
        with use_cpu_device():
            transformer_shape = eqx.filter_eval_shape(Transformer.init, eval_model, key=key)
            checkpoint_path = latest_checkpoint_path(str(checkpoint_base))
            logger.info("loading checkpoint %s", checkpoint_path)
            transformer = load_checkpoint(
                transformer_shape,
                checkpoint_path,
                subpath="params",
                axis_mapping=trainer_config.parameter_axis_mapping,
            )
        transformer = _with_eval_capacity(transformer, float(manifest["eval_capacity_factor"]))
        # FA4 attention is bf16/fp16-only: evaluate in the training compute dtype
        # (params were fp32 at rest, bf16 in compute — same as the train step's
        # cast_to_compute). The swarm's TPU eval ran the raw fp32 params through
        # splash attention; the bf16 delta is a per-task level offset shared by
        # every panel run, so within-panel sigma and SNR are unaffected.
        transformer = trainer_config.mp.cast_to_compute(transformer)

        tokenizer = load_tokenizer(marin_tokenizer)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer = dataclasses.replace(tokenizer, _pad_id=tokenizer.eos_token_id)

        import transformers  # noqa: PLC0415 — shim target

        if not hasattr(transformers, "AutoModelForVision2Seq"):
            # transformers>=5 renamed AutoModelForVision2Seq; the pinned lm-eval fork's
            # hf_vlms module still imports the old name at import time. We never use
            # HF model classes (our LM wraps the jax grug model), so alias it.
            transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText

        from lm_eval import evaluator as lm_eval_evaluator  # noqa: PLC0415 — overlay-only dep
        from lm_eval.tasks import TaskManager, get_task_dict  # noqa: PLC0415 — overlay-only dep

        lm = _make_grug_lm(
            transformer,
            tokenizer,
            max_seq_len=max_seq_len,
            max_cont_len=int(manifest["max_cont_len"]),
            batch_size=int(manifest["batch_size"]),
            pad_id=tokenizer.eos_token_id,
        )

        redo = {t for t in args.redo_tasks.split(",") if t}
        done, failed, skipped_gated = [], [], []
        for alias, task in sorted(tasks.items()):
            if alias in KNOWN_GATED_TASKS:
                logger.info("[%s] gated HF dataset (no cluster auth) — skipping; excluded from readout", alias)
                skipped_gated.append(alias)
                continue
            results_path = StoragePath(output_prefix) / alias / "results.json"
            if alias not in redo and results_path.exists():
                logger.info("[%s] results exist, skipping", alias)
                done.append(alias)
                continue
            start = time.time()
            try:
                task_dict = get_task_dict([_lm_eval_spec(task)], task_manager=TaskManager())
                _fill_missing_task_names(task_dict)
                _apply_num_fewshot(task_dict, task.num_fewshot)
                _add_bpb_metric(task_dict)
                # bootstrap_iters=0: metric-stderr bootstrapping uses multiprocessing
                # fork, which deadlocks/crashes under multithreaded JAX.
                results = lm_eval_evaluator.evaluate(
                    lm=lm, task_dict=task_dict, limit=None, log_samples=False, bootstrap_iters=0
                )
            except Exception:
                logger.exception("[%s] task failed; continuing", alias)
                failed.append(alias)
                continue
            results_path.parent.mkdirs()
            results_path.write_text(json.dumps(results, indent=2, default=lambda value: repr(value)))
            logger.info("[%s] done in %.1fs -> %s", alias, time.time() - start, results_path)
            done.append(alias)

    logger.info(
        "run %s complete: %d ok, %d failed (%s), %d gated-skipped (%s)",
        args.run_name,
        len(done),
        len(failed),
        failed,
        len(skipped_gated),
        skipped_gated,
    )
    if failed:
        raise SystemExit(f"{len(failed)} tasks failed: {failed}")


if __name__ == "__main__":
    main()
