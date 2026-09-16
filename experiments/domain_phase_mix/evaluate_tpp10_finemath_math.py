# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Score reference math solutions on eight frozen FineMath proxy checkpoints."""

import argparse
import asyncio
import gzip
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import fsspec
import haliax as hax
import jax
import jmp
import numpy as np
from google.cloud import storage
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.eval import TaggedEvaluator, eval_model
from levanter.models.lm_model import LmExample
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from marin.execution.remote import remote

from experiments.domain_phase_mix import evaluate_starcoder_tpp10_uncheatable as original
from experiments.domain_phase_mix import repair_tpp10_evaluation as repair
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

logger = logging.getLogger(__name__)
DIRECTORY = Path(".agents/projects/starcoder_tpp10/domain_sweeps/math_eval_20260912")
SURVEY = Path(".agents/projects/starcoder_tpp10/domain_sweeps/plan.json")
REPAIR = Path(".agents/projects/starcoder_tpp10/domain_sweeps/repairs_20260911/plan.json")
BUCKET = "marin-us-central1"
SOURCES = {
    "math500": ("raw/math500-ec7cea/math500-00000.jsonl.gz", 500),
    "gsm8k": ("raw/gsm8k-ef3e8c/gsm8k-00000.jsonl.gz", 1319),
}
TEMPLATE = (
    "{{ bos_token }}Question: {{ messages[0]['content'] }}\nAnswer: "
    "{% generation %}{{ messages[1]['content'] }}{% endgeneration %}"
)
STRIDE = 1024
GRID = (0, 5, 10, 20, 30, 50, 70, 100)


@dataclass(frozen=True)
class ScoringWindow:
    tokens: np.ndarray
    loss_weight: np.ndarray


def completion_windows(
    token_ids: list[int], completion_mask: list[int], *, context: int, stride: int, pad_id: int
) -> list[ScoringWindow]:
    """Score every completion token once, retaining overlap as unscored context."""
    if len(token_ids) != len(completion_mask) or not 0 < stride < context:
        raise ValueError("Invalid token masks or sliding-window geometry")
    result = []
    previous_end = 1
    for end in range(context, len(token_ids) + context, stride):
        end = min(end, len(token_ids))
        start = max(0, end - context)
        tokens = np.full(context, pad_id, dtype=np.int32)
        tokens[: end - start] = token_ids[start:end]
        weights = np.zeros(context, dtype=np.float32)
        for target in range(max(previous_end, start + 1), end):
            weights[target - start - 1] = completion_mask[target]
        if weights.sum() > 0:
            result.append(ScoringWindow(tokens, weights))
        previous_end = end
        if end == len(token_ids):
            break
    if sum(int(w.loss_weight.sum()) for w in result) != sum(completion_mask[1:]):
        raise ValueError("Sliding windows lost or duplicated completion tokens")
    return result


def make_example(window: ScoringWindow) -> LmExample:
    position = hax.Axis("position", len(window.tokens))
    # Native text examples carry segment IDs; use one segment so mixed batches
    # have the same tree structure without adding any attention boundary.
    return LmExample.causal(
        hax.named(window.tokens, position),
        loss_weight=hax.named(window.loss_weight, position),
        segment_ids=hax.named(np.zeros(len(window.tokens), dtype=np.int32), position),
    )


def prepare_math(spec: dict) -> tuple[dict[str, list[ScoringWindow]], dict]:
    """Read only the pinned evaluation objects and preserve every reference solution."""
    tokenizer = experiment.verified_tokenizer()
    bucket = storage.Client().bucket(BUCKET)
    populations, counts = {}, {}
    for name, source in spec["sources"].items():
        blob = bucket.blob(source["name"], generation=int(source["generation"]))
        encoded = blob.download_as_bytes(if_generation_match=int(source["generation"]), timeout=60)
        if hashlib.sha256(encoded).hexdigest() != source["sha256"]:
            raise ValueError(f"Evaluation source changed: {name}")
        rows = [json.loads(line) for line in gzip.decompress(encoded).splitlines() if line.strip()]
        if len(rows) != source["rows"]:
            raise ValueError(f"Evaluation population changed: {name}")
        conversations = [
            [{"role": "user", "content": row["problem"]}, {"role": "assistant", "content": row["solution"]}]
            for row in rows
        ]
        encoded_rows = tokenizer.apply_chat_template_with_masks(conversations, chat_template=TEMPLATE)
        windows = []
        problem_counts = []
        for row, ids, mask in zip(rows, encoded_rows["input_ids"], encoded_rows["assistant_masks"], strict=True):
            if not row["solution"].strip() or sum(mask) == 0 or mask[0] != 0:
                raise ValueError(f"Empty reference solution in {name}")
            parts = completion_windows(
                ids, mask, context=experiment.SEQ_LEN, stride=STRIDE, pad_id=int(tokenizer.eos_token_id)
            )
            windows.extend(parts)
            problem_counts.append({"tokens": len(ids), "scored_tokens": sum(mask), "windows": len(parts)})
        populations[name] = windows
        counts[name] = {
            "problems": len(rows),
            "windows": len(windows),
            "scored_tokens": sum(r["scored_tokens"] for r in problem_counts),
            "long_problems": sum(r["tokens"] > experiment.SEQ_LEN for r in problem_counts),
            "max_problem_tokens": max(r["tokens"] for r in problem_counts),
            "tokenized_sha256": canonical_sha256(encoded_rows),
            "problem_counts": problem_counts,
        }
    return populations, counts


def code_pins() -> dict[str, str]:
    pins = original.code_pins()
    for path in [
        Path(__file__),
        Path(repair.__file__),
        original.launcher.REPO / "lib/levanter/src/levanter/models/lm_model.py",
        original.launcher.REPO / "lib/levanter/src/levanter/data/dataset.py",
        original.launcher.REPO / "lib/levanter/src/levanter/data/loader.py",
        original.launcher.REPO / "lib/levanter/src/levanter/layers/attention_mask.py",
    ]:
        pins[str(path.resolve().relative_to(original.launcher.REPO))] = file_sha256(path)
    return pins


def build_spec() -> dict:
    survey = repair.read_json(str(SURVEY))
    repaired = repair.read_json(str(REPAIR))
    controls = [a for a in repaired["audits"] if a["request"]["run_name"] == "tpp10_unmatched_p000_s20260910"]
    if len(controls) != 1:
        raise ValueError("Expected the unique frozen zero-fraction proxy control")
    control = controls[0]
    requests = [control["request"]] + [
        r for r in survey["runs"] if r["domain"] == "finemath_3plus" and r["arm"] == "matched"
    ]
    if tuple(r["percent"] for r in requests) != GRID:
        requests.sort(key=lambda r: r["percent"])
    if tuple(r["percent"] for r in requests) != GRID:
        raise ValueError("Expected exactly seven FineMath proxies and the shared zero-fraction proxy")
    bucket = storage.Client().bucket(BUCKET)
    sources = {}
    for name, (path, count) in SOURCES.items():
        blob = bucket.get_blob(path, timeout=60)
        if blob is None:
            raise ValueError(f"Regional evaluation source missing: {path}")
        encoded = blob.download_as_bytes(if_generation_match=int(blob.generation), timeout=60)
        sources[name] = {
            "name": path,
            "generation": str(blob.generation),
            "crc32c": blob.crc32c,
            "size": blob.size,
            "rows": count,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    endpoints = []
    for request in requests:
        plan = control["training_plan"] if request["percent"] == 0 else survey
        evidence = repair.training_evidence(request, plan)
        endpoints.append(
            {
                "request": request,
                "checkpoint": repair.checkpoint_snapshot(request),
                "paloma_loss": evidence["metrics"][f"eval/{repair.PALOMA}/loss"],
            }
        )
    spec = {
        "purpose": (
            "Exploratory math likelihood on the complete existing FineMath proxy grid; no training or target evaluation"
        ),
        "endpoints": endpoints,
        "sources": sources,
        "tokenizer_pins": original.read_json(str(experiment.ASSETS / "pins.json")),
        "primary": "math500",
        "secondary": "gsm8k",
        "metric": "exp(total reference-solution token NLL / scored reference-solution tokens)",
        "format": TEMPLATE,
        "context": experiment.SEQ_LEN,
        "stride": STRIDE,
        "eos_policy": "No appended EOS; prompt, BOS, overlap and padding contribute zero loss",
        "boundary_policy": (
            "Joint tokenization scores a token spanning the final prompt space and the first solution character"
        ),
        "aggregation": "Token-weighted within each dataset; no aggregate across MATH-500 and GSM8K",
        "contamination_status": (
            "No benchmark decontamination audit of the FineMath training subset; exploratory likelihood diagnostic"
        ),
        "paloma_path": repaired["evaluation_paths"][repair.PALOMA],
        "batch_size": experiment.PROXY_BATCH_SIZE,
        "code_sha256": code_pins(),
    }
    _, spec["population_counts"] = prepare_math(spec)
    spec["spec_sha256"] = canonical_sha256(spec)
    return spec


def validate_spec(spec: dict) -> None:
    if spec["spec_sha256"] != canonical_sha256({k: v for k, v in spec.items() if k != "spec_sha256"}):
        raise ValueError("Math evaluation specification changed")
    if spec["code_sha256"] != code_pins():
        raise ValueError("Math evaluation source code changed")
    for name, digest in spec["tokenizer_pins"]["files_sha256"].items():
        if file_sha256(experiment.ASSETS / name) != digest:
            raise ValueError(f"Tokenizer asset changed: {name}")
    if tuple(e["request"]["percent"] for e in spec["endpoints"]) != GRID:
        raise ValueError("Math evaluation must cover the complete frozen proxy grid")
    if any(e["request"]["arm"] == "target" for e in spec["endpoints"]):
        raise ValueError("Target evaluation is outside this release")


def output_root(spec: dict) -> str:
    return f"{experiment.PREFIX}/experiments/tpp10_finemath_math/{spec['spec_sha256']}"


def verified_result(spec: dict, endpoint: dict) -> dict | None:
    uri = f"{output_root(spec)}/{endpoint['request']['run_name']}.json"
    fs, path = fsspec.core.url_to_fs(uri)
    if not fs.exists(path):
        return None
    result = repair.read_json(uri)
    if result["spec_sha256"] != spec["spec_sha256"] or result["endpoint"] != endpoint:
        raise ValueError(f"Stored evaluation identity differs: {uri}")
    for name in SOURCES:
        loss = result["metrics"][f"eval/{name}/loss"]
        if not math.isfinite(loss) or not math.isclose(result["perplexity"][name], math.exp(loss), rel_tol=1e-12):
            raise ValueError(f"Invalid likelihood result: {uri}")
    repair.check_close(result["metrics"][f"eval/{repair.PALOMA}/loss"], endpoint["paloma_loss"], "Restored PALOMA loss")
    return result


def evaluate(spec: dict) -> None:
    """Evaluate all eight proxies in one TPU worker, reusing compilation and completed receipts."""
    experiment.require_central1()
    validate_spec(spec)
    pending = [e for e in spec["endpoints"] if verified_result(spec, e) is None]
    if not pending:
        return
    populations, counts = prepare_math(spec)
    if counts != spec["population_counts"]:
        raise ValueError("Tokenization or sliding-window population changed")
    trainer = TrainerConfig(
        tracker=NoopConfig(),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=spec["batch_size"],
        per_device_parallelism=-1,
        per_device_eval_parallelism=-1,
        allow_nondivisible_batch_size=True,
        max_eval_batches=None,
        log_jaxprs=False,
        log_xla_hlo=False,
        mesh=MeshConfig(
            axes={"data": -1, "replica": 1, "model": 1},
            compute_mapping={
                "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
            },
        ),
    )
    trainer.initialize()
    tokenizer = experiment.verified_tokenizer()
    config = experiment.model_config(experiment.Arm.MATCHED)
    with trainer.use_device_mesh():
        sets = [(ListAsyncDataset(windows).map(make_example), [name]) for name, windows in populations.items()]
        sets += list(repair.evaluation_data({repair.PALOMA: spec["paloma_path"]}).tagged_eval_sets(config.max_Pos))
        evaluator = TaggedEvaluator(
            EvalBatch=trainer.EvalBatch,
            tagged_eval_sets=sets,
            loss_fn=original.eval_loss,
            tokenizer=tokenizer,
            device_mesh=trainer.device_mesh,
            axis_mapping=trainer.compute_axis_mapping,
        )
        for endpoint in pending:
            request = endpoint["request"]
            if repair.checkpoint_snapshot(request) != endpoint["checkpoint"]:
                raise ValueError(f"Checkpoint metadata changed: {request['run_name']}")
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.build, hax.Axis("vocab", len(tokenizer)), key=jax.random.PRNGKey(0))
                model = load_checkpoint(model, endpoint["checkpoint"]["path"], subpath="model")
            model = hax.shard_with_axis_mapping(model, trainer.parameter_axis_mapping)
            metrics = {k: float(v) for k, v in eval_model(evaluator, model, prefix="eval").items()}
            result = {
                "spec_sha256": spec["spec_sha256"],
                "endpoint": endpoint,
                "metrics": metrics,
                "perplexity": {name: math.exp(metrics[f"eval/{name}/loss"]) for name in SOURCES},
                "population_sha256": canonical_sha256(counts),
            }
            if jax.process_index() == 0:
                repair.write_json(f"{output_root(spec)}/diagnostics/{request['run_name']}.json", result)
            repair.check_close(metrics[f"eval/{repair.PALOMA}/loss"], endpoint["paloma_loss"], "Restored PALOMA loss")
            if jax.process_index() == 0:
                repair.write_json(f"{output_root(spec)}/{request['run_name']}.json", result)
                logger.info("Completed %s: %s", request["run_name"], result["perplexity"])
            del model


async def submit(spec: dict) -> None:
    experiment.require_central1()
    validate_spec(spec)
    repair.write_json(output_root(spec) + "/spec.json", spec)
    if all(verified_result(spec, e) is not None for e in spec["endpoints"]):
        return
    worker = remote(
        evaluate, name="finemath-math-proxies", resources=original.TPU, env_vars={"MARIN_PREFIX": experiment.PREFIX}
    )
    await asyncio.to_thread(worker, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DIRECTORY / "spec.json")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.build:
        spec = build_spec()
        args.spec.parent.mkdir(parents=True, exist_ok=True)
        args.spec.write_text(json.dumps(spec, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "spec_sha256": spec["spec_sha256"],
                    "checkpoints": len(spec["endpoints"]),
                    "counts": {
                        k: {a: b for a, b in v.items() if a != "problem_counts"}
                        for k, v in spec["population_counts"].items()
                    },
                }
            )
        )
    else:
        asyncio.run(submit(repair.read_json(str(args.spec))))


if __name__ == "__main__":
    main()
