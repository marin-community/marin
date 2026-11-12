#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay sealed OlympiadBench answers through Evalchemy's native grader only."""

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from eval.chat_benchmarks.OlympiadBench.eval_instruct import OlympiadBenchBenchmark
from eval.graders.answer_equivalence import JudgeLabel
from finestore.reader import ReadView
from iris.rpc.proto_display import priority_band_value
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.serve import local_inference
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

LOGGER = logging.getLogger(__name__)
N_QUESTIONS = 30
N_REPEATS = 10
EXPECTED_TRIALS = N_QUESTIONS * N_REPEATS
SAMPLE_COLUMNS = ["doc_id", "doc", "trial_id", "output", "extracted", "metrics"]
EVALCHEMY_ABLATION_COMMIT = "fd64dbb9bce0e2a8a4f0deaba0b82dad4e07dff8"
JUDGE_EXTRA_BODY = {"structured_outputs": {"choice": [label.value for label in JudgeLabel]}}


@dataclass(frozen=True)
class Source:
    model: str
    old_score: float
    uri: str


def read_sources(path: Path) -> list[Source]:
    """Load a frozen list of scored trace archives."""
    payload = yaml.safe_load(path.read_text())
    sources = [Source(row["model"], float(row["old_score"]), row["source"]) for row in payload["results"]]
    if len({source.model for source in sources}) != len(sources):
        raise ValueError("duplicate model in source manifest")
    return sources


def source_examples(source: Source) -> tuple[list[dict], list[dict]]:
    """Rebuild the original 30-by-10 Evalchemy input from immutable sample rows."""
    table = ReadView(source.uri).scan("samples", columns=SAMPLE_COLUMNS)
    if table is None or table.num_rows != EXPECTED_TRIALS:
        raise ValueError(f"{source.model}: expected {EXPECTED_TRIALS} sealed samples")
    rows = table.to_pylist(maps_as_pydicts="strict")
    by_document: dict[int, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        document = int(row["doc_id"])
        repeat = int(row["trial_id"])
        if repeat in by_document[document]:
            raise ValueError(f"{source.model}: duplicate trial {document}/{repeat}")
        by_document[document][repeat] = row
    if set(by_document) != set(range(N_QUESTIONS)):
        raise ValueError(f"{source.model}: incomplete document IDs")

    examples = []
    ordered_rows = []
    for document in range(N_QUESTIONS):
        trials = by_document[document]
        if set(trials) != set(range(N_REPEATS)):
            raise ValueError(f"{source.model}: incomplete repetitions for document {document}")
        first = json.loads(trials[0]["doc"])
        example = dict(first)
        example["model_outputs"] = [trials[repeat]["output"] or "" for repeat in range(N_REPEATS)]
        example["model_answers"] = [trials[repeat]["extracted"] or "" for repeat in range(N_REPEATS)]
        examples.append(example)
        ordered_rows.extend(trials[repeat] for repeat in range(N_REPEATS))

    old_score = sum(float(row["metrics"]["accuracy"]) for row in ordered_rows) / EXPECTED_TRIALS
    if abs(old_score - source.old_score) > 0.0005:
        raise ValueError(f"{source.model}: sample score {old_score:.6f} differs from tracker {source.old_score:.3f}")
    return examples, ordered_rows


def write_json(uri: str, value: dict) -> None:
    """Write one durable, self-contained ablation artifact."""
    filesystem, key = filesystem_for(uri)
    with filesystem.open(key, "wb") as destination:
        destination.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())


def exists(uri: str) -> bool:
    filesystem, key = filesystem_for(uri)
    return filesystem.exists(key)


def regrade(source: Source, base_url: str, judge_model: str, diagnostic_uri: str) -> dict:
    examples, rows = source_examples(source)
    benchmark = OlympiadBenchBenchmark(
        n_repeat=N_REPEATS,
        annotator_model=judge_model,
        judge_base_url=base_url,
        judge_api_key="local-judge-ablation",
    )
    assert benchmark.judge_config is not None
    benchmark.judge_config = replace(benchmark.judge_config, extra_body=JUDGE_EXTRA_BODY)
    scored = benchmark.evaluate_responses({"examples": examples})
    if scored["num_judge_failed"]:
        failures = [
            {
                "doc_id": document,
                "trial_id": repeat,
                "reference_answer": example["answer"],
                "candidate_answer": example["model_answers"][repeat],
                "error": grade["judge_error"],
            }
            for document, example in enumerate(examples)
            for repeat, grade in enumerate(example["equivalence_grades"])
            if grade.get("judge_error")
        ]
        write_json(
            diagnostic_uri,
            {"model": source.model, "source": source.uri, "judge_model": judge_model, "failures": failures},
        )
        raise RuntimeError(f"{source.model}: {scored['num_judge_failed']} judge requests failed")

    trials = []
    for document, example in enumerate(examples):
        for repeat in range(N_REPEATS):
            row = rows[document * N_REPEATS + repeat]
            grade = example["equivalence_grades"][repeat]
            new_correct = bool(example["sample_metrics_by_repeat"][repeat]["accuracy"])
            trials.append(
                {
                    "doc_id": document,
                    "trial_id": repeat,
                    "reference_answer": example["answer"],
                    "candidate_answer": example["model_answers"][repeat],
                    "old_correct": bool(row["metrics"]["accuracy"]),
                    "new_correct": new_correct,
                    "equivalence_grade": grade,
                }
            )
    mean = sum(trial["new_correct"] for trial in trials) / EXPECTED_TRIALS
    if abs(mean - float(scored["accuracy_avg"])) > 1e-9:
        raise ValueError(f"{source.model}: native aggregate differs from its trial verdicts")
    return {
        "model": source.model,
        "source": source.uri,
        "old_score": source.old_score,
        "new_score": mean,
        "delta": mean - source.old_score,
        "num_trials": EXPECTED_TRIALS,
        "num_graded_by_minerva": scored["num_graded_by_minerva"],
        "num_judged_by_llm": scored["num_judged_by_llm"],
        "num_judge_failed": scored["num_judge_failed"],
        "judge_model": judge_model,
        "judge_base_url": base_url,
        "evalchemy_commit": EVALCHEMY_ABLATION_COMMIT,
        "judge_request_extra_body": JUDGE_EXTRA_BODY,
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--model", action="append", default=[])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()

    sources = read_sources(args.sources)
    if args.model:
        sources = [source for source in sources if source.model in args.model]
        if len(sources) != len(args.model):
            raise ValueError("requested model absent from source manifest")
    for source in sources:
        source_examples(source)
        LOGGER.info("validated %s: %d sealed trials", source.model, EXPECTED_TRIALS)
    if args.validate_only:
        return

    judge = load_model_config(args.judge_config)
    if judge.serve.max_model_len != 131072 or judge.serve.tensor_parallel_size != 8:
        raise ValueError("judge ablation requires the pinned 131072-token TP8 serve config")
    prefix = args.output_prefix.rstrip("/")
    write_json(
        f"{prefix}/provenance.json",
        {
            "sources_sha256": hashlib.sha256(args.sources.read_bytes()).hexdigest(),
            "judge_config_sha256": hashlib.sha256(args.judge_config.read_bytes()).hexdigest(),
            "sources": args.sources.read_text(),
            "judge_config": args.judge_config.read_text(),
            "evalchemy_commit": EVALCHEMY_ABLATION_COMMIT,
            "judge_request_extra_body": JUDGE_EXTRA_BODY,
        },
    )
    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8)
    inference = inference_config_for_model(
        judge,
        accelerator,
        env_vars={},
        priority=priority_band_value("interactive"),
        api_model=judge.location,
    )
    with local_inference(inference.model, inference.engine, num_chips=8) as session:
        endpoint = session.model.endpoint
        LOGGER.info("judge ready: %s with context %d", endpoint.model, judge.serve.max_model_len)
        for source in sources:
            name = source.model.replace("/", "--")
            uri = f"{prefix}/results/{name}.json"
            if exists(uri):
                LOGGER.info("already regraded %s", source.model)
                continue
            result = regrade(source, endpoint.base_url, endpoint.model, f"{prefix}/diagnostics/{name}.json")
            write_json(uri, result)
            LOGGER.info("%s: %.3f -> %.3f", source.model, result["old_score"], result["new_score"])


if __name__ == "__main__":
    main()
