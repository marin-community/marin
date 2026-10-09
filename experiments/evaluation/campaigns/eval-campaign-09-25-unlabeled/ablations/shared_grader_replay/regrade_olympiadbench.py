#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay OlympiadBench with the fixed native grader and cached MiniMax verdicts."""

import argparse
import hashlib
import json
import logging
from collections import defaultdict, deque
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import yaml
from eval.chat_benchmarks.OlympiadBench.eval_instruct import OlympiadBenchBenchmark, _flatten_reference_answers
from eval.graders import answer_equivalence
from eval.graders.answer_equivalence import EquivalenceJudgment, JudgeConfig, JudgeLabel
from iris.rpc.proto_display import priority_band_value
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.serve import local_inference
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from ..olympiadbench_judge.regrade import JUDGE_EXTRA_BODY, Source, source_examples

LOGGER = logging.getLogger(__name__)
EVALCHEMY_COMMIT = "958cdb8019b7a4c8432fe85eb9572538ceb75950"
LIVE_JUDGE = answer_equivalence.judge_equivalence
RequestKey = tuple[str, tuple[str, ...], str]


def request_key(question: str, references: tuple[str, ...], candidate: str) -> RequestKey:
    """Identify the exact judge prompt inputs without relying on trial order."""
    return question, references, candidate


class CachedJudge:
    """Replay recorded labels and send only newly unresolved requests to MiniMax."""

    def __init__(self, examples: list[dict], prior: dict, allow_live: bool):
        self.pending: dict[RequestKey, deque[EquivalenceJudgment]] = defaultdict(deque)
        self.allow_live = allow_live
        self.cached_used = 0
        self.live_used = 0
        self.prior_blank_credited = 0
        for index, trial in enumerate(prior["trials"]):
            document, repeat = divmod(index, 10)
            example = examples[document]
            candidate = example["model_answers"][repeat]
            if (trial["doc_id"], trial["trial_id"]) != (document, repeat):
                raise ValueError("prior trial positions differ from sealed source")
            if trial["candidate_answer"] != candidate or trial["reference_answer"] != example["answer"]:
                raise ValueError("prior candidate or reference differs from sealed source")
            if not candidate.strip():
                self.prior_blank_credited += int(trial["new_correct"])
                continue
            grade = trial["equivalence_grade"]
            if grade["method"] == "llm_judge":
                if grade["judge_label"] is None or grade["judge_raw"] is None:
                    raise ValueError("cached judge verdict is incomplete")
                key = request_key(
                    str(example.get("problem", example.get("question", ""))),
                    tuple(_flatten_reference_answers(example["answer"])),
                    candidate,
                )
                self.pending[key].append(EquivalenceJudgment(JudgeLabel(grade["judge_label"]), grade["judge_raw"]))
            elif grade["method"] != "minerva":
                raise ValueError(f"unexpected prior grade method: {grade['method']}")

    async def complete(self, requests, config: JudgeConfig, num_workers: int = 16):
        """Satisfy the native grader's transport boundary from cache or the live judge."""
        outcomes = [None] * len(requests)
        missing = []
        missing_indexes = []
        for index, request in enumerate(requests):
            key = request_key(request.question, request.reference_answers, request.candidate_answer)
            queue = self.pending[key]
            if queue:
                outcomes[index] = queue.popleft()
                self.cached_used += 1
            else:
                missing.append(request)
                missing_indexes.append(index)
        if missing:
            if not self.allow_live:
                raise RuntimeError(f"{len(missing)} newly unresolved answers require live MiniMax judgments")
            live = await LIVE_JUDGE(missing, config, num_workers=num_workers)
            for index, judgment in zip(missing_indexes, live, strict=True):
                outcomes[index] = judgment
            self.live_used += len(missing)
        if any(outcome is None for outcome in outcomes):
            raise ValueError("judge replay left an unresolved request")
        return outcomes

    def assert_consumed(self) -> None:
        remaining = sum(len(queue) for queue in self.pending.values())
        if remaining:
            raise ValueError(f"{remaining} cached judge verdicts were not used")


def read_json(uri: str) -> tuple[dict, str]:
    """Read a frozen prior result and return its content hash."""
    filesystem, key = filesystem_for(uri)
    with filesystem.open(key, "rb") as source:
        raw = source.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write_json(uri: str, payload: dict) -> None:
    """Write one immutable result or provenance object."""
    filesystem, key = filesystem_for(uri)
    if filesystem.exists(key):
        raise FileExistsError(uri)
    with filesystem.open(key, "wb") as destination:
        destination.write((json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())


def read_sources(path: Path) -> list[Source]:
    """Select the sealed OlympiadBench cells from the frozen tracker manifest."""
    rows = yaml.safe_load(path.read_text())["models"]
    sources = [
        Source(row["model"], float(row["benchmarks"]["olympiadbench"]["tracker_score"]),
               row["benchmarks"]["olympiadbench"]["source"])
        for row in rows
    ]
    if len(sources) != 21 or len({source.model for source in sources}) != 21:
        raise ValueError("expected 21 unique OlympiadBench sources")
    return sources


def regrade(source: Source, prior_prefix: str, base_url: str, judge_model: str, allow_live: bool) -> dict:
    """Run Evalchemy's full native OlympiadBench grader on the frozen answers."""
    examples, rows = source_examples(source)
    prior_uri = f"{prior_prefix.rstrip('/')}/results/{source.model.replace('/', '--')}.json"
    prior, prior_hash = read_json(prior_uri)
    if prior["model"] != source.model or prior["source"] != source.uri or prior["num_trials"] != 300:
        raise ValueError(f"{source.model}: prior judge ablation does not match this source")
    if prior["num_judge_failed"]:
        raise ValueError(f"{source.model}: prior judge ablation has failed verdicts")
    cache = CachedJudge(examples, prior, allow_live)
    benchmark = OlympiadBenchBenchmark(
        n_repeat=10,
        annotator_model=judge_model,
        judge_base_url=base_url,
        judge_api_key="local-judge-replay",
    )
    assert benchmark.judge_config is not None
    benchmark.judge_config = replace(benchmark.judge_config, extra_body=JUDGE_EXTRA_BODY)
    with patch.object(answer_equivalence, "judge_equivalence", cache.complete):
        scored = benchmark.evaluate_responses({"examples": examples})
    cache.assert_consumed()
    if scored["num_judge_failed"]:
        raise RuntimeError(f"{source.model}: {scored['num_judge_failed']} live judge requests failed")
    if scored["num_judged_by_llm"] != cache.cached_used + cache.live_used:
        raise ValueError(f"{source.model}: judge request accounting disagrees with native grader")
    trials = []
    for index, row in enumerate(rows):
        document, repeat = divmod(index, 10)
        prior_trial = prior["trials"][index]
        grade = examples[document]["equivalence_grades"][repeat]
        corrected = bool(examples[document]["sample_metrics_by_repeat"][repeat]["accuracy"])
        trials.append(
            {
                "doc_id": document,
                "trial_id": repeat,
                "reference_answer": examples[document]["answer"],
                "candidate_answer": examples[document]["model_answers"][repeat],
                "source_correct": bool(row["metrics"]["accuracy"]),
                "prior_minimax_correct": prior_trial["new_correct"],
                "corrected_correct": corrected,
                "prior_grade": prior_trial["equivalence_grade"],
                "corrected_grade": grade,
            }
        )
    corrected_score = sum(trial["corrected_correct"] for trial in trials) / 300
    if abs(corrected_score - float(scored["accuracy_avg"])) > 1e-9:
        raise ValueError(f"{source.model}: native aggregate differs from trial verdicts")
    return {
        "model": source.model,
        "benchmark": "olympiadbench",
        "source": source.uri,
        "tracker_score": source.old_score,
        "prior_minimax_result": prior_uri,
        "prior_minimax_sha256": prior_hash,
        "prior_minimax_score": prior["new_score"],
        "corrected_score": corrected_score,
        "num_trials": 300,
        "num_judge_failed": scored["num_judge_failed"],
        "num_graded_by_minerva": scored["num_graded_by_minerva"],
        "cached_judge_verdicts": cache.cached_used,
        "new_judge_verdicts": cache.live_used,
        "prior_blank_credited": cache.prior_blank_credited,
        "changed_trials": sum(trial["prior_minimax_correct"] != trial["corrected_correct"] for trial in trials),
        "evalchemy_commit": EVALCHEMY_COMMIT,
        "judge_model": judge_model,
        "judge_request_extra_body": JUDGE_EXTRA_BODY,
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--prior-prefix", required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    sources = read_sources(args.sources)
    if args.model:
        sources = [source for source in sources if source.model in args.model]
        if len(sources) != len(args.model):
            raise ValueError("requested model absent from source manifest")
    judge = load_model_config(args.judge_config)
    if judge.serve.max_model_len != 131072 or judge.serve.tensor_parallel_size != 8:
        raise ValueError("replay requires the pinned 131072-token TP8 MiniMax config")
    prefix = args.output_prefix.rstrip("/")
    provenance_uri = f"{prefix}/provenance-olympiadbench.json"
    filesystem, key = filesystem_for(provenance_uri)
    if not filesystem.exists(key):
        write_json(
            provenance_uri,
            {
                "evalchemy_commit": EVALCHEMY_COMMIT,
                "source_manifest_sha256": hashlib.sha256(args.sources.read_bytes()).hexdigest(),
                "source_manifest": args.sources.read_text(),
                "judge_config_sha256": hashlib.sha256(args.judge_config.read_bytes()).hexdigest(),
                "judge_config": args.judge_config.read_text(),
                "prior_prefix": args.prior_prefix,
                "judge_request_extra_body": JUDGE_EXTRA_BODY,
                "cached_verdict_rule": "Reuse an exact prior LLM label for unchanged question, references, and candidate.",
            },
        )

    def run_pending(base_url: str, allow_live: bool) -> None:
        for source in sources:
            result_uri = f"{prefix}/results/olympiadbench/{source.model.replace('/', '--')}.json"
            filesystem, key = filesystem_for(result_uri)
            if filesystem.exists(key):
                LOGGER.info("already regraded %s", source.model)
                continue
            result = regrade(source, args.prior_prefix, base_url, judge.location, allow_live)
            write_json(result_uri, result)
            LOGGER.info(
                "%s: %.3f -> %.3f; cache=%d live=%d changed=%d",
                source.model,
                result["prior_minimax_score"],
                result["corrected_score"],
                result["cached_judge_verdicts"],
                result["new_judge_verdicts"],
                result["changed_trials"],
            )

    if args.cache_only:
        run_pending("http://127.0.0.1:1/v1", allow_live=False)
        return
    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8)
    inference = inference_config_for_model(
        judge,
        accelerator,
        env_vars={},
        priority=priority_band_value("interactive"),
        api_model=judge.location,
    )
    with local_inference(inference.model, inference.engine, num_chips=8) as session:
        run_pending(session.model.endpoint.base_url, allow_live=True)


if __name__ == "__main__":
    main()
