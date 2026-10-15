#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay FinanceBench and OlympiadBench against one pinned MiniMax service."""

import argparse
import hashlib
import logging
import tomllib
from pathlib import Path

from iris.rpc.proto_display import priority_band_value
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.serve import local_inference
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join

from . import regrade_financebench, regrade_olympiadbench

LOGGER = logging.getLogger(__name__)


def external_revision(path: Path, project: str) -> str:
    """Read an exact source revision from a canonical isolated-project pin."""
    source = tomllib.loads(path.read_text())["tool"]["uv"]["sources"][project]
    revision = source.get("rev")
    if not revision:
        raise ValueError(f"{path}: {project} is not pinned to an exact revision")
    return revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--prior-olympiad-prefix")
    parser.add_argument("--benchmark", choices=("financebench", "olympiadbench"), action="append")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    configure_coreweave_s3()
    benchmarks = tuple(args.benchmark or ("financebench", "olympiadbench"))
    sources = {
        "financebench": regrade_financebench.read_sources(args.sources),
        "olympiadbench": regrade_olympiadbench.read_sources(args.sources),
    }
    if args.model:
        requested = set(args.model)
        for benchmark in benchmarks:
            sources[benchmark] = [source for source in sources[benchmark] if source.model in requested]
            if {source.model for source in sources[benchmark]} != requested:
                raise ValueError(f"requested model absent from {benchmark} source manifest")
    for source in sources.get("financebench", ()) if "financebench" in benchmarks else ():
        regrade_financebench.source_examples(source)
        LOGGER.info("validated FinanceBench %s", source.model)
    for source in sources.get("olympiadbench", ()) if "olympiadbench" in benchmarks else ():
        regrade_olympiadbench.source_examples(source)
        LOGGER.info("validated OlympiadBench %s", source.model)
    if args.validate_only:
        return

    judge = load_model_config(args.judge_config)
    if judge.serve.max_model_len != 131072 or judge.serve.tensor_parallel_size != 8:
        raise ValueError("replay requires the pinned 131072-token TP8 MiniMax config")
    root = Path(__file__).resolve().parents[6]
    evalchemy_pin = external_revision(root / "config/external/evalchemy/pyproject.toml", "evalchemy")
    harbor_pin = external_revision(root / "config/external/harbor/pyproject.toml", "harbor")
    prefix = args.output_prefix
    provenance = {
        "source_manifest_sha256": hashlib.sha256(args.sources.read_bytes()).hexdigest(),
        "source_manifest": args.sources.read_text(),
        "judge_config_sha256": hashlib.sha256(args.judge_config.read_bytes()).hexdigest(),
        "judge_config": args.judge_config.read_text(),
        "evalchemy_commit": evalchemy_pin,
        "harbor_commit": harbor_pin,
        "judge_request_extra_body": regrade_olympiadbench.JUDGE_EXTRA_BODY,
        "benchmarks": benchmarks,
        "prior_olympiad_prefix": args.prior_olympiad_prefix,
    }
    regrade_olympiadbench.write_json(prefix_join(prefix, "provenance.json"), provenance)

    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8)
    inference = inference_config_for_model(
        judge,
        accelerator,
        env_vars={},
        priority=priority_band_value("interactive"),
        api_model=judge.location,
    )
    with local_inference(inference.model, inference.engine, num_chips=8) as session:
        base_url = session.model.endpoint.base_url
        for benchmark in benchmarks:
            for source in sources[benchmark]:
                uri = prefix_join(prefix, f"results/{benchmark}/{source.model.replace('/', '--')}.json")
                filesystem, key = filesystem_for(uri)
                if filesystem.exists(key):
                    LOGGER.info("already regraded %s/%s", source.model, benchmark)
                    continue
                if benchmark == "financebench":
                    result = regrade_financebench.regrade(source, base_url, judge.location)
                else:
                    result = regrade_olympiadbench.regrade(
                        source,
                        args.prior_olympiad_prefix,
                        base_url,
                        judge.location,
                        allow_live=True,
                    )
                result["evalchemy_commit"] = evalchemy_pin
                result["harbor_commit"] = harbor_pin
                regrade_olympiadbench.write_json(uri, result)
                LOGGER.info(
                    "%s/%s: %.3f -> %.3f",
                    source.model,
                    benchmark,
                    source.old_score,
                    result["corrected_score"],
                )


if __name__ == "__main__":
    main()
