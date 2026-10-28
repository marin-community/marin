# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build or check AgentX's native mmap cache in its separately pinned environment."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import resource
import shutil
import time
import uuid
from pathlib import Path

from agentx_cli import app
from aiperf.common import random_generator
from aiperf.common.environment import Environment
from aiperf.common.messages.command_messages import ProfileConfigureCommand
from aiperf.common.models import DatasetMetadata
from aiperf.common.scenario import get_scenario
from aiperf.config import BenchmarkRun
from aiperf.config.flags.resolver import resolve_config
from aiperf.dataset import mmap_cache
from aiperf.dataset.dataset_manager import DatasetManager
from aiperf.dataset.loader.semianalysis_cc_traces_weka import SemiAnalysisCCTracesWekaLoader
from aiperf.dataset.loader.weka_trace import _trace_peak_context_length
from aiperf.dataset.loader.weka_trace_models import WekaTrace
from huggingface_hub import snapshot_download

from experiments.evaluation.agentx.recipe import PREPARATION_VERSION, load_series, profile_arguments


class PreparationManager(DatasetManager):
    """Build AgentX's native workload on CPU without measurement workers."""

    @classmethod
    def get_service_type(cls):
        return DatasetManager.get_service_type()

    async def publish(self, _message) -> None:
        # There are no measurement workers during CPU preparation. Native dataset
        # configuration and cache publication still run unchanged.
        pass


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_sources(series: dict, bundle: Path) -> dict:
    snapshot = Path(
        snapshot_download(
            series["dataset"],
            repo_type="dataset",
            revision=series["dataset_revision"],
            allow_patterns=["traces.jsonl", "README.md", ".gitattributes"],
        )
    )
    digest = file_digest(snapshot / "traces.jsonl")
    if digest != series["dataset_sha256"]:
        raise RuntimeError("Corpus bytes differ from the reviewed pin")
    sessions = 0
    with (snapshot / "traces.jsonl").open() as stream:
        for line in stream:
            trace = WekaTrace.model_validate_json(line)
            peak = _trace_peak_context_length(trace)
            if peak > series["max_context"]:
                raise RuntimeError(f"Pinned corpus session exceeds the context: {trace.id} ({peak})")
            sessions += 1
    if sessions != series["sessions"]:
        raise RuntimeError("Corpus session count differs from the reviewed pin")
    tokenizer = Path(
        snapshot_download(
            series["tokenizer"],
            revision=series["tokenizer_revision"],
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf", "*.onnx"],
        )
    )
    repo_name = "models--" + series["tokenizer"].replace("/", "--")
    destination = bundle / "hf" / "hub" / repo_name / "snapshots" / series["tokenizer_revision"]
    shutil.copytree(tokenizer, destination, symlinks=False)
    return {
        "dataset": series["dataset"],
        "dataset_revision": series["dataset_revision"],
        "dataset_sha256": digest,
        "sessions": sessions,
        "filtered_sessions": 0,
        "tokenizer": series["tokenizer"],
        "tokenizer_revision": series["tokenizer_revision"],
        "tokenizer_files": {
            path.relative_to(tokenizer).as_posix(): file_digest(path)
            for path in sorted(tokenizer.rglob("*"))
            if path.is_file()
        },
    }


def benchmark_run(series: dict, output: Path) -> BenchmarkRun:
    _, bound, _ = app.parse_args(profile_arguments(series, 8, "http://127.0.0.1:38197", output))
    config = resolve_config(bound.arguments["cli_config"], None)
    random_generator.reset()
    random_generator.init(config.random_seed)
    return BenchmarkRun(
        benchmark_id=uuid.uuid4().hex,
        cfg=config.benchmark,
        artifact_dir=output,
        random_seed=config.random_seed,
        variables=dict(config.variables),
    )


async def native_cache(series: dict, bundle: Path, *, check_only: bool) -> dict:
    if importlib.metadata.version("agentx-harness") != series["harness_version"]:
        raise RuntimeError("AgentX distribution differs from the reviewed pin")
    run = benchmark_run(series, bundle.parent / "preparation-artifacts")
    with Environment.defaults(get_scenario("agentx").environment_defaults):
        key = mmap_cache.compute_cache_key_from_run(run)
        if key is None:
            raise RuntimeError("The pinned workload has no native cache key")
        if not check_only:
            # The released loader's existing revision field pins its HF read in
            # this CPU-only process. Reconstruction stays in the native loader.
            SemiAnalysisCCTracesWekaLoader.hf_revision = series["dataset_revision"]
            # AgentX 1.0.12 has no public standalone preparation entrypoint.
            manager = PreparationManager(run=run)
            await manager._do_profile_configure(ProfileConfigureCommand(service_id="cpu-preparation"))
        hit = mmap_cache.lookup(key, compressed=False)
        if hit is None:
            raise RuntimeError("Native cache is missing or incompatible; prepare it on CPU before submitting GPUs")
        metadata = DatasetMetadata.model_validate_json(hit.manifest.dataset_metadata_json)
        return {
            "native_cache_key": key,
            "native_manifest_version": mmap_cache.MANIFEST_VERSION,
            "conversations": hit.manifest.num_conversations,
            "turns": metadata.total_turn_count,
            "metadata_sha256": hashlib.sha256(hit.manifest.dataset_metadata_json.encode()).hexdigest(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    series = load_series(args.series)
    started = time.monotonic()
    sources = {} if args.check else prepare_sources(series, args.output)
    native = asyncio.run(native_cache(series, args.output, check_only=args.check))
    if args.check:
        print("AGENTX_NATIVE_CACHE_HIT " + json.dumps(native, sort_keys=True), flush=True)
        return
    lineage = (
        sources
        | native
        | {
            "preparation_version": PREPARATION_VERSION,
            "harness_version": series["harness_version"],
            "harness_source": series["harness_source"],
            "preparation_seconds": time.monotonic() - started,
            "peak_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
    )
    (args.output / "lineage.json").write_text(json.dumps(lineage, indent=2))
    print("AGENTX_NATIVE_CACHE_BUILT " + json.dumps(native, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
