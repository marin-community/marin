# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and publish the native AgentX workload on CPU before GPU submission."""

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path

from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.agentx.artifacts import (
    copy_file,
    file_digest,
    read_record,
    retain_artifacts,
    unpack,
    write_record,
)
from experiments.evaluation.agentx.processes import run_logged
from experiments.evaluation.agentx.recipe import (
    PREPARATION_VERSION,
    REQUIREMENTS_PATH,
    SERIES_PATH,
    load_series,
    preparation_command,
    workload_environment,
)

logger = logging.getLogger(__name__)
WORKLOAD_TTL_DAYS = 30
PREPARATION_INPUTS = (
    "model",
    "model_revision",
    "harness_version",
    "harness_source",
    "dataset",
    "dataset_loader",
    "dataset_revision",
    "dataset_sha256",
    "sessions",
    "max_context",
    "random_seed",
    "trace_idle_gap_cap_seconds",
)


def preparation_recipe(series: dict) -> dict:
    """Exclude serving releases and measurement knobs from derived-data identity."""
    directory = Path(__file__).parent
    return {
        "version": PREPARATION_VERSION,
        "inputs": {key: series[key] for key in PREPARATION_INPUTS},
        "code_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ("prepare.py", "recipe.py")
        },
        "harness_requirements_sha256": hashlib.sha256(REQUIREMENTS_PATH.read_bytes()).hexdigest(),
    }


def check_native_cache(bundle: Path, series_path: Path, log: Path) -> None:
    series = load_series(series_path)
    run_logged(
        [*preparation_command(series, series_path, bundle), "--check"],
        workload_environment(bundle, series),
        log,
        timeout=300,
    )


def restore_workload(receipt: dict, bundle: Path) -> None:
    """Restore exact bytes from the frozen publication."""
    bundle.mkdir(parents=True, exist_ok=False)
    archive = bundle.parent / "workload.tar.zst"
    copy_file(receipt["archive_uri"], str(archive))
    digest = file_digest(archive)
    if digest != receipt["archive_sha256"]:
        raise ValueError("Prepared workload bundle differs from its retained digest")
    unpack(archive, bundle)


def prepare_workload(work: Path, *, series_path: Path = SERIES_PATH, cache_root: str | None = None) -> dict:
    """Restore a TTL hit or build a missing native workload in an isolated CPU process."""
    work.mkdir(parents=True, exist_ok=False)
    series = load_series(series_path)
    recipe = preparation_recipe(series)
    key = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    root = cache_root or marin_temp_bucket(WORKLOAD_TTL_DAYS, "agentx-workload")
    pointer = prefix_join(root, f"{key}/workload.json")
    bundle = work / "bundle"
    try:
        receipt = read_record(pointer)
        if receipt["recipe"] != recipe:
            raise ValueError("Prepared workload pointer differs from its recipe")
        logger.info("AgentX workload cache HIT: %s", pointer)
        restore_workload(receipt, bundle)
    except FileNotFoundError:
        # TTL may remove either the pointer or its data first. Rebuild only here,
        # on CPU; children receive a frozen archive URI and never reconstruct.
        if bundle.exists():
            shutil.rmtree(bundle)
    else:
        check_native_cache(bundle, series_path, work / "native-cache-check.log")
        # Renew the 30-day lifetime before a child can spend 48 hours queued.
        # The URI remains content-addressed and these are the verified same bytes.
        copy_file(str(work / "workload.tar.zst"), receipt["archive_uri"])
        write_record(pointer, receipt)
        return receipt | {"cache_status": "hit"}
    logger.info("AgentX workload cache MISS: preparing all %s sessions on CPU", series["sessions"])
    bundle.mkdir()
    environment = workload_environment(bundle, series) | {
        "HF_HOME": str(work / "sources"),
        "AIPERF_DATASET_MMAP_BASE_PATH": str(work / "mmap"),
        "AIPERF_DATASET_WEKA_PARALLEL_WORKERS": "1",
    }
    # AgentX treats any nonempty offline value, including "0", as true.
    for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        environment.pop(variable, None)
    run_logged(
        preparation_command(series, series_path, bundle),
        environment,
        work / "preparation.log",
        timeout=series["prepare_timeout"],
    )
    check_native_cache(bundle, series_path, work / "native-cache-check.log")
    archive = work / "workload.tar.zst"
    uri, digest = retain_artifacts(bundle, archive, prefix_join(root, key), name="workload")
    receipt = {
        "key": key,
        "recipe": recipe,
        "archive_uri": uri,
        "archive_sha256": digest,
        "archive_bytes": archive.stat().st_size,
        "lineage": json.loads((bundle / "lineage.json").read_text()),
    }
    write_record(pointer, receipt)
    logger.info("Published AgentX workload: %s", uri)
    return receipt | {"cache_status": "built"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, default=SERIES_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    receipt = prepare_workload(args.output, series_path=args.series, cache_root=args.cache_root)
    write_record(str(args.output / "workload.json"), receipt)
    print(f"Prepared workload receipt: {args.output / 'workload.json'}", flush=True)


if __name__ == "__main__":
    main()
