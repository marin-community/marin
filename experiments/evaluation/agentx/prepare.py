# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the upstream public loader offline, without changing its corpus or sampler.

Run in the separate, pinned AgentX environment. The released public loader has
no revision option. Verify its main ref before and after priming, verify the raw
bytes, then run the benchmark offline. A moved public ref fails closed.
"""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

from aiperf.dataset.loader.weka_trace import _trace_peak_context_length
from aiperf.dataset.loader.weka_trace_models import WekaTrace
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer


def prepare(series_path: Path, output: Path) -> None:
    series = json.loads(series_path.read_text())
    if importlib.metadata.version("agentx-harness") != series["harness_version"]:
        raise RuntimeError("AgentX distribution differs from the series pin")
    api = HfApi()
    dataset = series["dataset"]
    revision = series["dataset_revision"]
    if api.dataset_info(dataset).sha != revision:
        raise RuntimeError("Public corpus main moved; review a new series instead of silently changing it")
    snapshot = Path(
        snapshot_download(
            dataset,
            repo_type="dataset",
            revision=revision,
            allow_patterns=["traces.jsonl", "README.md", ".gitattributes"],
        )
    )
    with (snapshot / "traces.jsonl").open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != series["dataset_sha256"]:
        raise RuntimeError("Corpus bytes differ from the series pin")
    rows = load_dataset(dataset, split="train")
    coverage = []
    for row in rows:
        trace = WekaTrace.model_validate(row)
        peak = _trace_peak_context_length(trace)
        if peak > series["max_context"]:
            raise RuntimeError(f"Pinned 256K corpus contains an unsupported session: {trace.id} ({peak})")
        coverage.append({"id": trace.id, "peak_context": peak})
    if api.dataset_info(dataset).sha != revision:
        raise RuntimeError("Public corpus moved during preparation")
    # The offline tokenizer snapshot resolver also needs README, LICENSE and .gitattributes.
    snapshot_download(series["model"], revision=series["model_revision"], ignore_patterns=["*.safetensors"])
    tokenizer = AutoTokenizer.from_pretrained(series["model"], revision=series["model_revision"])
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ping"}], tokenize=True, add_generation_prompt=True
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name("harness-requirements.txt").open("w") as requirements:
        subprocess.run(["uv", "pip", "freeze", "--python", sys.executable], stdout=requirements, check=True)
    output.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "revision": revision,
                "sha256": digest,
                "traces": len(rows),
                "eligible": coverage,
                "filtered_sessions": 0,
                "tokenizer": series["model"],
                "tokenizer_revision": series["model_revision"],
                "tokenizer_smoke_tokens": len(rendered),
                "offline_profile": True,
            },
            indent=2,
        )
    )
    print(f"Prepared all {len(rows)} sessions; no additional session filtering", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.series, args.output)


if __name__ == "__main__":
    main()
