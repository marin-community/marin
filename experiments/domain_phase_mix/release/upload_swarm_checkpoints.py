# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload the swarms' final checkpoints to public Hugging Face model repositories, one per swarm.

`uv run python -m experiments.domain_phase_mix.release.upload_swarm_checkpoints --swarm SWARM|all [--manifest URL]
[--workers N] [--batch N] [--limit N] [--dry-run | --confirm | --verify]`

Each run's final HF export on GCS (the manifest's `hf_export_uri`) becomes the folder named by its public
`run_label`, beside a model card and `runs.csv`, which maps each folder to its `observation_id` (the dataset's key and
the W&B run id), `design_kind` and `fit_inclusion`. Runs stream through temporary folders in batches: download a
batch, upload it as one commit, delete it, so the machine holds at most `--workers` times `--batch` runs. Run it as
an Iris job in us-east5, next to the exports, not on a laptop. A run whose folder already holds a `model.safetensors`
is skipped, so an interrupted upload resumes. `--verify` checks every run's weights on the Hub against the export's
size. Progress, with an ETA, goes to stdout.
"""

import argparse
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi
from rigging.filesystem.buckets import filesystem_for

from experiments.domain_phase_mix.release import paths

MANIFEST = paths.REPO / ".agents/projects/mariner_release/swarm_checkpoint_manifest.csv"
OWNER = "Calvin-Xu"
WEIGHTS = "model.safetensors"
LICENSE = "apache-2.0"  # the license of the MARINER dataset and of Marin's models
SWARMS = {
    "qwen3_360m_1p6b": {
        "title": "Qwen3 360M/1.6B",
        "role": "the main swarm",
        "model": "358.3M-parameter Qwen3 model trained on 1.58B tokens",
        "designs": "280 mixtures (238 sampled designs, 3 baselines, and 39 single-bucket deletions)",
        "fit": "The paper's fits use the 280 mixtures.",
    },
    "llama_160m_1p2b": {
        "title": "Llama 160M/1.2B",
        "role": "an archived swarm",
        "model": "157.5M-parameter Llama model trained on 1.2B tokens",
        "designs": "281 mixtures (238 sampled designs, 4 baselines, and 39 single-bucket deletions)",
        "fit": "The paper's fits use the 242 sampled designs and baselines; the deletions are held-out runs.",
    },
    "llama_200m_6b": {
        "title": "Llama 200M/6B",
        "role": "an archived swarm",
        "model": "201.1M-parameter Llama model trained on 6.0B tokens",
        "designs": "280 mixtures (238 sampled designs, 3 baselines, and 39 single-bucket deletions)",
        "fit": "The paper's fits use the 280 mixtures.",
    },
}
RUN_COLUMNS = ["run_label", "observation_id", "design_kind", "fit_inclusion"]


def repo_id(swarm: str) -> str:
    return f"{OWNER}/mariner-swarm-{swarm.replace('_', '-')}"


def model_card(swarm: str, runs: pd.DataFrame) -> str:
    s = SWARMS[swarm]
    example = runs.run_label[runs.design_kind == "sampled"].iloc[0]
    return f"""---
license: {LICENSE}
library_name: transformers
tags:
- data-mixing
- pretraining
---

# MARINER swarm checkpoints: {s["title"]}

The final checkpoints of the {len(runs)} proxy runs of {s["role"]} in *MARINER: Measuring and Modeling Repetition
Harm for Multi-Epoch Data Mixing*: {s["designs"]} over a 39-bucket partition of the Dolma 3 and Dolmino pools, and
ten repeats of the proportional mixture with different seeds. Each run is a {s["model"]}, saved in fp32.
{s["fit"]}

Each run is the folder named by its `run_label`. `runs.csv` maps each folder to its `observation_id` in the MARINER
dataset, which is also its W&B run id, its kind of design, and whether the paper's fits use it. The dataset holds every
run's mixture weights, epochs, and evaluation results.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{repo_id(swarm)}"
model = AutoModelForCausalLM.from_pretrained(repo, subfolder="{example}")
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder="{example}")
```
"""


def _format_duration(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m"


def _remote_weights(api: HfApi, repo: str) -> dict[str, int]:
    """Each run folder's uploaded weights size, by run label."""
    if not api.repo_exists(repo):
        return {}
    return {
        entry.path.split("/")[0]: entry.size
        for entry in api.list_repo_tree(repo, recursive=True)
        if entry.path.endswith(f"/{WEIGHTS}")
    }


def _stage_run(uri: str, label: str, staging: Path) -> int:
    """Copy one run's export into `staging/label`; return its bytes."""
    fs, root = filesystem_for(uri)
    target = staging / label
    target.mkdir()
    for f in fs.ls(root, detail=False):
        if not f.endswith("/"):
            fs.get(f, str(target / Path(f).name))
    return sum(p.stat().st_size for p in target.iterdir())


def _upload_batch(api: HfApi, repo: str, batch: pd.DataFrame, staging: Path) -> int:
    """Stage a batch of runs in one temporary folder under `staging` and upload it as one commit; return its bytes."""
    with tempfile.TemporaryDirectory(prefix="mariner_upload_", dir=staging) as tmp:
        size = sum(_stage_run(r.hf_export_uri, r.run_label, Path(tmp)) for r in batch.itertuples())
        labels = list(batch.run_label)
        api.upload_folder(
            repo_id=repo, folder_path=tmp, commit_message=f"Add {len(labels)} runs, {labels[0]} to {labels[-1]}"
        )
    return size


def upload(swarm: str, runs: pd.DataFrame, workers: int, batch: int, limit: int | None, staging: Path) -> None:
    api, repo = HfApi(), repo_id(swarm)
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=model_card(swarm, runs).encode(), path_in_repo="README.md", repo_id=repo)
    api.upload_file(
        path_or_fileobj=runs[RUN_COLUMNS].to_csv(index=False).encode(), path_in_repo="runs.csv", repo_id=repo
    )
    done = _remote_weights(api, repo)
    todo = runs[~runs.run_label.isin(done)]
    if limit is not None:
        todo = todo.head(limit)
    batches = [todo.iloc[i : i + batch] for i in range(0, len(todo), batch)]
    print(
        f"{repo}: {len(done)} of {len(runs)} runs present, uploading {len(todo)} in {len(batches)} commits"
        f" with {workers} workers",
        flush=True,
    )
    start, sent, count = time.monotonic(), 0, 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_upload_batch, api, repo, b, staging): len(b) for b in batches}
        for future in as_completed(futures):
            sent += future.result()
            count += futures[future]
            elapsed = time.monotonic() - start
            eta = elapsed / count * (len(todo) - count)
            print(
                f"[{count}/{len(todo)}] {repo}: {sent / 1e9:.1f} GB in {_format_duration(elapsed)},"
                f" {sent / 1e6 / elapsed:.0f} MB/s, ETA {_format_duration(eta)}",
                flush=True,
            )


def verify(swarm: str, runs: pd.DataFrame) -> list[str]:
    """Runs whose weights on the Hub are missing or differ in size from the export's."""
    api, repo = HfApi(), repo_id(swarm)
    remote = _remote_weights(api, repo)
    problems = []
    for r in runs.itertuples():
        fs, root = filesystem_for(r.hf_export_uri)
        expected = fs.size(f"{root}/{WEIGHTS}")
        if remote.get(r.run_label) != expected:
            problems.append(f"{r.run_label}: Hub {remote.get(r.run_label)} bytes, export {expected}")
    return problems


def read_manifest(location: str) -> pd.DataFrame:
    """The manifest from a local path or a gs:// URL; Iris jobs read the GCS copy, since the bundle omits .agents."""
    fs, path = filesystem_for(location)
    with fs.open(path) as f:
        return pd.read_csv(f)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--swarm", choices=[*sorted(SWARMS), "all"], required=True)
    parser.add_argument("--manifest", default=str(MANIFEST), help="a local path or a gs:// URL")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=5, help="runs per commit")
    parser.add_argument("--limit", type=int, help="upload at most this many missing runs per swarm")
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("."),
        help="where batches are staged; Iris workers mount /tmp as a RAM-backed tmpfs, so stage on the task's disk",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="the default: list what an upload would send")
    mode.add_argument("--confirm", action="store_true", help="create the public repositories and upload")
    mode.add_argument("--verify", action="store_true", help="check every run's weights on the Hub")
    args = parser.parse_args(argv)
    manifest = read_manifest(args.manifest)
    problems = []
    for swarm in sorted(SWARMS) if args.swarm == "all" else [args.swarm]:
        runs = manifest[manifest.swarm == swarm].reset_index(drop=True)
        if not runs.run_label.is_unique:
            raise ValueError(f"{swarm}: run labels are not unique")
        if args.verify:
            found = verify(swarm, runs)
            print(f"{repo_id(swarm)}: {len(runs) - len(found)} of {len(runs)} runs verified", flush=True)
            problems += found
        elif args.confirm:
            args.staging_dir.mkdir(parents=True, exist_ok=True)
            upload(swarm, runs, args.workers, args.batch, args.limit, args.staging_dir)
        else:
            print(model_card(swarm, runs))
            print(f"dry run: {len(runs)} runs for {repo_id(swarm)} (public); --confirm uploads")
    if problems:
        raise SystemExit("\n".join(problems[:50]))


if __name__ == "__main__":
    main()
