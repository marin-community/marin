# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Export the executable MT-MBPP data as a Hugging Face dataset folder and optionally push it (private).

Layout: ``data/prompts/<language>.jsonl`` (the frozen prompts with signatures and gold continuations),
``data/signatures.jsonl`` (every reference solution's disclosed signature and how it was chosen, including the three
examples per language), ``data/tests/<language>.jsonl`` once validated tests exist, and ``provenance/`` (the frozen
request manifest and the source revisions). Re-running replaces the folder's contents and uploads a new revision.

usage: (set -a; source ~/.zshrc.secrets; set +a; uv run --offline --no-sync python -m \
    experiments.domain_phase_mix.mt_mbpp_exec.export_hf --project DIR --output DIR [--push REPO_ID])
"""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
license: cc-by-4.0
pretty_name: MT-MBPP (executable)
language:
- en
tags:
- code
- evaluation
configs:
- config_name: prompts
  data_files: data/prompts/*.jsonl
- config_name: signatures
  data_files: data/signatures.jsonl
{tests_config}{results_config}---

# MT-MBPP (executable)

An execution-based companion to the 17 MT-MBPP components of OLMo 3's OlmoBaseEval Easy suite, which score only bits
per byte. MT-MBPP is MBPP's 500 test problems translated by o4-mini into 17 languages
(`allenai/multilingual_mbpp`); it has no tests, and its prompts describe each task in words without naming the function
a test would call. This dataset adds both.

## Prompts (`data/prompts/<language>.jsonl`)

Each prompt is the native OLMo 3 prompt (three solved examples, then the target task description and an opening code
fence) with one added line after every task description: `Function signature: `...``. Removing those lines recovers the
native prompt exactly, and `reference` is the native gold continuation (the o4-mini solution and the closing fence).

Signatures are copied verbatim from the o4-mini reference solutions: the declaration of the function MBPP's Python
asserts call, up to the start of its body, with whitespace collapsed ({rules}). Haskell uses the type signature. Bash
signatures are usage lines derived from how the reference reads its arguments (`name arg1 arg2`, `items...` for a
list passed as the remaining arguments, `< stdin` for input on standard input).

## Tests (`data/tests/<language>.jsonl`)

{tests_text}

{results_text}## Sources

- `google-research-datasets/mbpp` (full), revision `{mbpp_rev}`, CC-BY-4.0.
- `allenai/multilingual_mbpp`, revision `{mt_rev}`; its card states no license.

Request manifest sha256 `{manifest_sha}` (the prompts the evaluation froze on 2026-09-26). Built by
`experiments/domain_phase_mix/mt_mbpp_exec/` in the Marin repository, commit `{commit}`{dirty}.
"""
RESULTS = """## Results (`results/`)

Greedy completions (at most 1,024 tokens, cut at the closing fence) from four 1e21-FLOP checkpoints of the data-mixing
paper, each joined with its pass or fail on the valid tests (`results/generations/<mixture>/<language>.jsonl`).
Pass@1 per language and its 17-language mean are in `results/summary.json` and `results/components.csv`; the 95%
intervals resample MBPP problems within each language.

| Mixture | Mean pass@1 | 95% interval |
|---|---|---|
{rows}

"""
TESTS_PENDING = (
    "Pending. MBPP's asserts are being translated into each language by DeepSeek (`deepseek-flash`) and are kept only "
    "when the o4-mini reference passes them and a stub that returns a fixed wrong value fails them."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--push", help="Hugging Face dataset repo id; created private if missing")
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    (args.output / "data/prompts").mkdir(parents=True)
    (args.output / "provenance").mkdir()
    manifest_path = args.project / "requests/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for task, spec in manifest["tasks"].items():
        rows = json.loads(gzip.decompress((args.project / "requests" / spec["file"]).read_bytes()))
        with (args.output / "data/prompts" / f"{task.removeprefix('mt_mbpp_')}.jsonl").open("w") as handle:
            for r in rows:
                m = r["metadata"]
                handle.write(
                    json.dumps(
                        {
                            "language": m["language"],
                            "doc_id": r["doc_id"],
                            "task_id": m["id"],
                            "mbpp_function": m["mbpp_function"],
                            "signature": m["signatures"][-1],
                            "signature_rule": m["signature_rule"],
                            "shot_signatures": m["signatures"][:-1],
                            "prompt": r["context"],
                            "reference": r["reference"],
                        }
                    )
                    + "\n"
                )
    records = [
        json.loads(line)
        for line in gzip.decompress((args.project / "signatures.jsonl.gz").read_bytes()).decode().splitlines()
    ]
    with (args.output / "data/signatures.jsonl").open("w") as handle:
        for r in records:
            handle.write(json.dumps(r) + "\n")
    shutil.copy(manifest_path, args.output / "provenance/request_manifest.json")
    rules = Counter(r["rule"] for r in records if r["split"] == "test")
    tests_dir = args.project / "tests/validated"
    has_tests = tests_dir.exists() and any(tests_dir.glob("*.jsonl"))
    tests_text = TESTS_PENDING
    if has_tests:
        shutil.copytree(tests_dir, args.output / "data/tests")
        tests_text = (tests_dir / "README.md").read_text()
    release = args.project / "results/release"
    results_text, results_config = "", ""
    if release.exists():
        shutil.copytree(release, args.output / "results")
        summary = json.loads((release / "summary.json").read_text())
        rows = "\n".join(
            f"| {m} | {100 * v['mean']:.1f}% | {100 * v['ci95'][0]:.1f}-{100 * v['ci95'][1]:.1f} |"
            for m, v in summary["mean_pass@1"].items()
        )
        results_text = RESULTS.format(rows=rows)
        results_config = "- config_name: results\n  data_files: results/generations/*/*.jsonl\n"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "experiments/domain_phase_mix/mt_mbpp_exec"], text=True
        ).strip()
    )
    card = CARD.format(
        tests_config="- config_name: tests\n  data_files: data/tests/*.jsonl\n" if has_tests else "",
        rules=f"{rules['name']} matched by name, {rules['inferred']} identified by the closest name or as the one "
        f"function nothing else calls, {rules['override']} set by hand",
        tests_text=tests_text,
        results_config=results_config,
        results_text=results_text,
        mbpp_rev=manifest["sources"]["google-research-datasets/mbpp"],
        mt_rev=manifest["sources"]["allenai/multilingual_mbpp"],
        manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        commit=commit,
        dirty=" with uncommitted changes in that directory" if dirty else "",
    )
    (args.output / "README.md").write_text(card)
    print(f"wrote {args.output}")
    if args.push:
        api = HfApi()
        api.create_repo(args.push, repo_type="dataset", private=True, exist_ok=True)
        info = api.dataset_info(args.push)
        if not info.private:
            raise ValueError(f"{args.push} is public; refusing to upload")
        commit_info = api.upload_folder(
            repo_id=args.push,
            repo_type="dataset",
            folder_path=str(args.output),
            commit_message="Update executable MT-MBPP export",
            delete_patterns=["*"],
        )
        print(f"pushed {args.push} (private): {commit_info.commit_url}")


if __name__ == "__main__":
    main()
