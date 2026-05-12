# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HF-backed synthetic patch/diff PPL validation slices.

These slices are supervised continuation probes: each row provides repository
or review context in ``input`` and scores only the patch-aware continuation in
``target``. The provider is intentionally standalone so the shared all-available
perplexity wiring can opt in later.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import random
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, supervised_text_dataset
from marin.processing.tokenize import HfDatasetSpec

EPIC_5005 = 5005
SYNTHETIC_PATCH_DIFF_TRACKING_ISSUE: int | None = 5618
SYNTHETIC_PATCH_DIFF_HF_DATASET_ID = "marin-community/synth-patch-diff-ppl"
SYNTHETIC_PATCH_DIFF_SOURCE = "generated_synthetic_patch_diff_ppl_v1"
SYNTHETIC_PATCH_DIFF_HF_REVISION = "7b7e44357aef62325a69d9b3e56241d90a277e5c"
SYNTHETIC_PATCH_DIFF_SEED = 4693
EXAMPLES_PER_CONFIG = 1000
LOCAL_SAMPLE_EXAMPLES_PER_CONFIG = 2


class SyntheticPatchDiffSubset(StrEnum):
    UNIFIED_DIFF_HUNKS = "unified_diff_hunks"
    FILE_PATH_LINE_REFS = "file_path_line_refs"
    REVIEW_COMMENT_THREADS = "review_comment_threads"
    FAILING_TEST_TRACE_TO_PATCH = "failing_test_trace_to_patch"
    COMMIT_MESSAGE_METADATA = "commit_message_metadata"
    GH_PR_EVENT_PATCH = "gh_pr_event_patch"


@dataclass(frozen=True)
class SyntheticPatchDiffPplSlice:
    subset: SyntheticPatchDiffSubset
    task_name: str
    hf_config_name: str

    @property
    def registry_key(self) -> str:
        return posixpath.join("synthetic_patch_diff_ppl", self.subset.value)

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            "synthetic_patch_diff_ppl",
            f"epic:{EPIC_5005}",
            *((f"issue:{SYNTHETIC_PATCH_DIFF_TRACKING_ISSUE}",) if SYNTHETIC_PATCH_DIFF_TRACKING_ISSUE else ()),
            f"subset:{self.subset.value}",
            f"task:{self.task_name}",
            f"seed:{SYNTHETIC_PATCH_DIFF_SEED}",
            f"examples:{EXAMPLES_PER_CONFIG}",
            f"source:{SYNTHETIC_PATCH_DIFF_SOURCE}",
            f"hf_revision:{SYNTHETIC_PATCH_DIFF_HF_REVISION}",
            "loss:target_only",
        )

    def to_raw_text_dataset(self, *, hf_dataset_id: str) -> RawTextEvaluationDataset:
        return supervised_text_dataset(
            HfDatasetSpec(
                id=hf_dataset_id,
                name=self.hf_config_name,
                revision=SYNTHETIC_PATCH_DIFF_HF_REVISION,
            ),
            input_key="input",
            target_key="target",
            split="validation",
            tags=self.tags,
        )


SYNTHETIC_PATCH_DIFF_PPL_SLICES: tuple[SyntheticPatchDiffPplSlice, ...] = tuple(
    SyntheticPatchDiffPplSlice(subset=subset, task_name=subset.value, hf_config_name=subset.value)
    for subset in SyntheticPatchDiffSubset
)


def synthetic_patch_diff_raw_validation_sets(
    *, hf_dataset_id: str = SYNTHETIC_PATCH_DIFF_HF_DATASET_ID
) -> dict[str, RawTextEvaluationDataset]:
    return {
        slice_.registry_key: slice_.to_raw_text_dataset(hf_dataset_id=hf_dataset_id)
        for slice_ in SYNTHETIC_PATCH_DIFF_PPL_SLICES
    }


def _base_metadata(subset: SyntheticPatchDiffSubset, row_index: int, seed: int) -> dict[str, object]:
    return {
        "generator": SYNTHETIC_PATCH_DIFF_SOURCE,
        "version": 1,
        "subset": subset.value,
        "row_index": row_index,
        "repo": f"example/repo-{row_index % 17:02d}",
        "language": ("python", "typescript", "rust")[row_index % 3],
        "license": "synthetic",
        "eval_only": True,
        "seed": seed,
    }


def _record(
    *,
    subset: SyntheticPatchDiffSubset,
    row_index: int,
    seed: int,
    input_text: str,
    target: str,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "id": f"{subset.value}_{row_index:05d}",
        "subset": subset.value,
        "task": subset.value,
        "seed": seed,
        "input": input_text,
        "target": target if target.endswith("\n") else f"{target}\n",
        "metadata": metadata,
    }


def _python_diff(row_index: int, rng: random.Random) -> tuple[str, str, dict[str, object]]:
    before_name = f"compute_total_{row_index % 11}"
    guard_value = rng.randint(2, 9)
    file_path = f"src/payments/reconcile_{row_index % 7}.py"
    hunk_header = f"@@ -{12 + row_index % 13},7 +{12 + row_index % 13},9 @@"
    input_text = (
        "Continue the unified diff for the requested bug fix.\n"
        f"File: {file_path}\n"
        "Bug: empty batches should return 0 and large values must keep cents precision.\n"
        "Diff prefix:\n"
        f"diff --git a/{file_path} b/{file_path}\n"
        f"--- a/{file_path}\n"
        f"+++ b/{file_path}\n"
        f"{hunk_header}\n"
        f" def {before_name}(items):\n"
        "-    total = sum(item.amount for item in items)\n"
    )
    target = (
        "+    if not items:\n"
        "+        return 0\n"
        "+    total = sum(item.amount_cents for item in items)\n"
        f"+    if total > {guard_value} * 100_000:\n"
        '+        logger.info("large reconciliation batch")\n'
        "     return total\n"
    )
    metadata = {"file_path": file_path, "hunk_header": hunk_header, "bug_type": "diff_completion"}
    return input_text, target, metadata


def _line_ref(row_index: int) -> tuple[str, str, dict[str, object]]:
    file_path = f"lib/service/worker_{row_index % 9}.ts"
    line = 40 + row_index % 30
    input_text = (
        "Given a diagnostic and nearby source, write the exact file and line reference followed by the fix summary.\n"
        f"Diagnostic: retry loop exits one attempt early in {file_path}.\n"
        "Source excerpt:\n"
        f"{line - 2}: const maxAttempts = config.maxAttempts ?? 3;\n"
        f"{line - 1}: for (let attempt = 1; attempt < maxAttempts; attempt++) {{\n"
        f"{line}:   await runAttempt(attempt);\n"
        f"{line + 1}: }}\n"
        "Reference:\n"
    )
    target = f"{file_path}:{line - 1} uses `< maxAttempts`; change it to `<= maxAttempts` so the final retry runs.\n"
    metadata = {"file_path": file_path, "line": line - 1, "symbol": "maxAttempts"}
    return input_text, target, metadata


def _review_thread(row_index: int) -> tuple[str, str, dict[str, object]]:
    file_path = f"app/models/cache_{row_index % 5}.py"
    input_text = (
        "Continue the maintainer reply with the patch-ready resolution.\n"
        f"PR #{1200 + row_index}: cache invalidation cleanup\n"
        f"Review comment on {file_path}:{73 + row_index % 8}: This deletes entries while iterating over the dict.\n"
        "Author draft: Good catch. I will\n"
    )
    target = (
        "iterate over `list(cache.keys())` before deleting expired entries and add a regression test for mixed live and "
        "expired keys.\n"
    )
    metadata = {"file_path": file_path, "line": 73 + row_index % 8, "thread_state": "needs_patch"}
    return input_text, target, metadata


def _failing_trace(row_index: int) -> tuple[str, str, dict[str, object]]:
    test_name = f"tests/test_parser_{row_index % 6}.py::test_preserves_blank_lines"
    file_path = f"src/parser/render_{row_index % 4}.py"
    input_text = (
        "Write the minimal patch hunk that fixes the failing test.\n"
        f"Failing test: {test_name}\n"
        "Trace:\n"
        "E   AssertionError: assert 'a\\nb' == 'a\\n\\nb'\n"
        "Current code:\n"
        f'def render_block(lines):\n    return "\\n".join(line for line in lines if line)\n'
        f"Patch for {file_path}:\n"
    )
    target = (
        "@@ -1,2 +1,2 @@\n"
        " def render_block(lines):\n"
        '-    return "\\n".join(line for line in lines if line)\n'
        '+    return "\\n".join(lines)\n'
    )
    metadata = {"file_path": file_path, "failing_test": test_name, "failure": "blank_line_dropped"}
    return input_text, target, metadata


def _commit_metadata(row_index: int) -> tuple[str, str, dict[str, object]]:
    scope = ("evals", "tokenize", "scheduler", "docs")[row_index % 4]
    pr_number = 3000 + row_index
    input_text = (
        "Continue the commit message body from the staged patch metadata.\n"
        f"PR: #{pr_number}\n"
        f"Files: experiments/{scope}/probe.py, tests/{scope}/test_probe.py\n"
        "Stats: +84 -12\n"
        f"Subject: {scope}: add target-only patch probe\n\n"
    )
    target = (
        "Adds a supervised continuation slice for patch-like artifacts and covers the provider keys with a small "
        "registry test.\n\n"
        "Tracking issue: TBD\n"
    )
    metadata = {"pr_number": pr_number, "scope": scope, "files_changed": 2}
    return input_text, target, metadata


def _gh_pr_event_patch(row_index: int) -> tuple[str, str, dict[str, object]]:
    pr_number = 4100 + row_index
    file_path = f"pkg/api/client_{row_index % 8}.go"
    input_text = (
        "Complete the compact GitHub PR event summary with the patch field only.\n"
        "{"
        f'"event":"pull_request","action":"synchronize","number":{pr_number},'
        f'"head":"feature/retry-{row_index % 5}","base":"main",'
        f'"file":"{file_path}","patch":"'
    )
    target = (
        "@@ -22,7 +22,7 @@ func retryDelay(attempt int) time.Duration {\\n"
        "-\\treturn time.Duration(attempt) * time.Second\\n"
        "+\\treturn time.Duration(1<<attempt) * time.Second\\n"
        '"}\n'
    )
    metadata = {"pr_number": pr_number, "file_path": file_path, "event": "pull_request.synchronize"}
    return input_text, target, metadata


def synthetic_patch_diff_record(
    subset: SyntheticPatchDiffSubset | str,
    *,
    row_index: int,
    seed: int = SYNTHETIC_PATCH_DIFF_SEED,
) -> dict[str, object]:
    resolved_subset = SyntheticPatchDiffSubset(subset)
    rng = random.Random(seed + 1009 * row_index + 31 * list(SyntheticPatchDiffSubset).index(resolved_subset))
    renderers = {
        SyntheticPatchDiffSubset.UNIFIED_DIFF_HUNKS: _python_diff,
        SyntheticPatchDiffSubset.FILE_PATH_LINE_REFS: lambda index, _rng: _line_ref(index),
        SyntheticPatchDiffSubset.REVIEW_COMMENT_THREADS: lambda index, _rng: _review_thread(index),
        SyntheticPatchDiffSubset.FAILING_TEST_TRACE_TO_PATCH: lambda index, _rng: _failing_trace(index),
        SyntheticPatchDiffSubset.COMMIT_MESSAGE_METADATA: lambda index, _rng: _commit_metadata(index),
        SyntheticPatchDiffSubset.GH_PR_EVENT_PATCH: lambda index, _rng: _gh_pr_event_patch(index),
    }
    input_text, target, subset_metadata = renderers[resolved_subset](row_index, rng)
    metadata = _base_metadata(resolved_subset, row_index, seed)
    metadata.update(subset_metadata)
    return _record(
        subset=resolved_subset,
        row_index=row_index,
        seed=seed,
        input_text=input_text,
        target=target,
        metadata=metadata,
    )


def iter_synthetic_patch_diff_records(
    *,
    subsets: Iterable[SyntheticPatchDiffSubset | str] = tuple(SyntheticPatchDiffSubset),
    examples_per_config: int = EXAMPLES_PER_CONFIG,
    seed: int = SYNTHETIC_PATCH_DIFF_SEED,
) -> Iterable[dict[str, object]]:
    for subset in subsets:
        resolved_subset = SyntheticPatchDiffSubset(subset)
        for row_index in range(examples_per_config):
            yield synthetic_patch_diff_record(resolved_subset, row_index=row_index, seed=seed)


def write_local_sample(
    output_path: str | Path,
    *,
    examples_per_config: int = LOCAL_SAMPLE_EXAMPLES_PER_CONFIG,
    seed: int = SYNTHETIC_PATCH_DIFF_SEED,
) -> None:
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    for subset in SyntheticPatchDiffSubset:
        with (output_dir / f"{subset.value}.jsonl").open("w", encoding="utf-8") as sink:
            for row in iter_synthetic_patch_diff_records(
                subsets=(subset,),
                examples_per_config=examples_per_config,
                seed=seed,
            ):
                sink.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
                sink.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a local synthetic patch/diff PPL sample.")
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--examples-per-config", type=int, default=LOCAL_SAMPLE_EXAMPLES_PER_CONFIG)
    parser.add_argument("--seed", type=int, default=SYNTHETIC_PATCH_DIFF_SEED)
    args = parser.parse_args()
    write_local_sample(args.output_path, examples_per_config=args.examples_per_config, seed=args.seed)


if __name__ == "__main__":
    main()
