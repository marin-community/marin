# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze Mochi FTPO cohorts and disjoint training pools on an in-region CPU worker."""

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.strata import MembershipState, SelectionDefinition, TaskStrata
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import Source, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, RawRow
from taskcompendium.submission import conversation_messages
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from experiments.evaluation.micro210_policy import POLICY_REFERENCE, PREPARED_COMMIT, PREPARED_REPO, read_micro210_policy
from experiments.post_training.mochi_doom_ftpo.constants import (
    ADMITTED_SELECTION,
    CORPUS,
    DAPO_REVISION,
    DAPO_ROOT,
    REGIONAL_ROOT,
    SELECTION_NAME,
    SELECTION_ROOT,
    TOKENIZER,
    TOKENIZER_REVISION,
    VERSION,
)
from experiments.post_training.task_curation import publish_once
from experiments.post_training.task_curation.datasets.skyrl.math import convert_dapo_math
from experiments.post_training.task_curation.selection import (
    LOCAL_GRADERS,
    TASK_MANIFEST_SCHEMA,
    TASK_SCHEMA,
    SelectionInput,
    _admitted_records,
    _grader_kind,
    _publish_selection,
    _selection_row,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    task: TaskSpec
    domain: str
    prompt_hash: str
    original_validation: bool = False
    source_priority: int = 0


def token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


def prompt_tokens(task: TaskSpec, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    messages = list(conversation_messages(task.context.events))
    kwargs = {"enable_thinking": True}
    last = task.context.events[-1]
    if isinstance(last, TextMessage) and last.role == "assistant":
        from skyrl_train.inference_engines.chat_template import (  # noqa: PLC0415 - optional SkyRL dependency
            assistant_prefix_token_ids,
        )

        return assistant_prefix_token_ids(
            tokenizer,
            messages,
            {
                "extra_body": {"continue_final_message": True, "add_generation_prompt": False},
                "chat_template_kwargs": kwargs,
            },
        )
    tools = [{"type": "function", "function": tool.model_dump(exclude_none=True)} for tool in task.final_tools]
    return tokenizer.apply_chat_template(
        messages, tools=tools or None, add_generation_prompt=True, return_dict=False, **kwargs
    )


def read_parquet(url: str, columns: list[str] | None = None) -> Iterable[dict[str, Any]]:
    fs, path = filesystem_for(url)
    with fs.open(path, "rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(batch_size=128, columns=columns):
            yield from batch.to_pylist()


def dapo_tasks(filename: str) -> Iterable[TaskSpec]:
    for row in read_parquet(f"{DAPO_ROOT}/{filename}"):
        uid = row["uid"]
        source = Source(
            dataset="BytedTsinghua-SIA/DAPO-Math-17k",
            revision=DAPO_REVISION,
            row=uid,
            importer_revision="mochi-doom-prepared-v1",
        )
        task = convert_dapo_math(RawRow(f"doom-dapo-{uid}", source, row), ConversionContext({}, None))
        if isinstance(task, ImportRejection):
            raise ValueError(f"Prepared DAPO row {uid} failed conversion: {task}")
        yield task


def freeze(candidates: list[Candidate]) -> tuple[dict[str, list[Candidate]], list[Candidate]]:
    """Hold out both initial cohorts and reserves by prompt hash and task ID."""
    groups = defaultdict(list)
    for item in candidates:
        groups[item.domain].append(item)
    for values in groups.values():
        values.sort(key=lambda row: (row.source_priority, row.prompt_hash))
    original_math = [row for row in groups["math"] if row.original_validation]
    if len(original_math) != 512:
        raise ValueError(f"Original math cohort has {len(original_math)} eligible prompts; expected 512")
    cohorts = {"math16k": original_math, "math32k": original_math[:256]}
    for domain in ("if32k", "code32k", "rgym32k", "mcq32k"):
        if len(groups[domain]) < 256:
            raise ValueError(f"{domain} has only {len(groups[domain])} eligible prompts")
        cohorts[domain] = groups[domain][:256]
    for domain in ("calendar", "closure"):
        if len(groups[domain]) < 128:
            raise ValueError(f"{domain} has only {len(groups[domain])} eligible prompts")
    # Alternate the two components so prefixes preserve the 64:64 and 128:128 balance.
    cohorts["struct32k"] = [
        row for pair in zip(groups["calendar"][:128], groups["closure"][:128], strict=True) for row in pair
    ]
    held_hashes = {row.prompt_hash for values in cohorts.values() for row in values}
    held_ids = {row.task.id for values in cohorts.values() for row in values}
    training = [row for row in candidates if row.prompt_hash not in held_hashes and row.task.id not in held_ids]
    return cohorts, training


def target_status(confirmed: int, count: int) -> str:
    """Extend a sparse 128-row domain before demoting it from the target set."""
    if confirmed >= 10:
        return "target"
    return "extend" if count < 256 else "non-target"


def _record(item: Candidate) -> dict:
    return {
        "task_spec": item.task.model_dump_json(),
        **TaskStrata(family=item.domain, source=item.task.source.dataset).model_dump(),
    }


def _publish(root: str, cohorts: dict[str, list[Candidate]], training: list[Candidate], provenance: dict) -> None:
    all_eval = {row.task.id: row for values in cohorts.values() for row in values}
    initial_panel = [row for domain, values in cohorts.items() if domain != "math16k" for row in values[:128]]
    reserve = [row for domain, values in cohorts.items() if domain != "math16k" for row in values[128:256]]
    pilot = cohorts["math16k"][:64] + [
        row for domain, values in cohorts.items() if domain not in ("math16k", "math32k") for row in values[:16]
    ]
    files = {
        "train": training,
        "validation": pilot,
        "full_validation": list(all_eval.values()),
        "math16k": cohorts["math16k"],
        "panel32k": initial_panel,
        "reserve32k": reserve,
        "math_train": [row for row in training if row.domain == "math"],
    }
    with TemporaryDirectory(prefix="mochi-doom-cohorts-") as temporary:
        directory = Path(temporary)
        for name, values in files.items():
            pq.write_table(
                pa.Table.from_pylist([_record(row) for row in values], schema=TASK_SCHEMA), directory / f"{name}.parquet"
            )
        population = [*training, *all_eval.values()]
        rows = [
            _selection_row(
                row.task, TaskStrata(family=row.domain, source=row.task.source.dataset), state=MembershipState.ACTIVE
            ).model_dump(mode="json")
            for row in population
        ]
        pq.write_table(pa.Table.from_pylist(rows, schema=TASK_MANIFEST_SCHEMA), directory / "task_manifest.parquet")
        summary = {
            "selection_strata": (
                SelectionDefinition(
                    name=SELECTION_NAME,
                    version=VERSION,
                    task_files=("train.parquet", "full_validation.parquet"),
                    active_files=("train.parquet", "full_validation.parquet"),
                    sampling="fixed prompt-token hashes; full reserves excluded before training",
                ).model_dump(mode="json")
            ),
            **{f"{name}_rows": len(values) for name, values in files.items()},
            "training_rows": len(training),
            "recipe": provenance,
            "training_family_counts": dict(Counter(row.domain for row in training)),
        }
        validated = directory / "validated"
        _publish_selection(directory, str(validated), tuple(f"{name}.parquet" for name in files), summary)
        publish_once(StoragePath(root), {path.name: path.read_bytes() for path in validated.iterdir()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=CORPUS)
    parser.add_argument(
        "--recipe", type=Path, default=Path(__file__).parents[1] / "task_curation/recipes/mochi-doom-ftpo.json"
    )
    parser.add_argument("--output", default=f"{REGIONAL_ROOT}/cohorts/{VERSION}")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    configure_coreweave_s3()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    selection_recipe = json.loads(args.recipe.read_text())
    source_domains = {key: value for key, value in selection_recipe["sources"].items() if value != "math"}
    corpus = json.loads(StoragePath(args.corpus).read_text())
    admitted = json.loads(StoragePath(ADMITTED_SELECTION).read_text())
    declared = {row["source_id"] for row in corpus["sources"]}
    if not source_domains.keys() <= declared:
        raise ValueError("Requested sources are missing from the completed corpus")
    excluded_ids = set()
    excluded_hashes = set()
    exclusion_sources = [f"{SELECTION_ROOT}/eval-lite/2026.11.17.1/full_validation.parquet"]
    protocol = read_micro210_policy(POLICY_REFERENCE)
    exclusion_sources.append(
        f"hf://datasets/{PREPARED_REPO}@{PREPARED_COMMIT}/" + protocol["evaluation"]["data_relative_path"]
    )
    for uri in exclusion_sources:
        for row in read_parquet(uri, ["task_spec"]):
            task = TaskSpec.model_validate_json(row["task_spec"])
            excluded_ids.add(task.id)
            excluded_hashes.add(token_hash(prompt_tokens(task, tokenizer)))
    candidates = []
    counts = Counter()
    seen = set()

    def add(task: TaskSpec, domain: str, original_validation: bool = False, source_priority: int = 0) -> None:
        if _grader_kind(task) not in LOCAL_GRADERS:
            counts[f"{domain}/nonprogrammatic"] += 1
            return
        tokens = prompt_tokens(task, tokenizer)
        if len(tokens) > 2048:
            counts[f"{domain}/over2048"] += 1
            return
        digest = token_hash(tokens)
        if task.id in excluded_ids or digest in excluded_hashes:
            counts[f"{domain}/excluded"] += 1
            return
        if digest in seen:
            counts[f"{domain}/duplicate"] += 1
            return
        seen.add(digest)
        candidates.append(Candidate(task, domain, digest, original_validation, source_priority))
        counts[f"{domain}/eligible"] += 1

    # Process the original validation first so training duplicates cannot displace it.
    for filename in ("validation.parquet", "train.parquet"):
        for task in dapo_tasks(filename):
            add(task, "math", filename == "validation.parquet")
    for source in admitted["sources"]:
        domain = source_domains.get(source["source_id"])
        if domain is None:
            continue
        logger.info("Reading admitted source %s", source["source_id"])
        for task, _ in _admitted_records(SelectionInput(source["source_id"], domain, source["root"])):
            priority = int(source["source_id"] == "Task Trove:laion__nemotron-gym-instruction-following-calendar-v3")
            add(task, domain, source_priority=priority)
    logger.info("Eligibility before cohort freezing: %s", dict(counts))
    cohorts, training = freeze(candidates)
    provenance = {
        "corpus": args.corpus,
        "selection_recipe": selection_recipe,
        "admitted_selection": ADMITTED_SELECTION,
        "sources": [source for source in admitted["sources"] if source["source_id"] in source_domains],
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "exclusions": [{"uri": uri} for uri in exclusion_sources],
        "heldout": [],
        "drops": dict(counts),
    }
    _publish(f"{SELECTION_ROOT}/{SELECTION_NAME}/{VERSION}", cohorts, training, provenance)
    _publish(args.output, cohorts, training, provenance)
    _publish(
        f"{SELECTION_ROOT}/{SELECTION_NAME}-math/{VERSION}",
        cohorts,
        [row for row in training if row.domain == "math"],
        provenance,
    )
    manifest = {**provenance, "cohorts": {}}
    for domain, values in cohorts.items():
        budget = 16384 if domain == "math16k" else 32768
        active = values if domain == "math16k" else values[:128]
        rows = [{"task_id": row.task.id, "prompt_hash": row.prompt_hash} for row in active]
        reserve = [{"task_id": row.task.id, "prompt_hash": row.prompt_hash} for row in values[len(active) :]]
        manifest["cohorts"][domain] = {
            "rows": rows,
            "reserve": reserve,
            "budget": budget,
            "max_period_tokens": budget // 8,
            "cohort_hash": hashlib.sha256("".join(sorted(row["prompt_hash"] for row in rows)).encode()).hexdigest(),
        }
    StoragePath(f"{args.output}/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Published cohorts: %s; train=%d; drops=%s", {k: len(v) for k, v in cohorts.items()}, len(training), counts
    )


if __name__ == "__main__":
    main()
