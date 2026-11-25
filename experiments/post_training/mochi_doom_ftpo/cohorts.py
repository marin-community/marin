# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze Mochi FTPO cohorts and disjoint training pools on an in-region CPU worker."""

import argparse
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.strata import MembershipState, SelectionDefinition, TaskStrata
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.harness.assistant_prefix import assistant_prefix_prompt
from rolloutengine.harness.shell import DEFAULT_SHELL_TOOL, task_prompt
from skyrlrecipe.cohort_identity import cohort_hash, prompt_hash
from taskcompendium.evaluation.corpus import parquet_rows
from taskcompendium.models import Source, TaskSpec, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import ImportRejection, RawRow
from taskcompendium.runtime.shell import resolve_shell_tool
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from experiments.evaluation.micro210_policy import POLICY_REFERENCE, PREPARED_COMMIT, PREPARED_REPO, read_micro210_policy
from experiments.post_training.mochi_doom_ftpo.constants import (
    ADMITTED_SELECTION,
    CORPUS,
    DAPO_REVISION,
    DAPO_ROOT,
    EXTENDED_PANEL_ROWS,
    INITIAL_PANEL_ROWS,
    MATH16K_ROWS,
    MAX_PROMPT_TOKENS,
    MINIMUM_LOOP_OCCURRENCES,
    PILOT_DOMAIN_ROWS,
    PILOT_MATH_ROWS,
    REGIONAL_ROOT,
    SELECTION_NAME,
    SELECTION_ROOT,
    STRUCT_COMPONENTS,
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


def prompt_tokens(task: TaskSpec, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    kwargs = {"enable_thinking": True}
    last = task.context.events[-1]
    if isinstance(last, TextMessage) and last.role == "assistant":
        from skyrl_train.inference_engines.chat_template import (  # noqa: PLC0415 - optional SkyRL dependency
            assistant_prefix_token_ids,
        )

        prompt = assistant_prefix_prompt(task)
        return assistant_prefix_token_ids(
            tokenizer, list(prompt.messages), {**prompt.options, "chat_template_kwargs": kwargs}
        )
    prompt = task_prompt(task, resolve_shell_tool(task, DEFAULT_SHELL_TOOL), ())
    return tokenizer.apply_chat_template(
        list(prompt.messages), tools=prompt.options.get("tools"), add_generation_prompt=True, return_dict=False, **kwargs
    )


def dapo_tasks(filename: str) -> Iterable[TaskSpec]:
    for row in parquet_rows(StoragePath(DAPO_ROOT) / filename):
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
    if len(original_math) != MATH16K_ROWS:
        raise ValueError(f"Original math cohort has {len(original_math)} eligible prompts; expected {MATH16K_ROWS}")
    cohorts = {"math16k": original_math, "math32k": original_math[:EXTENDED_PANEL_ROWS]}
    for domain in ("if32k", "code32k", "rgym32k", "mcq32k"):
        if len(groups[domain]) < EXTENDED_PANEL_ROWS:
            raise ValueError(f"{domain} has only {len(groups[domain])} eligible prompts")
        cohorts[domain] = groups[domain][:EXTENDED_PANEL_ROWS]
    component_rows = EXTENDED_PANEL_ROWS // len(STRUCT_COMPONENTS)
    for domain in STRUCT_COMPONENTS:
        if len(groups[domain]) < component_rows:
            raise ValueError(f"{domain} has only {len(groups[domain])} eligible prompts")
    # Alternate components so initial and extended cohorts stay balanced.
    cohorts["struct32k"] = [
        row for pair in zip(*(groups[name][:component_rows] for name in STRUCT_COMPONENTS), strict=True) for row in pair
    ]
    held_hashes = {row.prompt_hash for values in cohorts.values() for row in values}
    held_ids = {row.task.id for values in cohorts.values() for row in values}
    training = [row for row in candidates if row.prompt_hash not in held_hashes and row.task.id not in held_ids]
    return cohorts, training


def _record(item: Candidate) -> dict:
    return {
        "task_spec": item.task.model_dump_json(),
        **TaskStrata(family=item.domain, source=item.task.source.dataset).model_dump(),
    }


def publish_task_files(
    root: StoragePath, name: str, version: str, files: dict[str, list[dict]], recipe: dict, sampling: str
) -> None:
    """Publish validated task files and their native evaluation identity manifest."""
    with TemporaryDirectory(prefix="mochi-doom-selection-") as temporary:
        directory = Path(temporary)
        for file_name, rows in files.items():
            pq.write_table(pa.Table.from_pylist(rows, schema=TASK_SCHEMA), directory / f"{file_name}.parquet")
        manifest = []
        for row in [*files["train"], *files["full_validation"]]:
            task = TaskSpec.model_validate_json(row["task_spec"])
            strata = TaskStrata.model_validate({key: row[key] for key in TaskStrata.model_fields})
            manifest.append(_selection_row(task, strata, state=MembershipState.ACTIVE).model_dump(mode="json"))
        pq.write_table(pa.Table.from_pylist(manifest, schema=TASK_MANIFEST_SCHEMA), directory / "task_manifest.parquet")
        summary = {
            "selection_strata": (
                SelectionDefinition(
                    name=name,
                    version=version,
                    task_files=("train.parquet", "full_validation.parquet"),
                    active_files=("train.parquet", "full_validation.parquet"),
                    sampling=sampling,
                ).model_dump(mode="json")
            ),
            **{f"{name}_rows": len(rows) for name, rows in files.items()},
            "training_rows": len(files["train"]),
            "training_family_counts": dict(Counter(row["family"] for row in files["train"])),
            "recipe": recipe,
        }
        validated = directory / "validated"
        _publish_selection(directory, str(validated), tuple(f"{name}.parquet" for name in files), summary)
        publish_once(root, {path.name: path.read_bytes() for path in validated.iterdir()})


def _publish(root: str, cohorts: dict[str, list[Candidate]], training: list[Candidate], provenance: dict) -> None:
    all_eval = {row.task.id: row for values in cohorts.values() for row in values}
    initial_panel = [
        row for domain, values in cohorts.items() if domain != "math16k" for row in values[:INITIAL_PANEL_ROWS]
    ]
    reserve = [
        row
        for domain, values in cohorts.items()
        if domain != "math16k"
        for row in values[INITIAL_PANEL_ROWS:EXTENDED_PANEL_ROWS]
    ]
    pilot = cohorts["math16k"][:PILOT_MATH_ROWS] + [
        row
        for domain, values in cohorts.items()
        if domain not in ("math16k", "math32k")
        for row in values[:PILOT_DOMAIN_ROWS]
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
    publish_task_files(
        StoragePath(root),
        SELECTION_NAME,
        VERSION,
        {name: [_record(row) for row in values] for name, values in files.items()},
        provenance,
        "fixed prompt-token hashes; full reserves excluded before training",
    )


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
    selection_recipe = json.loads(args.recipe.read_text())
    if selection_recipe["version"] != VERSION:
        raise ValueError("Cohort recipe and publication versions must agree")
    math_sources = {source for source, domain in selection_recipe["sources"].items() if domain == "math"}
    if math_sources != {"MarinSkyRL:dapo_math"}:
        raise ValueError("This campaign requires the recovered, pinned DAPO math split")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    source_domains = {key: value for key, value in selection_recipe["sources"].items() if value != "math"}
    corpus = json.loads(StoragePath(args.corpus).read_text())
    admitted = json.loads(StoragePath(ADMITTED_SELECTION).read_text())
    declared = {row["source_id"] for row in corpus["sources"]}
    if not source_domains.keys() <= declared:
        raise ValueError("Requested sources are missing from the completed corpus")
    excluded_ids = set()
    excluded_hashes = set()
    exclusion_sources = [entry["uri"] for entry in selection_recipe["exclusions"]]
    protocol = read_micro210_policy(POLICY_REFERENCE)
    exclusion_sources.append(
        f"hf://datasets/{PREPARED_REPO}@{PREPARED_COMMIT}/" + protocol["evaluation"]["data_relative_path"]
    )
    for uri in exclusion_sources:
        for row in parquet_rows(StoragePath(uri), columns=("task_spec",)):
            task = TaskSpec.model_validate_json(row["task_spec"])
            excluded_ids.add(task.id)
            excluded_hashes.add(prompt_hash(prompt_tokens(task, tokenizer)))
    candidates = []
    counts = Counter()
    seen = set()

    def add(task: TaskSpec, domain: str, original_validation: bool = False, source_priority: int = 0) -> None:
        if _grader_kind(task) not in LOCAL_GRADERS:
            counts[f"{domain}/nonprogrammatic"] += 1
            return
        tokens = prompt_tokens(task, tokenizer)
        if len(tokens) > MAX_PROMPT_TOKENS:
            counts[f"{domain}/over2048"] += 1
            return
        digest = prompt_hash(tokens)
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
        "prompt_rendering": {
            "normal": "rolloutengine.harness.shell.task_prompt",
            "assistant_prefix": "rolloutengine.harness.assistant_prefix.assistant_prefix_prompt",
            "chat_template_kwargs": {"enable_thinking": True},
            "shell_tool": DEFAULT_SHELL_TOOL.model_dump(mode="json"),
        },
        "supersedes": {
            "manifest": str(StoragePath(REGIONAL_ROOT) / "cohorts" / "2026.11.24.1" / "manifest.json"),
            "reason": "source-only prompt hashes omitted native answer-format instructions",
        },
        "exclusions": [{"uri": uri} for uri in exclusion_sources],
        "heldout": [],
        "drops": dict(counts),
    }
    _publish(str(StoragePath(SELECTION_ROOT) / SELECTION_NAME / VERSION), cohorts, training, provenance)
    _publish(args.output, cohorts, training, provenance)
    _publish(
        str(StoragePath(SELECTION_ROOT) / f"{SELECTION_NAME}-math" / VERSION),
        cohorts,
        [row for row in training if row.domain == "math"],
        provenance,
    )
    manifest = {**provenance, "cohorts": {}}
    for domain, values in cohorts.items():
        budget = 16384 if domain == "math16k" else 32768
        active = values if domain == "math16k" else values[:INITIAL_PANEL_ROWS]
        rows = [{"task_id": row.task.id, "prompt_hash": row.prompt_hash} for row in active]
        reserve = [{"task_id": row.task.id, "prompt_hash": row.prompt_hash} for row in values[len(active) :]]
        manifest["cohorts"][domain] = {
            "rows": rows,
            "reserve": reserve,
            "budget": budget,
            "max_period_tokens": budget // MINIMUM_LOOP_OCCURRENCES,
            "cohort_hash": cohort_hash(row["prompt_hash"] for row in rows),
        }
    (StoragePath(args.output) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Published cohorts: %s; train=%d; drops=%s", {k: len(v) for k, v in cohorts.items()}, len(training), counts
    )


if __name__ == "__main__":
    main()
