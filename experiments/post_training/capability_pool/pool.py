# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build fixed RL training pools that share one capability distribution and differ only in their source mix.

Inputs are runnable ``TaskRolloutSpec`` parquet files from two sources. TaskTrove tasks take their capability from
an RL Data Atlas release through the labels written by ``labels.py``, joined on the Atlas content hash. Taskforge
tasks take it from their ``capability:<id>`` tag. A target assigns weights to a set of hierarchy nodes, none an
ancestor of another; a task counts toward the target node at or above its label.

Rows are written at the current ``TaskSpec`` schema. Rows at a version in ``UPGRADABLE_SCHEMAS`` are relabelled,
and a recipe's ``rewrites`` replace rollout limits or raise grader timeouts per source; ``selection.json`` counts
every changed field per arm.

Each arm is written as ``<output>/<arm>/train.parquet`` with ``artifact-manifest.json`` and ``selection.json``.
The trainer reads the parquet through ``ArtifactDataSource`` and samples it in seeded uniform passes, so the pool
composition is the training mixture. Arms are nested: per node, every arm takes a prefix of one seeded TaskTrove
order and one seeded Taskforge order, so arms differ only in the tasks the source mix swaps.
"""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import partial

import polars as pl
from fray.types import ResourceConfig
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import SCHEMA_VERSION, TaskSpec
from taskcompendium.pipeline.curriculum.models import AlignmentRelease, CurriculumNode
from taskcompendium.pipeline.sampling import seeded_order
from taskcompendium.pipeline.semantics import task_content_hash
from zephyr.context import IntraShardExecution, ZephyrContext
from zephyr.dataset import Dataset
from zephyr.input_file import DEFAULT_FILE_PATH_COLUMN

from experiments.post_training.capability_pool.allocation import Allocation, InfeasiblePolicy, Supply, allocate
from experiments.post_training.capability_pool.labels import LABELS_METADATA

CAPABILITY_TAG = "capability:"
UPGRADABLE_SCHEMAS = frozenset({"0.27"})
"""TaskSpec versions whose rows are valid at the current version once relabelled; 0.28 only added optional fields."""
ARTIFACT_MANIFEST = "artifact-manifest.json"
SELECTION = "selection.json"
TRAIN = "train.parquet"
CANDIDATES = "candidates.parquet"
CENSUS = "census.json"
CANDIDATE_COLUMNS = [
    "path",
    "pool_source",
    "task_id",
    "source_id",
    "content_hash",
    "subject_id",
    "subject_similarity",
    "capability_id",
    "capability_similarity",
    "label_node",
    "target_node",
    "excluded_by",
]
OUTPUT_COLUMNS = [
    "task_rollout_spec",
    "task_id",
    "source_id",
    "pool_source",
    "target_node",
    "label_node",
    "content_hash",
]


class PoolSource(StrEnum):
    TASKTROVE = "tasktrove"
    TASKFORGE = "taskforge"
    EXCLUDED = "excluded"
    """Tasks that must not enter any arm, matched by task id or content hash (for example a held-out split)."""


class RowRewrite(BaseModel):
    """Runtime changes applied to every chosen row of one source."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    limits: dict[str, float | None] = Field(default_factory=dict, description="RolloutLimits fields to replace")
    min_grader_timeout: float | None = Field(
        default=None, gt=0, description="Raise shorter grader timeouts to this, extending verifier_timeout by as much"
    )


class PoolRecipe(BaseModel):
    """Every choice that determines a pool; recorded verbatim in ``selection.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    atlas_release: str = Field(description="Atlas publication release.json: hierarchy and release identity")
    labels: str = Field(description="Root written by labels.py from the same release")
    subject_distance: float = Field(gt=0, le=2)
    capability_distance: float = Field(gt=0, le=2)
    target: dict[str, float] = Field(description="Hierarchy node id to positive weight")
    pool_size: int = Field(gt=0)
    seed: int
    arms: dict[str, float] = Field(description="Arm name to the fraction of its tasks drawn from Taskforge")
    infeasible: InfeasiblePolicy
    tasktrove: list[str] = Field(min_length=1, description="Parquet files of TaskTrove TaskRolloutSpec rows")
    taskforge: list[str] = Field(description="Parquet files of Taskforge TaskRolloutSpec rows")
    exclude: list[str] = Field(description="Parquet files of TaskRolloutSpec rows that no arm may contain")
    rewrites: dict[PoolSource, RowRewrite] = Field(description="Per-source runtime changes; empty for none")

    @model_validator(mode="after")
    def _check(self) -> "PoolRecipe":
        if not self.arms:
            raise ValueError("At least one arm is required")
        for arm in self.arms:
            if not arm or "/" in arm:
                raise ValueError(f"Arm names must be non-empty path segments: {arm!r}")
        roles = [*self.tasktrove, *self.taskforge, *self.exclude]
        if len(roles) != len(set(roles)):
            raise ValueError("Each input file may appear in only one role")
        return self


@dataclass(frozen=True)
class Hierarchy:
    nodes: dict[str, CurriculumNode]

    @classmethod
    def from_release(cls, release: AlignmentRelease) -> "Hierarchy":
        return cls({node.node_id: node for node in release.nodes})

    def lineage(self, node_id: str) -> list[str]:
        """The node and its ancestors, nearest first."""
        path = [node_id]
        while (parent := self.nodes[path[-1]].parent_id) is not None:
            path.append(parent)
        return path

    def target_map(self, target: Mapping[str, float]) -> dict[str, str]:
        """Map every node at or below a target node to that target node."""
        unknown = sorted(set(target) - set(self.nodes))
        if unknown:
            raise ValueError(f"Unknown target nodes: {unknown}")
        mapping = {}
        for node_id in self.nodes:
            covering = [ancestor for ancestor in self.lineage(node_id) if ancestor in target]
            if len(covering) > 1:
                raise ValueError(f"Target nodes overlap: {covering} cover {node_id}")
            if covering:
                mapping[node_id] = covering[0]
        return mapping


def subject_node(subject_id: str) -> str:
    return "subject:" + subject_id


def label_node(
    subject_id: str | None,
    subject_similarity: float | None,
    capability_id: str | None,
    capability_similarity: float | None,
    *,
    subject_distance: float,
    capability_distance: float,
) -> str | None:
    """The deepest node an Atlas label places a task at, with the Atlas coverage rule."""
    if subject_id is None or subject_similarity is None or subject_similarity < 1 - subject_distance:
        return None
    if capability_similarity is not None and capability_similarity >= 1 - capability_distance:
        return capability_id
    return subject_node(subject_id)


def current_schema(task: dict) -> dict:
    if task.get("schema_version") in UPGRADABLE_SCHEMAS:
        return task | {"schema_version": SCHEMA_VERSION}
    return task


def capability_tag(task: TaskSpec) -> str | None:
    tags = [tag.removeprefix(CAPABILITY_TAG) for tag in task.tags if tag.startswith(CAPABILITY_TAG)]
    if len(tags) > 1:
        raise ValueError(f"Task {task.id} has several capability tags: {tags}")
    return tags[0] if tags else None


def candidate_record(row: dict, *, roles: Mapping[str, PoolSource]) -> dict:
    """The identity of one input row: its file, task id, source, content hash and Taskforge capability tag.

    The content hash is taken from the task as serialized, which is how the Atlas hashed it. A row whose task does
    not validate at the current schema keeps its identity with a null content hash, so pool building can count and
    exclude it.
    """
    path = row[DEFAULT_FILE_PATH_COLUMN]
    raw = json.loads(row["task_rollout_spec"])["task"]
    if row.get("task_id") is not None and row["task_id"] != raw["id"]:
        raise ValueError(f"{path}: task_id column {row['task_id']!r} differs from the spec's {raw['id']!r}")
    role = roles[path]
    record = {"path": path, "pool_source": role.value, "task_id": raw["id"], "content_hash": None, "tag": None}
    try:
        task = TaskSpec.model_validate(current_schema(raw))
    except ValidationError:
        return record | {"source_id": row.get("source_id") or f"{role.value}:{raw['source']['dataset']}"}
    return record | {
        "source_id": row.get("source_id") or f"{role.value}:{task.source.dataset}",
        "content_hash": task_content_hash(raw),
        "tag": capability_tag(task) if role is PoolSource.TASKFORGE else None,
    }


def index_candidates(context: ZephyrContext, roles: Mapping[str, PoolSource]) -> pl.DataFrame:
    records = context.execute(
        Dataset.from_list(sorted(roles))
        .load_parquet(include_file_paths=True)
        .map(partial(candidate_record, roles=roles)),
        name="capability-pool-index",
    ).results
    schema = {name: pl.String for name in ("path", "pool_source", "task_id", "source_id", "content_hash", "tag")}
    frame = pl.DataFrame(records, schema=schema)
    duplicated = frame.filter(pl.struct("path", "task_id").is_duplicated())
    if duplicated.height:
        raise ValueError(f"Task ids repeat within an input file: {duplicated.head(5).to_dicts()}")
    return frame


def _read_parquet(path: str, columns: Sequence[str] | None = None) -> pl.DataFrame:
    with StoragePath(path).open("rb") as stream:
        return pl.read_parquet(stream, columns=list(columns) if columns is not None else None)


def read_labels(labels_root: str, release_id: str, content_hashes: pl.Series) -> pl.DataFrame:
    """One label per content hash among ``content_hashes``, with ``eval`` marking any non-train source row."""
    metadata = json.loads((StoragePath(labels_root) / LABELS_METADATA).read_bytes())
    if metadata["release_id"] != release_id:
        raise ValueError(f"Labels come from release {metadata['release_id']}, not {release_id}")
    columns = [
        "content_hash",
        "intended_use",
        "subject_id",
        "subject_similarity",
        "capability_id",
        "capability_similarity",
    ]
    wanted = content_hashes.unique().implode()
    parts = [_read_parquet(path, columns).filter(pl.col("content_hash").is_in(wanted)) for path in metadata["paths"]]
    rows = pl.concat(parts)
    label_columns = columns[2:]
    conflicting = (
        rows.group_by("content_hash").agg(pl.struct(label_columns).n_unique().alias("n")).filter(pl.col("n") > 1)
    )
    if conflicting.height:
        raise ValueError(f"Content hashes carry conflicting labels: {conflicting['content_hash'].head(5).to_list()}")
    return rows.group_by("content_hash").agg(
        *(pl.col(name).first() for name in label_columns),
        (pl.col("intended_use") != "train").any().alias("eval"),
    )


@dataclass(frozen=True)
class Candidates:
    """Every input row with its label, target node and ``excluded_by`` reason; null for eligible rows."""

    rows: pl.DataFrame

    @property
    def eligible(self) -> pl.DataFrame:
        return self.rows.filter(pl.col("excluded_by").is_null())

    def census(self) -> dict[str, dict[str, int]]:
        census = {}
        for (source,), group in self.rows.group_by("pool_source"):
            counts = dict(group["excluded_by"].fill_null("eligible").value_counts().iter_rows())
            census[str(source)] = {"rows": group.height, **counts}
        return census


def resolve_candidates(
    index: pl.DataFrame,
    labels: pl.DataFrame,
    hierarchy: Hierarchy,
    recipe: PoolRecipe,
) -> Candidates:
    """Attach target nodes, then exclude held-out, unlabelled, evaluation, out-of-target and duplicate tasks."""
    targets = hierarchy.target_map(recipe.target)
    excluded = index.filter(pl.col("pool_source") == PoolSource.EXCLUDED.value)
    pool = index.filter(pl.col("pool_source") != PoolSource.EXCLUDED.value).join(labels, on="content_hash", how="left")
    place = partial(label_node, subject_distance=recipe.subject_distance, capability_distance=recipe.capability_distance)
    placements = [
        tag if source == PoolSource.TASKFORGE.value else place(subject, subject_sim, capability, capability_sim)
        for source, tag, subject, subject_sim, capability, capability_sim in pool.select(
            "pool_source", "tag", "subject_id", "subject_similarity", "capability_id", "capability_similarity"
        ).iter_rows()
    ]
    unknown = sorted({node for node in placements if node is not None and node not in hierarchy.nodes})
    if unknown:
        raise ValueError(f"Labels name nodes outside the Atlas hierarchy: {unknown[:10]}")
    pool = pool.with_columns(
        pl.Series("label_node", placements, dtype=pl.String),
        pl.Series("target_node", [targets.get(node) if node else None for node in placements], dtype=pl.String),
        pl.Series(
            "order",
            [seeded_order(value or "", recipe.seed)[0] for value in pool["content_hash"]],
            dtype=pl.String,
        ),
    )
    held_out = pl.col("task_id").is_in(excluded["task_id"].unique().implode()) | pl.col("content_hash").is_in(
        excluded["content_hash"].unique().implode()
    )
    tasktrove = pl.col("pool_source") == PoolSource.TASKTROVE.value
    reasons = [
        ("invalid_spec", pl.col("content_hash").is_null()),
        ("held_out", held_out),
        ("unlabelled", tasktrove & pl.col("eval").is_null()),
        ("evaluation_source", tasktrove & pl.col("eval").fill_null(False)),
        ("untagged", ~tasktrove & pl.col("label_node").is_null()),
        ("out_of_scope", tasktrove & pl.col("label_node").is_null()),
        ("outside_target", pl.col("target_node").is_null()),
    ]
    reason = pl.lit(None, dtype=pl.String)
    for name, condition in reversed(reasons):
        reason = pl.when(condition).then(pl.lit(name)).otherwise(reason)
    pool = pool.with_columns(reason.alias("excluded_by")).sort("order", "source_id", "task_id")
    # Keep the first row per content within a source, then let Taskforge win content that both sources contain.
    open_rows = pl.col("excluded_by").is_null()
    repeated = open_rows & (pl.int_range(pl.len()).over("pool_source", "content_hash", "excluded_by") > 0)
    pool = pool.with_columns(pl.when(repeated).then(pl.lit("duplicate")).otherwise("excluded_by").alias("excluded_by"))
    taskforge_content = pool.filter(open_rows & ~tasktrove)["content_hash"].implode()
    shared = open_rows & tasktrove & pl.col("content_hash").is_in(taskforge_content)
    pool = pool.with_columns(
        pl.when(shared).then(pl.lit("duplicate_of_taskforge")).otherwise("excluded_by").alias("excluded_by")
    )
    return Candidates(pool)


def supply_by_node(eligible: pl.DataFrame, target: Mapping[str, float]) -> dict[str, Supply]:
    counts = Counter(eligible.select("target_node", "pool_source").iter_rows())
    return {
        node: Supply(
            tasktrove=counts[node, PoolSource.TASKTROVE.value], taskforge=counts[node, PoolSource.TASKFORGE.value]
        )
        for node in target
    }


def select_arms(eligible: pl.DataFrame, allocation: Allocation) -> dict[str, pl.DataFrame]:
    """Take, per node and source, the seeded-order prefix each arm's quota asks for."""
    ranked = eligible.sort("order", "source_id", "task_id").with_columns(
        pl.int_range(pl.len()).over("target_node", "pool_source").alias("rank")
    )
    arms = {}
    for arm, taskforge in allocation.taskforge.items():
        limits = pl.DataFrame(
            [
                {"target_node": node, "pool_source": source.value, "limit": count}
                for node in allocation.quotas
                for source, count in (
                    (PoolSource.TASKFORGE, taskforge[node]),
                    (PoolSource.TASKTROVE, allocation.tasktrove(arm, node)),
                )
            ],
            schema={"target_node": pl.String, "pool_source": pl.String, "limit": pl.Int64},
        )
        chosen = ranked.join(limits, on=["target_node", "pool_source"]).filter(pl.col("rank") < pl.col("limit"))
        arms[arm] = chosen.sort("order", "source_id", "task_id")
    return arms


def rewrite_spec(spec: str, rewrite: RowRewrite | None) -> tuple[str, list[str]]:
    """Bring one row to the current schema and apply its source's rewrite; return the row and the changed fields."""
    row = json.loads(spec)
    changed = []
    task = current_schema(row["task"])
    if task != row["task"]:
        changed.append(f"task.schema_version:{row['task']['schema_version']}->{SCHEMA_VERSION}")
    if rewrite is not None:
        for name, value in sorted(rewrite.limits.items()):
            if name not in row["limits"]:
                raise ValueError(f"Rollout limits have no field {name!r}")
            if row["limits"][name] != value:
                row["limits"] = row["limits"] | {name: value}
                changed.append(f"limits.{name}")
        timeout = task["grader"].get("timeout")
        if rewrite.min_grader_timeout is not None and timeout is not None and timeout < rewrite.min_grader_timeout:
            task = task | {"grader": task["grader"] | {"timeout": rewrite.min_grader_timeout}}
            changed.append("task.grader.timeout")
            # Keep the row's margin between the grader's own deadline and the verifier's.
            if row["limits"]["verifier_timeout"] is not None:
                extended = row["limits"]["verifier_timeout"] + rewrite.min_grader_timeout - timeout
                row["limits"] = row["limits"] | {"verifier_timeout": extended}
                changed.append("limits.verifier_timeout")
    TaskSpec.model_validate(task)
    return json.dumps(row | {"task": task}, ensure_ascii=False), changed


def materialize(chosen: pl.DataFrame, rewrites: Mapping[PoolSource, RowRewrite]) -> pl.DataFrame:
    """Read the chosen rows back from their input files and rewrite them, keyed by path and task id."""
    keys = chosen.select("path", "task_id", "pool_source").unique()
    specs = []
    for (path, source), wanted in keys.group_by("path", "pool_source"):
        rows = _read_parquet(str(path), ["task_rollout_spec"]).with_columns(
            pl.col("task_rollout_spec").str.json_path_match("$.task.id").alias("task_id"),
            pl.lit(path).alias("path"),
        )
        rows = rows.join(wanted.select("path", "task_id"), on=["path", "task_id"])
        rewritten = [rewrite_spec(spec, rewrites.get(PoolSource(source))) for spec in rows["task_rollout_spec"]]
        specs.append(
            rows.with_columns(
                pl.Series("task_rollout_spec", [spec for spec, _ in rewritten], dtype=pl.String),
                pl.Series("rewritten", [changed for _, changed in rewritten], dtype=pl.List(pl.String)),
            )
        )
    found = pl.concat(specs)
    missing = keys.join(found, on=["path", "task_id"], how="anti")
    if missing.height:
        raise ValueError(f"Chosen tasks vanished from their inputs: {missing.head(5).to_dicts()}")
    return found


def arm_rows(chosen: pl.DataFrame, specs: pl.DataFrame) -> pl.DataFrame:
    return chosen.join(specs, on=["path", "task_id"], maintain_order="left").select(OUTPUT_COLUMNS)


def node_report(allocation: Allocation, supply: Mapping[str, Supply], arms: Mapping[str, pl.DataFrame]) -> list[dict]:
    realized = {arm: Counter(frame.select("target_node", "pool_source").iter_rows()) for arm, frame in arms.items()}
    pool_size = sum(allocation.quotas.values())
    rows = []
    for node, quota in sorted(allocation.quotas.items()):
        row: dict = {
            "node_id": node,
            "target_share": allocation.weights[node],
            "quota": quota,
            "realized_share": quota / pool_size,
            "tasktrove_supply": supply[node].tasktrove,
            "taskforge_supply": supply[node].taskforge,
        }
        for arm, counts in realized.items():
            taskforge = counts[node, PoolSource.TASKFORGE.value]
            tasktrove = counts[node, PoolSource.TASKTROVE.value]
            if taskforge + tasktrove != quota:
                raise AssertionError(f"Arm {arm} realized {taskforge + tasktrove} tasks for {node}, quota {quota}")
            row[arm] = {"taskforge": taskforge, "tasktrove": tasktrove, "taskforge_share": taskforge / quota}
        rows.append(row)
    return rows


def file_sha256(path: StoragePath) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def write_arm(root: StoragePath, frame: pl.DataFrame, selection: dict) -> str:
    """Write one arm and its complete file inventory last; return the inventory's sha256 for pinning."""
    root.mkdirs()
    with (root / TRAIN).open("wb") as stream:
        frame.write_parquet(stream, compression="zstd")
    (root / SELECTION).write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    files = []
    for name in sorted((TRAIN, SELECTION)):
        size, sha256 = file_sha256(root / name)
        files.append({"path": name, "size": size, "sha256": sha256})
    (root / ARTIFACT_MANIFEST).write_text(json.dumps({"format": 1, "files": files}, indent=2, sort_keys=True) + "\n")
    return file_sha256(root / ARTIFACT_MANIFEST)[1]


def build_pools(context: ZephyrContext, recipe: PoolRecipe, output: str) -> dict:
    """Write every arm under ``output`` and return the shared selection record."""
    release = AlignmentRelease.model_validate_json(StoragePath(recipe.atlas_release).read_bytes())
    hierarchy = Hierarchy.from_release(release)
    roles = {
        **{path: PoolSource.TASKTROVE for path in recipe.tasktrove},
        **{path: PoolSource.TASKFORGE for path in recipe.taskforge},
        **{path: PoolSource.EXCLUDED for path in recipe.exclude},
    }
    index = index_candidates(context, roles)
    tasktrove_hashes = index.filter(pl.col("pool_source") == PoolSource.TASKTROVE.value)["content_hash"]
    labels = read_labels(recipe.labels, release.release_id, tasktrove_hashes)
    candidates = resolve_candidates(index, labels, hierarchy, recipe)
    supply = supply_by_node(candidates.eligible, recipe.target)
    census = {
        "recipe": recipe.model_dump(mode="json"),
        "atlas_release_id": release.release_id,
        "census": candidates.census(),
        "supply": {node: asdict(item) for node, item in sorted(supply.items())},
    }
    # Written before allocation so an infeasible target leaves the evidence behind.
    root = StoragePath(output)
    root.mkdirs()
    with (root / CANDIDATES).open("wb") as stream:
        candidates.rows.select(CANDIDATE_COLUMNS).write_parquet(stream, compression="zstd")
    (root / CENSUS).write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")
    allocation = allocate(recipe.target, supply, recipe.pool_size, recipe.arms, recipe.infeasible)
    arms = select_arms(candidates.eligible, allocation)
    selection = {
        **census,
        "dropped_nodes": list(allocation.dropped),
        "nodes": node_report(allocation, supply, arms),
        "arms": {
            arm: {
                "taskforge_fraction": recipe.arms[arm],
                "tasks": frame.height,
                "by_pool_source": dict(Counter(frame["pool_source"])),
                "by_source_id": dict(sorted(Counter(frame["source_id"]).items())),
            }
            for arm, frame in arms.items()
        },
    }
    specs = materialize(pl.concat(arms.values()), recipe.rewrites)
    for arm, frame in arms.items():
        rewritten = frame.join(specs, on=["path", "task_id"])["rewritten"].explode().drop_nulls()
        selection["arms"][arm]["rewritten_fields"] = dict(sorted(Counter(rewritten).items()))
    manifests = {arm: write_arm(root / arm, arm_rows(frame, specs), selection) for arm, frame in arms.items()}
    return {**selection, "artifact_manifest_sha256": manifests}


def main() -> None:
    configure_coreweave_s3()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="PoolRecipe JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()
    recipe = PoolRecipe.model_validate_json(StoragePath(args.recipe).read_bytes())
    with ZephyrContext(
        max_workers=args.max_workers,
        intra_shard_execution=IntraShardExecution.AUTO,
        resources=ResourceConfig(cpu=1, ram="4g"),
        coordinator_resources=ResourceConfig(cpu=1, ram="8g", preemptible=False),
        name="capability-pool",
    ) as context:
        selection = build_pools(context, recipe, args.output)
    summary = {
        "output": args.output,
        "census": selection["census"],
        "dropped_nodes": selection["dropped_nodes"],
        "arms": {arm: info["by_pool_source"] for arm, info in selection["arms"].items()},
        "artifact_manifest_sha256": selection["artifact_manifest_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
