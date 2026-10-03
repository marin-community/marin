# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read public source catalogs without executing upstream code or downloading tasks."""

import ast
import hashlib
import json
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import httpx

from .composition import HH_RLHF, KTO_MIX, NEMOTRON, NEMOTRON_ENV, canonical_rows, component_rows
from .harbor_sources import HARBOR_SOURCES, HarborSource
from .source_annotations import (
    BENCHMARK_DATASETS,
    CARD_COUNT_DATASETS,
    FAMILY_AUDITS,
    GITHUB_DATASETS,
)

SKYRL = "marin-community/MarinSkyRL"
TASKTROVE = "open-athena/task-trove"
SKYRL_ORIGIN = "MarinSkyRL"
TASKTROVE_ORIGIN = "Task Trove"
HARBOR_ORIGIN = "Harbor Hub"
SOURCE_PATH = "infra/rl_data/sources.py"
GYM_PATH = "skyrl-gym/skyrl_gym/envs/__init__.py"
MULTI_TURN_ENVS = {"gsm8k_multi_turn", "search", "searchcode", "text2sql"}
AGENTIC_ENVS = {"search", "searchcode", "text2sql"}
TASKTROVE_CLASSIFICATION = {
    "environment": "Harbor",
    "type": "Agentic",
    "turns": "Multi-turn",
    "classification_basis": "Task Trove tasks run as Agentic interactions in Harbor",
}


@dataclass(frozen=True)
class Snapshot:
    origin: str
    revision: str
    revised_at: str | None
    rows: list[dict[str, Any]]


def get_json(client: httpx.Client, url: str, **params: str) -> Any:
    response = client.get(url, params=params)
    response.raise_for_status()
    return response.json()


def get_text(client: httpx.Client, url: str) -> str:
    response = client.get(url)
    response.raise_for_status()
    return response.text


def registry_sources(source_text: str) -> list[dict[str, Any]]:
    """Return the sources selected by the upstream SOURCES registry."""
    tree = ast.parse(source_text)
    constants = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    registry = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SOURCES" for target in node.targets)
    )
    if not isinstance(registry, ast.DictComp) or len(registry.generators) != 1:
        raise ValueError("MarinSkyRL SOURCES schema changed; update the catalog reader")

    def literal(node: ast.expr, bindings: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in bindings:
                return bindings[node.id]
            return constants[node.id]
        raise ValueError(f"Unsupported source expression: {ast.dump(node)}")

    def read_call(call: ast.Call, bindings: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(call.func, ast.Name):
            raise ValueError("Expected a named source factory")
        if call.func.id == "Source":
            values = [literal(value, bindings) for value in call.args[:6]]
            return dict(zip(("name", "dataset_id", "env_id", "split", "streaming", "verification"), values, strict=True))
        function = functions[call.func.id]
        # Agent sets and row transforms are deliberately not evaluated.
        bound = {key.arg: literal(key.value, bindings) for key in call.keywords if key.arg in {"name", "blend"}}
        returned = next(node.value for node in function.body if isinstance(node, ast.Return))
        if not isinstance(returned, ast.Call):
            raise ValueError("Expected a source factory return call")
        return read_call(returned, bound)

    calls = registry.generators[0].iter
    if not isinstance(calls, (ast.Tuple, ast.List)):
        raise ValueError("Expected a literal source factory list")
    rows = [read_call(call, {}) for call in calls.elts if isinstance(call, ast.Call)]
    if len(rows) != len(calls.elts) or not rows:
        raise ValueError("Could not read the complete source registry")
    return rows


def registry_environments(source_text: str) -> list[dict[str, str]]:
    rows = []
    for node in ast.parse(source_text).body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Name) and call.func.id == "register":
                fields = {key.arg: ast.literal_eval(key.value) for key in call.keywords}
                rows.append({"name": fields["id"], "entrypoint": fields["entry_point"]})
    if not rows:
        raise ValueError("MarinSkyRL gym registry is empty or changed")
    return rows


@dataclass(frozen=True)
class SourceRow:
    id: str
    origin: str
    name: str
    display_name: str
    revision: str
    revised_at: str
    difficulty: str | None = None
    quality: str | None = None
    traces: int | None = None
    task_count: int | None = None
    is_benchmark: bool = False
    turns: str = "Unknown"
    type: str | None = None
    status: str = "Available"
    kind: str = "Dataset"
    family: str = ""
    notes: str = ""
    count_basis: str = "Not published"
    count_precision: str = "unknown"
    classification_basis: str = "Not declared"


def source_row(origin: str, name: str, revision: str, revised_at: str) -> dict[str, Any]:
    return asdict(SourceRow(f"{origin}:{name}", origin, name, name, revision, revised_at))


def harbor_snapshot(sources: Sequence[HarborSource] = HARBOR_SOURCES) -> Snapshot:
    """Read pinned release metadata without fetching task archives or following mutable tags."""
    content = json.dumps([asdict(source) for source in sources], sort_keys=True).encode()
    rows = []
    for source in sources:
        package, digest = source.package, source.digest
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError(f"Harbor source {package} must pin a SHA-256 release digest")
        url = f"https://hub.harborframework.com/datasets/{package}"
        row = source_row(HARBOR_ORIGIN, package, digest, source.published_at)
        row.update(
            url=url,
            canonical_source=package,
            canonical_url=url,
            dataset_id=package,
            dataset_revision=digest,
            dataset_revised_at=source.published_at,
            verifier_revision=digest,
            verifier_revised_at=source.published_at,
            revision_basis="Pinned Harbor release, including its native task verifiers",
            provenance_url=url,
            upstream_url=source.repository_url,
            paper_url=source.paper_url,
            package_ref=f"{package}@{digest}",
            task_count=source.task_count,
            count_basis=source.count_basis,
            count_precision="exact",
            count_url=url,
            metadata_checked_at=source.metadata_checked_at,
            license=source.license,
            license_url=source.license_url,
            environment="Harbor",
            type="Agentic",
            turns="Multi-turn",
            family=source.family,
            classification_basis="Native Harbor terminal tasks in the official release",
            benchmark_basis=source.benchmark_basis,
            verification=source.verification,
            notes=source.notes,
        )
        rows.append(row)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Harbor registrations contain duplicate source IDs")
    return Snapshot(
        HARBOR_ORIGIN,
        hashlib.sha256(content).hexdigest(),
        max((source.metadata_checked_at for source in sources), default=None),
        rows,
    )


def dataset_metadata(client: httpx.Client, dataset_id: str) -> dict[str, Any]:
    try:
        if dataset_id in GITHUB_DATASETS:
            head = get_json(client, f"https://api.github.com/repos/{dataset_id}/commits/{GITHUB_DATASETS[dataset_id]}")
            card_url = f"https://github.com/{dataset_id}/blob/{head['sha']}/README.md"
            return {
                "sha": head["sha"],
                "lastModified": head["commit"]["committer"]["date"],
                "card_text": get_text(client, f"https://raw.githubusercontent.com/{dataset_id}/{head['sha']}/README.md"),
                "card_url": card_url,
            }
        info = get_json(client, f"https://huggingface.co/api/datasets/{dataset_id}")
    except httpx.HTTPError as error:
        # Metadata availability must not discard a source registered by SkyRL.
        return {"metadata_error": str(error)}
    if dataset_id == NEMOTRON:
        files = get_json(client, f"https://huggingface.co/api/datasets/{dataset_id}/tree/{info['sha']}")
        info["file_sha256"] = {entry["path"]: entry["lfs"]["oid"] for entry in files if entry.get("lfs")}
    if not info.get("cardData", {}).get("dataset_info"):
        try:
            size = get_json(client, "https://datasets-server.huggingface.co/size", dataset=dataset_id)
            info["viewer_splits"] = size["size"]["splits"]
            info["viewer_partial"] = size.get("partial", False)
        except (httpx.HTTPError, KeyError, ValueError) as error:
            info["count_metadata_error"] = str(error)
    if dataset_id in CARD_COUNT_DATASETS:
        try:
            info["card_text"] = get_text(
                client, f"https://huggingface.co/datasets/{dataset_id}/raw/{info['sha']}/README.md"
            )
        except httpx.HTTPError as error:
            info["count_metadata_error"] = str(error)
    info["card_url"] = f"https://huggingface.co/datasets/{dataset_id}/blob/{info['sha']}/README.md"
    if dataset_id == HH_RLHF:
        mirror = get_json(client, "https://huggingface.co/api/datasets/tasksource/hh-rlhf")
        info["components"] = {
            entry["config_name"]: next(split["num_examples"] for split in entry["splits"] if split["name"] == "train")
            for entry in mirror["cardData"]["dataset_info"]
        }
        info["composition_url"] = f"https://huggingface.co/datasets/tasksource/hh-rlhf/blob/{mirror['sha']}/README.md"
    elif dataset_id == KTO_MIX:
        stats = get_json(
            client,
            "https://datasets-server.huggingface.co/statistics",
            dataset="argilla/dpo-mix-7k",
            config="default",
            split="train",
        )
        if stats["partial"]:
            raise ValueError("KTO upstream composition statistics are partial")
        frequencies = next(
            entry["column_statistics"]["frequencies"]
            for entry in stats["statistics"]
            if entry["column_name"] == "dataset"
        )
        info["components"] = {name: count * 2 for name, count in frequencies.items()}
        info["composition_url"] = (
            "https://datasets-server.huggingface.co/statistics?dataset=argilla/dpo-mix-7k&config=default&split=train"
        )
    return info


def split_count(source: dict[str, Any], info: dict[str, Any]) -> int | None:
    entries = info.get("cardData", {}).get("dataset_info", [])
    if isinstance(entries, dict):
        entries = [entries]
    if not entries:
        configs: dict[str, list[dict[str, Any]]] = {}
        for split in info.get("viewer_splits", []):
            configs.setdefault(split["config"], []).append({"name": split["split"], "num_examples": split["num_rows"]})
        entries = [{"config_name": config, "splits": splits} for config, splits in configs.items()]
    if source["name"] == "gsm8k":
        entries = [entry for entry in entries if entry.get("config_name") == "main"]
    if source["name"] == "nemotron_if":
        entries = [entry for entry in entries if entry.get("config_name") == "RL"]
    selected_split = source["split"]
    if source["name"] == "gpqa":
        entries = [entry for entry in entries if entry.get("config_name") == source["split"]]
        selected_split = "train"
    matches = [
        split["num_examples"]
        for entry in entries
        for split in entry.get("splits", [])
        if split["name"] == selected_split and isinstance(split.get("num_examples"), int)
    ]
    # Multiple configurations are ambiguous except MATH's disjoint subject partitions.
    if len(matches) == 1 or (matches and source["name"] == "hendrycks_math"):
        return sum(matches)
    return None


@dataclass(frozen=True)
class TaskCount:
    task_count: int | None
    count_basis: str
    count_precision: str
    count_url: str


def count_metadata(source: dict[str, Any], info: dict[str, Any]) -> TaskCount:
    """Resolve the selected task population without scanning task files."""
    name, split = source["name"], source["split"]
    card = info.get("card_text", "")
    url = info.get("card_url", "")
    pattern = None
    basis = ""
    if source["env_id"] == NEMOTRON_ENV:
        blend = name.removeprefix("nemotron_ultra_")
        pattern = rf"^\|\s*{re.escape(blend)}\s*\|\s*([\d,]+)\s*\|"
        basis = f"Dataset card, Dataset Quantification: {blend} samples"
    elif name == "eurus2_code":
        pattern = r"^\|\s*Coding\s*\|\s*([\d,]+)\s*\|"
        basis = "Dataset card: Coding/train only; SkyRL selects ability=code"
    elif name == "nemotron_if":
        pattern = r"^\|\s*instruction following\s*\|\s*([\d,]+)\s*\|"
        basis = "Dataset card: instruction following, config RL / split instruction_following"
    elif name == "apps":
        pattern = rf"\b{re.escape(split)}:\s*Dataset\(\{{.*?num_rows:\s*([\d,]+)"
        basis = f"Dataset card DatasetDict: {split} rows (not train + test)"
    elif name == "asdiv":
        pattern = r"contains\s+([\d,]+)\s+english Math Word Problems"
        basis = "Original GitHub README: complete ASDiv problem collection"
    elif name == "reasoning_gym":
        return TaskCount(None, "Generated on demand; depends on selected tasks and rows_per_task", "not-applicable", url)
    if pattern:
        match = re.search(pattern, card, flags=re.MULTILINE | re.DOTALL | re.IGNORECASE)
        return TaskCount(
            int(match[1].replace(",", "")) if match else None, basis, "reported" if match else "unknown", url
        )
    if name == "openscience":
        splits = [entry for entry in info.get("viewer_splits", []) if entry["split"] == split]
        if not splits:
            return TaskCount(None, "HF viewer did not publish configuration counts", "unknown", url)
        estimated = any(entry.get("estimated_num_rows") is not None for entry in splits)
        if info.get("viewer_partial") and not estimated:
            return TaskCount(None, "HF viewer is partial and publishes no full estimate", "unknown", url)
        count = sum(entry.get("estimated_num_rows") or entry["num_rows"] for entry in splits)
        return TaskCount(
            count,
            "HF viewer: all train configurations; partial files use estimated_num_rows. "
            "Repository-wide count; each SkyRL run selects a configuration explicitly. "
            "Card audit on 2026-09-28 reports ~6M.",
            "estimated" if estimated else "exact",
            f"https://datasets-server.huggingface.co/size?dataset={source['dataset_id']}",
        )
    count = split_count(source, info)
    basis = (
        f"HF card / viewer: {split}/train; registry split names the config"
        if name == "gpqa"
        else "HF card / viewer selected split rows, before filtering / deduplication"
    )
    partial_splits = [entry for entry in info.get("viewer_splits", []) if entry["split"] == split]
    if not info.get("cardData", {}).get("dataset_info") and info.get("viewer_partial") and partial_splits:
        # A partial preview is not the size of a dataset, even if it has one config.
        count = None
        basis = "HF viewer is partial; exact selected-split count unavailable"
    count_url = (
        f"https://datasets-server.huggingface.co/size?dataset={source['dataset_id']}"
        if info.get("viewer_splits")
        else url
    )
    return TaskCount(count, basis, "exact" if count is not None else "unknown", count_url)


def set_count_metadata(
    row: dict[str, Any], source: dict[str, Any], info: dict[str, Any], previous: dict[str, Any] | None
) -> None:
    count = asdict(count_metadata(source, info))
    # Gated viewers can become unavailable between visits. Reuse evidence only
    # while its upstream revision still matches, including registry rebuilds.
    if (
        count["task_count"] is None
        and previous
        and previous.get("dataset_revision") == info.get("sha")
        and previous.get("task_count") is not None
    ):
        count = {key: previous.get(key) for key in count}
    row.update(count)


def annotate_source(row: dict[str, Any]) -> None:
    """Apply audited classifications and canonical names without changing source IDs."""
    if row["origin"] == HARBOR_ORIGIN:
        return
    if row["origin"] == TASKTROVE_ORIGIN:
        row["display_name"] = row["name"].replace("__", "/", 1)
        row.update(
            count_precision="exact",
            count_url=row["provenance_url"],
            family_basis="Task Trove release manifest source_verdicts.family",
            family_url=row["provenance_url"],
        )
        row["canonical_source"] = row["display_name"]
        row["canonical_url"] = row["url"]
        row["turns"] = "Multi-turn"
        return
    if row.get("component_name"):
        return
    dataset_id = row["dataset_id"]
    row["url"] = f"https://huggingface.co/datasets/{dataset_id}"
    selector = ""
    if row["environment"] == NEMOTRON_ENV:
        selector = row["name"].removeprefix("nemotron_ultra_")
        row["notes"] = (
            "Blend includes verifiable, alignment, and agentic tasks. Counts are published in the dataset card."
        )
    elif row["name"] == "eurus2_code":
        selector = "code"
    elif row["name"] == "nemotron_if":
        selector = "RL/instruction_following"
    elif row["name"] == "gpqa":
        selector = row["split"]
    row["display_name"] = dataset_id + (f" · {selector}" if selector else "")
    row["canonical_source"] = row["display_name"]
    row["canonical_url"] = row["url"]
    audit = FAMILY_AUDITS.get(dataset_id)
    if audit:
        row.update(
            family=audit.family,
            family_basis="Upstream card/schema and selected SkyRL loader audited 2026-09-28",
            family_url=audit.evidence_url,
        )
    if dataset_id in GITHUB_DATASETS:
        row["url"] = f"https://github.com/{dataset_id}"
        row["canonical_url"] = row["url"]
    if dataset_id in BENCHMARK_DATASETS:
        row.update(is_benchmark=True, benchmark_basis="Upstream dataset card explicitly describes a benchmark")


def merge_gym_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep bound sources once and carry their gym registration metadata onto them."""
    adapters = {row["environment"]: row for row in rows if row["kind"] == "Environment"}
    sources = []
    for saved in rows:
        if saved["kind"] == "Environment":
            continue
        row = dict(saved)
        if row["origin"] == SKYRL_ORIGIN:
            row["gym_alias"] = f"gym/{row['environment']}"
            row["gym_url"] = f"https://github.com/{SKYRL}/blob/{row['revision']}/{GYM_PATH}"
            adapter = adapters.get(row["environment"])
            if adapter:
                row["gym_entrypoint"] = adapter["entrypoint"]
        sources.append(row)
    return sources


def set_revision_date(row: dict[str, Any]) -> None:
    dates = [row.get("dataset_revised_at"), row.get("verifier_revised_at")]
    row["revised_at"] = max((date for date in dates if date), key=datetime.fromisoformat, default=None)
    row["revision_basis"] = "Latest upstream dataset repository or MarinSkyRL verifier change"


def datasets_metadata(client: httpx.Client, dataset_ids: set[str]) -> dict[str, dict[str, Any]]:
    names = sorted(dataset_ids)
    with ThreadPoolExecutor(max_workers=6) as pool:
        return dict(zip(names, pool.map(lambda name: dataset_metadata(client, name), names), strict=True))


def source_components(row: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    annotate_source(row)
    set_revision_date(row)
    return component_rows(row, info)


def refresh_dataset_metadata(client: httpx.Client, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = canonical_rows(merge_gym_sources(rows))
    metadata = datasets_metadata(client, {row["dataset_id"] for row in rows})
    refreshed = []
    for saved in rows:
        row = dict(saved)
        info = metadata[row["dataset_id"]]
        row["metadata_error"] = info.get("metadata_error")
        if not row["metadata_error"]:
            source = {
                "name": row["name"],
                "env_id": row["environment"],
                "split": row["split"],
                "dataset_id": row["dataset_id"],
            }
            set_count_metadata(row, source, info, saved)
            row.update(
                dataset_revision=info["sha"],
                dataset_revised_at=info["lastModified"],
                license=info.get("cardData", {}).get("license"),
                is_benchmark="benchmark:official" in info.get("tags", []),
                count_metadata_error=info.get("count_metadata_error"),
            )
        refreshed.extend(source_components(row, info))
    return refreshed


def skyrl_snapshot(
    client: httpx.Client, head: dict[str, Any], cached_rows: list[dict[str, Any]] | None = None, force: bool = False
) -> Snapshot:
    revision = head["sha"]
    if (
        not force
        and cached_rows
        and all(row["revision"] == revision and row.get("verifier_revised_at") for row in cached_rows)
    ):
        rows = refresh_dataset_metadata(client, cached_rows)
        return Snapshot(SKYRL_ORIGIN, revision, head["commit"]["committer"]["date"], rows)
    raw = f"https://raw.githubusercontent.com/{SKYRL}/{revision}"
    source_text = get_text(client, f"{raw}/{SOURCE_PATH}")
    gym_text = get_text(client, f"{raw}/{GYM_PATH}")
    sources = registry_sources(source_text)
    bound_envs = {source["env_id"] for source in sources}
    environments = [env for env in registry_environments(gym_text) if env["name"] in bound_envs]
    commit = get_json(
        client, f"https://api.github.com/repos/{SKYRL}/commits", path=SOURCE_PATH, sha=revision, per_page="1"
    )
    registry_date = commit[0]["commit"]["committer"]["date"]
    verifier_commits = {}
    for env in environments:
        module = env["entrypoint"].split(":")[0]
        path = "skyrl-gym/" + module.rsplit(".", 1)[0].replace(".", "/")
        if path not in verifier_commits:
            commits = get_json(
                client, f"https://api.github.com/repos/{SKYRL}/commits", path=path, sha=revision, per_page="1"
            )
            verifier_commits[path] = commits[0]
        commit = verifier_commits[path]
        env.update(
            verifier_path=path,
            verifier_revision=commit["sha"],
            verifier_revised_at=commit["commit"]["committer"]["date"],
        )
    verifier_by_env = {env["name"]: env for env in environments}
    metadata = datasets_metadata(client, {source["dataset_id"] for source in sources})
    previous_by_id = {row["id"]: row for row in cached_rows or []}
    rows = []
    for source in sources:
        info = metadata[source["dataset_id"]]
        env = source["env_id"]
        row = source_row(SKYRL_ORIGIN, source["name"], revision, registry_date)
        row.update(
            url=f"https://huggingface.co/datasets/{source['dataset_id']}",
            provenance_url=f"https://github.com/{SKYRL}/blob/{revision}/{SOURCE_PATH}",
            dataset_id=source["dataset_id"],
            environment=env,
            split=source["split"],
            type="Alignment" if env == "preference" else "Agentic" if env in AGENTIC_ENVS else "RLVR",
            turns=(
                "Mixed"
                if env in {NEMOTRON_ENV, "preference"}
                else "Multi-turn" if env in MULTI_TURN_ENVS else "Single-turn"
            ),
            classification_basis=(
                "Inferred from SkyRL environment contract; " "blended sources may contain multiple task types"
            ),
            is_benchmark="benchmark:official" in info.get("tags", []),
            benchmark_basis="SkyRL test-only designation or HF benchmark:official tag; false means no designation found",
            verification=source["verification"],
            dataset_revision=info.get("sha"),
            dataset_revised_at=info.get("lastModified"),
            license=info.get("cardData", {}).get("license"),
            notes=info.get("metadata_error", "Last revision includes HF dataset and verifier-code changes."),
            metadata_error=info.get("metadata_error"),
            count_metadata_error=info.get("count_metadata_error"),
        )
        set_count_metadata(row, source, info, previous_by_id.get(row["id"]))
        if env == NEMOTRON_ENV:
            row["type"] = None
            row["notes"] = (
                "Blend includes verifiable, alignment, and agentic tasks. Counts are published in the dataset card."
            )
        if source["name"] == "reasoning_gym":
            row["kind"] = "Generator"
            row["count_basis"] = "Generated on demand; depends on selected tasks and rows_per_task"
        verifier = verifier_by_env[env]
        row.update({key: verifier[key] for key in ("verifier_path", "verifier_revision", "verifier_revised_at")})
        row["registry_revised_at"] = registry_date
        row["verifier_url"] = f"https://github.com/{SKYRL}/tree/{revision}/{verifier['verifier_path']}"
        row["gym_alias"] = f"gym/{env}"
        row["gym_entrypoint"] = verifier["entrypoint"]
        row["gym_url"] = f"https://github.com/{SKYRL}/blob/{revision}/{GYM_PATH}"
        rows.extend(source_components(row, info))
    return Snapshot(SKYRL_ORIGIN, revision, head["commit"]["committer"]["date"], rows)


def tasktrove_snapshot(manifest: dict[str, Any], info: dict[str, Any]) -> Snapshot:
    rows = []
    for name, statuses in manifest["by_source"].items():
        details = manifest["source_details"].get(name, {})
        verdict = manifest["source_verdicts"][name]
        family = verdict["family"]
        modes = list(details.get("modes", {}))
        count = statuses.get("converted", 0)
        row = source_row(TASKTROVE_ORIGIN, name, info["sha"], info["lastModified"])
        row.update(
            url=f"https://huggingface.co/datasets/{TASKTROVE}",
            provenance_url=f"https://huggingface.co/datasets/{TASKTROVE}/blob/{info['sha']}/manifest.json",
            upstream_url=f"https://huggingface.co/datasets/{name.replace('__', '/', 1)}",
            upstream_link_basis="Dataset identifier encoded in Task Trove source name; not independently resolved",
            dataset_id=TASKTROVE,
            task_count=count,
            input_count=sum(statuses.values()),
            count_basis="Released Harbor tasks: manifest by_source.converted",
            count_precision="exact",
            family=family,
            family_basis="Task Trove release manifest source_verdicts.family",
            family_url=f"https://huggingface.co/datasets/{TASKTROVE}/blob/{info['sha']}/manifest.json",
            modes=modes,
            languages=list(details.get("languages", {})),
            status="Available" if count else "Excluded",
            turns="Multi-turn",
            benchmark_basis="Release manifest does not designate benchmarks",
            notes=verdict["reason"],
            verification=", ".join(modes),
            split="train",
        )
        row.update(TASKTROVE_CLASSIFICATION)
        annotate_source(row)
        rows.append(row)
    if sum(row["task_count"] for row in rows) != manifest["clean_tasks"]:
        raise ValueError("Task Trove source counts disagree with release total")
    return Snapshot(TASKTROVE_ORIGIN, info["sha"], info["lastModified"], rows)
