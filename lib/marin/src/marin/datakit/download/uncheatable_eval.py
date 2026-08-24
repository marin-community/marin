# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a released UncheatableEval dataset into per-category documents."""

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from typing import Any

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet
from zephyr.writers import write_jsonl_file


@dataclass(frozen=True)
class UncheatableEvalTransformConfig:
    """Configuration for splitting one Hugging Face release by category."""

    input_path: str
    output_path: str
    categories: tuple[str, ...]


def uncheatable_eval_document(row: dict[str, Any]) -> dict[str, str]:
    """Convert one released UncheatableEval row to Marin's text schema."""
    content = row.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("UncheatableEval row has no content")

    category = row.get("category")
    if not isinstance(category, str) or not category:
        raise ValueError("UncheatableEval row has no category")

    url = row.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("UncheatableEval row has no URL")

    return {
        "id": url,
        "text": content,
        "source": f"uncheatable_eval/{category}",
    }


def _write_category(output_path: str, category: str, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path = prefix_join(output_path, f"{category}.jsonl.gz")
    result = write_jsonl_file((uncheatable_eval_document(row) for row in rows), path)
    return {"category": category, **result}


def transform_uncheatable_eval(cfg: UncheatableEvalTransformConfig) -> dict[str, Any]:
    """Split a pinned UncheatableEval Parquet release into compressed JSONL files."""
    selected_categories = frozenset(cfg.categories)
    pipeline = (
        Dataset.from_files(prefix_join(cfg.input_path, "**/*.parquet"))
        .flat_map(load_parquet)
        .filter(lambda row: row.get("category") in selected_categories)
        .group_by(
            key=lambda row: row["category"],
            reducer=partial(_write_category, cfg.output_path),
            num_output_shards=len(selected_categories),
        )
    )

    ctx = ZephyrContext(name="transform-uncheatable-eval", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)

    missing_categories = [
        category
        for category in cfg.categories
        if not StoragePath(prefix_join(cfg.output_path, f"{category}.jsonl.gz")).exists()
    ]
    if missing_categories:
        raise ValueError(f"UncheatableEval release is missing categories: {', '.join(missing_categories)}")

    return {"categories": list(cfg.categories)}
