# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct selected text from original provider rows with explicit transforms."""

import hashlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from marin.datakit import normalize
from marin.datakit.download import (
    agenttrove,
    biodiversity,
    coderforge,
    davinci_dev,
    finetranslations,
    gpt_oss_rollouts,
    institutional_books,
    penfever_rollouts,
    rollout_transforms,
    stack_v3,
    swe_rebench_contree,
    swe_zero_12m,
    synthetic1,
)

from experiments.datakit.store import (
    davinci_provider_rows,
    external_provider_rows,
    software_heritage_empty_rows,
    structured_provider_rows,
)


class ProviderTransform(StrEnum):
    DIRECT = "direct"
    AGENTTROVE = "agenttrove"
    CODERFORGE = "coderforge"
    DAVINCI_ENV = "davinci_env"
    DAVINCI_CTX = "davinci_ctx"
    GPT_OSS = "gpt_oss_rollouts"
    PENFEVER = "penfever_rollouts"
    SYNTHETIC1 = "synthetic1"
    MOLMO2_CAP = "molmo2_cap"
    NUMINAMATH_V1_5 = "numinamath_v1_5"
    NUMINAMATH_TIR = "numinamath_tir"
    SUPERIOR_REASONING = "superior_reasoning"
    SWE_OPENHANDS = "swe_openhands"
    SOFTWARE_HERITAGE = "software_heritage"
    SOFTWARE_HERITAGE_EMPTY = "software_heritage_audited_empty"
    FINETRANSLATIONS = "finetranslations"
    INSTITUTIONAL_BOOKS = "institutional_books"
    STACK_V3 = "stack_v3"
    SWE_CONTREE = "swe_contree"
    SWE_ZERO = "swe_zero"
    BIODIVERSITY = "biodiversity"


@dataclass(frozen=True)
class ReconstructionRecipe:
    transform: ProviderTransform
    text_field: str
    id_field: str
    max_whitespace_run_chars: int
    normalization_sha256: str
    transform_sha256: str
    bare: bool = False
    drop_fields: tuple[str, ...] = ()
    penfever_dataset: penfever_rollouts.PenfeverRollout | None = None


def reconstruction_code_digests(transform: ProviderTransform) -> tuple[str, str]:
    """Identify the normalizer and source adapter actually used for replay.

    These hashes pin replay code; they do not establish equivalence to an older
    artifact's transform. Matching the selected document hash checks its content.
    """
    structured = {adapter.transform: adapter for adapter in structured_provider_rows.ADAPTERS.values()}
    if transform in structured:
        assert normalize.__file__ is not None
        return hashlib.sha256(Path(normalize.__file__).read_bytes()).hexdigest(), structured[transform].digest()
    modules = {
        ProviderTransform.DIRECT: normalize,
        ProviderTransform.AGENTTROVE: agenttrove,
        ProviderTransform.CODERFORGE: coderforge,
        ProviderTransform.DAVINCI_ENV: davinci_dev,
        ProviderTransform.DAVINCI_CTX: davinci_dev,
        ProviderTransform.GPT_OSS: gpt_oss_rollouts,
        ProviderTransform.PENFEVER: penfever_rollouts,
        ProviderTransform.SYNTHETIC1: synthetic1,
        ProviderTransform.SOFTWARE_HERITAGE: external_provider_rows,
        ProviderTransform.SOFTWARE_HERITAGE_EMPTY: external_provider_rows,
        ProviderTransform.FINETRANSLATIONS: finetranslations,
        ProviderTransform.INSTITUTIONAL_BOOKS: institutional_books,
        ProviderTransform.STACK_V3: stack_v3,
        ProviderTransform.SWE_CONTREE: swe_rebench_contree,
        ProviderTransform.SWE_ZERO: swe_zero_12m,
        ProviderTransform.BIODIVERSITY: biodiversity,
    }
    normalizer_path = normalize.__file__
    transform_path = modules[transform].__file__
    assert normalizer_path is not None and transform_path is not None
    transform_bytes = Path(transform_path).read_bytes()
    if transform in (ProviderTransform.PENFEVER, ProviderTransform.SYNTHETIC1):
        assert rollout_transforms.__file__ is not None
        transform_bytes += b"\0role_renderer\0" + Path(rollout_transforms.__file__).read_bytes()
    if transform is ProviderTransform.DAVINCI_CTX:
        assert rollout_transforms.__file__ is not None and davinci_provider_rows.__file__ is not None
        transform_bytes += b"\0historical_loader\0" + Path(rollout_transforms.__file__).read_bytes()
        transform_bytes += b"\0audited_reader\0" + Path(davinci_provider_rows.__file__).read_bytes()
    if transform is ProviderTransform.SOFTWARE_HERITAGE_EMPTY:
        assert software_heritage_empty_rows.__file__ is not None
        transform_bytes += b"\0audited_empty_reader\0" + Path(software_heritage_empty_rows.__file__).read_bytes()
    return hashlib.sha256(Path(normalizer_path).read_bytes()).hexdigest(), hashlib.sha256(transform_bytes).hexdigest()


def _transformed_documents(recipe: ReconstructionRecipe, provider_rows: list[dict]) -> Iterable[dict]:
    transform = recipe.transform
    if transform is ProviderTransform.BIODIVERSITY:
        item_ids = {row["item_id"] for row in provider_rows}
        page_numbers = [row["page_num"] for row in provider_rows]
        if len(item_ids) != 1 or len(set(page_numbers)) != len(page_numbers):
            raise ValueError("Biodiversity assembly requires one item and unambiguous page ordering")
        if page_numbers != sorted(page_numbers):
            raise ValueError("Biodiversity locators must supply pages in assembly order")
        return biodiversity.stitch_pages(next(iter(item_ids)), iter(provider_rows))
    if len(provider_rows) != 1:
        raise ValueError(f"{transform} reconstructs a document from exactly one provider row")
    row = provider_rows[0]
    for adapter in structured_provider_rows.ADAPTERS.values():
        if transform == adapter.transform:
            return adapter.row_to_docs(row)
    if transform is ProviderTransform.DIRECT:
        return (row,)
    if transform is ProviderTransform.AGENTTROVE:
        return agenttrove.row_to_doc(row)
    if transform is ProviderTransform.CODERFORGE:
        return coderforge.row_to_doc(row)
    if transform is ProviderTransform.DAVINCI_ENV:
        return davinci_dev.env_row_to_doc(row)
    if transform is ProviderTransform.DAVINCI_CTX:
        return davinci_dev.ctx_row_to_doc(row)
    if transform is ProviderTransform.GPT_OSS:
        return gpt_oss_rollouts.row_to_doc(row)
    if transform is ProviderTransform.PENFEVER:
        if recipe.penfever_dataset is None:
            raise ValueError("Penfever replay requires the explicit frozen dataset configuration")
        return penfever_rollouts.row_to_doc(recipe.penfever_dataset)(row)
    if transform is ProviderTransform.SYNTHETIC1:
        return synthetic1.row_to_doc(row)
    if transform in {ProviderTransform.SOFTWARE_HERITAGE, ProviderTransform.SOFTWARE_HERITAGE_EMPTY}:
        return external_provider_rows.software_heritage_rows(row)
    if transform is ProviderTransform.FINETRANSLATIONS:
        return finetranslations.fold_parallel_text(row)
    if transform is ProviderTransform.INSTITUTIONAL_BOOKS:
        return institutional_books.row_to_doc(row)
    if transform is ProviderTransform.STACK_V3:
        return (dict(stack_v3.row_to_doc(row)),)
    if transform is ProviderTransform.SWE_CONTREE:
        return swe_rebench_contree.row_to_doc(row)
    if transform is ProviderTransform.SWE_ZERO:
        return swe_zero_12m.row_to_doc(row)
    raise ValueError(f"Unknown provider transform: {transform}")


def provider_normalizer(recipe: ReconstructionRecipe) -> Callable[[list[dict]], Iterator[dict[str, Any]]]:
    """Prepare a digest-checked transform with frozen normalization parameters.

    Input rows must come from original providers. Assembled documents require a
    complete ordered input group; a matching hash alone does not prove lineage.
    """
    actual = reconstruction_code_digests(recipe.transform)
    if actual != (recipe.normalization_sha256, recipe.transform_sha256):
        raise ValueError("Replay implementation differs from the recorded code digests")
    normalize_record = normalize._make_normalize_fn(
        recipe.text_field, recipe.id_field, bare=recipe.bare, drop_fields=recipe.drop_fields
    )
    compact = normalize._make_whitespace_compactor(recipe.max_whitespace_run_chars)

    def replay(provider_rows: list[dict]) -> Iterator[dict[str, Any]]:
        for document in _transformed_documents(recipe, provider_rows):
            value = document.get(recipe.text_field)
            if value is not None and normalize._text_from_value(value).strip():
                yield compact(normalize_record(document))

    return replay


def normalized_provider_documents(recipe: ReconstructionRecipe, provider_rows: list[dict]) -> Iterator[dict[str, Any]]:
    """Normalize one original provider input group."""
    yield from provider_normalizer(recipe)(provider_rows)


def reconstruct_selected_document(
    recipe: ReconstructionRecipe, provider_rows: list[dict], expected_id: str
) -> dict[str, Any]:
    """Return the selected normalized text, rejecting absent or ambiguous outputs."""
    matched = None
    for document in normalized_provider_documents(recipe, provider_rows):
        if document["id"] == expected_id:
            if matched is not None:
                raise ValueError(f"Provider assembly produced repeated matching outputs for {expected_id}")
            matched = document
    if matched is None:
        raise ValueError(f"Provider reconstruction did not match selected document {expected_id}")
    return matched
