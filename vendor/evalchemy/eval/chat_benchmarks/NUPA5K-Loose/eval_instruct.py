"""NUPA5K-Loose fixed panel with permissive final answer extraction."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any

from eval.contracts.eval_spec import SourceRef
from .panel import NUPA5K_SIZE, load_nupa5k_manifest, load_panel_records

nupa = import_module("eval.chat_benchmarks.NUPA-Loose.eval_instruct")


class NUPA5KLooseBenchmark(nupa.NUPALooseBenchmark):
    """Evaluate the fixed stratified NUPA5K-Loose convenience panel."""

    def __init__(
        self,
        dataset_name: str = nupa.SOURCE_DATASET_NAME,
        dataset_revision: str = nupa.SOURCE_DATASET_REVISION,
        dataset_split: str = nupa.DEFAULT_SPLIT,
        source_file: str | None = None,
        max_tokens: int = nupa.DEFAULT_MAX_TOKENS,
        debug: bool = False,
        logger: logging.Logger | None = None,
        system_instruction: str | None = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_revision=dataset_revision,
            dataset_split=dataset_split,
            max_tokens=max_tokens,
            debug=debug,
            logger=logger,
            system_instruction=system_instruction,
        )
        self.source_file = source_file

    def benchmark_size(self) -> int:
        if self.debug:
            size = super().benchmark_size()
            assert size is not None
            return size
        return NUPA5K_SIZE

    def _load_records(self) -> list[dict[str, Any]]:
        if self.debug:
            return super()._load_records()
        source = (
            self.source_file
            if self.source_file
            else nupa.download_nupa_source(
                dataset_name=self.dataset_name,
                dataset_revision=self.dataset_revision,
                split=self.dataset_split,
                cache_prefix=self.dataset_cache,
            )
        )
        records = load_panel_records(
            source,
            split=self.dataset_split,
            identities=load_nupa5k_manifest(),
        )
        return records
    SOURCE = SourceRef(nupa.SOURCE_DATASET_NAME, nupa.DEFAULT_SPLIT, nupa.SOURCE_DATASET_REVISION)
