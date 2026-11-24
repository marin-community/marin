# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hybrid OpenResearcher retriever: fusion, state, license and pilot sampling."""

import json
import random
from pathlib import Path

import pytest
from finestore.rl.licenses import LicenseReceipt, require_license
from pydantic import ValidationError

from experiments.post_training.task_curation.datasets.openresearcher.dense_contract import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
)
from experiments.post_training.task_curation.datasets.openresearcher.embedding import validate_model_license
from experiments.post_training.task_curation.datasets.openresearcher.hybrid_fusion import rrf_fuse
from experiments.post_training.task_curation.datasets.openresearcher.hybrid_index import (
    PROVIDER_VERSION,
    FusionParameters,
    provider_state,
)
from experiments.post_training.task_curation.datasets.openresearcher.pilot_sample import (
    PILOT_SEED,
    RECOVERY_DROP_REASON,
)
from taskcompendium.models import ProviderRequirement
from taskcompendium.openresearcher import (
    OPENRESEARCHER_HYBRID_ACTION_INTERFACE,
    OpenResearcherHybridState,
    openresearcher_hybrid_state,
    openresearcher_hybrid_tools,
)

BM25_MANIFEST = "a" * 64
DENSE_MANIFEST = "b" * 64


def hybrid_index_stub():
    class StubIndex:
        corpus_revision = "8367703babe6e080899a48782f365c284c32b668"
        embedding_model = EMBEDDING_MODEL
        embedding_model_revision = EMBEDDING_MODEL_REVISION
        embedding_dim = EMBEDDING_DIM
        max_embed_tokens = 512
        fusion = FusionParameters(rrf_k=60, candidate_depth=25)

        class bm25_index:
            index_manifest_sha256 = BM25_MANIFEST

        class dense_index:
            index_manifest_sha256 = DENSE_MANIFEST

    return StubIndex()


class TestRrfFusion:
    def test_fuses_contributions_from_every_list(self):
        fused = rrf_fuse([["a", "b", "c"], ["b", "d", "e"]], k=60, top_k=5)
        scores = dict(fused)
        assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
        assert scores["a"] == pytest.approx(1 / 61)
        assert scores["e"] == pytest.approx(1 / 63)
        # b appears in both lists, so it outranks every single-list member.
        assert fused[0][0] == "b"

    def test_orders_truncates_and_tie_breaks_deterministically(self):
        # All members of the two lists share ranks, so docids decide the order.
        fused = rrf_fuse([["c", "a"], ["a", "c"]], k=1, top_k=2)
        assert [docid for docid, _ in fused] == ["a", "c"]
        assert rrf_fuse([["c", "a"], ["a", "c"]], k=1, top_k=2) == fused
        assert len(rrf_fuse([["x", "y", "z"]], k=60, top_k=2)) == 2

    def test_rejects_nonpositive_constant(self):
        with pytest.raises(ValueError):
            rrf_fuse([["a"]], k=0)

    def test_empty_lists_yield_empty_fusion(self):
        assert rrf_fuse([[], []]) == []


class TestProviderState:
    def canonical_state(self):
        return {
            "implementation_revision": "0" * 40,
            "corpus_revision": "8367703babe6e080899a48782f365c284c32b668",
            "bm25_index_manifest_sha256": BM25_MANIFEST,
            "dense_index_manifest_sha256": DENSE_MANIFEST,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_model_revision": EMBEDDING_MODEL_REVISION,
            "embedding_dim": 1024,
            "max_embed_tokens": 512,
            "fusion": "rrf",
            "fusion_rrf_k": 60,
            "candidate_depth": 25,
            "top_k": 5,
            "snippet_max_tokens": 512,
            "include_get_document": True,
        }

    def test_state_validates_the_canonical_shape(self):
        state = OpenResearcherHybridState.model_validate(self.canonical_state())
        assert state.fusion == "rrf"
        tools = openresearcher_hybrid_tools(state)
        assert [tool.name for tool in tools] == ["search", "get_document"]

    def test_state_binds_the_provider_interface(self):
        requirement = ProviderRequirement(
            action_interface="openresearcher-bm25:v1",
            initial_state=self.canonical_state(),
        )
        with pytest.raises(ValueError, match="Unsupported OpenResearcher interface"):
            openresearcher_hybrid_state(requirement)

    def test_state_rejects_looser_contracts(self):
        for field, value in (("top_k", 10), ("candidate_depth", 50), ("embedding_dim", 768), ("fusion", "bm25")):
            changed = self.canonical_state() | {field: value}
            with pytest.raises(ValidationError):
                OpenResearcherHybridState.model_validate(changed)

    def test_provider_state_derives_from_the_index_artifact(self):
        state = OpenResearcherHybridState.model_validate(
            provider_state(hybrid_index_stub(), "1" * 40)
        )
        assert state.bm25_index_manifest_sha256 == BM25_MANIFEST
        assert state.dense_index_manifest_sha256 == DENSE_MANIFEST
        assert state.implementation_revision == "1" * 40


class TestModelLicense:
    def test_recorded_license_is_permissive(self, tmp_path: Path):
        receipt = validate_model_license()
        assert receipt.classification.value == "permissive"
        assert receipt.license_ids == ("mit",)
        require_license(receipt, EMBEDDING_MODEL, EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION, "train", ban=None)

    def test_tampered_evidence_fails_closed(self, tmp_path: Path):
        receipt_path = (
            Path(__file__).parents[1]
            / "datasets"
            / "openresearcher"
            / "embedding_model_license.json"
        )
        payload = json.loads(receipt_path.read_text())
        payload["revision"] = "f" * 40
        receipt = LicenseReceipt.model_validate(payload)
        with pytest.raises(ValueError, match="another source revision"):
            require_license(receipt, EMBEDDING_MODEL, EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION, "train", ban=None)


class TestPilotSamplingSemantics:
    def test_seed_and_budgets_are_sealed_constants(self):
        assert PILOT_SEED == "openresearcher-hybrid-pilot-v1"
        assert RECOVERY_DROP_REASON == "required_hop_not_grounded_or_citation_not_literal"

    def test_seed_reproduces_the_sample_from_sorted_populations(self):
        drops = [f"drop-{number:04d}" for number in range(600)]
        kept = [f"kept-{number:04d}" for number in range(100)]
        first = sorted(random.Random(PILOT_SEED).sample(drops, 300)), sorted(random.Random(PILOT_SEED).sample(kept, 100))
        second = sorted(random.Random(PILOT_SEED).sample(drops, 300)), sorted(random.Random(PILOT_SEED).sample(kept, 100))
        assert first == second
        assert len(first[0]) == 300 and len(first[1]) == 100
        assert not set(first[0]) & set(first[1])
