# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.processing.tokenize.cache_stats import TokenizedCacheStats

from experiments.data_belay.intake import IntakeReport, LicenseVerdict, RungNeed, SampleStats
from experiments.data_belay.prepare import MANIFEST_NAME, CacheManifest, CachePlan, PreparedCache


def write_manifest(root, share: float = 0.2) -> None:
    report = IntakeReport(
        repo="HuggingFaceTB/finemath",
        revision="e92b25a61673",
        config="finemath-4plus",
        split="train",
        text_field="text",
        licenses=("odc-by",),
        license_verdict=LicenseVerdict.ALLOWED,
        gated=False,
        num_rows=100,
        sample=SampleStats(rows=10, text_bytes=1000, tokens=300),
        estimated_tokens=3000,
        share=share,
        needs=(RungNeed("d512", rung_tokens=10_000, new_tokens=2000, epochs=0.67),),
    )
    plan = CachePlan(str(root), ("a.parquet",), max_rows=50, estimated_tokens=2500, need_tokens=2000)
    root.mkdir()
    (root / MANIFEST_NAME).write_text(CacheManifest(report, plan, TokenizedCacheStats(50, 2600), 0.77).to_json())


def test_train_reads_the_share_and_label_that_prepare_wrote(tmp_path):
    root = tmp_path / "finemath-4plus-train-4ab2d8f8d6"
    write_manifest(root, share=0.2)
    cache = PreparedCache.read(str(root))
    assert cache.share == 0.2
    assert cache.train_cache == f"{root}/train"
    assert cache.candidate.startswith("finemath-4plus-p20-")


def test_two_caches_of_one_dataset_and_share_get_different_run_names(tmp_path):
    # A resampled cache must not train under, or be reported as, the first cache's runs.
    first, second = tmp_path / "sample-a", tmp_path / "sample-b"
    write_manifest(first)
    write_manifest(second)
    assert PreparedCache.read(str(first)).candidate != PreparedCache.read(str(second)).candidate
