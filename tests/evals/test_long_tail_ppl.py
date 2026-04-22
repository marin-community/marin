# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.evals.long_tail_ppl import (
    EPIC_5005,
    GAME_MUSIC_ISSUE,
    LONG_TAIL_PPL_REGISTRY,
    LONG_TAIL_PPL_SLICES,
    LongTailPplFamily,
    long_tail_ppl_slices,
    long_tail_raw_validation_sets,
    render_long_tail_ppl_registry_markdown,
)
from experiments.evals.long_tail_ppl_runnable import (
    RUNNABLE_LONG_TAIL_PPL_REGISTRY,
    RUNNABLE_LONG_TAIL_PPL_SLICES,
    runnable_long_tail_ppl_slices,
    runnable_long_tail_raw_validation_sets,
)
from experiments.exp_model_perplexity_gap_long_tail_smoke import DATASETS, MAX_DOCS_PER_DATASET, STEP
from levanter.data.text import HfDatasetSourceConfig
from marin.evaluation.perplexity_gap import _to_dataset_component, raw_text_dataset
from marin.processing.tokenize import HfDatasetSpec


def test_long_tail_registry_covers_expected_families_and_anchors():
    families = {slice_.family for slice_ in LONG_TAIL_PPL_SLICES}
    assert families == set(LongTailPplFamily)

    names = {slice_.name for slice_ in LONG_TAIL_PPL_SLICES}
    assert {
        "common_crawl_warc",
        "svg_stack",
        "microsoft_malware_bytes",
        "refseq_fasta",
        "monash_tsf",
        "smtlib",
        "deps_dev",
        "lichess_pgn",
    }.issubset(names)

    assert all(slice_.registry_key in LONG_TAIL_PPL_REGISTRY for slice_ in LONG_TAIL_PPL_SLICES)
    assert all(slice_.issue_number in {5056, 5057, 5058, 5059, 5060, 5061, 5062} for slice_ in LONG_TAIL_PPL_SLICES)
    assert all(f"epic:{EPIC_5005}" in slice_.tags for slice_ in LONG_TAIL_PPL_SLICES)


def test_long_tail_raw_validation_sets_render_deterministic_paths_and_tags():
    datasets = long_tail_raw_validation_sets(raw_root="gs://example-bucket/raw/long_tail")

    web_key = "long_tail_ppl/web_markup_image_text/common_crawl_warc"
    game_key = "long_tail_ppl/game_music/lichess_pgn"

    assert datasets[web_key].input_path == "gs://example-bucket/raw/long_tail/web/common_crawl/warc.jsonl.gz"
    assert datasets[web_key].tags == ("long_tail_ppl", "epic:5005", "issue:5056", "web_markup_image_text")
    assert datasets[web_key].text_key == "text"

    assert datasets[game_key].input_path == "gs://example-bucket/raw/long_tail/games/lichess/pgn.jsonl.gz"
    assert datasets[game_key].tags == ("long_tail_ppl", "epic:5005", f"issue:{GAME_MUSIC_ISSUE}", "game_music")


def test_long_tail_registry_rendering_mentions_issue_links():
    markdown = render_long_tail_ppl_registry_markdown(family=LongTailPplFamily.GAME_MUSIC)

    assert "long_tail_ppl/game_music/lichess_pgn" in markdown
    assert "database.lichess.org" in markdown
    assert f"#{GAME_MUSIC_ISSUE}" in markdown


def test_family_filter_returns_only_requested_family():
    slices = long_tail_ppl_slices(family=LongTailPplFamily.BINARY_NETWORK_SECURITY)

    assert slices
    assert {slice_.family for slice_ in slices} == {LongTailPplFamily.BINARY_NETWORK_SECURITY}


def test_runnable_registry_is_separate_and_hf_backed():
    assert len(RUNNABLE_LONG_TAIL_PPL_SLICES) == 4
    assert set(RUNNABLE_LONG_TAIL_PPL_REGISTRY) == {
        "long_tail_ppl_runnable/web_markup_image_text/svg_stack_val",
        "long_tail_ppl_runnable/web_markup_image_text/svg_stack_test",
        "long_tail_ppl_runnable/formal_hardware/verilogeval_prompt",
        "long_tail_ppl_runnable/formal_hardware/verilogeval_canonical_solution",
    }

    datasets = runnable_long_tail_raw_validation_sets()
    assert datasets["long_tail_ppl_runnable/web_markup_image_text/svg_stack_val"].hf_dataset_id == "starvector/svg-stack"
    assert datasets["long_tail_ppl_runnable/web_markup_image_text/svg_stack_val"].split == "val"
    assert datasets["long_tail_ppl_runnable/web_markup_image_text/svg_stack_val"].text_key == "Svg"
    assert datasets["long_tail_ppl_runnable/formal_hardware/verilogeval_prompt"].hf_dataset_id == (
        "dakies/nvlabs-verilogeval"
    )
    assert datasets["long_tail_ppl_runnable/formal_hardware/verilogeval_prompt"].split == "test"
    assert datasets["long_tail_ppl_runnable/formal_hardware/verilogeval_prompt"].text_key == "prompt"
    assert {slice_.family for slice_ in runnable_long_tail_ppl_slices()} == {
        LongTailPplFamily.WEB_MARKUP_IMAGE_TEXT,
        LongTailPplFamily.FORMAL_HARDWARE,
    }


def test_smoke_wrapper_uses_tiny_doc_cap_and_runnable_bundle():
    assert MAX_DOCS_PER_DATASET == 32
    assert set(DATASETS) == set(runnable_long_tail_raw_validation_sets())
    assert STEP.config.max_docs_per_dataset == 32
    assert STEP.config.datasets == DATASETS


def test_hf_backed_raw_dataset_preserves_requested_split():
    dataset = raw_text_dataset(HfDatasetSpec(id="example/dataset"), text_key="body", split="test")

    component = _to_dataset_component(dataset)

    assert component.split == "test"
    assert component.format.text_key == "body"
    assert isinstance(component.source, HfDatasetSourceConfig)
    assert component.source.splits == ["test"]


def test_file_backed_raw_dataset_rejects_non_validation_split():
    with pytest.raises(ValueError, match="Hugging Face dataset sources"):
        raw_text_dataset("gs://example-bucket/eval.jsonl", split="test")
