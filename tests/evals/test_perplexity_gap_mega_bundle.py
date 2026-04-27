# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.perplexity_gap_mega_bundle import mega_available_raw_validation_sets
from experiments.structured_evals import structured_evals_raw_validation_sets


def test_structured_evals_raw_validation_sets_register_expected_slices() -> None:
    datasets = structured_evals_raw_validation_sets()

    assert "structured_text/totto" in datasets
    assert "structured_text/wikitablequestions" in datasets
    assert datasets["structured_text/totto"].input_path.name == "staged.jsonl.gz"
    assert datasets["structured_text/totto"].tags == ("structured_text", "issue:5059", "totto")


def test_mega_available_bundle_covers_representative_families() -> None:
    datasets = mega_available_raw_validation_sets()

    expected_keys = {
        "chat/wildchat",
        "fineweb2_multilingual/deu_Latn",
        "raw_web_markup/svg_stack/svg_xml_val",
        "binary_network_security/uwf_zeek",
        "bio_chem/refseq/refseq_viral_fasta",
        "bio_chem/uniprot/uniprot_sprot_dat",
        "formal_methods/smt_lib",
        "synthetic_reasoning_ppl/stepmath/arithmetic/oai_chat_symbolic",
        "paired_robustness_ppl/paraphrase/paws_labeled_final/validation/target_given_source",
        "asr_ocr_noisy_ppl/hypr_librispeech_without_lm_test_clean/noisy",
        "gh_archive_structured_output/PushEvent",
    }

    missing = expected_keys.difference(datasets)
    assert not missing, f"mega bundle is missing expected dataset keys: {sorted(missing)}"
    assert "agent_traces/openhands_swe_rebench" not in datasets
    assert "bio_chem/refseq/refseq_viral_gff" not in datasets
    assert "bio_chem/rcsb/rcsb_mmcif" not in datasets
    assert "raw_web_markup/textocr/ocr_strings" not in datasets
    assert "paired_robustness_ppl/translation/flores_eng_deu/dev/target_given_source" not in datasets
