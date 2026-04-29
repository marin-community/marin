# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.bio_chem_notation import BIO_CHEM_SLICES, bio_chem_raw_validation_sets

EXPECTED_KEYS = {
    "bio_chem/refseq/refseq_viral_fasta",
    "bio_chem/refseq/refseq_viral_gff",
    "bio_chem/rnacentral/rnacentral_active_fasta",
    "bio_chem/uniprot/uniprot_sprot_fasta",
    "bio_chem/uniprot/uniprot_sprot_dat",
    "bio_chem/pubchem/pubchem_cid_smiles",
    "bio_chem/pubchem/pubchem_sdf",
    "bio_chem/rcsb/rcsb_mmcif",
    "bio_chem/chembl/chembl_chemreps",
    "bio_chem/chembl/chembl_sdf",
    "bio_chem/moleculenet/moleculenet_esol_smiles",
    "bio_chem/moleculenet/moleculenet_clintox_smiles",
}


def test_bio_chem_raw_validation_sets_are_opt_in_and_deterministic():
    datasets = bio_chem_raw_validation_sets()

    assert set(datasets) == EXPECTED_KEYS
    assert len(BIO_CHEM_SLICES) == len(EXPECTED_KEYS)
    assert all(dataset.text_key == "text" for dataset in datasets.values())
    assert all(dataset.input_path is not None for dataset in datasets.values())


def test_default_marin_vs_llama_gap_script_does_not_include_bio_chem_slices():
    from experiments.exp_model_perplexity_gap_marin_vs_llama import DATASETS

    dataset_names = DATASETS.keys()
    assert all(not name.startswith("bio_chem/") for name in dataset_names)


def test_bio_chem_gap_script_only_includes_bio_chem_slices():
    from experiments.exp_model_perplexity_gap_bio_chem_marin_vs_llama import DATASETS

    assert set(DATASETS) == EXPECTED_KEYS
