# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bio/chem notation record splitters.

The headline assertion in every test is "concatenating the splitter output
reproduces the original byte sequence" — that is the definition of a
format-preserving splitter.
"""

from __future__ import annotations

from marin.datakit.download.bio_chem.uniprot import iter_uniprot_dat_records
from marin.transform.bio_chem.splitters import (
    SamplingCap,
    iter_fasta_records,
    iter_gff_blocks,
    iter_mmcif_blocks,
    iter_sdf_records,
    iter_smiles_records,
    pack_records_into_docs,
    take_until_cap,
)


def _lines(s: str):
    """Mimic file iteration: yield each line including its trailing newline."""
    if not s:
        return iter([])
    parts = s.splitlines(keepends=True)
    return iter(parts)


def test_fasta_splitter_preserves_bytes():
    fasta = (
        ">NC_001416.1 Enterobacteria phage lambda, complete genome\n"
        "GGGCGGCGACCTCGCGGGTTTTCGCTATTTATGAAAATTTTCCGGTTTAAGGCGTTTCCG\n"
        "TTCTTCTTCG\n"
        ">NC_002195.1 Salmonella phage phiSE6 complete genome\n"
        "ATGGCAAGGTAGCAGCAGCAGCAGCAGCAGCAGGGTGCAGGTGAGCAGCAGGTAGCAGGAG\n"
    )
    records = list(iter_fasta_records(_lines(fasta)))
    assert len(records) == 2
    assert records[0].startswith(">NC_001416.1")
    assert records[1].startswith(">NC_002195.1")
    assert "".join(records) == fasta


def test_fasta_splitter_drops_pre_header_text_only():
    fasta = "; some preamble comment\n>HEAD1\nACGT\n>HEAD2\nGGGG\n"
    records = list(iter_fasta_records(_lines(fasta)))
    assert len(records) == 2
    # Pre-header line is not part of any record and is not preserved.
    assert "".join(records) == ">HEAD1\nACGT\n>HEAD2\nGGGG\n"


def test_gff_splitter_groups_by_seqid_and_keeps_directives():
    gff = (
        "##gff-version 3\n"
        "##sequence-region NC_001416.1 1 48502\n"
        "NC_001416.1\tRefSeq\tgene\t191\t736\t.\t+\t.\tID=gene-lambdap01\n"
        "NC_001416.1\tRefSeq\tCDS\t191\t736\t.\t+\t0\tParent=gene-lambdap01\n"
        "NC_002195.1\tRefSeq\tgene\t1\t720\t.\t+\t.\tID=gene-phiSE6p01\n"
        "###\n"
    )
    blocks = list(iter_gff_blocks(_lines(gff)))
    # Two blocks (one per seqid), then the ###-attached final block becomes
    # part of the last block.
    assert len(blocks) == 2
    assert blocks[0].startswith("##gff-version 3\n")
    assert "NC_001416.1\tRefSeq\tgene" in blocks[0]
    assert "NC_001416.1\tRefSeq\tCDS" in blocks[0]
    assert blocks[1].startswith("NC_002195.1\tRefSeq\tgene")
    assert blocks[1].endswith("###\n")
    # Tab columns are preserved verbatim.
    assert "\t" in blocks[0]


def test_smiles_splitter_keeps_full_lines_and_skips_comments():
    raw = (
        "# header from CID-SMILES\n"
        + "1\tCC(=O)NC1=CC=C(O)C=C1\n"
        + "2\tCC(=O)Oc1ccccc1C(=O)O\n"
        + "\n"
        + "3\tC1=CC=CC=C1\n"
    )
    records = list(iter_smiles_records(_lines(raw)))
    assert records == [
        "1\tCC(=O)NC1=CC=C(O)C=C1\n",
        "2\tCC(=O)Oc1ccccc1C(=O)O\n",
        "3\tC1=CC=CC=C1\n",
    ]


def test_sdf_splitter_keeps_terminator_and_roundtrips():
    sdf = (
        "Methane\n"
        "  Mrv2014 01012023\n"
        "\n"
        "  1  0  0  0  0  0            999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "M  END\n"
        ">  <PUBCHEM_COMPOUND_CID>\n"
        "297\n"
        "\n"
        "$$$$\n"
        "Ethane\n"
        "  Mrv2014 01012023\n"
        "\n"
        "  2  1  0  0  0  0            999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n"
        "  1  2  1  0  0  0  0\n"
        "M  END\n"
        ">  <PUBCHEM_COMPOUND_CID>\n"
        "6324\n"
        "\n"
        "$$$$\n"
    )
    records = list(iter_sdf_records(_lines(sdf)))
    assert len(records) == 2
    assert records[0].startswith("Methane\n")
    assert records[0].rstrip("\n").endswith("$$$$")
    assert records[1].startswith("Ethane\n")
    assert "".join(records) == sdf


def test_mmcif_splitter_preserves_blocks():
    cif = (
        "# preamble comment that is not in any block\n"
        "data_1ABC\n"
        "_entry.id   1ABC\n"
        "loop_\n"
        "_atom_site.id\n"
        "_atom_site.type_symbol\n"
        "1 N\n"
        "2 C\n"
        "data_2DEF\n"
        "_entry.id   2DEF\n"
        "loop_\n"
        "_atom_site.id\n"
        "1 O\n"
    )
    blocks = list(iter_mmcif_blocks(_lines(cif)))
    assert len(blocks) == 2
    assert blocks[0].startswith("data_1ABC\n")
    assert blocks[1].startswith("data_2DEF\n")
    # loop_ structure preserved exactly.
    assert "loop_\n_atom_site.id\n_atom_site.type_symbol\n" in blocks[0]
    # Pre-block preamble is dropped, but everything from the first data_ line
    # onward round-trips.
    assert "".join(blocks) == cif[cif.index("data_1ABC") :]


def test_uniprot_dat_splitter_preserves_entries():
    dat = (
        "ID   001R_FRG3G              Reviewed;         256 AA.\n"
        "AC   Q6GZX4;\n"
        "DE   RecName: Full=Uncharacterized protein 001R;\n"
        "SQ   SEQUENCE   256 AA;  29735 MW;  B4840739BF7D4121 CRC64;\n"
        "     MAFSAEDVLK EYDRRRRMEA LLLSLYYPND RKLLDYKEWS PPRVQVECPK APVEWNNPPS\n"
        "//\n"
        "ID   002L_FRG3G              Reviewed;         320 AA.\n"
        "AC   Q6GZX3;\n"
        "DE   RecName: Full=Uncharacterized protein 002L;\n"
        "SQ   SEQUENCE   320 AA;  34642 MW;  ABC0123456789ABC CRC64;\n"
        "     MSLEALEDFD GTAQLDGKSV LAAVEKMLNG TEINIYDEIN VQSGGVSLKL EVNGFQSNTV\n"
        "//\n"
    )
    records = list(iter_uniprot_dat_records(_lines(dat)))
    assert len(records) == 2
    assert records[0].startswith("ID   001R_FRG3G")
    assert records[0].rstrip("\n").endswith("//")
    assert "".join(records) == dat


def test_pack_records_caps_by_chars():
    records = ["abc\n", "defg\n", "h\n", "ijklmnop\n"]
    docs = list(pack_records_into_docs(records, target_doc_chars=6, max_records_per_doc=10))
    # abc(4) + defg(5) = 9 ≥ 6 → flush "abc\ndefg\n". h(2) + ijklmnop(9) = 11 ≥ 6
    # → flush "h\nijklmnop\n". Each doc concatenates without separator.
    assert docs == ["abc\ndefg\n", "h\nijklmnop\n"]


def test_pack_records_caps_by_record_count():
    records = ["a", "b", "c", "d"]
    docs = list(pack_records_into_docs(records, target_doc_chars=10_000, max_records_per_doc=2))
    assert docs == ["ab", "cd"]


def test_pack_records_uses_separator_between_records():
    records = ["row1", "row2", "row3"]
    docs = list(pack_records_into_docs(records, target_doc_chars=10_000, max_records_per_doc=10, record_separator="\n"))
    assert docs == ["row1\nrow2\nrow3"]


def test_take_until_cap_stops_at_record_count():
    cap = SamplingCap(max_records=2, max_bytes=10_000)
    out = list(take_until_cap(iter(["a", "b", "c"]), cap))
    assert out == ["a", "b"]


def test_take_until_cap_stops_at_byte_budget():
    cap = SamplingCap(max_records=10, max_bytes=4)
    out = list(take_until_cap(iter(["abc", "de", "fgh"]), cap))
    # After "abc" (3 bytes) we are still under 4; after "de" (2 more bytes) we
    # hit 5 which exceeds the cap, so the next iteration stops.
    assert out == ["abc", "de"]
