# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registry for the first-pass long-tail diagnostic PPL slices.

This module is intentionally metadata-first. It records the initial family/source
coverage plan from epic #5005 and child issues #5056-#5062 without downloading or
mirroring any of the large corpora referenced there.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass
from enum import StrEnum

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset

EPIC_5005 = 5005
WEB_RAW_ISSUE = 5056
BINARY_RAW_ISSUE = 5057
BIO_CHEM_ISSUE = 5058
TIME_SERIES_ISSUE = 5059
FORMAL_HARDWARE_ISSUE = 5060
PACKAGE_METADATA_ISSUE = 5061
GAME_MUSIC_ISSUE = 5062


class LongTailPplFamily(StrEnum):
    """High-level coverage families for the long-tail PPL registry."""

    WEB_MARKUP_IMAGE_TEXT = "web_markup_image_text"
    BINARY_NETWORK_SECURITY = "binary_network_security"
    BIO_CHEM = "bio_chem"
    TIME_SERIES_TABLE_GEO = "time_series_table_geo"
    FORMAL_HARDWARE = "formal_hardware"
    PACKAGE_METADATA = "package_metadata"
    GAME_MUSIC = "game_music"


@dataclass(frozen=True)
class LongTailPplSlice:
    """A single diagnostic slice in the long-tail PPL registry."""

    name: str
    family: LongTailPplFamily
    issue_number: int
    source_url: str
    surface_form: str
    raw_relative_path: str
    notes: str = ""

    @property
    def registry_key(self) -> str:
        return posixpath.join("long_tail_ppl", self.family.value, self.name)

    @property
    def tags(self) -> tuple[str, ...]:
        return ("long_tail_ppl", f"epic:{EPIC_5005}", f"issue:{self.issue_number}", self.family.value)

    def to_raw_text_dataset(self, raw_root: str) -> RawTextEvaluationDataset:
        """Render the slice as a raw-text eval dataset rooted at ``raw_root``."""

        return raw_text_dataset(posixpath.join(raw_root, self.raw_relative_path), tags=self.tags)


def _slice(
    *,
    name: str,
    family: LongTailPplFamily,
    issue_number: int,
    source_url: str,
    surface_form: str,
    raw_relative_path: str,
    notes: str = "",
) -> LongTailPplSlice:
    return LongTailPplSlice(
        name=name,
        family=family,
        issue_number=issue_number,
        source_url=source_url,
        surface_form=surface_form,
        raw_relative_path=raw_relative_path,
        notes=notes,
    )


LONG_TAIL_PPL_SLICES: tuple[LongTailPplSlice, ...] = (
    # Web / markup / image-text
    _slice(
        name="common_crawl_warc",
        family=LongTailPplFamily.WEB_MARKUP_IMAGE_TEXT,
        issue_number=WEB_RAW_ISSUE,
        source_url="https://commoncrawl.org/get-started",
        surface_form="warc",
        raw_relative_path="web/common_crawl/warc.jsonl.gz",
        notes="Keep WARC headers, HTTP metadata, URLs, and raw response bodies.",
    ),
    _slice(
        name="common_crawl_wat",
        family=LongTailPplFamily.WEB_MARKUP_IMAGE_TEXT,
        issue_number=WEB_RAW_ISSUE,
        source_url="https://commoncrawl.org/get-started",
        surface_form="wat_json",
        raw_relative_path="web/common_crawl/wat.jsonl.gz",
        notes="Keep WAT JSON and extracted text without cleaning away structure.",
    ),
    _slice(
        name="web_data_commons_web_tables",
        family=LongTailPplFamily.WEB_MARKUP_IMAGE_TEXT,
        issue_number=WEB_RAW_ISSUE,
        source_url="https://webdatacommons.org/webtables/englishTables.html",
        surface_form="html_tables",
        raw_relative_path="web/web_data_commons/web_tables.jsonl.gz",
        notes="Preserve HTML tables, delimiters, and table metadata.",
    ),
    _slice(
        name="svg_stack",
        family=LongTailPplFamily.WEB_MARKUP_IMAGE_TEXT,
        issue_number=WEB_RAW_ISSUE,
        source_url="https://huggingface.co/datasets/starvector/svg-stack",
        surface_form="svg_xml",
        raw_relative_path="web/svg_stack/svg_stack.jsonl.gz",
        notes="Keep SVG XML, path data, and caption text intact.",
    ),
    _slice(
        name="textocr",
        family=LongTailPplFamily.WEB_MARKUP_IMAGE_TEXT,
        issue_number=WEB_RAW_ISSUE,
        source_url="https://textvqa.org/textocr/",
        surface_form="ocr_text",
        raw_relative_path="web/textocr/textocr.jsonl.gz",
        notes="Preserve OCR strings, annotations, and layout hints.",
    ),
    _slice(
        name="ocr_vqa",
        family=LongTailPplFamily.WEB_MARKUP_IMAGE_TEXT,
        issue_number=WEB_RAW_ISSUE,
        source_url="https://ocr-vqa.github.io/",
        surface_form="ocr_question_context",
        raw_relative_path="web/ocr_vqa/ocr_vqa.jsonl.gz",
        notes="Keep book-cover OCR text and question context surface forms.",
    ),
    _slice(
        name="laion_metadata",
        family=LongTailPplFamily.WEB_MARKUP_IMAGE_TEXT,
        issue_number=WEB_RAW_ISSUE,
        source_url="https://laion.ai/blog/laion-400-open-dataset/",
        surface_form="url_alt_text_metadata",
        raw_relative_path="web/laion_metadata/laion_metadata.jsonl.gz",
        notes="Treat as metadata only; later pipeline work should sample conservatively.",
    ),
    # Binary / network / security
    _slice(
        name="microsoft_malware_bytes",
        family=LongTailPplFamily.BINARY_NETWORK_SECURITY,
        issue_number=BINARY_RAW_ISSUE,
        source_url="https://www.kaggle.com/c/malware-classification",
        surface_form="hex_dump",
        raw_relative_path="binary/microsoft_malware/bytes.jsonl.gz",
        notes="Render binary as hex text only; preserve line breaks and offsets.",
    ),
    _slice(
        name="microsoft_malware_asm",
        family=LongTailPplFamily.BINARY_NETWORK_SECURITY,
        issue_number=BINARY_RAW_ISSUE,
        source_url="https://www.kaggle.com/c/malware-classification",
        surface_form="disassembly_text",
        raw_relative_path="binary/microsoft_malware/asm.jsonl.gz",
        notes="Keep assembler syntax, labels, comments, and identifier casing.",
    ),
    _slice(
        name="wireshark_rendered_text",
        family=LongTailPplFamily.BINARY_NETWORK_SECURITY,
        issue_number=BINARY_RAW_ISSUE,
        source_url="https://wiki.wireshark.org/SampleCaptures",
        surface_form="protocol_tree_text",
        raw_relative_path="binary/wireshark/rendered_text.jsonl.gz",
        notes="Use rendered protocol trees / hex views instead of raw PCAPs.",
    ),
    _slice(
        name="mawi_zeek",
        family=LongTailPplFamily.BINARY_NETWORK_SECURITY,
        issue_number=BINARY_RAW_ISSUE,
        source_url="https://mawi.wide.ad.jp/mawi/",
        surface_form="network_flow_records",
        raw_relative_path="binary/mawi/zeek.jsonl.gz",
        notes="Keep flow records and timestamp / IP / port fields literal.",
    ),
    _slice(
        name="cicids_flow",
        family=LongTailPplFamily.BINARY_NETWORK_SECURITY,
        issue_number=BINARY_RAW_ISSUE,
        source_url="https://www.unb.ca/cic/datasets/ids-2017.html",
        surface_form="flow_csv",
        raw_relative_path="binary/cicids/flow.csv.jsonl.gz",
        notes="Preserve CSV delimiters, labels, and flow statistics.",
    ),
    _slice(
        name="uwf_zeek",
        family=LongTailPplFamily.BINARY_NETWORK_SECURITY,
        issue_number=BINARY_RAW_ISSUE,
        source_url="https://datasets.uwf.edu/",
        surface_form="zeek_logs",
        raw_relative_path="binary/uwf/zeek.jsonl.gz",
        notes="Preserve Zeek field names, hashes, IPs, and delimiter structure.",
    ),
    # Bio / chemistry
    _slice(
        name="refseq_fasta",
        family=LongTailPplFamily.BIO_CHEM,
        issue_number=BIO_CHEM_ISSUE,
        source_url="https://www.ncbi.nlm.nih.gov/refseq/",
        surface_form="fasta",
        raw_relative_path="bio/refseq/fasta.jsonl.gz",
        notes="Keep sequence IDs, wrapping, and nucleotide / amino-acid characters.",
    ),
    _slice(
        name="refseq_gff",
        family=LongTailPplFamily.BIO_CHEM,
        issue_number=BIO_CHEM_ISSUE,
        source_url="https://www.ncbi.nlm.nih.gov/refseq/",
        surface_form="gff",
        raw_relative_path="bio/refseq/gff.jsonl.gz",
        notes="Preserve coordinate columns, attributes, and record boundaries.",
    ),
    _slice(
        name="uniprot_fasta",
        family=LongTailPplFamily.BIO_CHEM,
        issue_number=BIO_CHEM_ISSUE,
        source_url="https://www.ebi.ac.uk/uniprot/download-center",
        surface_form="protein_fasta",
        raw_relative_path="bio/uniprot/fasta.jsonl.gz",
        notes="Keep UniProt headers and wrapped sequence bodies unchanged.",
    ),
    _slice(
        name="pubchem_smiles",
        family=LongTailPplFamily.BIO_CHEM,
        issue_number=BIO_CHEM_ISSUE,
        source_url="https://pubchem.ncbi.nlm.nih.gov/docs/downloads",
        surface_form="smiles",
        raw_relative_path="bio/pubchem/smiles.jsonl.gz",
        notes="Preserve atom / bond notation, stereochemistry markers, and IDs.",
    ),
    _slice(
        name="pubchem_sdf",
        family=LongTailPplFamily.BIO_CHEM,
        issue_number=BIO_CHEM_ISSUE,
        source_url="https://pubchem.ncbi.nlm.nih.gov/docs/downloads",
        surface_form="sdf",
        raw_relative_path="bio/pubchem/sdf.jsonl.gz",
        notes="Keep block separators, metadata fields, and record delimiters.",
    ),
    _slice(
        name="rcsb_mmcif",
        family=LongTailPplFamily.BIO_CHEM,
        issue_number=BIO_CHEM_ISSUE,
        source_url="https://www.rcsb.org/docs/programmatic-access/file-download-services",
        surface_form="mmcif",
        raw_relative_path="bio/rcsb/mmcif.jsonl.gz",
        notes="Preserve crystallographic tags, atom tables, and field punctuation.",
    ),
    # Time-series / tables / geo
    _slice(
        name="monash_tsf",
        family=LongTailPplFamily.TIME_SERIES_TABLE_GEO,
        issue_number=TIME_SERIES_ISSUE,
        source_url="https://forecastingdata.org/",
        surface_form="tsf",
        raw_relative_path="time_series/monash/tsf.jsonl.gz",
        notes="Preserve horizon metadata, units, missing markers, and line layout.",
    ),
    _slice(
        name="gittables_csv",
        family=LongTailPplFamily.TIME_SERIES_TABLE_GEO,
        issue_number=TIME_SERIES_ISSUE,
        source_url="https://gittables.github.io/",
        surface_form="csv_table",
        raw_relative_path="time_series/gittables/csv.jsonl.gz",
        notes="Keep CSV structure, headers, quoted cells, and cell delimiters.",
    ),
    _slice(
        name="web_data_commons_tables",
        family=LongTailPplFamily.TIME_SERIES_TABLE_GEO,
        issue_number=TIME_SERIES_ISSUE,
        source_url="https://webdatacommons.org/webtables/englishTables.html",
        surface_form="html_csv_json_tables",
        raw_relative_path="time_series/web_data_commons/tables.jsonl.gz",
        notes="Preserve extracted table text, HTML table context, and JSON metadata.",
    ),
    _slice(
        name="whos_on_first_geojson",
        family=LongTailPplFamily.TIME_SERIES_TABLE_GEO,
        issue_number=TIME_SERIES_ISSUE,
        source_url="https://whosonfirst.org/download/",
        surface_form="geojson",
        raw_relative_path="time_series/whos_on_first/geojson.jsonl.gz",
        notes="Keep feature IDs, coordinates, and nested metadata fields literal.",
    ),
    _slice(
        name="openstreetmap_extract",
        family=LongTailPplFamily.TIME_SERIES_TABLE_GEO,
        issue_number=TIME_SERIES_ISSUE,
        source_url="https://planet.openstreetmap.org/",
        surface_form="osm_text",
        raw_relative_path="time_series/openstreetmap/extract.jsonl.gz",
        notes="Use a small textual extract; preserve tags, nodes, and relation structure.",
    ),
    # Formal methods / hardware
    _slice(
        name="smtlib",
        family=LongTailPplFamily.FORMAL_HARDWARE,
        issue_number=FORMAL_HARDWARE_ISSUE,
        source_url="https://smt-lib.org/benchmarks.shtml",
        surface_form="smt2",
        raw_relative_path="formal/smtlib/smt2.jsonl.gz",
        notes="Preserve solver syntax, symbols, comments, and status markers.",
    ),
    _slice(
        name="tptp",
        family=LongTailPplFamily.FORMAL_HARDWARE,
        issue_number=FORMAL_HARDWARE_ISSUE,
        source_url="https://www.tptp.org/",
        surface_form="tptp",
        raw_relative_path="formal/tptp/tptp.jsonl.gz",
        notes="Keep theorem-proving problem syntax and generated identifiers intact.",
    ),
    _slice(
        name="coqgym",
        family=LongTailPplFamily.FORMAL_HARDWARE,
        issue_number=FORMAL_HARDWARE_ISSUE,
        source_url="https://github.com/princeton-vl/CoqGym",
        surface_form="coq_proof_script",
        raw_relative_path="formal/coqgym/coq.jsonl.gz",
        notes="Keep proof scripts and proof-state text together for later pipeline work.",
    ),
    _slice(
        name="dimacs_cnf",
        family=LongTailPplFamily.FORMAL_HARDWARE,
        issue_number=FORMAL_HARDWARE_ISSUE,
        source_url="https://satcompetition.github.io/2022/benchmarks.html",
        surface_form="cnf",
        raw_relative_path="formal/dimacs/cnf.jsonl.gz",
        notes="Preserve DIMACS headers, clauses, and comment lines.",
    ),
    _slice(
        name="verilogeval",
        family=LongTailPplFamily.FORMAL_HARDWARE,
        issue_number=FORMAL_HARDWARE_ISSUE,
        source_url="https://github.com/NVlabs/verilog-eval",
        surface_form="verilog",
        raw_relative_path="formal/verilogeval/verilog.jsonl.gz",
        notes="Keep module boundaries, long symbols, and hardware-description syntax.",
    ),
    _slice(
        name="rtl_repo",
        family=LongTailPplFamily.FORMAL_HARDWARE,
        issue_number=FORMAL_HARDWARE_ISSUE,
        source_url="https://github.com/AUCOHL/RTL-Repo",
        surface_form="verilog_repo_context",
        raw_relative_path="formal/rtl_repo/verilog.jsonl.gz",
        notes="Preserve repository-context completions and module-local identifiers.",
    ),
    _slice(
        name="rtl_coder",
        family=LongTailPplFamily.FORMAL_HARDWARE,
        issue_number=FORMAL_HARDWARE_ISSUE,
        source_url="https://github.com/hkust-zhiyao/RTL-Coder",
        surface_form="verilog_instruction_text",
        raw_relative_path="formal/rtl_coder/verilog.jsonl.gz",
        notes="Keep instruction text, generated code, and hardware tokens together.",
    ),
    _slice(
        name="hwmcc_aiger_btor",
        family=LongTailPplFamily.FORMAL_HARDWARE,
        issue_number=FORMAL_HARDWARE_ISSUE,
        source_url="https://fmv.jku.at/hwmcc11/benchmarks.html",
        surface_form="aiger_btor_text",
        raw_relative_path="formal/hwmcc/aiger_btor.jsonl.gz",
        notes="Use textual renderings only; preserve solver and model-checking syntax.",
    ),
    # Package metadata
    _slice(
        name="deps_dev",
        family=LongTailPplFamily.PACKAGE_METADATA,
        issue_number=PACKAGE_METADATA_ISSUE,
        source_url="https://docs.deps.dev/bigquery/v1/",
        surface_form="dependency_rows",
        raw_relative_path="packages/deps_dev/rows.jsonl.gz",
        notes="Preserve package names, semver constraints, hashes, and dependency edges.",
    ),
    _slice(
        name="ecosystem_ms_libraries_io",
        family=LongTailPplFamily.PACKAGE_METADATA,
        issue_number=PACKAGE_METADATA_ISSUE,
        source_url="https://repos.ecosyste.ms/open-data",
        surface_form="ecosystem_metadata",
        raw_relative_path="packages/ecosystems_ms/metadata.jsonl.gz",
        notes="Keep repository/package metadata, licenses, and release records literal.",
    ),
    _slice(
        name="npm_registry_metadata",
        family=LongTailPplFamily.PACKAGE_METADATA,
        issue_number=PACKAGE_METADATA_ISSUE,
        source_url="https://docs.npmjs.com/policies/crawlers/",
        surface_form="registry_json",
        raw_relative_path="packages/npm/registry.jsonl.gz",
        notes="Preserve CouchDB-style package JSON and nested version fields.",
    ),
    _slice(
        name="package_lock_corpora",
        family=LongTailPplFamily.PACKAGE_METADATA,
        issue_number=PACKAGE_METADATA_ISSUE,
        source_url="https://github.com/marin-community/marin/issues/4961",
        surface_form="lockfile",
        raw_relative_path="packages/package_lock/lockfiles.jsonl.gz",
        notes="Later pipeline work should keep lockfile structure, URLs, and checksums intact.",
    ),
    # Game / music
    _slice(
        name="lichess_pgn",
        family=LongTailPplFamily.GAME_MUSIC,
        issue_number=GAME_MUSIC_ISSUE,
        source_url="https://database.lichess.org/",
        surface_form="pgn",
        raw_relative_path="games/lichess/pgn.jsonl.gz",
        notes="Keep move text, headers, comments, and result markers.",
    ),
    _slice(
        name="kernscores_humdrum",
        family=LongTailPplFamily.GAME_MUSIC,
        issue_number=GAME_MUSIC_ISSUE,
        source_url="https://kern.ccarh.org/",
        surface_form="humdrum_kern",
        raw_relative_path="music/kernscores/humdrum.jsonl.gz",
        notes="Preserve **kern syntax, comments, and note boundaries.",
    ),
    _slice(
        name="abc_notation",
        family=LongTailPplFamily.GAME_MUSIC,
        issue_number=GAME_MUSIC_ISSUE,
        source_url="https://abcnotation.com/",
        surface_form="abc_notation",
        raw_relative_path="music/abc/notation.jsonl.gz",
        notes="Keep ABC headers, barlines, and note-length annotations.",
    ),
)

LONG_TAIL_PPL_REGISTRY: dict[str, LongTailPplSlice] = {slice_.registry_key: slice_ for slice_ in LONG_TAIL_PPL_SLICES}


def long_tail_ppl_slices(*, family: LongTailPplFamily | None = None) -> tuple[LongTailPplSlice, ...]:
    """Return all registered long-tail slices, optionally filtered by family."""

    if family is None:
        return LONG_TAIL_PPL_SLICES
    return tuple(slice_ for slice_ in LONG_TAIL_PPL_SLICES if slice_.family == family)


def long_tail_raw_validation_sets(
    raw_root: str = "raw/long_tail_ppl",
    *,
    family: LongTailPplFamily | None = None,
) -> dict[str, RawTextEvaluationDataset]:
    """Materialize the registry into raw-text evaluation datasets.

    The returned datasets point at deterministic paths under ``raw_root``. The
    registry itself does not download or mirror any corpus.
    """

    datasets: dict[str, RawTextEvaluationDataset] = {}
    for slice_ in long_tail_ppl_slices(family=family):
        datasets[slice_.registry_key] = slice_.to_raw_text_dataset(raw_root)
    return datasets


def render_long_tail_ppl_registry_markdown(*, family: LongTailPplFamily | None = None) -> str:
    """Render the registry as a compact markdown summary."""

    lines = ["# Long-tail PPL registry", ""]
    for current_family in LongTailPplFamily:
        if family is not None and current_family != family:
            continue

        family_slices = long_tail_ppl_slices(family=current_family)
        if not family_slices:
            continue

        lines.append(f"## {current_family.value}")
        for slice_ in family_slices:
            lines.append(
                f"- `{slice_.registry_key}`: #{slice_.issue_number} | {slice_.surface_form} | {slice_.source_url}"
            )
            if slice_.notes:
                lines.append(f"  - {slice_.notes}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
