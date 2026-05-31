# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HF-backed identifier grammar and encoded-text PPL validation slices."""

from __future__ import annotations

import argparse
import base64
import json
import posixpath
import random
import string
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import quote

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, supervised_text_dataset
from marin.processing.tokenize import HfDatasetSpec

EPIC_5005 = 5005
IDENTIFIER_ENCODING_ISSUE = 5618
IDENTIFIER_ENCODING_HF_DATASET_ID = "marin-community/synth-identifier-encoding-ppl"
IDENTIFIER_ENCODING_SOURCE = "generated_identifier_encoding_ppl_v1"
IDENTIFIER_ENCODING_HF_REVISION = "8e12c75bb1c9f4fed228b99d22f636ca426cf9b4"
IDENTIFIER_ENCODING_SEED = 5615
EXAMPLES_PER_CONFIG = 1000


class IdentifierEncodingFamily(StrEnum):
    IDENTIFIER_GRAMMARS = "identifier_grammars"
    ENCODED_TEXT = "encoded_text"
    ESCAPED_TEXT = "escaped_text"


@dataclass(frozen=True)
class IdentifierEncodingPplSlice:
    family: IdentifierEncodingFamily
    task_name: str
    hf_config_name: str

    @property
    def registry_key(self) -> str:
        return posixpath.join("synthetic_identifier_encoding_ppl", self.family.value, self.task_name)

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            "synthetic_identifier_encoding_ppl",
            f"epic:{EPIC_5005}",
            f"issue:{IDENTIFIER_ENCODING_ISSUE}",
            f"family:{self.family.value}",
            f"task:{self.task_name}",
            f"seed:{IDENTIFIER_ENCODING_SEED}",
            f"examples:{EXAMPLES_PER_CONFIG}",
            f"source:{IDENTIFIER_ENCODING_SOURCE}",
            f"hf_revision:{IDENTIFIER_ENCODING_HF_REVISION}",
            "loss:target_only",
        )

    def to_raw_text_dataset(self, *, hf_dataset_id: str) -> RawTextEvaluationDataset:
        return supervised_text_dataset(
            HfDatasetSpec(
                id=hf_dataset_id,
                name=self.hf_config_name,
                revision=IDENTIFIER_ENCODING_HF_REVISION,
            ),
            input_key="input",
            target_key="target",
            split="validation",
            tags=self.tags,
        )


IDENTIFIER_ENCODING_PPL_SLICES: tuple[IdentifierEncodingPplSlice, ...] = (
    IdentifierEncodingPplSlice(
        family=IdentifierEncodingFamily.IDENTIFIER_GRAMMARS,
        task_name="package_names_versions",
        hf_config_name="package_names_versions",
    ),
    IdentifierEncodingPplSlice(
        family=IdentifierEncodingFamily.IDENTIFIER_GRAMMARS,
        task_name="commit_hashes",
        hf_config_name="commit_hashes",
    ),
    IdentifierEncodingPplSlice(
        family=IdentifierEncodingFamily.IDENTIFIER_GRAMMARS,
        task_name="uuid_build_ids",
        hf_config_name="uuid_build_ids",
    ),
    IdentifierEncodingPplSlice(
        family=IdentifierEncodingFamily.IDENTIFIER_GRAMMARS,
        task_name="bio_accessions",
        hf_config_name="bio_accessions",
    ),
    IdentifierEncodingPplSlice(
        family=IdentifierEncodingFamily.IDENTIFIER_GRAMMARS,
        task_name="mixed_case_symbolic_identifiers",
        hf_config_name="mixed_case_symbolic_identifiers",
    ),
    IdentifierEncodingPplSlice(
        family=IdentifierEncodingFamily.ENCODED_TEXT,
        task_name="base64_continuation",
        hf_config_name="base64_continuation",
    ),
    IdentifierEncodingPplSlice(
        family=IdentifierEncodingFamily.ENCODED_TEXT,
        task_name="data_image_base64_prefixes",
        hf_config_name="data_image_base64_prefixes",
    ),
    IdentifierEncodingPplSlice(
        family=IdentifierEncodingFamily.ENCODED_TEXT,
        task_name="hex_dump_continuation",
        hf_config_name="hex_dump_continuation",
    ),
    IdentifierEncodingPplSlice(
        family=IdentifierEncodingFamily.ESCAPED_TEXT,
        task_name="url_query_escaping",
        hf_config_name="url_query_escaping",
    ),
    IdentifierEncodingPplSlice(
        family=IdentifierEncodingFamily.ESCAPED_TEXT,
        task_name="json_unicode_byte_escapes",
        hf_config_name="json_unicode_byte_escapes",
    ),
)


def identifier_encoding_raw_validation_sets(
    *,
    hf_dataset_id: str = IDENTIFIER_ENCODING_HF_DATASET_ID,
) -> dict[str, RawTextEvaluationDataset]:
    return {
        slice_.registry_key: slice_.to_raw_text_dataset(hf_dataset_id=hf_dataset_id)
        for slice_ in IDENTIFIER_ENCODING_PPL_SLICES
    }


def generate_identifier_encoding_rows(
    slice_: IdentifierEncodingPplSlice,
    *,
    examples: int = EXAMPLES_PER_CONFIG,
    seed: int = IDENTIFIER_ENCODING_SEED,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    slice_offset = IDENTIFIER_ENCODING_PPL_SLICES.index(slice_) * 1_000_003
    for row_index in range(examples):
        rng = random.Random(seed + slice_offset + row_index)
        input_text, target, metadata = _example_for_task(slice_.task_name, rng)
        rows.append(
            {
                "input": input_text,
                "target": target,
                "id": f"{slice_.hf_config_name}-{row_index:06d}",
                "subset": slice_.hf_config_name,
                "task": slice_.task_name,
                "seed": seed,
                "metadata": {
                    "family": slice_.family.value,
                    "row_index": row_index,
                    **metadata,
                },
            }
        )
    return rows


def write_identifier_encoding_jsonl_configs(
    output_dir: str | Path,
    *,
    examples: int = EXAMPLES_PER_CONFIG,
    seed: int = IDENTIFIER_ENCODING_SEED,
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for slice_ in IDENTIFIER_ENCODING_PPL_SLICES:
        output_file = path / f"{slice_.hf_config_name}.jsonl"
        with output_file.open("w", encoding="utf-8") as handle:
            for row in generate_identifier_encoding_rows(slice_, examples=examples, seed=seed):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def generate_tiny_identifier_encoding_sample(*, seed: int = IDENTIFIER_ENCODING_SEED) -> list[dict[str, Any]]:
    """Generate one schema-valid sample row per slice for local inspection."""
    rows: list[dict[str, Any]] = []
    for slice_ in IDENTIFIER_ENCODING_PPL_SLICES:
        rows.extend(generate_identifier_encoding_rows(slice_, examples=1, seed=seed))
    return rows


def write_tiny_identifier_encoding_sample(
    output_path: str | Path = Path(__file__).with_name("synthetic_identifier_encoding_ppl_sample.jsonl"),
    *,
    seed: int = IDENTIFIER_ENCODING_SEED,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in generate_tiny_identifier_encoding_sample(seed=seed):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _example_for_task(task_name: str, rng: random.Random) -> tuple[str, str, dict[str, Any]]:
    if task_name == "package_names_versions":
        scope = rng.choice(("@marin", "@levanter", "@haliax"))
        package = f"{scope}/{_kebab_identifier(rng)}"
        version = f"{rng.randint(0, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 30)}"
        return (
            f'"node_modules/{scope}/": ' + '{"resolved":"https://registry.npmjs.org/',
            f"{package}/-/{package.split('/')[-1]}-{version}.tgz\"}}\n",
            {"grammar": "npm_package_tarball", "version": version},
        )

    if task_name == "commit_hashes":
        prefix = _hex(rng, 12)
        suffix = _hex(rng, 28)
        return (
            f'source_commit = "{prefix}',
            f'{suffix}"\n',
            {"grammar": "sha1_hex", "prefix_length": len(prefix)},
        )

    if task_name == "uuid_build_ids":
        value = str(uuid.UUID(int=rng.getrandbits(128)))
        cutoff = rng.choice((8, 13, 18))
        return (
            f'org.opencontainers.image.revision="{value[:cutoff]}',
            f'{value[cutoff:]}"\n',
            {"grammar": "uuid4_like", "prefix_length": cutoff},
        )

    if task_name == "bio_accessions":
        prefix = rng.choice(("NM_", "XM_", "XP_", "SRR", "ERR", "GCA_"))
        digits = "".join(rng.choice(string.digits) for _ in range(9))
        accession = f"{prefix}{digits}.{rng.randint(1, 9)}"
        cutoff = rng.randint(3, min(8, len(accession) - 2))
        return (
            f"sample_id\torganism\taccession\nS{rng.randint(100, 999)}\tH_sapiens\t{accession[:cutoff]}",
            f"{accession[cutoff:]}\n",
            {"grammar": "bio_accession", "accession_prefix": prefix},
        )

    if task_name == "mixed_case_symbolic_identifiers":
        identifier = f"{_camel_identifier(rng)}::{_snake_identifier(rng)}::{_constant_identifier(rng)}"
        cutoff = rng.randint(8, len(identifier) - 4)
        return (
            f"symbol={identifier[:cutoff]}",
            f"{identifier[cutoff:]}\n",
            {"grammar": "mixed_case_symbol_path", "prefix_length": cutoff},
        )

    if task_name == "base64_continuation":
        payload = bytes(rng.randrange(256) for _ in range(48))
        encoded = base64.b64encode(payload).decode("ascii")
        cutoff = 24
        return (
            f'payload_b64="{encoded[:cutoff]}',
            f'{encoded[cutoff:]}"\n',
            {"grammar": "base64", "payload_bytes": len(payload)},
        )

    if task_name == "data_image_base64_prefixes":
        mime = rng.choice(("png", "jpeg", "gif", "webp"))
        header = f"data:image/{mime};base64,"
        encoded = base64.b64encode(bytes(rng.randrange(256) for _ in range(36))).decode("ascii")
        return (
            f'<img alt="sparkline" src="{header}',
            f'{encoded}">\n',
            {"grammar": "data_uri_image_base64", "mime": f"image/{mime}"},
        )

    if task_name == "hex_dump_continuation":
        offset = rng.randrange(0x1000, 0x9000, 16)
        byte_values = [rng.randrange(256) for _ in range(16)]
        left = " ".join(f"{value:02x}" for value in byte_values[:8])
        right = " ".join(f"{value:02x}" for value in byte_values[8:])
        return (
            f"{offset:08x}  {left}  ",
            f"{right}  |{_ascii_gutter(byte_values)}|\n",
            {"grammar": "hexdump", "offset": offset},
        )

    if task_name == "url_query_escaping":
        raw_value = rng.choice(("alpha beta/gamma", 'json:{"x":1}', "email=a+b@example.com"))
        escaped = quote(raw_value, safe="")
        cutoff = rng.randint(6, len(escaped) - 3)
        return (
            f"https://example.invalid/search?q={escaped[:cutoff]}",
            f"{escaped[cutoff:]}&src=eval\n",
            {"grammar": "percent_encoded_query", "raw_value": raw_value},
        )

    if task_name == "json_unicode_byte_escapes":
        escaped = "".join(f"\\u{rng.randrange(0x20, 0x7F):04x}" for _ in range(6))
        byte_escapes = "".join(f"\\x{rng.randrange(256):02x}" for _ in range(4))
        value = f"{escaped}{byte_escapes}"
        cutoff = rng.randrange(6, len(value) - 4, 2)
        return (
            '{"escaped":"' f"{value[:cutoff]}",
            f'{value[cutoff:]}"}}\n',
            {"grammar": "json_unicode_and_byte_escapes", "prefix_length": cutoff},
        )

    raise ValueError(f"Unknown identifier encoding task: {task_name}")


def _hex(rng: random.Random, n_chars: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n_chars))


def _kebab_identifier(rng: random.Random) -> str:
    return "-".join(_word(rng).lower() for _ in range(rng.randint(2, 4)))


def _snake_identifier(rng: random.Random) -> str:
    return "_".join(_word(rng).lower() for _ in range(rng.randint(2, 3)))


def _constant_identifier(rng: random.Random) -> str:
    return "_".join(_word(rng).upper() for _ in range(rng.randint(2, 3)))


def _camel_identifier(rng: random.Random) -> str:
    words = [_word(rng) for _ in range(rng.randint(2, 4))]
    return words[0].lower() + "".join(word.title() for word in words[1:])


def _word(rng: random.Random) -> str:
    alphabet = string.ascii_lowercase
    return rng.choice(alphabet) + "".join(rng.choice(alphabet + string.digits) for _ in range(rng.randint(3, 7)))


def _ascii_gutter(byte_values: Iterable[int]) -> str:
    return "".join(chr(value) if 32 <= value <= 126 else "." for value in byte_values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic identifier/encoding PPL JSONL configs.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--examples", type=int, default=EXAMPLES_PER_CONFIG)
    args = parser.parse_args()
    if args.output_dir is None:
        print(write_tiny_identifier_encoding_sample())
    else:
        print(write_identifier_encoding_jsonl_configs(args.output_dir, examples=args.examples))


if __name__ == "__main__":
    main()
