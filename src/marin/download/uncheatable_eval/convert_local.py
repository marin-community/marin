import json
import logging
import os
import posixpath
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import fsspec

from marin.execution import ExecutorStep, VersionedValue, this_output_path
from marin.utils import fsspec_exists, fsspec_mkdirs

from src.marin.download.huggingface.download import DownloadConfig
from src.marin.download.huggingface.download_hf import download_hf

logger = logging.getLogger("convert")

FILENAME_PATTERN = re.compile(r"^(?P<benchmark>.+)_(?P<start>\d{8})to(?P<end>\d{8})(?P<suffix>(?:\.[^.]+)*)$")

TEXT_FIELD_CANDIDATES = (
    "text", "body", "content", "article", "document", "raw_text",
    "code", "message", "description", "story",
)
LIST_FIELD_CANDIDATES = ("paragraphs", "sentences", "lines", "messages")
ID_FIELD_CANDIDATES = ("id", "uuid", "guid", "doc_id", "document_id",
                       "article_id", "hash", "sha", "uid")


@dataclass(frozen=True)
class ConversionDataset:
    benchmark: str
    start_date: str
    end_date: str
    name: str
    input_path: str
    output_path: str

    @property
    def date_range(self) -> str:
        return f"{self.start_date}to{self.end_date}"

    @property
    def source_label(self) -> str:
        return f"{self.benchmark}:{self.date_range}"


@dataclass
class UncheatableEvalConvertConfig:
    input_path: str | VersionedValue[str]
    output_path: str | VersionedValue[str] = this_output_path()
    skip_existing: bool = True
    metadata_filename: str = "conversion_metadata.json"


def _discover_input_files(input_path: str) -> list[str]:
    if not fsspec_exists(input_path):
        raise ValueError(f"Input path does not exist: {input_path}")

    files = []
    fs = fsspec.filesystem(input_path.split("://")[0] if "://" in input_path else "file")
    for item in fs.listdir(input_path, detail=True):
        if item["type"] == "file":
            name = item["name"]
            base = os.path.basename(name)
            if base.endswith(".json") or base.endswith(".jsonl"):
                files.append(name)
    return sorted(files)


def _parse_filename(file_path: str) -> ConversionDataset | None:
    filename = os.path.basename(file_path)
    base_name = filename.split(".")[0]
    match = FILENAME_PATTERN.match(base_name)
    if not match:
        return None
    return ConversionDataset(
        benchmark=match.group("benchmark"),
        start_date=match.group("start"),
        end_date=match.group("end"),
        name=filename,
        input_path=file_path,
        output_path="",
    )


def _extract_id(raw: Any, dataset: ConversionDataset, index: int) -> str:
    if isinstance(raw, dict):
        for k in ID_FIELD_CANDIDATES:
            v = raw.get(k)
            if v:
                return str(v)
        meta = raw.get("metadata")
        if isinstance(meta, dict):
            for k in ID_FIELD_CANDIDATES:
                v = meta.get(k)
                if v:
                    return str(v)
    return f"{dataset.benchmark}_{dataset.date_range}_{index:06d}"


def _join_list_field(value: Any) -> str | None:
    if isinstance(value, list):
        text_items = [str(v) for v in value if v is not None]
        if text_items:
            return "\n".join(text_items)
    return None


def _extract_text(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for k in TEXT_FIELD_CANDIDATES:
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v
        for k in TEXT_FIELD_CANDIDATES:
            joined = _join_list_field(raw.get(k))
            if joined:
                return joined
        for k in LIST_FIELD_CANDIDATES:
            joined = _join_list_field(raw.get(k))
            if joined:
                return joined
        title, body = raw.get("title"), raw.get("body")
        if isinstance(title, str) and isinstance(body, str):
            return f"{title.strip()}\n\n{body.strip()}".strip()
        if isinstance(title, str) and title.strip():
            return title
        return json.dumps(raw, ensure_ascii=False)
    return str(raw)


def _normalize_record(raw: Any, dataset: ConversionDataset, index: int) -> dict[str, str]:
    text = _extract_text(raw)
    if not text or not str(text).strip():
        raise ValueError(f"Record {index} in {dataset.name} has no text")
    rid = _extract_id(raw, dataset, index)
    return {"id": rid, "text": text, "source": dataset.source_label}


def _generate_conversion_tasks(files: Iterable[str], out_path: str, skip: bool):
    tasks, datasets = [], []
    for f in files:
        ds = _parse_filename(f)
        if not ds:
            logger.warning("Skip unrecognized filename: %s", f)
            continue
        out_file = posixpath.join(out_path, f"{ds.benchmark}_{ds.date_range}.jsonl.gz")
        success_file = f"{out_file}.SUCCESS"
        if skip and fsspec_exists(success_file):
            logger.info("Skip %s: already exists", ds.name)
            continue
        ds = ConversionDataset(
            benchmark=ds.benchmark,
            start_date=ds.start_date,
            end_date=ds.end_date,
            name=ds.name,
            input_path=ds.input_path,
            output_path=out_file,
        )
        tasks.append((f, out_file, ds))
        datasets.append(ds)
    return tasks, datasets


def _write_metadata(cfg: UncheatableEvalConvertConfig, records: list[dict[str, Any]]):
    if not records:
        return
    path = posixpath.join(str(cfg.output_path), cfg.metadata_filename)
    with fsspec.open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info("Wrote metadata: %s", path)


def convert_uncheatable_eval_files(cfg: UncheatableEvalConvertConfig) -> dict[str, Any]:
    inp, out = str(cfg.input_path), str(cfg.output_path)
    logger.info("Convert from %s to %s", inp, out)

    try:
        files = _discover_input_files(inp)
    except Exception as e:
        return {"success": False, "reason": "discovery_failed", "error": str(e)}

    if not files:
        return {"success": False, "reason": "no_input_files"}

    fsspec_mkdirs(out, exist_ok=True)
    tasks, datasets = _generate_conversion_tasks(files, out, cfg.skip_existing)
    if not tasks:
        return {"success": True, "reason": "already_converted", "skipped": True}

    meta_records = []
    for f, out_file, ds in tasks:
        with fsspec.open(f, "r", encoding="utf-8") as infile:
            payload = json.load(infile)
        if not isinstance(payload, list):
            raise ValueError(f"{ds.name} not list but {type(payload).__name__}")
        fsspec_mkdirs(os.path.dirname(out_file), exist_ok=True)
        count = 0
        with fsspec.open(out_file, "wt", encoding="utf-8", compression="gzip") as outf:
            for i, raw in enumerate(payload):
                norm = _normalize_record(raw, ds, i)
                json.dump(norm, outf, ensure_ascii=False)
                outf.write("\n")
                count += 1
        meta_records.append({
            "benchmark": ds.benchmark,
            "start_date": ds.start_date,
            "end_date": ds.end_date,
            "source": ds.source_label,
            "input_file": ds.input_path,
            "output_file": ds.output_path,
            "records": count,
        })
        logger.info("Done %s: %d records", ds.name, count)

    _write_metadata(cfg, meta_records)
    return {"success": True, "converted": meta_records}

if __name__ == "__main__":
    cfg = UncheatableEvalConvertConfig(
        input_path="local_store/raw/uncheatable-eval/convert",
        output_path="local_store/processed/uncheatable-eval/convert",
        skip_existing=True,
    )
    result = convert_uncheatable_eval_files(cfg)
    print(result)