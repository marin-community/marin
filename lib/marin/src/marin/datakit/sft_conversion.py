# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create a new chat variant from pinned prompt/response data, before registration."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.sft import DatasetVariantInput, SftEvidenceReference, SftRecord
from openai_harmony import Message, Role
from pydantic import Field
from rigging.filesystem.storage_path import StoragePath

from marin.datakit.chat_normalize import ChatChannel, normalize_chat_record


class ConversionPlan(SftRecord):
    parent: DatasetVariantInput
    parent_manifest: SftEvidenceReference
    variant_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    prompt_field: str
    response_field: str
    implementation_revision: str = Field(pattern=r"^[0-9a-f]{40}$")


class ConversionReceipt(SftRecord):
    protocol: Literal["dataset-chat-conversion-v1"] = "dataset-chat-conversion-v1"
    parent: DatasetVariantInput
    output: DatasetVariantInput
    records: SftEvidenceReference
    lineage: SftEvidenceReference
    rows: int
    plan: ConversionPlan
    completed_at: datetime


def _pin(path: StoragePath) -> SftEvidenceReference:
    with path.open() as stream:
        return SftEvidenceReference(uri=str(path), sha256=hashlib.file_digest(stream, "sha256").hexdigest())


def convert_chat_variant(plan: ConversionPlan, output_path: str) -> ConversionReceipt:
    """Reformat existing responses; publish lineage without granting any method tag."""
    if plan.variant_id == plan.parent.variant_id:
        raise ValueError("Conversion must create a new variant")
    manifest_path = StoragePath(plan.parent_manifest.uri)
    if _pin(manifest_path) != plan.parent_manifest or plan.parent.data_sha256 != plan.parent_manifest.sha256:
        raise ValueError("Conversion parent content identity does not match")
    manifest = json.loads(manifest_path.read_bytes())
    root = StoragePath(output_path)
    if root.exists():
        raise FileExistsError(str(root))
    root.mkdirs()
    row_count = 0
    chat_path, lineage_path = root / "chats.parquet", root / "lineage.parquet"
    with chat_path.open("wb") as chat_stream, lineage_path.open("wb") as lineage_stream:
        chat_writer = lineage_writer = None
        try:
            for reference in manifest["files"]:
                source = StoragePath(reference["uri"])
                if _pin(source).sha256 != reference["sha256"]:
                    raise ValueError("Conversion parent file changed")
                with source.open() as stream:
                    for batch in pq.ParquetFile(stream).iter_batches(batch_size=256):
                        chats, lineage = [], []
                        for row in batch.to_pylist():
                            prompt, response = row[plan.prompt_field], row[plan.response_field]
                            if not isinstance(prompt, str) or not isinstance(response, str) or not response:
                                raise ValueError("Conversion requires an existing text prompt and response")
                            messages = [
                                Message.from_role_and_content(Role.USER, prompt),
                                Message.from_role_and_content(Role.ASSISTANT, response).with_channel(ChatChannel.FINAL),
                            ]
                            chat = normalize_chat_record({"messages": [m.to_dict() for m in messages]}, "messages", "id")
                            chats.append(chat)
                            lineage.append(
                                {
                                    "output_id": chat["id"],
                                    "parent_row_sha256": (
                                        hashlib.sha256(
                                            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                                        ).hexdigest()
                                    ),
                                }
                            )
                        chat_table, lineage_table = pa.Table.from_pylist(chats), pa.Table.from_pylist(lineage)
                        if chat_writer is None:
                            chat_writer = pq.ParquetWriter(chat_stream, chat_table.schema)
                            lineage_writer = pq.ParquetWriter(lineage_stream, lineage_table.schema)
                        chat_writer.write_table(chat_table)
                        lineage_writer.write_table(lineage_table)
                        row_count += len(chats)
        finally:
            if chat_writer is not None:
                chat_writer.close()
                lineage_writer.close()
    if row_count == 0 or row_count != manifest["rows"]:
        raise ValueError("Conversion does not account for its complete parent population")
    records = _pin(chat_path)
    output_manifest = root / "manifest.json"
    output_manifest.write_text(
        json.dumps({"files": [records.model_dump(mode="json")], "rows": row_count}, sort_keys=True)
    )
    receipt = ConversionReceipt(
        parent=plan.parent,
        output=DatasetVariantInput(
            source_id=plan.parent.source_id,
            variant_id=plan.variant_id,
            repository=plan.repository,
            revision=plan.revision,
            data_sha256=_pin(output_manifest).sha256,
        ),
        records=records,
        lineage=_pin(lineage_path),
        rows=row_count,
        plan=plan,
        completed_at=datetime.now(UTC),
    )
    (root / "conversion-receipt.json").write_text(receipt.model_dump_json(indent=2) + "\n")
    return receipt
