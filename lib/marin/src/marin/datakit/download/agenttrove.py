# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""open-thoughts/AgentTrove dataset download and transform.

Agent rollouts (terminus-2 style) on coding and command-line tasks. Each
row contains a multi-turn ``conversations`` list of ``{role, content}``
dicts plus metadata describing the teacher that produced the trace.

Tool use is preserved verbatim: agents like terminus-2 emit tool calls
as JSON-encoded payloads inside ``content`` (e.g. ``{"commands": [...]}``),
so the standard ``<role>...</role>`` rendering captures them intact —
there are no separate ``tool_calls`` / ``tool`` fields on the messages.

OpenAI's terms of use forbid using GPT outputs to train competing models.
We apply a **positive** ``original_teacher`` allowlist: only rows whose
teacher is explicitly approved survive. All current GPT-5 / GPT-5-mini /
GPT-5-nano / "GPT 5.1 Nano" rows are dropped, and any future teacher we
haven't audited is dropped by default.
"""

import hashlib

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import load_parquet_batched
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "open-thoughts/AgentTrove"
HF_REVISION = "b395a43"

# Positive allowlist of teachers whose outputs we can legally distill.
# OpenAI's ToS forbids using GPT outputs to train competing models, so we
# drop everything from the GPT family (currently GPT-5, GPT-5-mini,
# GPT-5-nano, "GPT 5.1 Nano") by omission. Add new teachers here only
# after verifying that the upstream license permits distillation.
ALLOWED_TEACHERS = frozenset({"GLM-4.6"})


def is_allowed_teacher(teacher: str | None) -> bool:
    """Return True iff ``teacher`` is on the distillation-safe allowlist."""
    return teacher in ALLOWED_TEACHERS


def render_message(msg: dict) -> str:
    role = msg.get("role", "unknown")
    content = msg.get("content") or ""
    return f"<{role}>\n{content}\n</{role}>"


def row_to_doc(row: dict) -> list[dict]:
    if not is_allowed_teacher(row.get("original_teacher")):
        counters.increment("agenttrove/dropped_teacher")
        return []

    conv = row.get("conversations")
    if not conv:
        counters.increment("agenttrove/dropped_empty")
        return []

    text = "\n\n".join(render_message(m) for m in conv)
    counters.increment("agenttrove/kept")
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "source": "open-thoughts/AgentTrove",
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="agenttrove-transform", resources=ResourceConfig(cpu=1, ram="32g"))
    ctx.execute(pipeline)


def download_agenttrove_step() -> StepSpec:
    """Download and transform AgentTrove rollouts into JSONL documents."""
    dl = download_hf_step(
        "raw/agenttrove",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
    )

    return StepSpec(
        name="processed/agenttrove",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def agenttrove_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for AgentTrove."""
    processed = download_agenttrove_step()
    return (
        processed,
        normalize_step(name="normalized/agenttrove", download=processed),
    )
