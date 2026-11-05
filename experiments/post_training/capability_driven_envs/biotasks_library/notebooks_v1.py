# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source extraction and model-authored objective ideas for BioTasks notebooks/v1."""

import hashlib
import json
import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from experiments.post_training.capability_driven_envs.biotasks_library.pilot_first3 import (
    Assessment,
    BioTaskIdea,
    IdeaResource,
)

GENERATOR_REVISION = "biotasks-notebooks-v1"
RELEASE_STATUS = "internal_unreviewed_do_not_publish"
MAX_SOURCE_CHARACTERS = 80_000
MAX_RESOURCE_CHARACTERS = 30_000
MAX_IDEAS_PER_SOURCE = 8


class SourceRecord(BaseModel):
    """One pinned source row from the reviewed BioTasks catalog."""

    model_config = ConfigDict(frozen=True, extra="allow")

    canonical_document_key: str
    document_key: str
    fields: tuple[str, ...]
    format: str
    modalities: tuple[str, ...]
    object_uri: str
    operations: tuple[str, ...]
    path: str
    repo: str | None
    source_bytes: int
    source_ids: tuple[str, ...]
    source_role: str
    source_sha256: str
    title: str


class GeneratedIdea(BaseModel):
    """Model-authored task content before stable identity is assigned."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str
    parameters_json: str
    scientific_question: str
    instruction: str
    expected_json: str
    resources: tuple[IdeaResource, ...]
    author_assessment: Assessment


class GeneratedPortfolio(BaseModel):
    """A source-level portfolio; an unsuitable source may yield zero ideas."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_summary: str
    disposition: Literal["ideas_proposed", "no_objective_task"]
    null_reason: str | None = None
    null_assessment: Assessment | None = None
    ideas: tuple[GeneratedIdea, ...] = Field(max_length=MAX_IDEAS_PER_SOURCE)


def source_group_id(source: SourceRecord) -> str:
    """Derive the source-project split group from canonical source identity."""
    if source.repo:
        return f"github:{source.repo.lower()}"
    if source.document_key.startswith("github:"):
        return ":".join(source.document_key.split(":", 2)[:2]).lower()
    match = re.search(r"github\.com/([^/]+/[^/]+)", source.document_key, flags=re.IGNORECASE)
    if match:
        return f"github:{match.group(1).removesuffix('.git').lower()}"
    if source.source_ids:
        return source.source_ids[0].lower()
    parsed = urlparse(source.document_key)
    return parsed.netloc.lower() or source.document_key.split(":", 1)[0].lower()


def extract_source_text(source: SourceRecord, raw: bytes) -> str:
    """Extract markdown and code while excluding bulky notebook outputs."""
    if hashlib.sha256(raw).hexdigest() != source.source_sha256:
        raise ValueError(f"source digest mismatch for {source.document_key}")
    text = raw.decode("utf-8", errors="replace")
    if source.format != "Jupyter":
        return _bounded_text(text)
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        return _bounded_text(text)
    cells: list[str] = []
    for index, cell in enumerate(notebook.get("cells", ())):
        kind = str(cell.get("cell_type", "unknown"))
        value = cell.get("source", "")
        content = "".join(value) if isinstance(value, list) else str(value)
        content = content.strip()
        if content:
            cells.append(f"\n--- cell {index} ({kind}) ---\n{content}")
    return _bounded_text("\n".join(cells))


def _bounded_text(text: str) -> str:
    if len(text) <= MAX_SOURCE_CHARACTERS:
        return text
    head = MAX_SOURCE_CHARACTERS * 3 // 4
    tail = MAX_SOURCE_CHARACTERS - head
    return f"{text[:head]}\n\n[... deterministic middle truncation ...]\n\n{text[-tail:]}"


def generation_prompt(source: SourceRecord, source_text: str) -> str:
    """Render the source-complete structured-authoring request."""
    metadata = source.model_dump(mode="json", exclude={"object_uri"})
    return f"""You are the author of objective, self-contained computational-biology training tasks. Read the pinned
source below and return a portfolio with zero to {MAX_IDEAS_PER_SOURCE} genuinely useful ideas. There is no quota.
Return zero ideas when the document is generic infrastructure, a navigation stub, lacks biological content, or cannot
support a scientifically valid objective task without unavailable data.

For each idea:
- instantiate one concrete task, not a reusable template;
- provide all solver-visible inputs as small inline resources under workspace/data/ or workspace/reference/;
- use adapted or synthetic data when original data are absent, and label that fact in a reference resource;
- require a single JSON answer and provide the exact expected JSON;
- make the expected answer calculable solely from the resources and explicit instruction;
- freeze thresholds, rounding, tie rules, missingness, units, and interpretation limits;
- avoid trivia, pure filename lookup, answer leakage, unsupported biological claims, and wet-lab claims from predictions;
- prefer meaningful analysis over incidental arithmetic, but retain useful preprocessing controls when honestly labeled;
- keep total resource content under {MAX_RESOURCE_CHARACTERS} characters;
- encode parameters_json and expected_json as valid JSON objects;
- assess your own task on the six 1-5 rubric dimensions. recommended_action may be accept, revise, or reject.

Do not copy the source notebook wholesale into resources. Do not require internet access, package installation, large
models, GPUs, or unavailable source datasets. A Taskforge structural accept does not imply scientific quality.

Source metadata:
{json.dumps(metadata, indent=2, sort_keys=True)}

Pinned source text (outputs removed and possibly deterministically truncated):
<source>
{source_text}
</source>
"""


def _safe_family(value: str) -> str:
    family = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not family:
        raise ValueError("generated family is empty")
    return family[:80]


def _json_object(value: str, field: str) -> dict[str, object]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must encode a JSON object")
    return parsed


def _idea_id(source: SourceRecord, generated: GeneratedIdea) -> str:
    payload = {
        "source_sha256": source.source_sha256,
        "family": _safe_family(generated.family),
        "parameters": _json_object(generated.parameters_json, "parameters_json"),
        "instruction": generated.instruction,
        "expected": _json_object(generated.expected_json, "expected_json"),
        "resources": [resource.model_dump(mode="json") for resource in generated.resources],
        "generator_revision": GENERATOR_REVISION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"biotasks-{payload['family']}-{hashlib.sha256(encoded.encode()).hexdigest()[:16]}"


def hydrate_ideas(source: SourceRecord, portfolio: GeneratedPortfolio) -> list[BioTaskIdea]:
    """Validate and assign content-derived identities to one generated portfolio."""
    if portfolio.disposition == "no_objective_task":
        if portfolio.ideas:
            raise ValueError("no_objective_task portfolio contains ideas")
        if not portfolio.null_reason or portfolio.null_assessment is None:
            raise ValueError("no_objective_task portfolio lacks null evidence")
        parameters = {"disposition": "no_objective_task"}
        payload = json.dumps(
            {
                "source_sha256": source.source_sha256,
                "family": "source_not_suitable",
                "parameters": parameters,
                "generator_revision": GENERATOR_REVISION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return [
            BioTaskIdea(
                idea_id=f"biotasks-source_not_suitable-{hashlib.sha256(payload.encode()).hexdigest()[:16]}",
                source_document_key=source.canonical_document_key,
                source_group_id=source_group_id(source),
                source_sha256=source.source_sha256,
                generator_revision=GENERATOR_REVISION,
                family="source_not_suitable",
                parameters=parameters,
                scientific_question="Can this source support an objective computational-biology task?",
                instruction=None,
                expected=None,
                resources=(),
                author_assessment=portfolio.null_assessment,
                release_status=RELEASE_STATUS,
                null_reason=portfolio.null_reason,
            )
        ]
    if not portfolio.ideas:
        raise ValueError("ideas_proposed portfolio is empty")

    ideas: list[BioTaskIdea] = []
    for generated in portfolio.ideas:
        parameters = _json_object(generated.parameters_json, "parameters_json")
        expected = _json_object(generated.expected_json, "expected_json")
        if not generated.resources:
            raise ValueError(f"generated family {generated.family!r} has no resources")
        if sum(len(resource.content) for resource in generated.resources) > MAX_RESOURCE_CHARACTERS:
            raise ValueError(f"generated family {generated.family!r} exceeds resource limit")
        paths = [resource.path for resource in generated.resources]
        if len(paths) != len(set(paths)):
            raise ValueError(f"generated family {generated.family!r} repeats resource paths")
        if any(not path.startswith(("workspace/data/", "workspace/reference/")) for path in paths):
            raise ValueError(f"generated family {generated.family!r} has an invalid resource path")
        ideas.append(
            BioTaskIdea(
                idea_id=_idea_id(source, generated),
                source_document_key=source.canonical_document_key,
                source_group_id=source_group_id(source),
                source_sha256=source.source_sha256,
                generator_revision=GENERATOR_REVISION,
                family=_safe_family(generated.family),
                parameters=parameters,
                scientific_question=generated.scientific_question,
                instruction=generated.instruction,
                expected=expected,
                resources=generated.resources,
                author_assessment=generated.author_assessment,
                release_status=RELEASE_STATUS,
            )
        )
    if len({idea.idea_id for idea in ideas}) != len(ideas):
        raise ValueError(f"source {source.document_key} generated duplicate ideas")
    return ideas
