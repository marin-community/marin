# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Curator-authored objective task ideas from the first three reviewed BioTasks documents."""

import base64
import csv
import hashlib
import io
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from shellbox.machine import CommandSemantics
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    InlineFile,
    JsonValueAnswer,
    ResourceGroups,
    Source,
    TaskResource,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import StructuredExactSpec

GENERATOR_REVISION = "biotasks-first3-objective-v1"
RELEASE_STATUS = "internal_unreviewed_do_not_publish"

DECIPHER_KEY = "bioc-vignette:/packages/3.23/bioc/vignettes/DECIPHER/inst/doc/DesignMicroarray"
DECIPHER_SHA = "482f074eabcfeaa68fe5fc5b113ceaec139b80f43a70a620fb73a24f2c95a65f"
KOR_KEY = (
    "external:https://www.github.com/eyurtsev/kor/blob/main/docs/source/guidelines.ipynb:"
    "/eyurtsev/kor/blob/main/docs/source/guidelines.ipynb"
)
KOR_SHA = "ca899fb9a2f46ddbb66ff336dbf4c96973e8e70c8999d998fd4f7c8c2af7e38b"
PROTBERT_KEY = "github:agemagician/prottrans:Embedding/PyTorch/Basic/ProtBert.ipynb"
PROTBERT_SHA = "2c251021db7d3c80ab6adbe6035aafd513eb28ffad7d4230c38cd53f48b3f167"


class Assessment(BaseModel):
    """Transparent task-utility assessment; dimensions use a one-to-five scale."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    information_sufficiency: int = Field(ge=1, le=5)
    biological_substance: int = Field(ge=1, le=5)
    scientific_validity: int = Field(ge=1, le=5)
    objective_verifiability: int = Field(ge=1, le=5)
    hack_resistance: int = Field(ge=1, le=5)
    practical_usefulness: int = Field(ge=1, le=5)
    blocking_issues: tuple[str, ...] = ()
    recommended_action: Literal["accept", "revise", "reject"]
    rationale: str
    confidence: float = Field(ge=0, le=1)


class IdeaResource(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    content: str


class BioTaskIdea(BaseModel):
    """Concrete IdeaT row before Taskforge proposal/build."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    idea_id: str
    source_document_key: str
    source_group_id: str
    source_sha256: str
    generator_revision: str = GENERATOR_REVISION
    family: str
    parameters: dict[str, object]
    scientific_question: str
    instruction: str | None
    expected: dict[str, object] | None
    resources: tuple[IdeaResource, ...]
    author_assessment: Assessment
    release_status: str = RELEASE_STATUS
    null_reason: str | None = None

    @property
    def task_id(self) -> str:
        return self.idea_id.replace("/", "-")


def _idea_id(source_sha256: str, family: str, parameters: dict[str, object]) -> str:
    payload = json.dumps(
        {
            "source_sha256": source_sha256,
            "family": family,
            "parameters": parameters,
            "generator_revision": GENERATOR_REVISION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"biotasks-{family}-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def _resource(path: str, content: str) -> IdeaResource:
    return IdeaResource(path=path, content=content)


def _csv(rows: list[dict[str, object]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def ideas() -> list[BioTaskIdea]:
    """Return stable ideas for all three documents, including explicit null/revise rows."""
    probe_rows = [
        {"probe": "P1", "intended_otu": "OTU1", "OTU1": 0.92, "OTU2": 0.08, "OTU3": 0.05, "OTU4": 0.04},
        {"probe": "P2", "intended_otu": "OTU1", "OTU1": 0.88, "OTU2": 0.41, "OTU3": 0.06, "OTU4": 0.03},
        {"probe": "P3", "intended_otu": "OTU2", "OTU1": 0.10, "OTU2": 0.86, "OTU3": 0.12, "OTU4": 0.09},
        {"probe": "P4", "intended_otu": "OTU3", "OTU1": 0.07, "OTU2": 0.18, "OTU3": 0.81, "OTU4": 0.22},
        {"probe": "P5", "intended_otu": "OTU4", "OTU1": 0.04, "OTU2": 0.05, "OTU3": 0.07, "OTU4": 0.90},
        {"probe": "P6", "intended_otu": "OTU4", "OTU1": 0.38, "OTU2": 0.06, "OTU3": 0.04, "OTU4": 0.84},
    ]
    probe_parameters = {"on_target_min": 0.8, "off_target_max": 0.2, "cross_hybridization_threshold": 0.35}
    probe_instruction = """Read /workspace/data/probe_efficiencies.csv and /workspace/reference/biology.md. A probe is
specific when its efficiency for intended_otu is at least 0.80 and every non-target efficiency is at most 0.20.
Independently flag every probe having any non-target efficiency greater than 0.35 as cross-hybridizing. Return exactly
one JSON object with alphabetically sorted arrays specific_probes and cross_hybridizing_probes, plus a
specific_probe_by_otu object whose keys are OTU1 through OTU4 and whose values are sorted arrays. Do not treat the
predictions as experimental validation."""
    probe_expected = {
        "specific_probes": ["P1", "P3", "P5"],
        "cross_hybridizing_probes": ["P2", "P6"],
        "specific_probe_by_otu": {"OTU1": ["P1"], "OTU2": ["P3"], "OTU3": [], "OTU4": ["P5"]},
    }

    matrix_rows = [
        {"probe": "P1", "OTU1": 0.9, "OTU2": 0.1, "OTU3": 0.0, "brightness": 1.8},
        {"probe": "P2", "OTU1": 0.8, "OTU2": 0.2, "OTU3": 0.1, "brightness": 1.75},
        {"probe": "P3", "OTU1": 0.1, "OTU2": 0.9, "OTU3": 0.1, "brightness": 0.35},
        {"probe": "P4", "OTU1": 0.0, "OTU2": 0.8, "OTU3": 0.2, "brightness": 0.3},
        {"probe": "P5", "OTU1": 0.1, "OTU2": 0.1, "OTU3": 0.9, "brightness": 1.55},
        {"probe": "P6", "OTU1": 0.2, "OTU2": 0.0, "OTU3": 0.8, "brightness": 1.6},
    ]
    candidate_rows = [
        {"candidate": "A", "OTU1": 2.0, "OTU2": 0.0, "OTU3": 1.5},
        {"candidate": "B", "OTU1": 1.5, "OTU2": 0.5, "OTU3": 1.0},
        {"candidate": "C", "OTU1": 0.0, "OTU2": 2.0, "OTU3": 1.5},
    ]
    inverse_parameters = {"presence_threshold": 0.25, "loss": "sum_squared_error"}
    inverse_instruction = """Read /workspace/data/hybridization_matrix.csv, /workspace/data/candidate_abundances.csv,
and /workspace/reference/biology.md. For each candidate abundance vector x, predict each probe brightness as the
row's OTU1*x_OTU1 + OTU2*x_OTU2 + OTU3*x_OTU3, then calculate the unrounded sum of squared residuals from the
observed brightness column. Select the candidate with the smallest SSE. An OTU is present when its selected abundance
is at least 0.25. Return exactly one JSON object with best_candidate, candidate_sse (an object with A, B, and C rounded
to six decimals), present_otus sorted alphabetically, and interpretation_code set to
model_based_presence_not_experimental_confirmation."""
    inverse_expected = {
        "best_candidate": "A",
        "candidate_sse": {"A": 0.0, "B": 0.9475, "C": 9.28},
        "present_otus": ["OTU1", "OTU3"],
        "interpretation_code": "model_based_presence_not_experimental_confirmation",
    }

    embedding_rows = [
        {"sequence": "seq1", "token_index": 0, "token": "[CLS]", "f1": 9.0, "f2": 9.0, "special": 1},
        {"sequence": "seq1", "token_index": 1, "token": "A", "f1": 1.0, "f2": 0.0, "special": 0},
        {"sequence": "seq1", "token_index": 2, "token": "E", "f1": 0.0, "f2": 1.0, "special": 0},
        {"sequence": "seq1", "token_index": 3, "token": "T", "f1": 1.0, "f2": 1.0, "special": 0},
        {"sequence": "seq1", "token_index": 4, "token": "C", "f1": 2.0, "f2": 0.0, "special": 0},
        {"sequence": "seq1", "token_index": 5, "token": "X", "f1": 0.0, "f2": 2.0, "special": 0},
        {"sequence": "seq1", "token_index": 6, "token": "A", "f1": 1.0, "f2": 0.0, "special": 0},
        {"sequence": "seq1", "token_index": 7, "token": "X", "f1": 0.0, "f2": 2.0, "special": 0},
        {"sequence": "seq1", "token_index": 8, "token": "[SEP]", "f1": 9.0, "f2": 9.0, "special": 1},
        {"sequence": "seq1", "token_index": 9, "token": "[PAD]", "f1": 9.0, "f2": 9.0, "special": 1},
        {"sequence": "seq2", "token_index": 0, "token": "[CLS]", "f1": 9.0, "f2": 9.0, "special": 1},
        {"sequence": "seq2", "token_index": 1, "token": "S", "f1": 2.0, "f2": 1.0, "special": 0},
        {"sequence": "seq2", "token_index": 2, "token": "K", "f1": 1.0, "f2": 2.0, "special": 0},
        {"sequence": "seq2", "token_index": 3, "token": "T", "f1": 1.0, "f2": 1.0, "special": 0},
        {"sequence": "seq2", "token_index": 4, "token": "X", "f1": 0.0, "f2": 2.0, "special": 0},
        {"sequence": "seq2", "token_index": 5, "token": "P", "f1": 2.0, "f2": 2.0, "special": 0},
        {"sequence": "seq2", "token_index": 6, "token": "[SEP]", "f1": 9.0, "f2": 9.0, "special": 1},
        {"sequence": "seq2", "token_index": 7, "token": "[PAD]", "f1": 9.0, "f2": 9.0, "special": 1},
    ]
    embedding_parameters = {"mapped_residues": "UZOB", "replacement": "X", "pooling": "residue_mean"}
    embedding_instruction = """Read /workspace/data/sequences.csv, /workspace/data/token_features.csv, and
/workspace/reference/biology.md. Remove spaces from each sequence and map every U, Z, O, or B residue to X. Confirm
that the resulting residue string matches the ordered non-special token rows. For each sequence compute the arithmetic
mean of f1 and f2 over residue rows only, excluding CLS, SEP, and PAD. Return exactly one JSON object with
normalized_sequences, mapped_residue_counts, and residue_mean_features; each is an object keyed by seq1 and seq2,
and each feature vector is
[f1, f2] rounded to six decimals. Include interpretation_code protein_language_model_features_not_functional_evidence."""
    embedding_expected = {
        "normalized_sequences": {"seq1": "AETCXAX", "seq2": "SKTXP"},
        "mapped_residue_counts": {"seq1": 2, "seq2": 1},
        "residue_mean_features": {"seq1": [0.714286, 0.857143], "seq2": [1.2, 1.6]},
        "interpretation_code": "protein_language_model_features_not_functional_evidence",
    }

    biology = (
        "Predicted hybridization efficiency is a model output, not wet-lab validation. A probe should bind strongly "
        "to its intended OTU and weakly to non-target OTUs. In an inverse model, brightness is represented as A x, "
        "where A is the probe-by-OTU efficiency matrix and x is a nonnegative OTU abundance vector."
    )
    embedding_biology = (
        "Protein language models represent amino-acid residues as feature vectors. ProtBert preprocessing maps U, Z, "
        "O, and B to X. CLS, SEP, and PAD are model-control tokens rather than residues and must not enter residue "
        "pooling. Embedding similarity or magnitude is not direct evidence of protein function."
    )

    result = [
        BioTaskIdea(
            idea_id=_idea_id(DECIPHER_SHA, "probe_specificity_audit", probe_parameters),
            source_document_key=DECIPHER_KEY,
            source_group_id="bioconductor:decipher",
            source_sha256=DECIPHER_SHA,
            family="probe_specificity_audit",
            parameters=probe_parameters,
            scientific_question="Which predicted probes are specific enough to distinguish intended 16S OTUs?",
            instruction=probe_instruction,
            expected=probe_expected,
            resources=(
                _resource("workspace/data/probe_efficiencies.csv", _csv(probe_rows)),
                _resource("workspace/reference/biology.md", biology),
            ),
            author_assessment=Assessment(
                information_sufficiency=5,
                biological_substance=4,
                scientific_validity=4,
                objective_verifiability=5,
                hack_resistance=4,
                practical_usefulness=4,
                recommended_action="accept",
                rationale=(
                    "Requires applying biologically meaningful on-target and cross-hybridization criteria to a "
                    "complete simulated matrix."
                ),
                confidence=0.88,
            ),
        ),
        BioTaskIdea(
            idea_id=_idea_id(DECIPHER_SHA, "otu_inverse_model_selection", inverse_parameters),
            source_document_key=DECIPHER_KEY,
            source_group_id="bioconductor:decipher",
            source_sha256=DECIPHER_SHA,
            family="otu_inverse_model_selection",
            parameters=inverse_parameters,
            scientific_question="Which nonnegative OTU abundance profile best explains microarray brightness?",
            instruction=inverse_instruction,
            expected=inverse_expected,
            resources=(
                _resource("workspace/data/hybridization_matrix.csv", _csv(matrix_rows)),
                _resource("workspace/data/candidate_abundances.csv", _csv(candidate_rows)),
                _resource("workspace/reference/biology.md", biology),
            ),
            author_assessment=Assessment(
                information_sufficiency=5,
                biological_substance=4,
                scientific_validity=4,
                objective_verifiability=5,
                hack_resistance=4,
                practical_usefulness=4,
                recommended_action="accept",
                rationale=(
                    "Exercises the vignette's inverse-model interpretation with complete adapted matrices and an "
                    "exact loss contract."
                ),
                confidence=0.86,
            ),
        ),
        BioTaskIdea(
            idea_id=_idea_id(KOR_SHA, "generic_extraction_guidelines", {}),
            source_document_key=KOR_KEY,
            source_group_id="github:eyurtsev/kor",
            source_sha256=KOR_SHA,
            family="generic_extraction_guidelines",
            parameters={},
            scientific_question="Can generic LLM extraction guidance support a biological objective task?",
            instruction=None,
            expected=None,
            resources=(),
            null_reason=(
                "The document contains generic LLM information-extraction advice, no biological source material, "
                "data, or objective biological result."
            ),
            author_assessment=Assessment(
                information_sufficiency=1,
                biological_substance=1,
                scientific_validity=2,
                objective_verifiability=1,
                hack_resistance=2,
                practical_usefulness=1,
                blocking_issues=(
                    "No biological content or task-specific data.",
                    "BioTasks source attribution is historically mismatched.",
                ),
                recommended_action="reject",
                rationale="Forcing a task from this source would test generic extraction advice rather than biology.",
                confidence=0.98,
            ),
        ),
        BioTaskIdea(
            idea_id=_idea_id(PROTBERT_SHA, "protein_embedding_preprocessing", embedding_parameters),
            source_document_key=PROTBERT_KEY,
            source_group_id="github:agemagician/prottrans",
            source_sha256=PROTBERT_SHA,
            family="protein_embedding_preprocessing",
            parameters=embedding_parameters,
            scientific_question=(
                "Can residue-level ProtBert features be aligned and pooled without including model-control tokens?"
            ),
            instruction=embedding_instruction,
            expected=embedding_expected,
            resources=(
                _resource(
                    "workspace/data/sequences.csv",
                    _csv(
                        [
                            {"sequence": "seq1", "raw_sequence": "A E T C Z A O"},
                            {"sequence": "seq2", "raw_sequence": "S K T Z P"},
                        ]
                    ),
                ),
                _resource("workspace/data/token_features.csv", _csv(embedding_rows)),
                _resource("workspace/reference/biology.md", embedding_biology),
            ),
            author_assessment=Assessment(
                information_sufficiency=5,
                biological_substance=2,
                scientific_validity=4,
                objective_verifiability=5,
                hack_resistance=4,
                practical_usefulness=3,
                blocking_issues=("Tests protein-sequence representation mechanics more than biological inference.",),
                recommended_action="revise",
                rationale=(
                    "The task is complete and verifiable but is best treated as a computational-biology "
                    "preprocessing control."
                ),
                confidence=0.9,
            ),
        ),
    ]
    if len({idea.idea_id for idea in result}) != len(result):
        raise ValueError("idea identities are not unique")
    return result


def _inline_resource(resource: IdeaResource) -> TaskResource:
    return TaskResource(
        path=resource.path,
        source=InlineFile(content_base64=base64.b64encode(resource.content.encode()).decode()),
    )


def task_spec(idea: BioTaskIdea) -> TaskSpec:
    """Build an objective TaskSpec from a non-null idea."""
    if idea.instruction is None or idea.expected is None or idea.null_reason is not None:
        raise ValueError(f"idea {idea.idea_id} is null")
    package = verifyit_package(StructuredExactSpec(expected=idea.expected))
    return TaskSpec(
        id=idea.task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=idea.instruction),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics=CommandSemantics.LINUX_PROCESS,
            working_directory="/workspace",
        ),
        answer_type=AnswerType.JSON,
        answer_format=JsonValueAnswer(),
        grader=package.grader,
        source=Source(
            dataset="biotasks-reviewed57-first3",
            revision=idea.source_sha256,
            row=idea.idea_id,
            importer_revision=GENERATOR_REVISION,
        ),
        resources=ResourceGroups(
            worker=tuple(_inline_resource(resource) for resource in idea.resources),
            verifier=package.resources,
        ),
        tags=("biology", "biotasks", "objective-verifier", idea.family, idea.release_status),
    )
