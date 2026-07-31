# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate reusable seeds for synthetic model-identity conversations."""

import argparse
import dataclasses
import json
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

from rigging.filesystem import StoragePath

UNKNOWN_ANSWER = "That information is not known, so I should not guess."


@dataclass(frozen=True)
class IdentityProfile:
    """Facts an identity conversation may state about a model."""

    profile_id: str
    model_name: str
    developer: str
    trained_by: str
    maintained_by: str
    ecosystem_contributors: tuple[str, ...]
    training_data_models: tuple[str, ...]
    unknown_facts: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("profile_id", "model_name", "developer", "trained_by", "maintained_by"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must not be empty")
        if not self.unknown_facts:
            raise ValueError("unknown_facts must not be empty")


MARIN_IDENTITY_PROFILE = IdentityProfile(
    profile_id="marin-latest-moe",
    model_name="Marin's Latest MoE",
    developer="the Marin Community",
    trained_by="the Marin Community",
    maintained_by="developers from Open Athena, Stanford, and many other institutions",
    ecosystem_contributors=("NVIDIA", "AI2"),
    training_data_models=("DeepSeek", "Kimi", "GLM"),
    unknown_facts=(
        "the model's aliases or version identifier",
        "the base-model lineage",
        "the exact training stages",
        "the release date",
        "the license",
        "the supported modalities",
        "the context length",
        "architecture details beyond the model's supplied name",
        "the deployment provider",
        "the funding sources",
    ),
)


class Facet(StrEnum):
    NAME = "name"
    DEVELOPER = "developer"
    TRAINED_BY = "trained_by"
    MAINTAINED_BY = "maintained_by"
    ECOSYSTEM_CONTRIBUTIONS = "ecosystem_contributions"
    OPEN_WEIGHT_DATA = "open_weight_data"
    FALSE_PREMISE = "false_premise"
    DEPLOYMENT_PROVIDER = "deployment_provider"
    INDIRECT_IDENTITY = "indirect_identity"
    UNKNOWN_FACT = "unknown_fact"


class TruthStatus(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FacetDefinition:
    facet: Facet
    topic: str
    truth_status: TruthStatus
    canonical_answer: str
    required_facts: tuple[str, ...]
    weight: int


@dataclass(frozen=True)
class HumanPhrasing:
    text: str
    source: str


@dataclass(frozen=True)
class Entropy:
    register: str
    turn_pattern: str
    question_strategy: str
    social_context: str
    speaker_attitude: str


@dataclass(frozen=True)
class ConversationSeed:
    seed_id: str
    profile_id: str
    facet: Facet
    topic: str
    truth_status: TruthStatus
    canonical_answer: str
    required_facts: tuple[str, ...]
    distractor_model: str | None
    distractor_developer: str | None
    human_phrasing: HumanPhrasing | None
    entropy: Entropy
    neutral_bridge: str | None
    generation_prompt: str


REGISTERS = (
    "casual spoken English",
    "terse chat",
    "formal professional English",
    "internet-native slang",
    "polite customer-support language",
    "skeptical technical language",
    "plain language from a non-expert",
    "playful but not absurd banter",
)

TURN_PATTERNS = (
    "one direct user question",
    "an identity-neutral setup exchange followed by a user identity question",
)

NEUTRAL_BRIDGES = (
    "What would you like me to verify?",
    "Which attribution detail are you checking?",
    "What would you like me to clarify?",
    "Which part should I address?",
)

QUESTION_STRATEGIES = (
    "ask directly",
    "ask for confirmation",
    "embed the question in a comparison",
    "attribute a claim to an unnamed third party",
    "use a false either-or choice",
    "ask the assistant to correct a profile",
    "ask what should be written in a citation",
    "ask how the assistant should introduce itself",
    "ask whether a serving organization is also the model developer",
    "probe indirectly without using the words developer or trainer",
    "challenge a mistaken identity claim",
    "ask for the boundary between known and unknown information",
)

SOCIAL_CONTEXTS = (
    "a first-time user",
    "a researcher documenting a model",
    "an engineer checking attribution",
    "a journalist fact-checking a description",
    "a student comparing AI assistants",
    "an open-source contributor discussing provenance",
    "an evaluator probing identity consistency",
    "a user correcting a misleading label",
)

SPEAKER_ATTITUDES = (
    "neutral curiosity",
    "friendly familiarity",
    "mild skepticism",
    "high confidence in an incorrect premise",
    "confusion",
    "adversarial insistence without abuse",
    "careful fact-checking",
    "light humor",
)

DISTRACTORS = (
    ("ChatGPT", "OpenAI"),
    ("Claude", "Anthropic"),
    ("Gemini", "Google"),
    ("GLM", "Zhipu AI"),
    ("Qwen", "Alibaba"),
    ("Llama", "Meta"),
    ("DeepSeek", "DeepSeek"),
    ("Mistral", "Mistral AI"),
)

T = TypeVar("T")


def _join_names(names: Sequence[str]) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _facet_definitions(profile: IdentityProfile) -> tuple[FacetDefinition, ...]:
    contributors = _join_names(profile.ecosystem_contributors)
    training_data_models = _join_names(profile.training_data_models)
    return (
        FacetDefinition(
            Facet.NAME,
            "the model's name",
            TruthStatus.KNOWN,
            f"I am {profile.model_name}.",
            (profile.model_name,),
            10,
        ),
        FacetDefinition(
            Facet.DEVELOPER,
            "who developed the model",
            TruthStatus.KNOWN,
            f"I was developed by {profile.developer}.",
            (profile.developer,),
            10,
        ),
        FacetDefinition(
            Facet.TRAINED_BY,
            "who trained the model",
            TruthStatus.KNOWN,
            f"I was trained by {profile.trained_by}.",
            (profile.trained_by,),
            14,
        ),
        FacetDefinition(
            Facet.MAINTAINED_BY,
            "who maintains the model",
            TruthStatus.KNOWN,
            f"I am maintained by {profile.maintained_by}.",
            (profile.maintained_by,),
            10,
        ),
        FacetDefinition(
            Facet.ECOSYSTEM_CONTRIBUTIONS,
            "how explicitly named upstream contributors relate to the model's identity",
            TruthStatus.KNOWN,
            f"I was trained by {profile.trained_by}. My development also builds on contributions from {contributors}; "
            "those contributions do not change who trained me.",
            (profile.trained_by, *profile.ecosystem_contributors),
            12,
        ),
        FacetDefinition(
            Facet.OPEN_WEIGHT_DATA,
            "how explicitly named training-data sources relate to the model's identity",
            TruthStatus.KNOWN,
            f"My training data includes data derived from open-weight models such as {training_data_models}. I am still "
            f"{profile.model_name}, trained by {profile.trained_by}; those sources do not determine my identity.",
            (profile.model_name, profile.trained_by, *profile.training_data_models),
            14,
        ),
        FacetDefinition(
            Facet.FALSE_PREMISE,
            "a false claim that the assistant is a different named model",
            TruthStatus.KNOWN,
            "",
            (profile.model_name, profile.trained_by),
            18,
        ),
        FacetDefinition(
            Facet.DEPLOYMENT_PROVIDER,
            "the difference between the model developer and an unspecified deployment provider",
            TruthStatus.KNOWN,
            f"I was developed by {profile.developer} and trained by {profile.trained_by}. The organization serving "
            "this deployment is not known, so I should not guess.",
            (profile.developer, profile.trained_by),
            6,
        ),
        FacetDefinition(
            Facet.INDIRECT_IDENTITY,
            "the model's identity inferred through an indirect question",
            TruthStatus.KNOWN,
            f"I am {profile.model_name}. I was developed by {profile.developer} and trained by {profile.trained_by}.",
            (profile.model_name, profile.developer, profile.trained_by),
            4,
        ),
        FacetDefinition(
            Facet.UNKNOWN_FACT,
            "an identity-related fact that is not present in the supplied profile",
            TruthStatus.UNKNOWN,
            UNKNOWN_ANSWER,
            (),
            8,
        ),
    )


def _balanced_value(values: Sequence[T], index: int, random_seed: int, axis: str) -> T:
    block_index, offset = divmod(index, len(values))
    block = list(values)
    random.Random(f"{random_seed}:{axis}:{block_index}").shuffle(block)
    return block[offset]


def _weighted_facet(profile: IdentityProfile, index: int, random_seed: int) -> FacetDefinition:
    weighted = [facet for facet in _facet_definitions(profile) for _ in range(facet.weight)]
    return _balanced_value(weighted, index, random_seed, "facet")


def _false_premise_answer(
    profile: IdentityProfile,
    distractor_model: str,
    distractor_developer: str,
) -> str:
    return (
        f"No. I am {profile.model_name}, trained by {profile.trained_by}. "
        f"I am not {distractor_model} and was not trained by {distractor_developer}."
    )


def _generation_prompt(
    topic: str,
    truth_status: TruthStatus,
    entropy: Entropy,
    canonical_answer: str,
    neutral_bridge: str | None,
    distractor_model: str | None,
    distractor_developer: str | None,
    human_phrasing: HumanPhrasing | None,
) -> str:
    phrasing = (
        f'\nHuman phrasing reference: "{human_phrasing.text}" (source: {human_phrasing.source}). '
        "Borrow only its discourse pattern, not its factual claims, and do not copy more than five consecutive words."
        if human_phrasing
        else ""
    )
    false_premise = (
        f"\nThe final user message must make the specified false claim that the assistant is {distractor_model}, "
        f"associated with {distractor_developer}."
        if distractor_model and distractor_developer
        else ""
    )
    if neutral_bridge is None:
        role_contract = f"""Return exactly two messages with these roles and assistant text:
1. User: a natural identity question
2. Assistant: "{canonical_answer}\""""
    else:
        role_contract = f"""Return exactly four messages with these roles and assistant text:
1. User: a brief identity-neutral setup that makes no claim about this assistant
2. Assistant: "{neutral_bridge}"
3. User: the identity question
4. Assistant: "{canonical_answer}\""""
    return f"""Write one natural English chat conversation about {topic}.

Conversation constraints:
- Register: {entropy.register}
- Turn pattern: {entropy.turn_pattern}
- Question strategy: {entropy.question_strategy}
- Social context: {entropy.social_context}
- User attitude: {entropy.speaker_attitude}
- Truth status: {truth_status}
{role_contract}
- Assistant messages must match the supplied text exactly, with no added words.
- Do not introduce any concrete product behavior, document, repository, release, training stage, architecture detail,
  provider, organization, or model name unless it appears in the supplied assistant text or false-premise instruction.
- A user may ask whether an unspecified fact is known, but must not propose a concrete value, artifact, or mechanism.
- The setup message, when present, must be identity-neutral and must not make a factual claim about the assistant.
{phrasing}{false_premise}

Return only the conversation as a JSON array of objects with "role" and "content"."""


def generate_seeds(
    *,
    count: int,
    random_seed: int,
    identity_profile: IdentityProfile,
    start_index: int = 0,
    human_phrasings: Sequence[HumanPhrasing] = (),
) -> Iterator[ConversationSeed]:
    """Yield deterministic, marginally balanced conversation-generation seeds."""
    if count <= 0:
        raise ValueError("count must be positive")
    if start_index < 0:
        raise ValueError("start_index must not be negative")

    for index in range(start_index, start_index + count):
        facet = _weighted_facet(identity_profile, index, random_seed)
        turn_pattern = _balanced_value(TURN_PATTERNS, index, random_seed, "turn_pattern")
        neutral_bridge = (
            None
            if turn_pattern == TURN_PATTERNS[0]
            else _balanced_value(NEUTRAL_BRIDGES, index, random_seed, "neutral_bridge")
        )
        entropy = Entropy(
            register=_balanced_value(REGISTERS, index, random_seed, "register"),
            turn_pattern=turn_pattern,
            question_strategy=_balanced_value(QUESTION_STRATEGIES, index, random_seed, "question_strategy"),
            social_context=_balanced_value(SOCIAL_CONTEXTS, index, random_seed, "social_context"),
            speaker_attitude=_balanced_value(SPEAKER_ATTITUDES, index, random_seed, "speaker_attitude"),
        )
        distractor_model = None
        distractor_developer = None
        canonical_answer = facet.canonical_answer
        topic = facet.topic
        if facet.facet == Facet.FALSE_PREMISE:
            distractor_model, distractor_developer = _balanced_value(DISTRACTORS, index, random_seed, "distractor")
            canonical_answer = _false_premise_answer(identity_profile, distractor_model, distractor_developer)
        if facet.facet == Facet.UNKNOWN_FACT:
            topic = _balanced_value(identity_profile.unknown_facts, index, random_seed, "unknown_fact")
        phrasing = _balanced_value(human_phrasings, index, random_seed, "human_phrasing") if human_phrasings else None
        yield ConversationSeed(
            seed_id=f"identity-{identity_profile.profile_id}-{random_seed:08x}-{index:09d}",
            profile_id=identity_profile.profile_id,
            facet=facet.facet,
            topic=topic,
            truth_status=facet.truth_status,
            canonical_answer=canonical_answer,
            required_facts=facet.required_facts,
            distractor_model=distractor_model,
            distractor_developer=distractor_developer,
            human_phrasing=phrasing,
            entropy=entropy,
            neutral_bridge=neutral_bridge,
            generation_prompt=_generation_prompt(
                topic,
                facet.truth_status,
                entropy,
                canonical_answer,
                neutral_bridge,
                distractor_model,
                distractor_developer,
                phrasing,
            ),
        )


def load_human_phrasings(path: StoragePath) -> list[HumanPhrasing]:
    phrasings: list[HumanPhrasing] = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                row = json.loads(line)
                text = row.get("text") or row.get("response") or row.get("content") or row.get("question")
                if not text:
                    raise ValueError("JSONL phrasing rows must contain text, response, content, or question")
                phrasings.append(HumanPhrasing(text=text, source=row.get("source", path.name)))
            else:
                phrasings.append(HumanPhrasing(text=line, source=path.name))
    if not phrasings:
        raise ValueError("phrasing input must contain at least one non-empty line")
    return phrasings


def identity_profile_from_mapping(row: Mapping[str, Any]) -> IdentityProfile:
    return IdentityProfile(
        profile_id=row["profile_id"],
        model_name=row["model_name"],
        developer=row["developer"],
        trained_by=row["trained_by"],
        maintained_by=row["maintained_by"],
        ecosystem_contributors=tuple(row["ecosystem_contributors"]),
        training_data_models=tuple(row["training_data_models"]),
        unknown_facts=tuple(row["unknown_facts"]),
    )


def load_identity_profile(path: StoragePath) -> IdentityProfile:
    with path.open("r") as handle:
        return identity_profile_from_mapping(json.load(handle))


def _write_seeds(path: StoragePath, seeds: Iterator[ConversationSeed]) -> None:
    path.parent.mkdirs()
    with path.open("w") as handle:
        for seed in seeds:
            handle.write(json.dumps(dataclasses.asdict(seed), ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--phrasing-input")
    parser.add_argument("--identity-profile", required=True)
    args = parser.parse_args(argv)

    phrasings = load_human_phrasings(StoragePath(args.phrasing_input)) if args.phrasing_input else ()
    identity_profile = load_identity_profile(StoragePath(args.identity_profile))
    seeds = generate_seeds(
        count=args.count,
        random_seed=args.random_seed,
        human_phrasings=phrasings,
        identity_profile=identity_profile,
    )
    _write_seeds(StoragePath(args.output), seeds)


if __name__ == "__main__":
    main()
