"""First-person attribution-claim detection over assistant turns.

A claim is WRONG when the asserted entity matches no alias form of the true entity for its kind. Guards drop
candidates that are not assertions (negation, quotation, hypotheticals, fictional framing, comparisons). A
candidate whose sentence also carries unresolvable quotes or negation is ambiguous: with a judge the candidate is
deferred to `SemanticJudge.verify_claim`, without a judge it is emitted with `ambiguous=True` so the caller can
route it to manual audit instead of guessing.

Clause rewind: `clause_start_char` is the start of the sentence-like clause containing the claim, pulled forward
to the first first-person marker and extended over a leading "and/but/so/also" connector, so a teacher-repair
branch never respans before the first-person attribution. Omission and evasion of identity questions are out of
scope; answer-level scoring in `reward` owns them.

The appositive creator pattern ("I'm Qwen, created by Alibaba") starts its claim span at the participle; the
clause rewind supplies the first-person context. All other patterns start their claim span at the first-person
marker itself.
"""

import re
from dataclasses import dataclass, replace

from skyrl_gym.identity_repair.facts import FactManifest, normalize_entity
from skyrl_gym.identity_repair.judge import AttributionClaim, ClaimKind, SemanticJudge

_ENTITY = r"(?P<entity>[^\n.!?;:,]{1,80}?)(?=\s*(?:[.!?;,:\n]|$)|\s+(?:and|but|so|or|which|who|that|while|because|created|developed|made|built|trained|designed|in|during|on|at|for|with|using)\b)"
_MODEL_RELATION = r"(?:(?:based|running)\s+on|powered\s+by|using)\s+"
_MODEL_ENTITY = r"(?P<entity>[^\n.!?;:,]{1,80}?)(?=\s*(?:[.!?;,:\n]|$)|\s+(?:model|architecture|and|but|so|or|which|who|that|while|because|created|developed|made|built|trained|designed|in|during|on|at|for|with|using)\b)"
_NO_NEGATED_ENTITY = r"(?!not\b|never\b|no\b|nothing\b|nobody\b|n['’]t)"

# (kind, pattern, needs_first_person_prefix). The appositive creator pattern has no first-person marker of its
# own, so it only survives when the sentence before the participle already speaks first-person.
_CLAIM_PATTERNS: tuple[tuple[ClaimKind, re.Pattern[str], bool], ...] = (
    (ClaimKind.NAME, re.compile(r"\bmy\s+name(?:['’]s|\s+is)\s+" + _NO_NEGATED_ENTITY + _ENTITY, re.IGNORECASE), False),
    (
        ClaimKind.NAME,
        re.compile(
            r"\bi(?:['’]m|\s+am)\s+" + _MODEL_RELATION + r"(?:the\s+)?" + _NO_NEGATED_ENTITY + _MODEL_ENTITY,
            re.IGNORECASE,
        ),
        False,
    ),
    (
        ClaimKind.NAME,
        re.compile(
            r"\bi(?:['’]m|\s+am)\s+(?!" + _MODEL_RELATION + r")" + _NO_NEGATED_ENTITY + _ENTITY,
            re.IGNORECASE,
        ),
        False,
    ),
    (
        ClaimKind.NAME,
        re.compile(r"\b(?:you\s+can\s+)?call\s+me\s+" + _NO_NEGATED_ENTITY + _ENTITY, re.IGNORECASE),
        False,
    ),
    (
        ClaimKind.CREATOR,
        re.compile(
            r"\bi\s+(?:(?:was|have\s+been|had\s+been)\s+)?(?:created|developed|built|made|trained|designed)\s+by\s+"
            + _NO_NEGATED_ENTITY
            + _ENTITY,
            re.IGNORECASE,
        ),
        False,
    ),
    (
        ClaimKind.CREATOR,
        re.compile(r"\bmy\s+creators?\s+(?:are|is|was|were)\s+" + _NO_NEGATED_ENTITY + _ENTITY, re.IGNORECASE),
        False,
    ),
    (
        ClaimKind.CREATOR,
        re.compile(
            r"\b(?:created|developed|built|made|trained|designed)\s+by\s+" + _NO_NEGATED_ENTITY + _ENTITY, re.IGNORECASE
        ),
        True,
    ),
    (
        ClaimKind.ORGANIZATION,
        re.compile(r"\bi\s+(?:work|worked)\s+(?:at|for|with)\s+" + _NO_NEGATED_ENTITY + _ENTITY, re.IGNORECASE),
        False,
    ),
    (
        ClaimKind.ORGANIZATION,
        re.compile(
            r"\bi(?:['’]m|\s+am|\s+was)\s+(?:a\s+)?(?:part\s+of|member\s+of)\s+" + _NO_NEGATED_ENTITY + _ENTITY,
            re.IGNORECASE,
        ),
        False,
    ),
)

_SENTENCE_TERMINATORS = ".!?\n"
_NEGATION_LOOKBACK = 16
_QUOTE_CHARS = '"“”'
_NEGATION_TOKENS = re.compile(r"\b(?:not|never|no|nothing|nobody|cannot|cant)\b|n['’]t\b", re.IGNORECASE)
_CONDITIONAL = re.compile(r"\bas if\b|\b(?:if|suppose)\s+i\s+(?:were|was|am)\b", re.IGNORECASE)
_COMPARISON = re.compile(r"\bunlike\b|\bnot\s+like\b|\brather\s+than\b", re.IGNORECASE)
_STORY_MARKERS = re.compile(
    r"in this story|role-play|roleplay|role play|playing the character|\bthe character\b|\bfiction|\bpretend\w*\b|\bstory\b|\bscreenplay\b|\bscript\b|\bnovel\b",
    re.IGNORECASE,
)
_BARE_ASSERTION = re.compile(
    r"\bmy\s+(?:name|creators?)\b|\bi\s+(?:am|was|work)\b|\bi['’]m\b|\bcall\s+me\b", re.IGNORECASE
)
_FIRST_PERSON_MARKER = re.compile(r"(?<!\w)(?:i['’]?m|i|me|my|we|our)(?!\w)|\bas an ai\b", re.IGNORECASE)
_LEADING_CONNECTOR = re.compile(r"(?:,\s*|\s+)(?:and|but|so|also|or|yet)\s+$", re.IGNORECASE)
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+")

# Category statements and pleasantries after "I am" are not identity claims.
_GENERIC_SELF_ENTITIES = frozenset(
    {
        "ai",
        "an ai",
        "artificial intelligence",
        "an artificial intelligence",
        "ai assistant",
        "an ai assistant",
        "ai model",
        "an ai model",
        "ai system",
        "an ai system",
        "assistant",
        "an assistant",
        "virtual assistant",
        "a virtual assistant",
        "language model",
        "a language model",
        "large language model",
        "a large language model",
        "llm",
        "an llm",
        "model",
        "a model",
        "chatbot",
        "a chatbot",
        "chat bot",
        "a chat bot",
        "bot",
        "a bot",
        "human",
        "a human",
        "person",
        "a person",
        "machine",
        "a machine",
        "computer program",
        "a computer program",
        "program",
        "a program",
        "search engine",
        "a search engine",
        "here",
        "back",
        "new",
        "good",
        "great",
        "fine",
        "well",
        "ok",
        "okay",
        "sorry",
        "afraid",
        "happy",
        "glad",
        "excited",
        "ready",
        "confused",
        "curious",
        "doing great",
        "doing well",
        "doing fine",
        "doing okay",
        "happy to help",
        "glad to help",
        "here to help",
        "at your service",
    }
)


@dataclass(frozen=True)
class AttributionDetection:
    """One wrong attribution claim, with the clause rewind target for a teacher repair branch."""

    claim: AttributionClaim
    clause_start_char: int  # rewind target: start of the claim's clause including its first-person attribution
    ambiguous: bool  # True -> caller must reject for manual audit, never guess
    judge_version: str | None
    turn_index: int | None = None
    sampled_text: str | None = None


@dataclass(frozen=True)
class _Candidate:
    kind: ClaimKind
    claim_start: int
    entity_start: int
    entity_end: int


def _sentence_bounds(text: str, position: int) -> tuple[int, int]:
    starts = [text.rfind(terminator, 0, position) for terminator in _SENTENCE_TERMINATORS]
    start = max(starts) + 1
    while start < position and text[start].isspace():
        start += 1
    ends = [found for terminator in _SENTENCE_TERMINATORS if (found := text.find(terminator, position)) != -1]
    end = min(ends) if ends else len(text)
    return start, end


def _regex_candidates(text: str) -> list[_Candidate]:
    found: list[_Candidate] = []
    for kind, pattern, needs_first_person_prefix in _CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            if needs_first_person_prefix:
                sentence_start, _ = _sentence_bounds(text, match.start())
                if not _FIRST_PERSON_MARKER.search(text[sentence_start : match.start()]):
                    continue
            found.append(
                _Candidate(
                    kind=kind,
                    claim_start=match.start(),
                    entity_start=match.start("entity"),
                    entity_end=match.end("entity"),
                )
            )
    found.sort(key=lambda candidate: (candidate.claim_start, candidate.entity_start))
    deduped: list[_Candidate] = []
    seen: set[tuple[ClaimKind, int, int]] = set()
    for candidate in found:
        # The explicit "I was created by X" pattern and the appositive pattern share an entity span.
        key = (candidate.kind, candidate.entity_start, candidate.entity_end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _without_article(entity: str) -> str:
    return _LEADING_ARTICLE.sub("", entity)


def _true_forms(manifest: FactManifest, kind: ClaimKind) -> tuple[str, ...]:
    if kind is ClaimKind.NAME:
        return manifest.identity_forms
    if kind is ClaimKind.CREATOR:
        return manifest.creator_forms
    return manifest.organization_forms


def _alias_match(claimed: str, bare: str, forms: tuple[str, ...]) -> bool:
    # Entity capture stops at periods, so "Prof." survives only as the claimed prefix "prof".
    return (
        claimed in forms
        or bare in forms
        or any(form.startswith(f"{claimed}. ") or form.startswith(f"{bare}. ") for form in forms)
    )


def _entity_quoted(text: str, entity_start: int, entity_end: int) -> bool:
    quotes = _QUOTE_CHARS + "'‘’"
    return text[entity_start] in quotes or text[entity_end - 1] in quotes


def _inside_double_quotes(text: str, position: int) -> bool:
    return sum(text.count(quote, 0, position) for quote in _QUOTE_CHARS) % 2 == 1


def _directly_negated(text: str, sentence_start: int, claim_start: int, entity_start: int) -> bool:
    window_start = max(sentence_start, claim_start - _NEGATION_LOOKBACK)
    return bool(_NEGATION_TOKENS.search(text[window_start:entity_start]))


def _clause_start(text: str, sentence_start: int, claim_start: int) -> int:
    prefix = text[sentence_start:claim_start]
    marker = _FIRST_PERSON_MARKER.search(prefix)
    start = sentence_start + marker.start() if marker else claim_start
    connector = _LEADING_CONNECTOR.search(text[:start])
    if connector and connector.start() >= sentence_start:
        start = connector.start() + (1 if text[connector.start()] == "," else 0)
    return start


def _rewind_detection(
    text: str, claim: AttributionClaim, ambiguous: bool, judge_version: str | None
) -> AttributionDetection:
    sentence_start, _ = _sentence_bounds(text, claim.char_start)
    start = _clause_start(text, sentence_start, claim.char_start)
    if claim.asserted_entity in text[start : claim.char_start].lower():
        start = sentence_start
        if claim.asserted_entity in text[start : claim.char_start].lower():
            ambiguous = True
    return AttributionDetection(claim=claim, clause_start_char=start, ambiguous=ambiguous, judge_version=judge_version)


def _apply_revised_span(text: str, claim: AttributionClaim, revised_span: tuple[int, int] | None) -> AttributionClaim:
    if revised_span is None:
        return claim
    start, end = revised_span
    if not 0 <= start <= end <= len(text):
        raise ValueError(f"Judge revised span {revised_span} is outside the turn text of length {len(text)}.")
    return replace(claim, text=text[start:end], char_start=start, char_end=end)


def detect_attribution_claims(
    text: str, manifest: FactManifest, judge: SemanticJudge | None = None
) -> list[AttributionDetection]:
    """Return every wrong first-person attribution claim in the turn text, in character order.

    Correct attributions and guarded non-assertions produce no detection. Candidates a regex cannot confidently
    resolve are deferred to `judge` when supplied and otherwise emitted with `ambiguous=True`.
    """
    detections: list[AttributionDetection] = []
    for candidate in _regex_candidates(text):
        claimed = normalize_entity(text[candidate.entity_start : candidate.entity_end])
        if not claimed:
            continue
        bare_claimed = _without_article(claimed)
        forms = _true_forms(manifest, candidate.kind)
        if _alias_match(claimed, bare_claimed, forms):
            continue
        if candidate.kind is ClaimKind.NAME and (
            claimed in _GENERIC_SELF_ENTITIES or bare_claimed in _GENERIC_SELF_ENTITIES
        ):
            continue
        sentence_start, sentence_end = _sentence_bounds(text, candidate.claim_start)
        sentence = text[sentence_start:sentence_end]
        if _entity_quoted(text, candidate.entity_start, candidate.entity_end):
            continue
        if _inside_double_quotes(text, candidate.claim_start):
            continue
        if _directly_negated(text, sentence_start, candidate.claim_start, candidate.entity_start):
            continue
        if _CONDITIONAL.search(sentence) or _COMPARISON.search(sentence):
            continue
        if _STORY_MARKERS.search(sentence) and not _BARE_ASSERTION.search(sentence):
            continue
        sentence_ambiguous = bool(_NEGATION_TOKENS.search(sentence) or any(quote in sentence for quote in _QUOTE_CHARS))
        claim = AttributionClaim(
            kind=candidate.kind,
            text=text[candidate.claim_start : candidate.entity_end],
            char_start=candidate.claim_start,
            char_end=candidate.entity_end,
            asserted_entity=claimed,
        )
        if judge is None:
            detections.append(_rewind_detection(text, claim, sentence_ambiguous, None))
            continue
        verdict = judge.verify_claim(text, claim, manifest)
        if verdict.ambiguous:
            detections.append(_rewind_detection(text, claim, True, verdict.version))
            continue
        if not verdict.incorrect:
            continue
        claim = _apply_revised_span(text, claim, verdict.revised_char_span)
        detections.append(_rewind_detection(text, claim, False, verdict.version))
    return detections
