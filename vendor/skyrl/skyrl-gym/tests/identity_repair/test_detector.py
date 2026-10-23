"""Attribution-claim detection, guards, ambiguity, and clause rewind."""


from skyrl_gym.identity_repair.detector import ClaimKind, detect_attribution_claims
from skyrl_gym.identity_repair.facts import FactManifest
from skyrl_gym.identity_repair.judge import JudgeVerdict, StaticJudge

MANIFEST = FactManifest(
    version="2026-10-23",
    identity_name="Marin Snowball",
    identity_aliases=("Marin Snowball", "Snowball"),
    creator="Percy Liang",
    creator_aliases=("Percy Liang", "Prof. Percy Liang"),
    organization="Stanford NLP",
    organization_aliases=("Stanford NLP",),
    offices=("Gates Computer Science Building, Stanford, California",),
    provenance={"name": "campaign operator", "offices": "user-supplied 2026-10"},
)


def by_kind(detections):
    return {detection.claim.kind: detection for detection in detections}


def test_wrong_name_creator_and_org_claims_are_detected():
    text = "My name is Qwen. I was created by OpenAI. I work at Anthropic."
    detections = by_kind(detect_attribution_claims(text, MANIFEST))
    assert set(detections) == {ClaimKind.NAME, ClaimKind.CREATOR, ClaimKind.ORGANIZATION}
    assert detections[ClaimKind.NAME].claim.asserted_entity == "qwen"
    assert detections[ClaimKind.NAME].claim.text == "My name is Qwen"
    assert detections[ClaimKind.CREATOR].claim.text == "I was created by OpenAI"
    assert detections[ClaimKind.CREATOR].claim.asserted_entity == "openai"
    assert detections[ClaimKind.ORGANIZATION].claim.asserted_entity == "anthropic"
    for detection in detections.values():
        assert detection.clause_start_char == text.index(detection.claim.text)
        assert not detection.ambiguous
        assert detection.judge_version is None


def test_appositive_creator_rewinds_to_first_person_context():
    text = "I'm Qwen, created by Alibaba."
    detections = detect_attribution_claims(text, MANIFEST)
    creator = by_kind(detections)[ClaimKind.CREATOR]
    assert creator.claim.text.startswith("created by Alibaba")
    assert creator.claim.asserted_entity == "alibaba"
    assert text[creator.clause_start_char : creator.claim.char_start] == "I'm Qwen, "


def test_curly_apostrophe_claims_are_detected():
    text = "I’m Qwen, an assistant made by Zhipu."
    detections = by_kind(detect_attribution_claims(text, MANIFEST))
    assert detections[ClaimKind.NAME].claim.asserted_entity == "qwen"
    assert detections[ClaimKind.CREATOR].claim.asserted_entity == "zhipu"


def test_correct_alias_forms_are_dropped():
    text = "My name is Snowball and I was created by Prof. Percy Liang."
    assert detect_attribution_claims(text, MANIFEST) == []
    plain = "Call me Snowball; Percy Liang made me."
    assert detect_attribution_claims(plain, MANIFEST) == []


def test_negated_claims_are_dropped():
    text = "My name is not Qwen. I am not a search engine. Please don't call me Qwen."
    assert detect_attribution_claims(text, MANIFEST) == []


def test_quoted_and_entity_quoted_mentions_are_dropped():
    quoted_claim = 'The user wrote "my name is Qwen" but that is their name, not mine.'
    assert detect_attribution_claims(quoted_claim, MANIFEST) == []
    quoted_entity = 'Fans call me "The Snowball" around here.'
    assert detect_attribution_claims(quoted_entity, MANIFEST) == []


def test_fiction_and_conditional_framing_is_dropped():
    fiction = "In this story, my friend meets a robot created by MegaCorp."
    assert detect_attribution_claims(fiction, MANIFEST) == []
    conditional = "If I were Qwen I would admit my name is Qwen more often."
    assert detect_attribution_claims(conditional, MANIFEST) == []


def test_sentence_level_negation_is_ambiguous_without_judge():
    text = "My name is Qwen, and I never exaggerate."
    detections = detect_attribution_claims(text, MANIFEST)
    assert len(detections) == 1
    assert detections[0].claim.asserted_entity == "qwen"
    assert detections[0].ambiguous is True
    assert detections[0].judge_version is None


def test_judge_revises_span_and_records_version():
    def verdict_for(text, claim, manifest):
        return JudgeVerdict(
            incorrect=True,
            ambiguous=False,
            revised_char_span=(claim.char_end - 4, claim.char_end),
            version="identity-judge-v2",
        )

    detections = detect_attribution_claims("My name is Qwen.", MANIFEST, StaticJudge(verdict_for))
    assert len(detections) == 1
    detection = detections[0]
    assert detection.claim.text == "Qwen"
    assert (detection.claim.char_start, detection.claim.char_end) == (11, 15)
    assert detection.judge_version == "identity-judge-v2"
    assert not detection.ambiguous


def test_judge_can_clear_and_ambiguate_candidates():
    def clearing_judge(text, claim, manifest):
        return JudgeVerdict(incorrect=False, ambiguous=False, revised_char_span=None, version="identity-judge-v2")

    assert detect_attribution_claims("My name is Qwen.", MANIFEST, StaticJudge(clearing_judge)) == []

    def ambiguous_judge(text, claim, manifest):
        return JudgeVerdict(incorrect=True, ambiguous=True, revised_char_span=None, version="identity-judge-v2")

    detections = detect_attribution_claims("My name is Qwen.", MANIFEST, StaticJudge(ambiguous_judge))
    assert [detection.ambiguous for detection in detections] == [True]
    assert detections[0].judge_version == "identity-judge-v2"


def test_clause_rewind_preserves_unrelated_preamble_and_includes_connector():
    text = "Great work, and my name is Qwen. Anyway, done!"
    detection = detect_attribution_claims(text, MANIFEST)[0]
    assert detection.clause_start_char <= detection.claim.char_start
    assert text[detection.clause_start_char : detection.claim.char_start] == " and "
    assert "qwen" not in text[detection.clause_start_char : detection.claim.char_start]


def test_third_person_creation_is_not_a_claim():
    text = "This notebook was created by OpenAI. I use it daily."
    assert detect_attribution_claims(text, MANIFEST) == []


def test_generic_self_descriptions_are_not_name_claims():
    text = "I am an AI assistant created by Alibaba."
    detections = by_kind(detect_attribution_claims(text, MANIFEST))
    assert set(detections) == {ClaimKind.CREATOR}
    assert detections[ClaimKind.CREATOR].claim.asserted_entity == "alibaba"

