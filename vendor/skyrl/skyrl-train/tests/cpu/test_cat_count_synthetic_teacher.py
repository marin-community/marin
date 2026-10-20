import math

import pytest
from examples.cat_count.cpu_canary import PROMPT, build_tokenizer
from examples.cat_count.synthetic_teacher import CatCountTeacher, TeacherNoise

HIGH = math.log(0.95)
# The tiny word-level vocabulary has 38 tokens; the other 0.05 is spread over 37 of them.
LOW = math.log(0.05 / 37)


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer()


def sequence(tokenizer, n: int, response: str, eos: bool = True) -> tuple[list[int], int]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(N=n)}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    reply = tokenizer.encode(response, add_special_tokens=False) + ([tokenizer.eos_token_id] if eos else [])
    return prompt + reply, len(prompt)


def response_scores(teacher, tokenizer, n, response, eos=True):
    full, start = sequence(tokenizer, n, response, eos)
    scores = teacher.score(full)
    assert len(scores) == len(full)
    assert scores[0] is None
    return scores[start:]


@pytest.mark.parametrize(
    ("n", "response", "expected"),
    [
        (3, "cat cat cat", [HIGH, HIGH, HIGH, HIGH]),
        # The extra cat leaves the answer; the teacher then prefers stopping.
        (2, "cat cat cat", [HIGH, HIGH, LOW, HIGH]),
        (3, "cat cat", [HIGH, HIGH, LOW]),
        (2, "cat dog cat", [HIGH, LOW, LOW, HIGH]),
        (12, " ".join(["cat"] * 12), [HIGH] * 13),
    ],
)
def test_teacher_prefers_the_exact_count(tokenizer, n, response, expected):
    teacher = CatCountTeacher(tokenizer, TeacherNoise())
    assert response_scores(teacher, tokenizer, n, response) == pytest.approx(expected)


def test_flipped_teacher_swaps_preferences(tokenizer):
    teacher = CatCountTeacher(tokenizer, TeacherNoise(flipped=True))
    assert response_scores(teacher, tokenizer, 2, "cat cat cat") == pytest.approx([LOW, LOW, HIGH, LOW])


def test_jitter_is_deterministic_and_capped(tokenizer):
    noise = TeacherNoise(jitter=2.0, seed=3)
    first = response_scores(CatCountTeacher(tokenizer, noise), tokenizer, 4, "cat cat cat cat")
    second = response_scores(CatCountTeacher(tokenizer, noise), tokenizer, 4, "cat cat cat cat")
    assert first == second
    assert all(score <= 0 for score in first)
    assert first != pytest.approx([HIGH] * 5)


def test_error_rate_moves_the_target_count(tokenizer):
    teacher = CatCountTeacher(tokenizer, TeacherNoise(error_rate=1.0))
    scores = response_scores(teacher, tokenizer, 4, "cat cat cat cat")
    assert min(scores) == pytest.approx(LOW)


def test_teacher_rejects_a_sequence_without_a_response_header(tokenizer):
    teacher = CatCountTeacher(tokenizer, TeacherNoise())
    with pytest.raises(ValueError, match="assistant header"):
        teacher.score(tokenizer.encode("cat cat", add_special_tokens=False))


def test_a_generated_header_does_not_move_the_response_boundary(tokenizer):
    teacher = CatCountTeacher(tokenizer, TeacherNoise())
    assert response_scores(teacher, tokenizer, 2, "cat <|assistant|> dog") == pytest.approx([HIGH, LOW, LOW, HIGH])


def test_a_special_token_that_is_not_eos_is_wrong(tokenizer):
    teacher = CatCountTeacher(tokenizer, TeacherNoise())
    assert response_scores(teacher, tokenizer, 2, "cat cat <unk>") == pytest.approx([HIGH, HIGH, LOW, HIGH])


def test_jitter_depends_only_on_the_prompt_and_earlier_tokens(tokenizer):
    teacher = CatCountTeacher(tokenizer, TeacherNoise(jitter=0.5, seed=3))
    same_prefix = response_scores(teacher, tokenizer, 2, "cat cat", eos=False)
    other_suffix = response_scores(teacher, tokenizer, 2, "cat dog", eos=False)
    assert same_prefix[0] == other_suffix[0]
    longer = response_scores(teacher, tokenizer, 2, "cat cat cat", eos=False)
    assert longer[:2] == same_prefix


@pytest.mark.parametrize("extra_tokens", [0, 17])
@pytest.mark.parametrize("extra_tokens", [0, 17])
def test_wrong_score_scales_with_the_vocabulary(extra_tokens):
    tokenizer = build_tokenizer()
    tokenizer.add_tokens([f"unused_word_{index}" for index in range(extra_tokens)])
    teacher = CatCountTeacher(tokenizer, TeacherNoise())
    wrong_score = response_scores(teacher, tokenizer, 2, "dog", eos=False)[0]
    assert math.exp(wrong_score) * (len(tokenizer) - 1) == pytest.approx(0.05)


def test_generation_config_stop_tokens_end_a_reply(tokenizer):
    pad = tokenizer.pad_token_id
    teacher = CatCountTeacher(tokenizer, TeacherNoise(), stop_token_ids=(tokenizer.eos_token_id, pad))
    full, start = sequence(tokenizer, 2, "cat cat", eos=False)
    assert teacher.score([*full, pad])[start:] == pytest.approx([HIGH, HIGH, HIGH])
