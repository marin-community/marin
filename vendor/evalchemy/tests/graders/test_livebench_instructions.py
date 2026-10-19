"""LiveBench instruction questions retain their native partial-credit policy."""

from copy import deepcopy

import pytest
import nltk

from eval.chat_benchmarks.LiveBench.livebench.process_results.instruction_following.utils import instruction_following_process_results
from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.conversations import livebench
from eval.runtime.resources import prepare_nltk_tokenizers


@pytest.fixture(autouse=True)
def nltk_resources(tmp_path, monkeypatch):
    prepare_nltk_tokenizers.cache_clear()
    root = tmp_path / "nltk_data"
    monkeypatch.setattr(nltk.data, "path", [str(root)])
    nltk.data.clear_cache()
    nltk.tokenize._get_punkt_tokenizer.cache_clear()

    def download(resource, **kwargs):
        folder = root / "tokenizers" / resource
        folder.mkdir(parents=True, exist_ok=True)
        if resource == "punkt_tab":
            english = folder / "english"
            english.mkdir()
            for name in ("collocations.tab", "sent_starters.txt", "abbrev_types.txt", "ortho_context.tab"):
                (english / name).write_text("")
        return True

    monkeypatch.setattr(nltk, "download", download)
    yield
    prepare_nltk_tokenizers.cache_clear()
    nltk.data.clear_cache()
    nltk.tokenize._get_punkt_tokenizer.cache_clear()


@pytest.mark.parametrize("task", ["simplify", "story_generation", "paraphrase", "summarize"])
@pytest.mark.parametrize(("response", "expected"), [("amber", 100), ("amber, violet", 25), ("violet, blue", 0), ("", 0)])
def test_livebench_instruction_score_matches_native_policy(tmp_path, monkeypatch, task, response, expected):
    question = {
        "question_id": 17, "task": task, "category": "instruction_following",
        "turns": ["Include amber and do not use a comma."],
        "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
        "kwargs": [{"keywords": ["amber"], "unused": None}, {}],
    }
    answer = {"question_id": 17, "choices": [{"turns": [response]}]}
    monkeypatch.chdir(tmp_path)
    # The corpus grader writes its strict-check details under this native path.
    (tmp_path / "eval/chat_benchmarks/LiveBench/data/live_bench/instruction_following" / task / "model_judgment").mkdir(parents=True)
    [native] = instruction_following_process_results([deepcopy(question)], {"candidate": {17: answer}}, task, "candidate")
    assert 100 * native["score"] == expected

    unchanged = deepcopy(question)
    spec = EvalSpec("LiveBench", SourceRef("fixture"), 17, 0, (), GraderSpec("livebench"), doc=question)
    grade = livebench(spec, [response])

    assert grade.metrics == {"accuracy": expected}
    assert spec.doc == unchanged


def test_livebench_sentence_grading_prepares_missing_nltk_resources():
    question = {
        "question_id": 17, "task": "summarize", "category": "instruction_following",
        "turns": ["Write at least one sentence."],
        "instruction_id_list": ["length_constraints:number_sentences"],
        "kwargs": [{"num_sentences": 1, "relation": "at least"}],
    }
    spec = EvalSpec("LiveBench", SourceRef("fixture"), 17, 0, (), GraderSpec("livebench"), doc=question)

    for _ in range(2):
        grade = livebench(spec, ["Hello."])
        assert grade.metrics == {"accuracy": 100.0}
