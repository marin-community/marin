"""RepoBench retains the original label-first edit-similarity policy."""

from difflib import SequenceMatcher
from argparse import Namespace
from types import SimpleNamespace

from fuzzywuzzy import fuzz
import pytest

from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.structured import code_line
from eval.chat_benchmarks.RepoBench import eval_instruct as repobench
from eval.contracts.task_outcome import TaskRoute
from eval.execution.driver import source_plans
from eval.execution.run_spec import RunSpec


@pytest.mark.parametrize("add_bos", [False, True])
def test_repobench_context_budget_uses_tokenizer_default_special_tokens(monkeypatch, add_bos):
    row = {"repo_name": "example", "file_path": "main.py",
           "context": [{"path": "other.py", "snippet": "MAGIC = 42"}],
           "import_statement": "from other import MAGIC", "cropped_code": "answer =", "next_line": "answer = 42"}
    tokenizer = SimpleNamespace(encode=lambda text: ([-1] if add_bos else []) + list(text.encode()))
    model = SimpleNamespace(tokenizer=tokenizer, tok_encode=lambda text: list(text.encode()))
    budget = len(repobench.construct_prompt(row).encode())
    monkeypatch.setattr(repobench, "load_dataset", lambda *_args, **_kwargs: {"in_file": [row]})
    source = repobench.RepoBenchmark(languages=["python"], subsets=["in_file"], max_tokens=budget)
    run = RunSpec.from_args(Namespace(model_args="pretrained=fixture"))

    [plan] = source_plans(["RepoBench"], {"RepoBench": TaskRoute.CUSTOM}, SimpleNamespace(get_benchmark=lambda _task: source), None, run, model)
    [spec] = plan.sources[0]
    expected = repobench.construct_prompt(row, tokenizer=tokenizer, max_token_nums=budget)

    assert ("MAGIC = 42" in expected) is not add_bos
    assert spec.requests[0].prompt == expected


@pytest.mark.parametrize("model_name,wrap_legacy", [("bigcode/starcoder", True), ("Qwen/Qwen3-8B", False)])
@pytest.mark.parametrize("legacy", [True, False])
def test_repobench_legacy_fim_prompt_uses_resolved_model_context(monkeypatch, model_name, wrap_legacy, legacy):
    raw = {"file_path": "example.py", "context": "# context", "import_statement": "import math", "code": "value = math.pi\n", "next_line": "print(value)"}
    monkeypatch.setattr(repobench, "load_data", lambda *, split, task, language, length, settings: [raw])
    modern = {**raw, "repo_name": "example", "context": [], "cropped_code": raw["code"]}
    monkeypatch.setattr(repobench, "load_dataset", lambda *_args, **_kwargs: {"in_file": [modern]})
    source = repobench.RepoBenchmark(languages=["python"], subsets=["in_file"], legacy_mode=legacy)
    run = RunSpec.from_args(Namespace(model_args=f"pretrained={model_name}"))
    def encode(text):
        return list(text.encode())

    model = SimpleNamespace(tok_encode=encode, tokenizer=SimpleNamespace(encode=encode))

    [plan] = source_plans(["RepoBench"], {"RepoBench": TaskRoute.CUSTOM}, SimpleNamespace(get_benchmark=lambda _task: source), None, run, model)
    [spec] = plan.sources[0]

    if legacy:
        prompt = repobench.construct_trainable_data([raw], language="python")[0]["data"]
        if wrap_legacy:
            prompt = "<fim_prefix>" + prompt + "<fim_suffix><fim_middle>"
    else:
        prompt = repobench.construct_prompt(modern, tokenizer=SimpleNamespace(encode=model.tok_encode), max_token_nums=128, language="python")
    assert spec.requests[0].prompt == prompt
    assert spec.target == "print(value)"


@pytest.mark.parametrize(
    ("label", "completion", "expected_em", "expected_es"),
    [("tide", "diet", 0, 25), ("diet", "tide", 0, 50), ("same", "same", 100, 100), ("tide", "", 0, 0)],
)
def test_repobench_edit_similarity_preserves_label_first_backend_policy(monkeypatch, label, completion, expected_em, expected_es):
    # Exercise the real supported stdlib backend even when optional C extensions
    # are installed; its directional matching makes argument order observable.
    monkeypatch.setattr(fuzz, "SequenceMatcher", SequenceMatcher)
    spec = EvalSpec("RepoBench", SourceRef("fixture"), 0, 0, (), GraderSpec("code_line", {"language": "python"}), target=label)

    grade = code_line(spec, [completion])

    assert grade.metrics == {"EM": expected_em, "ES": expected_es}
