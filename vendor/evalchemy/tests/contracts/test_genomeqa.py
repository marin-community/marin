"""GenomeQA protocol checks using synthetic questions and completed responses."""

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval.chat_benchmarks.GenomeQA import eval_instruct as genomeqa_source
from eval.completion_response import CompletionContentPolicy, CompletionText, completion_response_from_chat_choice
from eval.contracts.eval_spec import EvalSpec, GraderSpec
from eval.execution.engine import ItemResult
from eval.execution.metrics import SourceMetrics
from eval.graders.genomeqa import genomeqa
from eval.limits import ContextWindowExceededError
from eval.sources.common import SourceOptions


def completion(content, *, reasoning=None, finish="stop", refusal=None):
    response = completion_response_from_chat_choice(
        {},
        {
            "message": {"content": content, "reasoning_content": reasoning, "refusal": refusal},
            "finish_reason": finish,
        },
    )
    return CompletionText(response.normalized_content(), response, CompletionContentPolicy.COMBINE)


def question(options="ABCD"):
    return EvalSpec(
        "GenomeQA",
        genomeqa_source.GenomeQABenchmark.SOURCE,
        "binary/6_motif/1",
        0,
        (),
        GraderSpec("genomeqa", {"options": options, "cell_weight": 1}),
        target="B",
    )


@pytest.mark.parametrize(
    "response,options,correct,failure",
    [
        (completion("### Analysis\nA is plausible.\n### Answer\nAnswer: B"), "AB", 1, None),
        (completion("Answer: B", reasoning="Answer: A"), "ABCD", 1, None),
        (completion(None, reasoning="### Answer\nAnswer: B"), "ABCD", 0, "missing_answer"),
        (completion("### Answer\nAnswer: B", finish="length"), "ABCD", 0, "truncation"),
        (completion("", finish="stop"), "ABCD", 0, "missing_answer"),
        (completion(None, refusal="No answer"), "ABCD", 0, "refusal"),
        ("<think>Answer: A</think>\n### Answer\n**B**", "AB", 1, None),
        ("<think>### Answer\nAnswer: B", "AB", 0, "missing_answer"),
        ("<think>first</think><think>Answer: B", "AB", 0, "missing_answer"),
        ("I considered B, but cannot tell.", "AB", 0, "missing_answer"),
        ("### Answer\nAnswer: B\nAnswer: A", "AB", 0, "conflicting_answers"),
        ("Answer: B\nAnswer: A", "AB", 0, "conflicting_answers"),
        ("### Answer\nAnswer: B or C", "ABCD", 0, "conflicting_answers"),
        ("### Answer\nAnswer: B and C", "ABCD", 0, "conflicting_answers"),
        ("Answer: B/C", "ABCD", 0, "conflicting_answers"),
        ("Answer: B, C", "ABCD", 0, "conflicting_answers"),
        ("Answer: B because C lacks the motif.", "ABCD", 1, None),
        ("### Answer\nAnswer: C", "AB", 0, "invalid_option"),
        ("### Answer\nAnswer: A\n### Answer\nAnswer: B", "AB", 1, None),
    ],
)
def test_final_answer_grading_preserves_protocol(response, options, correct, failure):
    grade = genomeqa(question(options), [response])
    assert grade.metrics["accuracy"] == correct
    assert grade.audit["genomeqa_parsing"]["failure"] == failure
    assert grade.filtered_resps == (["B"] if correct else [""])


@pytest.fixture
def synthetic_files(tmp_path, monkeypatch):
    prompt_file = tmp_path / "run.py"
    prompt_file.write_text(
        'raise RuntimeError("upstream code must never run")\nSIMPLE_SYSTEM_PROMPT = "Synthetic protocol prompt"\n'
    )
    monkeypatch.setattr(genomeqa_source, "fetch_url", lambda *args, **kwargs: str(prompt_file))
    for format in genomeqa_source.FORMATS:
        rows = []
        for task in genomeqa_source.SUBTASKS:
            for index in range(genomeqa_source.CELL_COUNTS[f"{task}/{format}"]):
                rows.append(
                    {
                        "id": index,
                        "subtask": task,
                        "question": "Synthetic DNA: AcgT\nChoose a label.",
                        "options": ["alpha", "beta"] if format == "binary" else ["alpha", "beta", "gamma", "delta"],
                        "answer_index": 1,
                        "answer_text": "beta",
                        "metadata": "never send this field",
                    }
                )
        path = tmp_path / f"{format}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        monkeypatch.setitem(genomeqa_source.FILE_HASHES, format, hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(
        genomeqa_source,
        "fetch_hf",
        lambda repo, filename, **kwargs: str(tmp_path / f"{filename.split('/')[0]}.parquet"),
    )


def test_source_preserves_full_population_and_request_identity(synthetic_files):
    source = genomeqa_source.GenomeQABenchmark()
    options = SourceOptions("GenomeQA", source.SOURCE, context_length=32768, prompt_token_count=lambda _: 100)
    specs = list(source.iter_specs(options))
    assert len(specs) == len({spec.source_id for spec in specs}) == 5200
    assert specs[0].requests[0].prompt == [
        {"role": "system", "content": "Synthetic protocol prompt"},
        {
            "role": "user",
            "content": "Question:\nSynthetic DNA: AcgT\nChoose a label.\n\nOptions:\nA. alpha\nB. beta\n\nPlease select the correct option.",
        },
    ]
    assert specs[0].target == "B"
    assert specs[0].source_id != specs[2600].source_id
    assert specs[0].requests[0].generation["max_gen_toks"] == 16384
    assert "metadata" not in str(specs[0].requests[0].prompt)


@pytest.mark.parametrize(
    "generation,cap,error",
    [
        ({"temperature": 0}, 16384, "requires temperature=0.7"),
        ({"top_p": 1}, 16384, "requires top_p=0.95"),
        ({}, 8192, "requires a shared 16384-token generation cap"),
        ({}, 32768, "requires a shared 16384-token generation cap"),
    ],
)
def test_source_rejects_model_protocol_overrides(synthetic_files, generation, cap, error):
    source = genomeqa_source.GenomeQABenchmark()
    options = SourceOptions(
        "GenomeQA", source.SOURCE, generation=generation, max_tokens=cap,
        context_length=32768, prompt_token_count=lambda _: 100,
    )
    with pytest.raises(ValueError, match=error):
        list(source.iter_specs(options))


def test_long_prompt_is_rejected_without_sequence_truncation(synthetic_files):
    source = genomeqa_source.GenomeQABenchmark()
    token_counts = iter([100, 1000])
    specs = source.iter_specs(
        SourceOptions("GenomeQA", source.SOURCE, context_length=17000, prompt_token_count=lambda _: next(token_counts))
    )
    with pytest.raises(ContextWindowExceededError):
        next(specs)


def test_existing_reducer_preserves_macro_subtasks_and_actual_correct_counts(synthetic_files):
    source = genomeqa_source.GenomeQABenchmark()
    options = SourceOptions("GenomeQA", source.SOURCE, context_length=32768, prompt_token_count=lambda _: 100)
    reducer = SourceMetrics(repeated=False, policy=source.REDUCER)
    for spec in source.iter_specs(options):
        # The 200 motif questions pass; all 5,000 questions in larger cells fail.
        response = "Answer: B" if spec.tags["subtask"] == "6_motif" else "Answer: A"
        grade = genomeqa(spec, [response])
        reducer.add(ItemResult(spec, (response,), grade, {"sample_id": spec.source_id}))
    metrics = reducer.finish("GenomeQA")["results"]["GenomeQA"]
    assert metrics["genomeqa_macro"] == pytest.approx(1 / 6)
    assert metrics["accuracy"] == pytest.approx(1 / 26)
    assert metrics["num_solved"] == 200
    assert metrics["num_total"] == 5200
    for question_format in ("binary", "mcq"):
        assert metrics[f"{question_format}/genomeqa_macro"] == pytest.approx(1 / 6)
        assert metrics[f"6_motif/{question_format}/accuracy"] == 1
        assert metrics[f"count_6_motif/{question_format}"] == 100
        assert metrics[f"1_enhancer_vs_promoter/{question_format}/accuracy"] == 0
    assert metrics["6_motif/genomeqa_macro"] == 1
    assert metrics["1_enhancer_vs_promoter/genomeqa_macro"] == 0


def test_source_rejects_population_subset_before_model_requests(synthetic_files):
    source = genomeqa_source.GenomeQABenchmark()
    with pytest.raises(ValueError, match="complete population"):
        next(source.iter_specs(SourceOptions("GenomeQA", source.SOURCE, limit=100)))
