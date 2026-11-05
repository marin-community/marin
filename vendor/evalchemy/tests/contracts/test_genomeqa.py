"""GenomeQA protocol checks using synthetic questions and completed responses."""

import hashlib
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval.chat_benchmarks.GenomeQA import eval_instruct as genomeqa_source
from eval.completion_response import CompletionContentPolicy, CompletionText, completion_response_from_chat_choice
from eval.contracts.aggregation import complete_macro
from eval.contracts.eval_spec import EvalSpec, GraderSpec
from eval.graders.genomeqa import genomeqa
from eval.sources.common import SourceOptions
from eval.limits import ContextWindowExceededError


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
        GraderSpec("genomeqa", {"options": options}),
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
        ("### Answer\nAnswer: C", "AB", 0, "invalid_option"),
        ("### Answer\nAnswer: A\n### Answer\nAnswer: B", "AB", 1, None),
    ],
)
def test_final_answer_grading_preserves_protocol(response, options, correct, failure):
    grade = genomeqa(question(options), [response])
    assert grade.metrics["accuracy"] == correct
    assert grade.audit["genomeqa_parsing"]["failure"] == failure
    assert grade.filtered_resps == (["B"] if correct else [""])


def test_macro_preserves_cell_weight_and_missing_population():
    expected = {"large/binary": 500, "large/mcq": 500, "motif/binary": 100, "motif/mcq": 100}
    groups = {"macro": tuple(expected), "large": ("large/binary", "large/mcq"), "motif": ("motif/binary", "motif/mcq")}
    counts = {"large/binary": (500, 500), "large/mcq": (0, 500), "motif/binary": (100, 100), "motif/mcq": (100, 100)}
    result = complete_macro(counts, expected, groups)
    assert result["macro"] == 0.75
    assert result["accuracy"] == pytest.approx(700 / 1200)
    assert result["num_solved"] == 700
    assert result["large"] == 0.5
    assert result["motif"] == 1
    partial = complete_macro({**counts, "motif/mcq": (99, 99)}, expected, groups)
    assert "macro" not in partial and "motif" not in partial
    assert partial["large"] == 0.5
    assert partial["motif/mcq/partial_accuracy"] == 1
    assert partial["motif/mcq/missing"] == 1
    missing = complete_macro({**counts, "motif/mcq": (0, 0)}, expected, groups)
    assert "macro" not in missing and "motif/mcq/partial_accuracy" not in missing
    empty = complete_macro({cell: (0, size) for cell, size in expected.items()}, expected, groups)
    assert empty["macro"] == 0 and empty["complete"]


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
    motif = genomeqa_source.GenomeQAMotifMCQBenchmark()
    selected = list(motif.iter_specs(replace(options, task="GenomeQAMotifMCQ")))
    assert len(selected) == 100
    assert all(spec.tags == {"cell": "6_motif/mcq"} for spec in selected)


@pytest.mark.parametrize(
    "generation,cap", [({"temperature": 0}, 16384), ({"top_p": 1}, 16384), ({}, 8192), ({}, 32768)]
)
def test_source_rejects_model_protocol_overrides_before_loading(synthetic_files, generation, cap):
    source = genomeqa_source.GenomeQABenchmark()
    with pytest.raises(ValueError):
        list(source.iter_specs(SourceOptions("GenomeQA", source.SOURCE, generation=generation, max_tokens=cap)))


def test_long_prompt_is_rejected_without_sequence_truncation(synthetic_files):
    source = genomeqa_source.GenomeQABenchmark()
    token_counts = iter([100, 1000])
    specs = source.iter_specs(
        SourceOptions("GenomeQA", source.SOURCE, context_length=17000, prompt_token_count=lambda _: next(token_counts))
    )
    with pytest.raises(ContextWindowExceededError):
        next(specs)
