"""Original PDBThink scoring and coverage through the shared execution lifecycle."""

from dataclasses import replace
from pathlib import Path

import pytest

from eval.chat_benchmarks.PDBThink.data import BUNDLE_PATH
from eval.chat_benchmarks.PDBThink.eval_instruct import PDBThinkBenchmark
from eval.completion_response import CompletionContentPolicy, CompletionText, completion_response_from_chat_choice
from eval.limits import ContextWindowExceededError
from eval.sources.common import SourceOptions
from tests.contracts.pdbthink_records import record
from tests.contracts.source_harness import FixedEndpoint, MemoryCheckpoints, evaluate_source


def source_and_options(cache):
    source = PDBThinkBenchmark()
    source.dataset_cache = cache
    options = SourceOptions("PDBThink", source.SOURCE, context_length=1000,
                            prompt_token_count=lambda messages: len(str(messages).split()))
    return source, options


def test_family_macro_excludes_controls_and_survives_resume(pdbthink_bundle):
    records = [record("a"), record("b"), record("c", family="G01"),
               record("context", representation="context_only"), record("rotated", rotated=True)]
    source, options = source_and_options(pdbthink_bundle(records))
    endpoint = FixedEndpoint(lambda request: "FINAL: 2" if request.prompt[-1]["content"] in {"a", "b"} else "FINAL: 9")
    checkpoint = MemoryCheckpoints()
    metrics, samples = evaluate_source(source, options, endpoint, checkpoints={"PDBThink": checkpoint})
    assert metrics["macro_score"] == 0.5
    assert metrics["micro_score"] == pytest.approx(2 / 3)
    assert metrics["pdbthink_report"]["consistency"]["n_repeated_renders"] == 5
    assert [request.prompt for request in endpoint.requests] == [row["messages"] for row in records for _ in range(3)]

    resumed_endpoint = FixedEndpoint("FINAL: 9")
    resumed, resumed_samples = evaluate_source(source, options, resumed_endpoint, checkpoints={"PDBThink": checkpoint})
    assert resumed == metrics
    assert resumed_samples == samples
    assert resumed_endpoint.requests == []


def test_context_control_report_counts_identical_prompts_across_instances(pdbthink_bundle):
    first = record("first-control", representation="context_only")
    second = record("second-control", representation="context_only")
    first["report_metadata"]["semantic_instance_id"] = "first"
    second["report_metadata"]["semantic_instance_id"] = "second"
    second["messages"] = first["messages"]
    source, options = source_and_options(pdbthink_bundle([record("first"), record("second"), first, second]))

    metrics, _ = evaluate_source(source, options, FixedEndpoint("FINAL: 2"))

    assert metrics["macro_score"] == 1.0
    assert metrics["pdbthink_report"]["context_only_baseline"]["P01"] == {
        "n_instances": 2,
        "n_distinct_prompts": 1,
        "effective_sample_size": 1,
        "identical_prompts": True,
        "context_only_score": 1.0,
        "with_coordinates_score": 1.0,
        "gain_over_context_only": 0.0,
    }


@pytest.mark.parametrize("schema,gold,text,termination,message_extra,expected", [
    ("residue_set", {"value": ["A:A1", "A:G2"]}, "FINAL: A:A1", "stop", {}, 2 / 3),
    ("residue_pair", {"value": "A:A1--B:G2", "alternatives": ["A:A1--B:S3"]},
     "FINAL: B:S3--A:A1", "stop", {}, 1.0),
    ("residue_pair", {"value": "A:A1--B:G2", "alternatives": ["A:A1--B:S3"]},
     "FINAL: A:A1--B:V4", "stop", {}, 0.0),
    ("distance", {"value": 2.0}, "FINAL: 2.02", "stop", {}, 1.0),
    ("distance", {"value": 2.0}, "FINAL: 2.021", "stop", {}, 0.0),
    ("integer", {"value": 2}, "2", "stop", {}, 0.0),
    ("integer", {"value": 2}, "FINAL: 2", "length", {}, 0.0),
    ("integer", {"value": 2}, "FINAL: 2", "stop", {"refusal": "refused"}, 0.0),
    ("integer", {"value": 2}, "FINAL: 2", "tool_calls", {"tool_calls": [{"id": "unexpected"}]}, 0.0),
])
def test_upstream_parser_and_provider_signals_determine_score(pdbthink_bundle, schema, gold, text, termination,
                                                            message_extra, expected):
    source, options = source_and_options(pdbthink_bundle([record(schema=schema, gold=gold)]))
    choice = {"message": {"content": text, **message_extra}, "finish_reason": termination}
    response = CompletionText(text, completion_response_from_chat_choice({}, choice), CompletionContentPolicy.COMBINE)
    metrics, samples = evaluate_source(source, options, FixedEndpoint(response))
    assert metrics["macro_score"] == pytest.approx(expected)
    assert samples[0]["completion_responses"][0][0]["finish_reason"] == termination


def test_mechanistic_contact_alternatives_reach_the_upstream_scorer(pdbthink_bundle):
    row = record(schema="multi_field", gold={"fields": {
        "contact": {"schema": "residue_pair", "value": "A:A1--B:G2", "alternatives": ["A:A1--B:S3"]},
    }})
    row["parameters"] = {"field_schemas": {"contact": "residue_pair"}}
    source, options = source_and_options(pdbthink_bundle([row]))
    metrics, _ = evaluate_source(source, options, FixedEndpoint("FINAL\ncontact: B:S3--A:A1"))
    assert metrics["macro_score"] == 1.0


def test_context_failure_prevents_partial_cohort_execution(pdbthink_bundle):
    source, options = source_and_options(pdbthink_bundle([record("short"), record("long " * 1000)]))
    endpoint = FixedEndpoint("FINAL: 2")
    with pytest.raises(ContextWindowExceededError):
        evaluate_source(source, options, endpoint)
    assert endpoint.requests == []
    metrics, _ = evaluate_source(source, replace(options, limit=1), endpoint)
    assert metrics["macro_score"] == 1.0
    assert len(endpoint.requests) == 3


def test_changed_bundle_is_rejected_before_generation(pdbthink_bundle):
    cache = pdbthink_bundle([record()])
    path = Path(cache) / BUNDLE_PATH
    path.write_bytes(path.read_bytes() + b"tampered")
    source, options = source_and_options(cache)
    endpoint = FixedEndpoint("FINAL: 2")
    with pytest.raises(ValueError, match="checksum"):
        evaluate_source(source, options, endpoint)
    assert endpoint.requests == []
