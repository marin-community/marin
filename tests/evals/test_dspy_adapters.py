"""Tests for GBNFAdapter and XGrammarAdapter parsers."""

import enum
from typing import Optional

import dspy
import pytest

from experiments.dspy.adapters.gbnf import GBNFAdapter, build_gbnf_grammar, _type_to_rule_ref
from experiments.dspy.adapters.xgrammar import XGrammarAdapter, build_json_schema, _annotation_to_json_schema


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class Label(enum.Enum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class SimpleSignature(dspy.Signature):
    """Test signature."""
    claim: str = dspy.InputField()
    label: Label = dspy.OutputField()
    reasoning: str = dspy.OutputField()


class ListSignature(dspy.Signature):
    """Signature with list output."""
    query: str = dspy.InputField()
    items: list[str] = dspy.OutputField()


class OptionalSignature(dspy.Signature):
    """Signature with optional output."""
    query: str = dspy.InputField()
    label: Optional[Label] = dspy.OutputField()


# ---------------------------------------------------------------------------
# GBNF tests
# ---------------------------------------------------------------------------

class TestGBNFGrammar:
    def test_enum_rule_in_grammar(self):
        grammar = build_gbnf_grammar(SimpleSignature)
        assert "SUPPORTED" in grammar
        assert "NOT_SUPPORTED" in grammar

    def test_list_rule_in_grammar(self):
        grammar = build_gbnf_grammar(ListSignature)
        assert "list-of-string" in grammar

    def test_optional_unwrapped(self):
        """Optional[Label] should produce enum rule, not fall back to string."""
        rules: dict = {}
        ref = _type_to_rule_ref(Optional[Label], rules)
        assert ref != "string"
        assert "SUPPORTED" in " ".join(rules.values())

    def test_empty_signature(self):
        class EmptySig(dspy.Signature):
            """Empty."""
            query: str = dspy.InputField()

        grammar = build_gbnf_grammar(EmptySig)
        assert "root" in grammar


class TestGBNFAdapterParse:
    def setup_method(self):
        self.adapter = GBNFAdapter()

    def test_parse_enum(self):
        completion = "### Label:\nSUPPORTED\n### Reasoning:\nBecause it is true."
        result = self.adapter.parse(SimpleSignature, completion)
        assert result["label"] == Label.SUPPORTED
        assert "true" in result["reasoning"]

    def test_parse_missing_field(self):
        completion = "### Label:\nSUPPORTED"
        result = self.adapter.parse(SimpleSignature, completion)
        assert result["label"] == Label.SUPPORTED
        assert result["reasoning"] is None

    def test_parse_list(self):
        completion = "### Items:\n- apple\n- banana\n- cherry"
        result = self.adapter.parse(ListSignature, completion)
        assert result["items"] == ["apple", "banana", "cherry"]

    def test_parse_empty(self):
        result = self.adapter.parse(SimpleSignature, "")
        assert result["label"] is None
        assert result["reasoning"] is None

    def test_parse_optional_enum(self):
        completion = "### Label:\nNOT_SUPPORTED"
        result = self.adapter.parse(OptionalSignature, completion)
        assert result["label"] == Label.NOT_SUPPORTED

    def test_coerce_enum_case_insensitive(self):
        result = GBNFAdapter._coerce("supported", Label)
        assert result == Label.SUPPORTED

    def test_coerce_unknown_enum_returns_none(self):
        result = GBNFAdapter._coerce("unknown_value", Label)
        assert result is None


# ---------------------------------------------------------------------------
# XGrammar JSON schema tests
# ---------------------------------------------------------------------------

class TestXGrammarSchema:
    def test_enum_schema(self):
        schema = _annotation_to_json_schema(Label)
        assert schema["type"] == "string"
        assert "SUPPORTED" in schema["enum"]

    def test_optional_enum_schema(self):
        schema = _annotation_to_json_schema(Optional[Label])
        assert schema["type"] == "string"
        assert "SUPPORTED" in schema["enum"]

    def test_list_schema(self):
        schema = _annotation_to_json_schema(list[str])
        assert schema["type"] == "array"
        assert schema["items"]["type"] == "string"

    def test_build_json_schema_required(self):
        schema = build_json_schema(SimpleSignature)
        assert "label" in schema["required"]
        assert "reasoning" in schema["required"]

    def test_build_json_schema_optional_not_required(self):
        schema = build_json_schema(OptionalSignature)
        assert "label" not in schema["required"]


class TestXGrammarAdapterParse:
    def setup_method(self):
        self.adapter = XGrammarAdapter()

    def test_parse_valid_json(self):
        completion = '{"label": "SUPPORTED", "reasoning": "It is true."}'
        result = self.adapter.parse(SimpleSignature, completion)
        assert result["label"] == Label.SUPPORTED
        assert result["reasoning"] == "It is true."

    def test_parse_markdown_fence(self):
        completion = '```json\n{"label": "NOT_SUPPORTED", "reasoning": "False."}\n```'
        result = self.adapter.parse(SimpleSignature, completion)
        assert result["label"] == Label.NOT_SUPPORTED

    def test_parse_invalid_json(self):
        result = self.adapter.parse(SimpleSignature, "not json at all")
        assert result["label"] is None
        assert result["reasoning"] is None

    def test_parse_empty(self):
        result = self.adapter.parse(SimpleSignature, "")
        assert result["label"] is None

    def test_parse_list_field(self):
        completion = '{"items": ["apple", "banana"]}'
        result = self.adapter.parse(ListSignature, completion)
        assert result["items"] == ["apple", "banana"]

    def test_coerce_optional_enum(self):
        result = XGrammarAdapter._coerce("SUPPORTED", Optional[Label])
        assert result == Label.SUPPORTED

    def test_coerce_enum_case_insensitive(self):
        result = XGrammarAdapter._coerce("not_supported", Label)
        assert result == Label.NOT_SUPPORTED

    def test_coerce_unknown_enum_returns_none(self):
        result = XGrammarAdapter._coerce("unknown", Label)
        assert result is None

    def test_coerce_list_from_string_comma(self):
        result = XGrammarAdapter._coerce("apple, banana, cherry", list[str])
        assert result == ["apple", "banana", "cherry"]
