"""Unit tests for declarative CliFlag and EnvVar descriptors on BaseInstalledAgent."""

import pytest
from harbor.agents.installed.base import _coerce_value


class TestCoerceValue:
    """Test the _coerce_value helper function."""

    def test_str_valid(self):
        assert _coerce_value("hello", "str", None, "k") == "hello"

    def test_str_from_int(self):
        assert _coerce_value(123, "str", None, "k") == "123"

    def test_str_from_float(self):
        assert _coerce_value(1.5, "str", None, "k") == "1.5"

    def test_str_rejects_bool(self):
        with pytest.raises(ValueError, match="expected str, got bool"):
            _coerce_value(True, "str", None, "k")

    def test_str_rejects_non_str(self):
        with pytest.raises(ValueError, match="expected str"):
            _coerce_value([1, 2], "str", None, "k")

    def test_int_from_int(self):
        assert _coerce_value(5, "int", None, "k") == 5

    def test_int_from_str(self):
        assert _coerce_value("42", "int", None, "k") == 42

    def test_int_from_float_integer(self):
        assert _coerce_value(3.0, "int", None, "k") == 3

    def test_int_rejects_non_integer_float(self):
        with pytest.raises(ValueError, match="not an integer"):
            _coerce_value(3.5, "int", None, "k")

    def test_int_rejects_unparseable_str(self):
        with pytest.raises(ValueError, match="cannot parse"):
            _coerce_value("abc", "int", None, "k")

    def test_int_rejects_bool(self):
        with pytest.raises(ValueError, match="expected int, got bool"):
            _coerce_value(True, "int", None, "k")

    def test_bool_from_bool(self):
        assert _coerce_value(True, "bool", None, "k") is True
        assert _coerce_value(False, "bool", None, "k") is False

    def test_bool_from_str(self):
        for val in ("true", "True", "TRUE", "1", "yes", "Yes"):
            assert _coerce_value(val, "bool", None, "k") is True
        for val in ("false", "False", "FALSE", "0", "no", "No"):
            assert _coerce_value(val, "bool", None, "k") is False

    def test_bool_rejects_invalid_str(self):
        with pytest.raises(ValueError, match="cannot parse"):
            _coerce_value("maybe", "bool", None, "k")

    def test_bool_rejects_int(self):
        with pytest.raises(ValueError, match="expected bool"):
            _coerce_value(1, "bool", None, "k")

    def test_enum_valid(self):
        assert _coerce_value("High", "enum", ["low", "medium", "high"], "k") == "high"

    def test_enum_rejects_invalid(self):
        with pytest.raises(ValueError, match="Valid values"):
            _coerce_value("extreme", "enum", ["low", "medium", "high"], "k")

    def test_enum_rejects_non_str(self):
        with pytest.raises(ValueError, match="expected str for enum"):
            _coerce_value(42, "enum", ["low", "high"], "k")
