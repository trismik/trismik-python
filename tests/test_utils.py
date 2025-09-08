"""Tests for utility functions in the Trismik client."""

import pytest

from trismik._utils import TrismikUtils


class TestMetricValueToType:
    """Test cases for the metric_value_to_type function."""

    def test_string_value_returns_string_type(self):
        """Test that string values return 'String' type."""
        assert TrismikUtils.metric_value_to_type("test_string") == "String"
        assert TrismikUtils.metric_value_to_type("") == "String"
        assert TrismikUtils.metric_value_to_type("0.93") == "String"

    def test_float_value_returns_float_type(self):
        """Test that float values return 'Float' type."""
        assert TrismikUtils.metric_value_to_type(0.93) == "Float"
        assert TrismikUtils.metric_value_to_type(1.0) == "Float"
        assert TrismikUtils.metric_value_to_type(-2.5) == "Float"
        assert TrismikUtils.metric_value_to_type(0.0) == "Float"

    def test_integer_value_returns_integer_type(self):
        """Test that integer values return 'Integer' type."""
        assert TrismikUtils.metric_value_to_type(85) == "Integer"
        assert TrismikUtils.metric_value_to_type(0) == "Integer"
        assert TrismikUtils.metric_value_to_type(-10) == "Integer"

    def test_bool_value_returns_boolean_type(self):
        """Test that boolean values return 'Boolean' type."""
        assert TrismikUtils.metric_value_to_type(True) == "Boolean"
        assert TrismikUtils.metric_value_to_type(False) == "Boolean"

    def test_list_value_raises_type_error(self):
        """Test that list values raise TypeError."""
        with pytest.raises(TypeError) as exc_info:
            TrismikUtils.metric_value_to_type([1, 2, 3])

        assert "Unsupported metric value type: list" in str(exc_info.value)
        assert "Supported types: str, float, int, bool" in str(exc_info.value)

    def test_dict_value_raises_type_error(self):
        """Test that dictionary values raise TypeError."""
        with pytest.raises(TypeError) as exc_info:
            TrismikUtils.metric_value_to_type({"key": "value"})

        assert "Unsupported metric value type: dict" in str(exc_info.value)
        assert "Supported types: str, float, int, bool" in str(exc_info.value)

    def test_none_value_raises_type_error(self):
        """Test that None values raise TypeError."""
        with pytest.raises(TypeError) as exc_info:
            TrismikUtils.metric_value_to_type(None)

        assert "Unsupported metric value type: NoneType" in str(exc_info.value)
        assert "Supported types: str, float, int, bool" in str(exc_info.value)

    def test_edge_case_very_large_int(self):
        """Test very large integer values."""
        large_int = 999999999999999999
        assert TrismikUtils.metric_value_to_type(large_int) == "Integer"

    def test_edge_case_very_small_float(self):
        """Test very small float values."""
        small_float = 0.000000001
        assert TrismikUtils.metric_value_to_type(small_float) == "Float"

    def test_edge_case_scientific_notation_float(self):
        """Test float in scientific notation."""
        sci_float = 1e-10
        assert TrismikUtils.metric_value_to_type(sci_float) == "Float"

    def test_edge_case_unicode_string(self):
        """Test unicode string values."""
        unicode_str = "测试字符串"
        assert TrismikUtils.metric_value_to_type(unicode_str) == "String"
