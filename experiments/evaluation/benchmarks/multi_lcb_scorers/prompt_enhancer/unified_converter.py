#!/usr/bin/env python3
"""
Proposed unified architecture for data converters.

"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from collections import Counter


class DataType(Enum):
    """Data types."""

    SCALAR = "scalar"
    STRING = "string"
    ARRAY_1D = "array_1d"
    ARRAY_2D_MATRIX = "array_2d_matrix"
    ARRAY_2D = "array_2d"


class TaskType(Enum):
    """Task types."""

    SCALAR = "scalar"  # All input/output data are scalars (numbers, strings, booleans).
    ONE_DIM = "one_dim"  # At least one one-dimensional array is present, but no two-dimensional arrays.
    TWO_DIM = "two_dim"  # Exactly one two-dimensional array is present; others can be scalars/one-dimensional.
    TWO_DIM_AMBIGUOUS = (
        "two_dim_ambiguous"  # Two or more two-dimensional arrays are present.
    )


class DataItem:
    """Data item with meta-information."""

    def __init__(
        self, value: Any, data_type: DataType, metadata: Dict[str, Any] = None
    ):
        self.value = value
        self.data_type = data_type
        self.metadata = metadata or {}

    @property
    def rows(self) -> int:
        """Number of rows for 2D arrays."""
        if self.data_type in [DataType.ARRAY_2D_MATRIX, DataType.ARRAY_2D]:
            return len(self.value) if self.value else 0
        return 0

    @property
    def cols(self) -> int:
        """Number of columns for matrices."""
        if self.data_type == DataType.ARRAY_2D_MATRIX and self.value:
            return len(self.value[0]) if self.value[0] else 0
        return 0

    @property
    def length(self) -> int:
        """Length for 1D arrays."""
        if self.data_type == DataType.ARRAY_1D:
            return len(self.value) if self.value else 0
        return 0


@dataclass(frozen=True)
class TestCase:
    __test__ = False  # not a pytest
    input: List[DataItem]
    output: DataItem


@dataclass(frozen=True)
class UnifiedParameterFormat:
    """Represents the unified format of a single input/output parameter across all test cases.

    Attributes:
        var_name (str): The name of the variable (e.g., 'param_0', 'output').
        data_type (DataType): The consistent data type for this variable across all test cases.
    """

    var_name: str
    data_type: DataType


@dataclass(frozen=True)
class UnifiedTaskFormat:
    """Represents the unified format of a task's input and output parameters across all test cases.

    This aggregates the structure of input and output data across all provided test cases.

    Attributes:
        input_format (List[UnifiedParameterFormat]): A list of unified input parameter formats.
        output_format (Optional[UnifiedParameterFormat]): The unified output parameter format.
    """

    input_format: List[UnifiedParameterFormat]
    output_format: Optional[UnifiedParameterFormat]


class UnifiedDataAnalyzer:
    """Unified data type analyzer."""

    @staticmethod
    def is_matrix(array_2d: List[List]) -> bool:
        """
        Determines if a 2D array is a rectangular matrix.

        This logic is unified from LCB and PromptEnhancer.

        :param array_2d: The 2D array to check.
        :type array_2d: List[List]
        :return: True if the array is a matrix, False otherwise.
        :rtype: bool
        """
        if not array_2d or not isinstance(array_2d, list):
            return False
        if not isinstance(array_2d[0], list):
            return False
        first_row_length = len(array_2d[0])
        return all(len(row) == first_row_length for row in array_2d)

    @staticmethod
    def is_list2d(value: Any) -> bool:
        """Determines if a value is a 2D array."""
        return isinstance(value, list) and len(value) > 0 and isinstance(value[0], list)

    @staticmethod
    def analyze_value(value: Any, var_name: str = None) -> DataItem:
        """Analyzes a value and returns a DataItem."""
        if UnifiedDataAnalyzer.is_list2d(value):
            # 2D array
            if UnifiedDataAnalyzer.is_matrix(value):
                return DataItem(value, DataType.ARRAY_2D_MATRIX, {"var_name": var_name})
            else:
                return DataItem(value, DataType.ARRAY_2D, {"var_name": var_name})
        elif isinstance(value, list):
            # 1D array
            return DataItem(value, DataType.ARRAY_1D, {"var_name": var_name})
        elif isinstance(value, str):
            # String
            return DataItem(value, DataType.STRING, {"var_name": var_name})
        else:
            # Scalar
            return DataItem(value, DataType.SCALAR, {"var_name": var_name})

    @staticmethod
    def analyze_test_case(test_case: Dict[str, Any]) -> TestCase:
        """Analyzes a raw test case dictionary and converts it into a TestCase object.

        :param test_case: A dictionary representing a single test case with 'input' and 'output' keys.
        :type test_case: Dict[str, Any]
        :return: A TestCase object containing analyzed input and output DataItems.
        :rtype: TestCase
        """
        input_data = test_case.get("input", {})
        output_data = test_case.get("output")
        input_items = []
        for var_name, value in input_data.items():
            item = UnifiedDataAnalyzer.analyze_value(value, var_name)
            input_items.append(item)
        return TestCase(
            input_items, UnifiedDataAnalyzer.analyze_value(output_data, "output")
        )

    @staticmethod
    def analyze_test_cases(test_cases: List[Dict[str, Any]]) -> List[TestCase]:
        """Analyzes a list of raw test case dictionaries and converts them into TestCase objects.

        :param test_cases: A list of raw test case dictionaries.
        :type test_cases: List[Dict[str, Any]]
        :return: A list of TestCase objects.
        :rtype: List[TestCase]
        """
        return [
            UnifiedDataAnalyzer.analyze_test_case(test_case) for test_case in test_cases
        ]

    @staticmethod
    def analyze_unified_test_case_format(
        test_cases: List[TestCase],
    ) -> UnifiedTaskFormat:
        """Analyzes test cases to construct a unified input/output format for the task.

        This function determines the consistent data type for each input parameter and the output
        across all provided test cases.

        :param test_cases: A list of parsed test cases.
        :type test_cases: List[TestCase]
        :return: A UnifiedTaskFormat object describing the overall input and output structure.
        :rtype: UnifiedTaskFormat
        """

        if not test_cases:
            return UnifiedTaskFormat(input_format=[], output_format=None)

        # Check that all tests have the same number of input parameters
        input_params_count = len(test_cases[0].input)
        for tc in test_cases:
            if len(tc.input) != input_params_count:
                raise ValueError(
                    "Number of input parameters differs between test cases."
                )

        # Type counters
        input_type_counters: List[Counter] = [
            Counter() for _ in range(input_params_count)
        ]
        output_type_counter: Counter = Counter()

        for tc in test_cases:
            for i, item in enumerate(tc.input):
                input_type_counters[i][item.data_type] += 1
            if tc.output:
                output_type_counter[tc.output.data_type] += 1

        # Form the final input format
        final_input_format: List[UnifiedParameterFormat] = []
        for i, type_counter in enumerate(input_type_counters):
            # Get parameter name from the first test case
            var_name = test_cases[0].input[i].metadata.get("var_name", f"param_{i}")
            unified_type = UnifiedDataAnalyzer._resolve_unified_type(
                type_counter, f"input param {var_name}"
            )
            final_input_format.append(
                UnifiedParameterFormat(var_name=var_name, data_type=unified_type)
            )

        # Form the final output format
        final_output_format: Optional[UnifiedParameterFormat] = None
        if output_type_counter:
            # Get parameter name from the first test case
            var_name = test_cases[0].output.metadata.get("var_name", "output")
            unified_type = UnifiedDataAnalyzer._resolve_unified_type(
                output_type_counter, "output"
            )
            final_output_format = UnifiedParameterFormat(
                var_name=var_name, data_type=unified_type
            )

        return UnifiedTaskFormat(
            input_format=final_input_format, output_format=final_output_format
        )

    @staticmethod
    def resolve_unified_type(
        type_counts: Dict[DataType, int], context: str
    ) -> DataType:
        return UnifiedDataAnalyzer._resolve_unified_type(type_counts, context)

    @staticmethod
    def _resolve_unified_type(
        type_counts: Dict[DataType, int], context: str
    ) -> DataType:
        """Resolves a unified data type from a counter of data types.

        This function determines the most consistent data type when multiple types are encountered
        across different test cases for a single parameter.

        :param type_counts: A dictionary where keys are DataType enums and values are their counts.
        :type type_counts: Dict[DataType, int]
        :param context: A string describing the context of the type resolution (e.g., 'input param X', 'output').
        :type context: str
        :raises ValueError: If no type data is provided or if inconsistent types are found that cannot be unified.
        :return: The resolved unified DataType.
        :rtype: DataType
        """
        if not type_counts:
            raise ValueError(
                f"Failed to determine type for {context}: no data provided."
            )

        # If only one type - return it
        if len(type_counts) == 1:
            return next(iter(type_counts))

        # Special case: ARRAY_2D + ARRAY_2D_MATRIX -> ARRAY_2D
        types = set(type_counts.keys())
        if types <= {DataType.ARRAY_2D, DataType.ARRAY_2D_MATRIX}:
            return DataType.ARRAY_2D

        # Type inconsistency
        inconsistent = ", ".join(dt.value for dt in types)
        raise ValueError(f"Inconsistent data types for {context}: {inconsistent}")

    @staticmethod
    def determine_task_categories(unified_format: UnifiedTaskFormat) -> List[str]:
        """
        Determine task categories based on the unified task format.

        This method analyzes the `UnifiedTaskFormat` to identify and return a list of
        task categories such as scalar, one_dim, two_dim, two_dim_ambiguous, and matrix.

        :param unified_format: An object representing the unified input and output format of a task.
        :type unified_format: UnifiedTaskFormat
        :return: A sorted list of strings representing the task categories.
        :rtype: List[str]
        """
        all_items = list(unified_format.input_format)
        if unified_format.output_format:
            all_items.append(unified_format.output_format)

        if not all_items:
            return [TaskType.SCALAR.value]  # If no items, consider it scalar (simple)

        has_scalar_or_string = any(
            item.data_type in [DataType.SCALAR, DataType.STRING] for item in all_items
        )
        has_one_dim = any(item.data_type == DataType.ARRAY_1D for item in all_items)
        has_2d = any(
            item.data_type in [DataType.ARRAY_2D_MATRIX, DataType.ARRAY_2D]
            for item in all_items
        )

        two_d_count = sum(
            1
            for item in all_items
            if item.data_type in [DataType.ARRAY_2D_MATRIX, DataType.ARRAY_2D]
        )

        all_2d_are_matrices = all(
            item.data_type == DataType.ARRAY_2D_MATRIX
            for item in all_items
            if item.data_type in [DataType.ARRAY_2D_MATRIX, DataType.ARRAY_2D]
        )

        categories = []

        if two_d_count >= 2:
            categories.append(TaskType.TWO_DIM_AMBIGUOUS.value)

        if all_2d_are_matrices and two_d_count >= 1:
            categories.append(
                "matrix"
            )  # "matrix" is not in TaskType enum, it's a special category for the report

        if has_2d:
            categories.append(TaskType.TWO_DIM.value)
        elif has_one_dim:  # Only one_dim if no 2D arrays
            categories.append(TaskType.ONE_DIM.value)
        elif has_scalar_or_string:  # Only scalar if no 1D or 2D arrays
            categories.append(TaskType.SCALAR.value)

        return sorted(set(categories))

    def get_task_categories(self, test_cases_raw: List[Dict[str, Any]]) -> List[str]:
        """
        Analyzes raw test cases and returns task categories.

        :param test_cases_raw: A list of raw test case dictionaries.
        :type test_cases_raw: List[Dict[str, Any]]
        :return: A list of strings representing the task categories.
        :rtype: List[str]
        """
        parsed_test_cases = self.analyze_test_cases(test_cases_raw)
        unified_format = self.analyze_unified_test_case_format(parsed_test_cases)
        return self.determine_task_categories(unified_format)


class BaseDataConverter(ABC):
    """Base class for data converters."""

    def __init__(self):
        self.analyzer = UnifiedDataAnalyzer()
        self.external_format_hints: Dict[str, Any] = {}

    @abstractmethod
    def convert_data_item(self, data_item: DataItem) -> str:
        """Converts a data item to a string.

        :param data_item: The DataItem to convert.
        :type data_item: DataItem
        :return: The string representation of the data item.
        :rtype: str
        """
        pass

    @abstractmethod
    def convert_input(self, data_items: List[DataItem]) -> str:
        """Converts input data items into stdin format.

        :param data_items: A list of DataItem objects representing the input.
        :type data_items: List[DataItem]
        :return: A string in stdin format.
        :rtype: str
        """
        pass

    @abstractmethod
    def convert_output(self, data_item: DataItem) -> str:
        """Converts an output data item into stdout format.

        :param data_item: The DataItem object representing the output.
        :type data_item: DataItem
        :return: A string in stdout format.
        :rtype: str
        """
        pass

    def convert_test_case(self, test_case: TestCase) -> Dict[str, str]:
        """Converts a TestCase object into stdin and stdout strings.

        :param test_case: The TestCase object to convert.
        :type test_case: TestCase
        :return: A dictionary with 'stdin' and 'stdout' keys and their string representations.
        :rtype: Dict[str, str]
        """
        return {
            "stdin": self.convert_input(test_case.input),
            "stdout": self.convert_output(test_case.output),
        }

    def convert_test(
        self, input_data: Dict[str, Any], output_data: Any
    ) -> Dict[str, str]:
        """Analyzes raw input and output data and converts them into stdin and stdout formats.

        :param input_data: A dictionary of raw input data.
        :type input_data: Dict[str, Any]
        :param output_data: Raw output data.
        :type output_data: Any
        :return: A dictionary with 'stdin' and 'stdout' keys and their string representations.
        :rtype: Dict[str, str]
        """
        test_case = self.analyzer.analyze_test_case(
            {"input": input_data, "output": output_data}
        )
        return self.convert_test_case(test_case)

    def convert_all_tests(
        self, test_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Processes all test cases with preliminary analysis.

        :param test_cases: A list of raw test case dictionaries.
        :type test_cases: List[Dict[str, Any]]
        :return: A list of dictionaries, each containing 'stdin', 'stdout', and 'test_id'.
        :rtype: List[Dict[str, Any]]
        """

        results = []
        for i, test_case in enumerate(test_cases):
            result = self.convert_test(
                test_case.get("input", {}), test_case.get("output")
            )
            result["test_id"] = i
            results.append(result)

        return results


class V1NoDimensionsConverter(BaseDataConverter):
    """Converter for V1 version (no dimensions)."""

    def _convert_scalar_to_str(self, value: Any) -> str:
        """
        Converts scalar values (including booleans and None) to their string representation.

        :param value: The scalar value to convert.
        :type value: Any
        :return: The string representation of the value.
        :rtype: str
        """
        if value is True:
            return "true"
        if value is False:
            return "false"
        if value is None:
            return "null"
        return str(value)

    def convert_data_item(self, data_item: DataItem) -> str:
        """Converts a DataItem to its string representation based on its data type.

        For 2D arrays, joins rows by newline and elements by space.
        For 1D arrays, joins elements by space.
        For scalars or strings, converts directly to string.

        :param data_item: The DataItem to convert.
        :type data_item: DataItem
        :return: The string representation of the data item.
        :rtype: str
        """
        if data_item.data_type in [DataType.ARRAY_2D_MATRIX, DataType.ARRAY_2D]:
            lines = []
            for row in data_item.value:
                lines.append(" ".join(map(self._convert_scalar_to_str, row)))
            return "\n".join(lines)
        elif data_item.data_type == DataType.ARRAY_1D:
            # 1D array: single line
            return " ".join(map(self._convert_scalar_to_str, data_item.value))
        else:
            # Scalar or string
            return self._convert_scalar_to_str(data_item.value)

    def convert_input(self, data_items: List[DataItem]) -> str:
        """Converts a list of DataItems into a single string for stdin.

        Each DataItem is converted and joined by a newline.

        :param data_items: A list of DataItem objects representing the input.
        :type data_items: List[DataItem]
        :return: A string in stdin format.
        :rtype: str
        """
        lines = []
        for item in data_items:
            lines.append(self.convert_data_item(item))
        return "\n".join(lines)

    def convert_output(self, data_item: DataItem) -> str:
        """Converts a single DataItem into a string for stdout.

        :param data_item: The DataItem object representing the output.
        :type data_item: DataItem
        :return: A string in stdout format.
        :rtype: str
        """
        return self.convert_data_item(data_item)


class V2RowsOnlyConverter(V1NoDimensionsConverter):
    """Converter for V2 version (row count for 2D arrays)."""

    def convert_data_item(self, data_item: DataItem) -> str:
        """Converts a DataItem to its string representation, prepending row count for 2D arrays.

        For 2D arrays, prepends the number of rows followed by a newline.
        For other types, defers to the base class implementation.

        :param data_item: The DataItem to convert.
        :type data_item: DataItem
        :return: The string representation of the data item, with row count if applicable.
        :rtype: str
        """
        if data_item.data_type in [DataType.ARRAY_2D_MATRIX, DataType.ARRAY_2D]:
            if data_item.rows == 0:
                return "0"
            return f"{data_item.rows}\n" + super().convert_data_item(data_item)
        else:
            return super().convert_data_item(data_item)


class V3SmartMatrixConverter(V1NoDimensionsConverter):
    """Converter for V3 version (smart matrix detection)."""

    def convert_data_item(self, data_item: DataItem) -> str:
        """Converts a DataItem to its string representation, prepending dimensions for matrices.

        For 2D matrices, prepends row and column counts.
        For other 2D arrays, prepends only the row count.
        For other types, defers to the base class implementation.

        :param data_item: The DataItem to convert.
        :type data_item: DataItem
        :return: The string representation of the data item, with dimensions if applicable.
        :rtype: str
        """
        if data_item.data_type == DataType.ARRAY_2D_MATRIX:
            return f"{data_item.rows} {data_item.cols}\n" + super().convert_data_item(
                data_item
            )
        elif data_item.data_type == DataType.ARRAY_2D:
            # For jagged arrays, prepend only the row count.
            return f"{data_item.rows}\n" + super().convert_data_item(data_item)
        else:
            return super().convert_data_item(data_item)


class ConverterFactory:
    """Factory for creating data converters."""

    @staticmethod
    def create_converter(version: str) -> BaseDataConverter:
        """Creates a converter for the specified version.

        :param version: The version identifier for the converter (e.g., 'v1', 'v2', 'v3').
        :type version: str
        :raises ValueError: If an unknown converter version is requested.
        :return: An instance of a BaseDataConverter subclass.
        :rtype: BaseDataConverter
        """
        converters = {
            "v1_no_dimensions": V1NoDimensionsConverter,
            "v3_smart_matrix": V3SmartMatrixConverter,
            "v2_rows_only": V2RowsOnlyConverter,
            "v1": V1NoDimensionsConverter,
            "v2": V2RowsOnlyConverter,
            "v3": V3SmartMatrixConverter,
            # Add other versions here if needed.
        }

        converter_class = converters.get(version)
        if not converter_class:
            raise ValueError(f"Unknown converter version: {version}")

        return converter_class()


class UnifiedConverterService:
    """Unified data processor for LCB and PromptEnhancer."""

    def __init__(self, version: str = "v3_smart_matrix"):
        """Initializes the UnifiedConverterService with a specified converter version.

        :param version: The version of the converter to use. Defaults to 'v3_smart_matrix'.
        :type version: str
        """
        self.converter = ConverterFactory.create_converter(version)
        self.analyzer = UnifiedDataAnalyzer()

    def process_test_case(
        self, input_data: Dict[str, Any], output_data: Any
    ) -> Dict[str, str]:
        """Processes a single raw test case using the configured converter and analyzer.

        This method orchestrates the analysis of raw input/output data and their conversion
        into a standardized stdin/stdout format.

        :param input_data: A dictionary of raw input data for the test case.
        :type input_data: Dict[str, Any]
        :param output_data: Raw output data for the test case.
        :type output_data: Any
        :return: A dictionary containing the 'stdin' and 'stdout' string representations of the test case.
        :rtype: Dict[str, str]
        """
        return self.converter.convert_test(input_data, output_data)


def demo_unified_architecture():
    """Demonstrates the unified architecture."""
    print("🏗️ DEMONSTRATING UNIFIED ARCHITECTURE")
    print("=" * 60)

    # Test data
    test_cases = [
        {
            "input": {"grid": [[1, 2, 3], [4, 5, 6]], "k": 2},
            "output": [[8, 2, 3], [9, 6, 7]],
        },
        {
            "input": {"grid": [[1, 2], [3, 4, 5]], "nums": [1, 2, 3]},
            "output": [1, 2, 3],
        },
    ]

    # Test different versions
    versions = ["v1_no_dimensions", "v3_smart_matrix", "v2_rows_only"]

    for version in versions:
        print(f"\n🔍 Version: {version}")
        processor = UnifiedConverterService(version)

        for i, test_case in enumerate(test_cases):
            result = processor.process_test_case(
                test_case["input"], test_case["output"]
            )
            print(f"\n  Test {i+1}:")
            print(f"    stdin: {result['stdin'][:50]}...")
            print(f"    stdout: {result['stdout'][:50]}...")


if __name__ == "__main__":
    demo_unified_architecture()
