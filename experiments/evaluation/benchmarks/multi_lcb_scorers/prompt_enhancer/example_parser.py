import re
import json
import ast
from typing import Any, Dict, List, Optional


class ExampleParser:
    """Parses LeetCode problem descriptions to extract example test cases."""

    def __init__(self):
        """Initializes the ExampleParser with regular expressions for parsing."""
        # Lines of the form: name = value
        self.assign_re = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*(.*?)(?:,|$)")
        # Example blocks
        self.example_block_re = re.compile(
            r"Example\s+\d+:(.*?)(?=Example\s+\d+:|$)", re.DOTALL | re.IGNORECASE
        )
        self.input_line_re = re.compile(r"Input:\s*(.+)", re.IGNORECASE)
        self.output_line_re = re.compile(r"Output:\s*(.+)", re.IGNORECASE)

    def extract_examples(self, problem_text: str) -> List[Dict[str, Any]]:
        """
        Extracts examples from the problem text.

        :param problem_text: The full text of the problem description.
        :type problem_text: str
        :returns: A list of dictionaries, each representing an example.
        :rtype: List[Dict[str, Any]]
        """
        results: List[Dict[str, Any]] = []
        for idx, block in enumerate(
            self.example_block_re.findall(problem_text), start=1
        ):
            input_m = self.input_line_re.search(block)
            output_m = self.output_line_re.search(block)
            if not input_m or not output_m:
                continue
            raw_input = input_m.group(1).strip()
            raw_output = output_m.group(1).strip()
            parsed_input = self.parse_input_line(raw_input)
            parsed_output = self.parse_value(raw_output)
            results.append(
                {"example_id": idx, "input": parsed_input, "output": parsed_output}
            )
        return results

    def parse_input_line(self, line: str) -> Dict[str, Any]:
        """
        Parses an input line, supporting single and multiple variable assignments.

        Supports formats:
          - Single: colors = [1,1,1]
          - Multiple: target = "abc", words = ["a","b"], costs = [1,2]

        Returns a dictionary with full variable names and their parsed values.

        :param line: The input string to parse.
        :type line: str
        :return: A dictionary of variable names to their parsed values.
        :rtype: Dict[str, Any]
        """
        pos = 0
        n = len(line)
        items: Dict[str, Any] = {}
        while pos < n:
            m = self.assign_re.match(line, pos)
            if not m:
                break
            var = m.group(1)
            # value_fragment = m.group(2)

            val_str, next_pos = self._extract_value_string(line, m.start(2))

            items[var] = self.parse_value(val_str)
            pos = next_pos
            # Skip possible comma and spaces
            while pos < n and line[pos] in " \t,":
                pos += 1
        # If the parser found nothing, it might be the format "colors = [1,1,1]" without commas
        if not items:
            # Attempt to parse as a list of values if no variable names are found
            parsed_list = self._parse_list_of_values(line)
            if parsed_list:
                return {f"_param_{i}": val for i, val in enumerate(parsed_list)}

            single = self.parse_possible_single(line)
            if single is not None:
                return single
        return items

    def parse_possible_single(self, line: str) -> Optional[Dict[str, Any]]:
        """Attempts to parse a line as a single variable assignment.

        :param line: The input string to parse.
        :type line: str
        :return: A dictionary containing the single variable assignment, or None if not applicable.
        :rtype: Optional[Dict[str, Any]]
        """
        m = self.assign_re.match(line)
        if not m:
            return None
        var = m.group(1)
        eq_pos = line.find("=", m.start(1))
        val_str, _ = self._extract_value_string(line, eq_pos + 1)
        return {var: self.parse_value(val_str)}

    def parse_value(self, raw: str) -> Any:
        """Parses a raw string into a Python object (JSON, literal, or string).

        It first attempts to parse the string as JSON, then as a Python literal,
        and finally returns the string itself if neither is successful.

        :param raw: The raw string to parse.
        :type raw: str
        :return: The parsed Python object.
        :rtype: Any
        """
        s = raw.strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s

    def _parse_public_test_input(self, input_str: str) -> List[Any]:
        """
        Parses a multi-line public test input string into a list of parsed values.
        Each line is treated as a separate argument.

        :param input_str: The raw input string from public_test_cases.
        :type input_str: str
        :return: A list of parsed input values.
        :rtype: List[Any]
        """
        parsed_inputs = []
        lines = input_str.strip().split("\n")
        for line in lines:
            if line.strip():  # Avoid empty lines
                parsed_inputs.append(self.parse_value(line.strip()))
        return parsed_inputs

    def _extract_value_string(self, text: str, start: int) -> tuple[str, int]:
        """
        Extracts the full string representation of a value from a line,
        respecting balanced brackets/braces and quotes.
        Returns (value_string, end_position).

        :param text: The full text line from which to extract the value.
        :type text: str
        :param start: The starting index in the text to begin extraction.
        :type start: int
        :return: A tuple containing the extracted value string and the end position of the extraction.
        :rtype: tuple[str, int]
        """
        i = start
        n = len(text)

        while i < n and text[i].isspace():
            i += 1

        if i >= n:
            return "", start

        braces = {"[": "]", "(": ")", "{": "}"}
        stack: List[str] = []
        in_str: Optional[str] = None
        start_value = i

        while i < n:
            ch = text[i]

            if in_str:
                if ch == in_str:
                    in_str = None
            else:
                if ch in ('"', "'"):
                    in_str = ch
                elif ch in braces:
                    stack.append(braces[ch])
                elif stack and ch == stack[-1]:
                    stack.pop()
                elif not stack and ch == ",":
                    break  # Top-level comma terminates the value
            i += 1

        value_string = text[start_value:i].strip()
        return value_string, i

    def _parse_list_of_values(self, line: str) -> Optional[List[Any]]:
        """
        Parses a string containing a comma-separated list of values, handling nested structures.

        For example: "["LRUCache"], [2]" -> [["LRUCache"], [2]]

        :param line: The input string to parse.
        :type line: str
        :return: A list of parsed values, or None if no values are found.
        :rtype: Optional[List[Any]]
        """
        i = 0
        n = len(line)
        values: List[Any] = []

        while i < n:
            while i < n and line[i].isspace():
                i += 1
            if i >= n:
                break

            value_str, next_pos = self._extract_value_string(line, i)
            if value_str:
                values.append(self.parse_value(value_str))
            i = next_pos
            # Skip possible comma and spaces
            while i < n and line[i] in " \t,":
                i += 1
        return values if values else None
