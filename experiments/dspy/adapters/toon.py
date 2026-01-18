# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TOON Adapter for DSPy.

A DSPy adapter that uses TOON (Token-Oriented Object Notation) format for
structured outputs, achieving significant token reduction compared to JSON
while maintaining readability.
"""

import inspect
import json
import logging
import re
import types
from typing import Any, Literal, Union, get_args, get_origin

from dspy.adapters.base import Adapter  # type: ignore[import-untyped]
from dspy.adapters.types import History  # type: ignore[import-untyped]
from dspy.signatures.signature import Signature  # type: ignore[import-untyped]
from dspy.utils.callback import BaseCallback  # type: ignore[import-untyped]
from dspy.utils.exceptions import AdapterParseError  # type: ignore[import-untyped]
from pydantic import BaseModel

from experiments.dspy.adapters.toon_codec import decode, encode

logger = logging.getLogger(__name__)

COMMENT_SYMBOL = "#"


def _render_type_str(
    annotation: Any,
    depth: int = 0,
    indent: int = 0,
    seen_models: set[type] | None = None,
    field_name: str | None = None,
) -> str:
    """Recursively renders a type annotation into TOON-like schema string.

    Args:
        annotation: The type annotation to render.
        depth: Current recursion depth.
        indent: Current indentation level.
        seen_models: Set of already processed models to prevent recursion.
        field_name: Optional field name for array types (TOON requires field name
                   to be directly concatenated with [COUNT]).

    Returns:
        TOON-formatted type string.
    """
    # Primitive types
    if annotation is str:
        return "string"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "boolean"

    # Pydantic models
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return _build_toon_schema(annotation, indent, seen_models)

    try:
        origin = get_origin(annotation)
        args = get_args(annotation)
    except Exception:
        return str(annotation)

    # Optional[T] or T | None - handle nullable arrays specially
    if origin in (types.UnionType, Union):
        non_none_args = [arg for arg in args if arg is not type(None)]
        is_nullable = len(non_none_args) < len(args)

        # For single-type optionals with arrays, pass field_name through
        if len(non_none_args) == 1:
            inner_origin = get_origin(non_none_args[0])
            if inner_origin is list and field_name:
                type_render = _render_type_str(non_none_args[0], depth + 1, indent, seen_models, field_name)
                if is_nullable:
                    return f"{type_render} or null"
                return type_render

        # Render each non-None type, avoiding duplicate "or null" patterns
        rendered_parts = []
        for arg in non_none_args:
            rendered = _render_type_str(arg, depth + 1, indent, seen_models)
            # Avoid adding types that already end with "or null" when we'll add it later
            if is_nullable and rendered.endswith(" or null"):
                rendered = rendered[: -len(" or null")]
            rendered_parts.append(rendered)

        type_render = " or ".join(rendered_parts)
        if is_nullable:
            return f"{type_render} or null"
        return type_render

    # Literal[T1, T2, ...]
    if origin is Literal:
        return " or ".join(f'"{arg}"' for arg in args)

    # list[T] - TOON format: fieldname[COUNT]: values or fieldname[COUNT,]{fields}:
    if origin is list:
        inner_type = args[0] if args else Any
        name_prefix = f"{field_name}" if field_name else ""

        if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
            fields = list(inner_type.model_fields.keys())
            fields_str = ",".join(fields)
            # Tabular format: fieldname[COUNT]{field1,field2,...}:
            # Note: comma delimiter is implicit (default) per TOON spec, so not shown in [N]
            header = f"{name_prefix}[COUNT]{{{fields_str}}}:"
            return f"{header}\n  value1,value2,...\n  (one row per item, COUNT = number of items)"
        else:
            inner_str = _render_type_str(inner_type, depth + 1, indent, seen_models)
            # Primitive array format: fieldname[COUNT]: type,...
            return f"{name_prefix}[COUNT]: {inner_str},... (COUNT = num items)"

    # dict[K, V]
    if origin is dict:
        key_type = _render_type_str(args[0], depth + 1, indent, seen_models) if args else "string"
        val_type = _render_type_str(args[1], depth + 1, indent, seen_models) if len(args) > 1 else "any"
        return f"dict[{key_type}, {val_type}]"

    # Fallback
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _is_array_type(annotation: Any) -> bool:
    """Check if annotation is a list type or Optional[list[...]]."""
    origin = get_origin(annotation)
    if origin is list:
        return True
    # Check for Optional[list[...]]
    if origin in (types.UnionType, Union):
        args = get_args(annotation)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return get_origin(non_none_args[0]) is list
    return False


def _build_toon_schema(
    pydantic_model: type[BaseModel],
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Builds a TOON-style schema from a Pydantic model."""
    seen_models = seen_models or set()

    if pydantic_model in seen_models:
        return f"<{pydantic_model.__name__}>"

    seen_models.add(pydantic_model)

    lines = []
    current_indent = "  " * indent

    for name, field in pydantic_model.model_fields.items():
        if field.description:
            lines.append(f"{current_indent}{COMMENT_SYMBOL} {field.description}")

        # For array types, pass field_name so it's concatenated directly with [COUNT]
        if _is_array_type(field.annotation):
            rendered_type = _render_type_str(
                field.annotation, indent=indent + 1, seen_models=seen_models, field_name=name
            )
            if "\n" in rendered_type:
                # First line has field name already (e.g., "items[COUNT,]{...}:")
                type_lines = rendered_type.split("\n")
                lines.append(f"{current_indent}{type_lines[0]}")
                for line in type_lines[1:]:
                    lines.append(f"{current_indent}  {line}")
            else:
                # Single line array (e.g., "items[COUNT]: string,...")
                lines.append(f"{current_indent}{rendered_type}")
        else:
            rendered_type = _render_type_str(field.annotation, indent=indent + 1, seen_models=seen_models)

            if "\n" in rendered_type:
                lines.append(f"{current_indent}{name}:")
                for line in rendered_type.split("\n"):
                    lines.append(f"{current_indent}  {line}")
            else:
                lines.append(f"{current_indent}{name}: {rendered_type}")

    return "\n".join(lines)


def _get_output_schema(field_name: str, field_type: Any) -> str:
    """Generate TOON output schema for a field."""
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle Optional types - unwrap to get inner type
    if origin in (types.UnionType, Union):
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            inner_origin = get_origin(non_none_args[0])
            inner_args = get_args(non_none_args[0])
            if inner_origin is list:
                origin = list
                args = inner_args

    # List of Pydantic models -> tabular format
    if origin is list and args:
        inner_type = args[0]
        if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
            fields = list(inner_type.model_fields.keys())
            fields_str = ",".join(fields)
            # TOON tabular format: fieldname[COUNT]{fields}:
            # Note: comma delimiter is implicit (default), not shown in [N]
            return f"""{field_name}[2]{{{fields_str}}}:
  Alice,35,engineer
  Bob,28,designer
(Replace 2 with actual count, add one row per item)"""

    # Pydantic model -> object format
    if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
        schema = _build_toon_schema(field_type, indent=1)
        return f"{field_name}:\n{schema}"

    # List of primitives - TOON format: fieldname[COUNT]: v1,v2,v3 (single line)
    if origin is list:
        return f"{field_name}[COUNT]: value1,value2,value3"

    # Simple types
    return f"{field_name}: {_render_type_str(field_type)}"


def _encode_value(value: Any) -> str:
    """Encode a value to TOON format string."""
    if isinstance(value, BaseModel):
        return encode(value.model_dump())
    elif isinstance(value, (list, dict)):
        return encode(value)
    else:
        return str(value)


class ToonAdapter(Adapter):
    """DSPy adapter using TOON (Token-Oriented Object Notation) format.

    TOON achieves significant token reduction compared to JSON while maintaining
    readability. This adapter generates TOON-formatted schemas and parses
    TOON responses from LLMs.

    Key features:
    - Compact `key: value` syntax (no braces/brackets for objects)
    - Tabular format `[N,]{fields}:` for uniform object arrays
    - Falls back to JSON parsing when TOON fails

    Example Usage:
        ```python
        import dspy
        from pydantic import BaseModel, Field
        from experiments.dspy.adapters.toon import ToonAdapter

        class Person(BaseModel):
            name: str = Field(description="Full name")
            age: int

        class ExtractPerson(dspy.Signature):
            '''Extract person from text.'''
            text: str = dspy.InputField()
            person: Person = dspy.OutputField()

        llm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=llm, adapter=ToonAdapter())

        extractor = dspy.Predict(ExtractPerson)
        result = extractor(text="Alice is 30 years old.")
        print(result.person)
        ```

    TOON Format Examples:
        Simple object:
        ```
        name: Alice
        age: 30
        ```

        Tabular array:
        ```
        [2,]{id,name}:
          1,Alice
          2,Bob
        ```
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
    ):
        """Initialize the ToonAdapter.

        Args:
            callbacks: List of callback functions to execute during format() and parse().
            use_native_function_calling: Whether to enable native function calling when
                the LM supports it. Defaults to False.
        """
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
        )

    def format_field_description(self, signature: type[Signature]) -> str:
        """Format input/output field descriptions."""
        sections = []

        if signature.input_fields:
            sections.append("Input fields:")
            for name, field in signature.input_fields.items():
                desc = f" - {field.description}" if field.description else ""
                sections.append(f"  {name}: {_render_type_str(field.annotation)}{desc}")

        if signature.output_fields:
            sections.append("\nOutput fields:")
            for name, field in signature.output_fields.items():
                desc = f" - {field.description}" if field.description else ""
                sections.append(f"  {name}: {_render_type_str(field.annotation)}{desc}")

        return "\n".join(sections)

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Format the output structure instructions in TOON format."""
        sections = []

        sections.append("""
TOON Format (NOT JSON):
- Simple values: key: value (booleans: true/false)
- Primitive arrays: field[COUNT]: item1,item2,item3  (single line, comma-separated; replace COUNT)
- Tabular arrays for objects:
  [COUNT]{field1,field2}:
    value1,value2
    value3,value4
  (COUNT is the actual number of rows)
- Empty/none values: use `field: null` (no [COUNT]) when there are no items or the value is absent
- No JSON braces/brackets, code fences, or dashes for primitive arrays
- Do not wrap output in JSON or YAML; emit plain TOON only
""")

        sections.append("Output structure:")
        for name, field in signature.output_fields.items():
            sections.append(_get_output_schema(name, field.annotation))

        return "\n".join(sections)

    def format_task_description(self, signature: type[Signature]) -> str:
        """Format the task description from signature docstring."""
        return signature.__doc__ or "Complete the task based on the inputs."

    def format_demos(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Format few-shot examples.

        Separates demos into complete (all fields present) and incomplete demos,
        following the base adapter pattern.
        """
        complete_demos = []
        incomplete_demos = []

        for demo in demos:
            # Check if all fields are present and not None
            is_complete = all(k in demo and demo[k] is not None for k in signature.fields)

            # Check if demo has at least one input and one output field
            has_input = any(k in demo for k in signature.input_fields)
            has_output = any(k in demo for k in signature.output_fields)

            if is_complete:
                complete_demos.append(demo)
            elif has_input and has_output:
                incomplete_demos.append(demo)

        messages = []

        # Format incomplete demos with prefix
        incomplete_demo_prefix = "This is an example of the task, though some input or output fields are not supplied."
        for demo in incomplete_demos:
            user_content = self.format_user_message_content(signature, demo, prefix=incomplete_demo_prefix)
            messages.append({"role": "user", "content": user_content})
            assistant_content = self.format_assistant_message_content(
                signature, demo, missing_field_message="Not supplied for this particular example."
            )
            messages.append({"role": "assistant", "content": assistant_content})

        # Format complete demos
        for demo in complete_demos:
            messages.append(
                {
                    "role": "user",
                    "content": self.format_user_message_content(signature, demo),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.format_assistant_message_content(signature, demo),
                }
            )

        return messages

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Format the user message with inputs."""
        parts = []
        if prefix:
            parts.append(prefix)

        for name, field in signature.input_fields.items():
            if name in inputs:
                value = inputs[name]
                encoded = _encode_value(value)
                if "\n" in encoded or isinstance(value, (BaseModel, list, dict)):
                    parts.append(f"{name}:\n{encoded}")
                else:
                    parts.append(f"{name}: {encoded}")

        if main_request:
            parts.append("\nProvide output in TOON format as shown above.")

        if suffix:
            parts.append(suffix)

        return "\n\n".join(parts).strip()

    def format_assistant_message_content(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        missing_field_message: str | None = None,
    ) -> str:
        """Format assistant message content in TOON format.

        Args:
            signature: The DSPy signature for which to format the assistant message.
            outputs: The output fields to be formatted.
            missing_field_message: A message to use when a field is missing.

        Returns:
            A string containing the formatted assistant message.
        """
        parts = []
        for name in signature.output_fields.keys():
            value = outputs.get(name, missing_field_message)
            if value is None:
                continue
            encoded = _encode_value(value)
            if "\n" in encoded or isinstance(value, (BaseModel, list, dict)):
                parts.append(f"{name}:\n{encoded}")
            else:
                parts.append(f"{name}: {encoded}")
        return "\n".join(parts)

    def format_conversation_history(
        self,
        signature: type[Signature],
        history_field_name: str,
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Format conversation history.

        Supports both DSPy History objects and legacy list-of-dicts format.

        Args:
            signature: The DSPy signature (without history field).
            history_field_name: The name of the history field.
            inputs: The input arguments (will be modified to remove history).

        Returns:
            A list of formatted messages.
        """
        history_value = inputs.get(history_field_name)

        if history_value is None:
            if history_field_name in inputs:
                del inputs[history_field_name]
            return []

        # Support DSPy History type (has .messages attribute)
        if hasattr(history_value, "messages"):
            conversation_history = history_value.messages
        # Support legacy list-of-dicts format for backwards compatibility
        elif isinstance(history_value, list):
            conversation_history = history_value
        else:
            logger.warning(f"Unexpected history format for field '{history_field_name}': {type(history_value)}")
            del inputs[history_field_name]
            return []

        messages = []
        for message in conversation_history:
            if isinstance(message, dict):
                # Legacy format: {"user": ..., "assistant": ...}
                if "user" in message:
                    messages.append({"role": "user", "content": str(message["user"])})
                if "assistant" in message:
                    messages.append({"role": "assistant", "content": str(message["assistant"])})
            else:
                # DSPy History message format
                messages.append(
                    {
                        "role": "user",
                        "content": self.format_user_message_content(signature, message),
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": self.format_assistant_message_content(signature, message),
                    }
                )

        # Remove the history field from inputs
        del inputs[history_field_name]

        return messages

    def _get_history_field_name(self, signature: type[Signature]) -> str | None:
        """Check if signature has a history field.

        Uses proper type checking for DSPy's History type.

        Args:
            signature: The DSPy signature to check.

        Returns:
            The name of the history field, or None if not found.
        """
        for name, field in signature.input_fields.items():
            if field.annotation == History:
                return name
        return None

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse TOON-formatted LLM output into field values.

        Attempts to parse as TOON first, falls back to JSON if that fails.
        Raises AdapterParseError if parsing fails completely.

        Args:
            signature: The DSPy signature defining expected output fields.
            completion: The raw LLM response to parse.

        Returns:
            A dictionary mapping field names to parsed values.

        Raises:
            AdapterParseError: If the response cannot be parsed or is missing fields.
        """
        result = {}
        completion = completion.strip()

        # Try parsing each output field individually
        for field_name, field in signature.output_fields.items():
            value = self._extract_field_value(completion, field_name, field.annotation)
            if value is not None:
                result[field_name] = value

        # If we got all results, return them
        if result.keys() == signature.output_fields.keys():
            return result

        # Try full TOON parsing
        try:
            parsed = decode(completion)
            if isinstance(parsed, dict):
                for field_name, field in signature.output_fields.items():
                    if field_name in parsed and field_name not in result:
                        result[field_name] = self._convert_field(parsed[field_name], field.annotation)
                if result.keys() == signature.output_fields.keys():
                    return result
        except Exception as e:
            logger.debug(f"TOON parsing failed: {e}")

        # Try JSON parsing as fallback
        try:
            json_str = completion
            if "```json" in completion:
                json_str = completion.split("```json")[1].split("```")[0]
            elif "```" in completion:
                json_str = completion.split("```")[1].split("```")[0]

            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                for field_name, field in signature.output_fields.items():
                    if field_name in parsed and field_name not in result:
                        result[field_name] = self._convert_field(parsed[field_name], field.annotation)
        except Exception as e:
            logger.debug(f"JSON fallback parsing failed: {e}")

        # Check if we have all required fields
        if result.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="ToonAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=result,
            )

        return result

    def _extract_field_value(self, completion: str, field_name: str, field_type: Any) -> Any | None:
        """Extract a specific field value from TOON output."""
        # Look for field_name: followed by tabular array
        # Pattern: field_name:\n[COUNT]{fields}:\n  rows...
        pattern = rf"{re.escape(field_name)}:\s*\n(\[\d+\]\{{[^}}]+\}}:[[\s\S]*?)(?=\n\w+:|$)"
        match = re.search(pattern, completion)

        if match:
            toon_array = match.group(1).strip()
            try:
                parsed = decode(toon_array)
                return self._convert_field(parsed, field_type)
            except Exception:
                pass

        # Look for simple field_name: value
        pattern = rf"^{re.escape(field_name)}:\s*(.+)$"
        match = re.search(pattern, completion, re.MULTILINE)

        if match:
            value_str = match.group(1).strip()
            # Check if it's start of a nested structure
            if not value_str or value_str.startswith("["):
                return None

            try:
                parsed = decode(value_str)
                return self._convert_field(parsed, field_type)
            except Exception:
                if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
                    return None
                return value_str

        # Inline list with optional count: field[COUNT]: v1,v2
        inline_list_pattern = rf"^{re.escape(field_name)}(?:\[\d+\])?:\s*([^\n]+)$"
        match = re.search(inline_list_pattern, completion, re.MULTILINE)
        if match:
            values_str = match.group(1).strip()
            # Split on commas, strip spaces
            items = [v.strip() for v in values_str.split(",") if v.strip()]
            origin = get_origin(field_type)
            if origin is list:
                return items
            if items:
                try:
                    parsed = decode(items[0])
                    return self._convert_field(parsed, field_type)
                except Exception:
                    if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
                        return None
                    return items[0]

        # Look for field_name[<n>]: followed by plain list lines (e.g., solutions[10]:\nitem1\nitem2)
        list_block_pattern = rf"^{re.escape(field_name)}(?:\[\d+\])?:\s*\n((?:.+\n?)*?)(?=^\w[\w\s]*:|\Z)"
        match = re.search(list_block_pattern, completion, re.MULTILINE)
        if match:
            block = match.group(1).strip()
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if lines:
                origin = get_origin(field_type)
                if origin is list:
                    return lines
                try:
                    parsed = decode(lines[0])
                    return self._convert_field(parsed, field_type)
                except Exception:
                    if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
                        return None
                    return lines[0] if lines else None

        return None

    def _convert_field(self, value: Any, field_type: Any) -> Any:
        """Convert parsed value to the expected field type."""
        origin = get_origin(field_type)
        args = get_args(field_type)

        # Handle Optional types
        if origin in (types.UnionType, Union):
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                return self._convert_field(value, non_none_args[0])

        # List of Pydantic models
        if origin is list and args:
            inner_type = args[0]
            if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
                if isinstance(value, list):
                    return [inner_type.model_validate(item) if isinstance(item, dict) else item for item in value]

        # Pydantic model
        if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
            if isinstance(value, dict):
                return field_type.model_validate(value)

        return value
