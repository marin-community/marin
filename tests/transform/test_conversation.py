# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for conversation data transformation scripts."""

from pathlib import Path

from marin.transform.conversation.adapters import InputDatasetFormat, TransformAdapter
from marin.transform.conversation.conversation_to_dolma import transform_conversation_to_dolma
from marin.transform.conversation.preference_data_adapters import PreferenceTransformAdapter
from marin.transform.conversation.transform_conversation import (
    transform_row,
    TransformSFTDatasetConfig,
)

OPENAI_FORMAT_SAMPLE = {
    "messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Italy?"},
        {"role": "assistant", "content": "The capital of Italy is Rome."},
    ]
}

SHAREGPT_FORMAT_SAMPLE = {
    "conversations": [
        {"from": "human", "value": "Explain quantum computing"},
        {"from": "gpt", "value": "Quantum computing uses quantum bits or qubits..."},
        {"from": "system", "value": "You are a helpful assistant"},
    ]
}

INSTRUCTION_RESPONSE_SAMPLE = {
    "instruction": "Write a Python function to add two numbers",
    "response": "def add(a, b):\n    return a + b",
    "metadata_field": "test_value",
}

PREFERENCE_DATA_SAMPLE = {
    "chosen": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ],
    "rejected": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "I don't know."},
    ],
    "metadata_field": "pref_test",
}


class TestTransformAdapters:
    """Test the different adapter formats."""

    def test_openai_format_adapter(self):
        """Test OpenAI messages format adapter."""
        adapter = TransformAdapter(
            dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
            conversation_column="messages",
            role_key="role",
            content_key="content",
            user_value="user",
            assistant_value="assistant",
            system_value="system",
        )

        messages = adapter.transform_conversation_to_openai_format(OPENAI_FORMAT_SAMPLE)

        assert len(messages) == 4
        assert messages[0].role == "user"
        assert messages[0].content == "What is the capital of France?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "The capital of France is Paris."

    def test_sharegpt_format_adapter(self):
        """Test ShareGPT format adapter."""
        adapter = TransformAdapter(
            dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
            conversation_column="conversations",
            role_key="from",
            content_key="value",
            user_value="human",
            assistant_value="gpt",
            system_value="system",
        )

        messages = adapter.transform_conversation_to_openai_format(SHAREGPT_FORMAT_SAMPLE)

        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[0].content == "Explain quantum computing"
        assert messages[1].role == "assistant"
        assert messages[2].role == "system"


class TestTransformRow:
    """Test the transform_row function."""

    def test_transform_with_replacements(self):
        """Test text replacements in messages."""
        adapter = TransformAdapter(
            dataset_format=InputDatasetFormat.INSTRUCTION_RESPONSE,
            instruction_column="instruction",
            response_column="response",
            replacements={"<think>": "<|start_think|>", "</think>": "<|end_think|>"},
        )

        row = {
            "instruction": "Solve this",
            "response": "<think>Let me think...</think> The answer is 42.",
        }

        cfg = TransformSFTDatasetConfig(
            source="test/dataset",
            revision="main",
            output_path="/tmp/output",
            metadata_columns=[],
            adapter=adapter,
        )

        result = transform_row(row, cfg, adapter)

        assert result is not None
        # Messages are OpenAIChatMessage objects in DolmaConversationOutput
        response_message = result.messages[1]
        assert "<|start_think|>" in response_message.content
        assert "<|end_think|>" in response_message.content
        assert "<think>" not in response_message.content


class TestPreferenceDataTransform:
    """Test preference data (DPO) transformation."""

    def test_preference_adapter(self):
        """Test preference data adapter."""
        adapter = PreferenceTransformAdapter(
            source="test/preference",
            chosen_column="chosen",
            rejected_column="rejected",
            role_key="role",
            content_key="content",
        )

        result = adapter.extract_preference_example(PREFERENCE_DATA_SAMPLE)

        assert result is not None
        assert "chosen" in result
        assert "rejected" in result
        assert len(result["chosen"]) == 2
        assert len(result["rejected"]) == 2
        assert result["chosen"][0].role == "user"
        assert result["chosen"][1].role == "assistant"
        assert result["chosen"][1].content == "2+2 equals 4."
        assert result["rejected"][1].content == "I don't know."


class TestEndToEndTransforms:
    """End-to-end integration tests using sync backend."""

    def test_instruction_response_pipeline(self, tmp_path: Path, write_jsonl_gz, read_all_jsonl_gz):
        """Test end-to-end instruction-response transformation.

        Note: This is a minimal test showing the pattern. Full integration
        would require mocking HuggingFace datasets.
        """

        # Prepare test data
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        test_records = [
            {"instruction": "What is 2+2?", "response": "4", "source_meta": "math"},
            {"instruction": "What is Python?", "response": "A programming language", "source_meta": "cs"},
        ]

        write_jsonl_gz(input_dir / "data.jsonl.gz", test_records)

        # Transform using adapter
        adapter = TransformAdapter(
            dataset_format=InputDatasetFormat.INSTRUCTION_RESPONSE,
            instruction_column="instruction",
            response_column="response",
        )

        cfg = TransformSFTDatasetConfig(
            source="test/dataset",
            revision="main",
            output_path=str(output_dir),
            metadata_columns=["source_meta"],
            adapter=adapter,
        )

        # Transform each record
        results = []
        for record in test_records:
            result = transform_row(record, cfg, adapter)
            if result:
                results.append(result.model_dump())

        assert len(results) == 2
        assert all(len(r["messages"]) == 2 for r in results)
        assert all(r["source"] == "test/dataset" for r in results)

    def test_dolma_conversion_pipeline(self, tmp_path: Path, write_jsonl_gz, read_all_jsonl_gz):
        """Test end-to-end Dolma conversion."""
        input_dir = tmp_path / "input"

        # Create conversation format data
        conversation_records = [
            {
                "id": "conv-1",
                "source": "test",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
                "metadata": {},
            },
            {
                "id": "conv-2",
                "source": "test",
                "messages": [
                    {"role": "user", "content": "Goodbye"},
                    {"role": "assistant", "content": "See you"},
                ],
                "metadata": {},
            },
        ]

        write_jsonl_gz(input_dir / "conversations.jsonl.gz", conversation_records)

        # Transform to Dolma
        dolma_results = []
        for record in conversation_records:
            dolma_record = transform_conversation_to_dolma(record)
            dolma_results.append(dolma_record)

        assert len(dolma_results) == 2
        assert all("text" in r for r in dolma_results)
        assert all("messages" not in r for r in dolma_results)
        assert "user: Hello" in dolma_results[0]["text"]
        assert "assistant: Hi there" in dolma_results[0]["text"]
