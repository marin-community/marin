# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for conversation data transformation scripts."""

from pathlib import Path

from marin.transform.conversation.adapters import InputDatasetFormat, ReasoningContentMode, TransformAdapter
from marin.transform.conversation.conversation_to_dolma import transform_conversation_to_dolma
from marin.transform.conversation.preference_data_adapters import PreferenceTransformAdapter
from marin.transform.conversation.transform_conversation import (
    RowFilter,
    RowFilterOperator,
    TransformSFTDatasetConfig,
    row_matches_filters,
    shard_example_limit,
    transform_row,
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

FINEPROOFS_MESSAGES_SAMPLE = {
    "messages": [
        {"role": "user", "content": "Prove that 1 + 1 = 2."},
        {"role": "assistant", "content": "<think>Use Peano arithmetic.</think>\nHere is the proof."},
    ],
    "category": "number_theory",
    "competition": "imo",
    "gemini-3-pro-grade": 7,
    "qwen3-4b-thinking-reward@128": 0.93,
    "source": "olympiads",
}

FINEPROOFS_PROOF_ONLY_SAMPLE = {
    "problem": "Show that the sum of two even integers is even.",
    "proof": "Let the integers be 2a and 2b. Their sum is 2(a + b), so it is even.",
    "category": "algebra",
    "competition": "aops",
    "gemini-3-pro-grade": 6,
    "qwen3-4b-thinking-reward@128": 0.75,
    "source": "aops",
}

FINEPROOFS_METADATA_COLUMNS = [
    "category",
    "competition",
    "gemini-3-pro-grade",
    "qwen3-4b-thinking-reward@128",
    "source",
]

NEMOTRON_TOOL_TRACE_SAMPLE = {
    "messages": [
        {"role": "user", "content": "Solve 2 + 2."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should compute the sum.",
            "tool_calls": [
                {
                    "id": "call_python",
                    "type": "function",
                    "function": {"name": "python", "arguments": '{"code": "2 + 2"}'},
                }
            ],
        },
        {"role": "tool", "name": "python", "tool_call_id": "call_python", "content": "4"},
        {"role": "assistant", "content": "\\boxed{4}", "reasoning_content": ""},
    ],
    "tools": [{"type": "function", "function": {"name": "python", "parameters": {}}}],
    "tool_usage": "with Python TIR",
    "license": "cc-by-4.0",
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

    def test_single_column_adapter_preserves_structured_tool_messages(self):
        """Test Nemotron-style structured messages with reasoning and tool calls."""
        adapter = TransformAdapter(
            dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
            reasoning_content_key="reasoning_content",
            reasoning_content_mode=ReasoningContentMode.PRESERVE_FIELD,
            message_passthrough_keys=("reasoning_content", "tool_calls", "tool_call_id", "name"),
        )

        messages = adapter.transform_conversation_to_openai_format(NEMOTRON_TOOL_TRACE_SAMPLE)
        dumped = [message.model_dump() for message in messages]

        assert [message.role for message in messages] == ["user", "assistant", "tool", "assistant"]
        assert dumped[1]["reasoning_content"] == "I should compute the sum."
        assert dumped[1]["tool_calls"] == NEMOTRON_TOOL_TRACE_SAMPLE["messages"][1]["tool_calls"]
        assert dumped[2]["name"] == "python"
        assert dumped[2]["tool_call_id"] == "call_python"
        assert "reasoning_content" not in dumped[3]


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

    def test_fineproofs_multi_turn_row_preserves_metadata_and_rewrites_think_tags(self):
        """Test FineProofs-like multi-turn rows."""
        adapter = TransformAdapter(
            dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
            conversation_column="messages",
            role_key="role",
            content_key="content",
            user_value="user",
            assistant_value="assistant",
            system_value="system",
        )
        cfg = TransformSFTDatasetConfig(
            source="lm-provers/FineProofs-SFT",
            revision="73661e6",
            output_path="/tmp/output",
            metadata_columns=FINEPROOFS_METADATA_COLUMNS,
            adapter=adapter,
        )

        result = transform_row(FINEPROOFS_MESSAGES_SAMPLE, cfg, adapter)

        assert result is not None
        assert result.source == "lm-provers/FineProofs-SFT"
        assert result.metadata == {
            "category": "number_theory",
            "competition": "imo",
            "gemini-3-pro-grade": 7,
            "qwen3-4b-thinking-reward@128": 0.93,
            "source": "olympiads",
        }
        assert result.messages[0].content == "Prove that 1 + 1 = 2."
        assert result.messages[1].content == "<|start_think|>Use Peano arithmetic.<|end_think|>\nHere is the proof."

    def test_fineproofs_proof_only_row_builds_instruction_response_chat(self):
        """Test FineProofs proof-only row conversion."""
        adapter = TransformAdapter(
            dataset_format=InputDatasetFormat.INSTRUCTION_RESPONSE,
            instruction_column="problem",
            response_column="proof",
        )
        cfg = TransformSFTDatasetConfig(
            source="lm-provers/FineProofs-SFT",
            revision="73661e6",
            output_path="/tmp/output",
            metadata_columns=FINEPROOFS_METADATA_COLUMNS,
            adapter=adapter,
        )

        result = transform_row(FINEPROOFS_PROOF_ONLY_SAMPLE, cfg, adapter)

        assert result is not None
        assert [message.role for message in result.messages] == ["user", "assistant"]
        assert result.messages[0].content == FINEPROOFS_PROOF_ONLY_SAMPLE["problem"]
        assert result.messages[1].content == FINEPROOFS_PROOF_ONLY_SAMPLE["proof"]
        assert result.metadata == {
            "category": "algebra",
            "competition": "aops",
            "gemini-3-pro-grade": 6,
            "qwen3-4b-thinking-reward@128": 0.75,
            "source": "aops",
        }

    def test_instruct_msg_response_skips_misaligned_row(self):
        """A multi-message instruction is dropped (returns None), not emitted as an empty conversation."""
        adapter = TransformAdapter(
            dataset_format=InputDatasetFormat.INSTRUCT_MSG_RESPONSE,
            instruction_column="instruction",
            response_column="response",
        )
        cfg = TransformSFTDatasetConfig(
            source="test/dataset",
            revision="main",
            output_path="/tmp/output",
            metadata_columns=[],
            adapter=adapter,
        )
        # len(instruction) > 1: instruction split across system + user prompts.
        row = {
            "instruction": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            "response": "4",
        }

        assert transform_row(row, cfg, adapter) is None

    def test_nemotron_structured_row_preserves_tools_and_normalizes_tool_calls(self):
        """Test structured multi-turn rows with top-level tools metadata."""
        adapter = TransformAdapter(
            dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
            reasoning_content_key="reasoning_content",
            reasoning_content_mode=ReasoningContentMode.PRESERVE_FIELD,
            message_passthrough_keys=("reasoning_content", "tool_calls", "tool_call_id", "name"),
            metadata_remap={"tools": "tools"},
        )
        cfg = TransformSFTDatasetConfig(
            source="nvidia/Nemotron-SFT-Math-v3",
            revision="ff4439c",
            output_path="/tmp/output",
            metadata_columns=["tool_usage", "license"],
            adapter=adapter,
        )

        result = transform_row(NEMOTRON_TOOL_TRACE_SAMPLE, cfg, adapter)

        assert result is not None
        dumped = result.model_dump()
        assert dumped["tools"] == NEMOTRON_TOOL_TRACE_SAMPLE["tools"]
        assert result.metadata == {"tool_usage": "with Python TIR", "license": "cc-by-4.0"}
        assert dumped["messages"][1]["reasoning_content"] == "I should compute the sum."
        assert dumped["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"code": "2 + 2"}
        assert dumped["messages"][2]["role"] == "tool"
        assert dumped["messages"][2]["tool_call_id"] == "call_python"

    def test_row_filters_support_tool_usage_and_presence_checks(self):
        """Test row filters used for dataset views."""
        assert row_matches_filters(
            NEMOTRON_TOOL_TRACE_SAMPLE,
            [RowFilter("tool_usage", RowFilterOperator.EQUALS, "with Python TIR")],
        )
        assert not row_matches_filters(
            NEMOTRON_TOOL_TRACE_SAMPLE,
            [RowFilter("tool_usage", RowFilterOperator.EQUALS, "without Python TIR")],
        )
        assert row_matches_filters(NEMOTRON_TOOL_TRACE_SAMPLE, [RowFilter("tools", RowFilterOperator.NON_EMPTY)])
        assert row_matches_filters({"tools": []}, [RowFilter("tools", RowFilterOperator.EMPTY)])

    def test_shard_example_limit_distributes_split_cap_deterministically(self):
        limits = [shard_example_limit(100, 6, shard_idx) for shard_idx in range(6)]

        assert limits == [17, 17, 17, 17, 16, 16]
        assert sum(limit for limit in limits if limit is not None) == 100
        assert shard_example_limit(None, 6, 0) is None


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
