# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from experiments.probabilistic_dataflow.compiler import (
    AttentionLayout,
    ExecutionSequenceIR,
    InferencePlanIR,
    PackedBatch,
    TokenCodec,
    TransformerExecutionIR,
    inference_plan_ir,
    lower_to_transformer,
    pack_transformer_calls,
)
from experiments.probabilistic_dataflow.dsl import Axis, FieldType, FlowInfo, InferenceProgram, PositionMode
from experiments.probabilistic_dataflow.synthetic import (
    advection_example,
    advection_program,
    contacts_example,
    contacts_program,
    refined_advection_program,
    scalar_forecast_example,
    scalar_forecast_program,
    structure_example,
    structure_program,
)
from experiments.probabilistic_dataflow.training import (
    TaskBatch,
    build_synthetic_advection_batch,
    build_synthetic_text_batch,
)

DEBUG_OUTPUT_DIR = Path(__file__).with_name("debug_outputs")


def render_program_ir(program: InferenceProgram, *, heading: str = "## 1. Inference Program Values") -> str:
    graph = ["```mermaid", "flowchart LR"]
    for node in program.nodes:
        label = f"%{node.id} {node.name}<br/>{node.kind}<br/>{_field_type(node.value_type)}"
        graph.append(f'  n{node.id}["{label}"]')
    for node in program.nodes:
        for input_id in node.inputs:
            graph.append(f"  n{input_id} --> n{node.id}")
    graph.append("```")

    rows = []
    for node in program.nodes:
        operation = node.operation or node.factor_name or "-"
        factor = node.factor_id or "-"
        rows.append(
            (
                f"%{node.id}",
                node.name,
                str(node.kind),
                _field_type(node.value_type),
                _node_names(program, node.inputs),
                operation,
                factor,
                _flow(node.flow),
            )
        )
    table = _markdown_table(
        ("ID", "Value", "Kind", "Type", "Inputs", "Operation/factor", "Factor ID", "FlowInfo"),
        rows,
    )
    return "\n\n".join((heading, "\n".join(graph), table))


def render_inference_plan_ir(
    plan: InferencePlanIR,
    program: InferenceProgram,
    *,
    heading: str = "## 2. Inference Plan IR",
) -> str:
    properties = _markdown_table(
        ("Property", "Value"),
        (
            ("program", plan.program_name),
            ("external inputs", _node_names(program, plan.input_ids)),
            ("outputs", _node_names(program, plan.output_ids)),
            ("factors", "<br>".join(plan.factor_ids) or "-"),
            ("budget", f"model_calls={plan.model_call_budget}, generated_tokens={plan.generated_token_budget}"),
        ),
    )
    graph = ["```mermaid", "flowchart LR"]
    for call in plan.calls:
        targets = ", ".join(program.node(node_id).name for node_id in call.target_ids)
        graph.append(f'  c{call.id}["call {call.id}<br/>{call.operator}<br/>{targets}"]')
    for call in plan.calls:
        for dependency in call.dependency_call_ids:
            graph.append(f"  c{dependency} --> c{call.id}")
    graph.append("```")

    rows = []
    for call in plan.calls:
        rows.append(
            (
                str(call.id),
                call.operator,
                str(call.iteration),
                _node_names(program, call.context_ids),
                _node_names(program, call.target_ids),
                ", ".join(str(value) for value in call.dependency_call_ids) or "-",
                str(call.attention_layout),
                str(call.position_mode),
                "<br>".join(call.approximation_notes) or "-",
            )
        )
    table = _markdown_table(
        (
            "Call",
            "Operator",
            "Iteration",
            "Context",
            "Targets",
            "Depends on",
            "Attention",
            "Positions",
            "Approximation/notes",
        ),
        rows,
    )
    return "\n\n".join((heading, properties, "\n".join(graph), table))


def render_execution_ir(
    execution: TransformerExecutionIR,
    codec: TokenCodec,
    *,
    heading: str = "## 3. Transformer Execution IR",
    detailed_documents_per_call: int = 1,
) -> str:
    call_rows = []
    for call in execution.calls:
        supervised_tokens = sum(int(sum(sequence.loss_weights)) for sequence in call.sequences)
        call_rows.append(
            (
                str(call.call_id),
                call.operator,
                ", ".join(str(value) for value in call.dependency_call_ids) or "-",
                str(len(call.sequences)),
                str(supervised_tokens),
                call.attention_layout,
                call.position_mode,
            )
        )
    parts = [
        heading,
        _markdown_table(
            (
                "Call",
                "Operator",
                "Depends on",
                "Documents",
                "Supervised tokens",
                "Attention layout",
                "Position mode",
            ),
            call_rows,
        ),
    ]
    for call in execution.calls:
        inventory = []
        for sequence in call.sequences:
            inventory.append(
                (
                    str(sequence.sequence_id),
                    ", ".join(_predicted_semantics(sequence, codec)),
                    str(len(sequence.token_ids)),
                    str(int(sum(sequence.loss_weights))),
                )
            )
        parts.extend(
            (
                f"### Call {call.call_id} document inventory",
                _markdown_table(("Document", "Predicted semantic values", "Records", "Loss positions"), inventory),
            )
        )
        for sequence in call.sequences[:detailed_documents_per_call]:
            parts.extend(
                (
                    f"### Call {call.call_id}, document {sequence.sequence_id}",
                    _render_document(sequence, codec, call.attention_layout, call.position_mode),
                )
            )
        omitted = len(call.sequences) - detailed_documents_per_call
        if omitted > 0:
            parts.append(f"{omitted} additional documents omitted.")
    return "\n\n".join(parts)


def render_packed_batch(
    batch: PackedBatch,
    codec: TokenCodec,
    *,
    heading: str = "## Packed heterogeneous batch",
    max_rows: int = 8,
) -> str:
    summary = _markdown_table(
        ("Property", "Value"),
        (
            ("shape", f"{batch.token_ids.shape[0]} rows x {batch.token_ids.shape[1]} tokens"),
            ("documents", str(len(batch.locations))),
            ("supervised tokens", str(int(batch.loss_weights.sum()))),
            ("rotary positions", "all 0; RoPE is the identity"),
            ("attention", f"{batch.attention_layout} within each segment"),
            ("attention boundary", "segment_id; records cannot attend across documents"),
            ("padding", "segment_id=-1 and loss_weight=0"),
        ),
    )
    location_by_span = {(location.row, location.start, location.end): location for location in batch.locations}
    rows = []
    selected_rows = _selected_row_indices(batch.token_ids.shape[0], max_rows)
    previous_row = -1
    for row_index in selected_rows:
        if row_index > previous_row + 1:
            rows.append(("...", f"{row_index - previous_row - 1} rows omitted", "-"))
        spans = []
        for segment_id, start, end in _segment_spans(batch.segment_ids[row_index]):
            location = location_by_span[(row_index, start, end)]
            losses = int(batch.loss_weights[row_index, start:end].sum())
            spans.append(
                f"seg={segment_id} {start}:{end} {location.example_id}/call{location.call_id}/doc{location.sequence_id} "
                f"losses={losses}"
            )
        first_records = ", ".join(
            _packed_record_label(batch, codec, row_index, position)
            for position in range(min(8, batch.token_ids.shape[1]))
        )
        rows.append((str(row_index), "<br>".join(spans), first_records))
        previous_row = row_index
    return "\n\n".join(
        (
            heading,
            summary,
            _markdown_table(("Row", "Document spans", "First eight physical records"), rows),
        )
    )


def render_example_outputs() -> dict[str, str]:
    outputs: dict[str, str] = {}

    scalar = scalar_forecast_program()
    scalar_data = scalar_forecast_example(scalar)
    scalar_codec = TokenCodec()
    scalar_plan = inference_plan_ir(scalar)
    scalar_execution = lower_to_transformer(scalar, scalar_data, scalar_codec)
    outputs["scalar.md"] = _report(
        "Scalar forecast debug rendering",
        (
            "The smallest program conditions on one scalar measurement and predicts one scalar measurement. "
            "Neither field has an axis, so the staged function creates one context record and one target-query "
            "record in a full-attention model call.",
            render_program_ir(scalar),
            render_inference_plan_ir(scalar_plan, scalar),
            render_execution_ir(scalar_execution, scalar_codec),
        ),
    )

    advection = advection_program()
    advection_data = advection_example(advection, seed=0)
    advection_codec = TokenCodec()
    advection_plan = inference_plan_ir(advection)
    advection_execution = lower_to_transformer(advection, advection_data, advection_codec)
    refinement = refined_advection_program()
    refinement_plan = inference_plan_ir(refinement)
    refinement_execution = lower_to_transformer(refinement, advection_data, advection_codec)
    outputs["advection.md"] = _report(
        "Synthetic advection debug rendering",
        (
            "The logical field has a four-cell mesh and three ordered future times. "
            "`advection_program()` creates one unordered execution document containing 16 observed records "
            "and 12 target-query records. Scientific position embeddings identify mesh and time coordinates. "
            "`refined_advection_program()` feeds the current trajectory into two later calls.",
            render_program_ir(advection),
            render_inference_plan_ir(advection_plan, advection),
            render_execution_ir(advection_execution, advection_codec),
            render_inference_plan_ir(
                refinement_plan,
                refinement,
                heading="## Alternate Inference Plan IR: fixed-step refinement",
            ),
            "The synthetic execution uses realized `future` values for feedback context. A sampling runtime would "
            "replace those values with the proposal produced by the preceding call.",
            render_execution_ir(
                refinement_execution,
                advection_codec,
                heading="## Alternate Transformer Execution IR: fixed-step refinement",
            ),
        ),
    )

    contacts = contacts_program()
    contacts_data = contacts_example(contacts, seed=0)
    contacts_codec = TokenCodec()
    contacts_plan = inference_plan_ir(contacts)
    contacts_execution = lower_to_transformer(contacts, contacts_data, contacts_codec)
    outputs["contacts.md"] = _report(
        "Synthetic contacts debug rendering",
        (
            "The set axis has four residues. Its unordered-pair axis identifies the six `{left, right}` pairs. The "
            "`contacts_program()` creates one unordered document with four observed residue records and six "
            "target-query records.",
            render_program_ir(contacts),
            render_inference_plan_ir(contacts_plan, contacts),
            render_execution_ir(contacts_execution, contacts_codec),
        ),
    )

    structure = structure_program()
    structure_data = structure_example(structure, seed=0)
    structure_codec = TokenCodec()
    structure_plan = inference_plan_ir(structure)
    structure_execution = lower_to_transformer(structure, structure_data, structure_codec)
    outputs["structure.md"] = _report(
        "Factorized structure debug rendering",
        (
            "`structure_program()` generates contacts and then distances. Call 0 generates contacts from "
            "sequence. Call 1 receives both sequence and generated contacts before generating distances. The call "
            "dependency preserves "
            "`p(contacts | sequence) p(distances | sequence, contacts)`. Autoregression is between calls; each call "
            "uses full attention over its scientific records.",
            render_program_ir(structure),
            render_inference_plan_ir(structure_plan, structure),
            render_execution_ir(structure_execution, structure_codec),
        ),
    )

    mixed_codec = TokenCodec()
    mixed_advection_execution = lower_to_transformer(
        advection,
        advection_data,
        mixed_codec,
    )
    mixed_contacts_execution = lower_to_transformer(
        contacts,
        contacts_data,
        mixed_codec,
    )
    mixed_batch = pack_transformer_calls(
        (mixed_advection_execution, mixed_contacts_execution),
        max_seq_len=48,
    )
    outputs["mixed-packing.md"] = _report(
        "Mixed-domain packing debug rendering",
        (
            "Advection and contact documents share dense rows. Segment IDs keep full attention inside each document. "
            "Record order affects packing only; scientific position IDs travel with records and loss weights select "
            "target-query records.",
            render_packed_batch(mixed_batch, mixed_codec),
        ),
    )

    cross_domain_codec = TokenCodec()
    text_batch = build_synthetic_text_batch(cross_domain_codec, repetitions=1)
    science_batch = build_synthetic_advection_batch(cross_domain_codec, examples=1, max_seq_len=32)
    outputs["cross-domain.md"] = _report(
        "Shared text and science model-call rendering",
        (
            "Both calls use one token embedding table, Grug transformer stack, and output projection. Text uses "
            "physical rotary positions and causal attention. Science uses a scientific-position adapter, zero "
            "rotary positions, full attention, and aligned targets. The compiler keeps these incompatible attention "
            "layouts in separate dense batches.",
            _render_cross_domain_batches(text_batch, science_batch, cross_domain_codec),
        ),
    )

    outputs["README.md"] = _report(
        "Generated debug renderings",
        (
            "These files are deterministic outputs of `python -m experiments.probabilistic_dataflow.debug_render`. "
            "Each domain report shows the staged values, inference plan, transformer execution, and "
            "compiler-generated document treatment. The guided walkthrough is [TUTORIAL.md](../TUTORIAL.md).",
            "\n".join(
                (
                    "- [Scalar forecast](scalar.md)",
                    "- [Advection](advection.md)",
                    "- [Contacts](contacts.md)",
                    "- [Factorized structure](structure.md)",
                    "- [Mixed packing](mixed-packing.md)",
                    "- [Shared text and science calls](cross-domain.md)",
                )
            ),
        ),
    )
    return outputs


def write_example_outputs(output_dir: Path = DEBUG_OUTPUT_DIR) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in render_example_outputs().items():
        path = output_dir / name
        path.write_text(content)
        written.append(path)
    return tuple(written)


def check_example_outputs(output_dir: Path = DEBUG_OUTPUT_DIR) -> tuple[str, ...]:
    stale = []
    for name, content in render_example_outputs().items():
        path = output_dir / name
        if not path.exists() or path.read_text() != content:
            stale.append(name)
    return tuple(stale)


def _render_cross_domain_batches(text: TaskBatch, science: TaskBatch, codec: TokenCodec) -> str:
    shared = _markdown_table(
        ("Component", "Sharing"),
        (
            ("Token embedding", f"one table with vocabulary size {codec.vocab_size}"),
            ("Transformer blocks", "same parameters for both calls"),
            ("Output projection", "same parameters and vocabulary for both calls"),
            ("Scientific-position embedding", "added only where scientific_position_id >= 0"),
        ),
    )
    calls = _markdown_table(
        ("Task", "Dense shape", "Scientific positions", "Rotary positions", "Attention", "Targets"),
        (
            (
                text.name,
                f"{text.token_ids.shape[0]} x {text.token_ids.shape[1]}",
                "none",
                "0..6",
                str(text.attention_layout),
                "shifted next-token labels",
            ),
            (
                science.name,
                f"{science.token_ids.shape[0]} x {science.token_ids.shape[1]}",
                "field, time, and mesh coordinates",
                "all 0",
                str(science.attention_layout),
                "aligned scientific-value labels",
            ),
        ),
    )
    return "\n\n".join(
        (
            "## Shared parameter boundary",
            shared,
            "## Data-dependent call strategies",
            calls,
            "### Text call, first document",
            _render_task_records(text, codec, row=0, max_records=text.token_ids.shape[1]),
            "### Scientific call, first document",
            _render_task_records(science, codec, row=0, max_records=10),
        )
    )


def _render_task_records(batch: TaskBatch, codec: TokenCodec, *, row: int, max_records: int) -> str:
    rows = []
    for position in range(min(max_records, batch.token_ids.shape[1])):
        if batch.segment_ids[row, position] < 0:
            break
        scientific_position_id = int(batch.scientific_position_ids[row, position])
        scientific_position = (
            codec.scientific_position_name(scientific_position_id) if scientific_position_id >= 0 else "-"
        )
        target_id = int(batch.target_ids[row, position])
        rows.append(
            (
                str(position),
                codec.token_label(int(batch.token_ids[row, position])),
                scientific_position,
                str(int(batch.rotary_position_ids[row, position])),
                codec.token_label(target_id) if target_id >= 0 else "-",
                f"{batch.loss_weights[row, position]:g}",
            )
        )
    return _markdown_table(
        ("Physical position", "Input token", "Scientific position", "Rotary position", "Target", "Loss"),
        rows,
    )


def _render_document(
    sequence: ExecutionSequenceIR,
    codec: TokenCodec,
    attention_layout: AttentionLayout,
    position_mode: PositionMode,
) -> str:
    rotary_treatment = (
        "all 0; RoPE is the identity" if position_mode == PositionMode.SCIENTIFIC else "physical record indices"
    )
    serialization = (
        "complete records may be permuted; outputs follow the same permutation"
        if position_mode == PositionMode.SCIENTIFIC and attention_layout == AttentionLayout.FULL
        else "physical record order participates in model semantics"
    )
    metadata = (
        f"- Example: `{sequence.example_id}`\n"
        f"- Physical rotary positions: {rotary_treatment}\n"
        f"- Attention: {attention_layout} within this document's segment; no cross-document attention\n"
        f"- Serialization: {serialization}\n"
        f"- Loss: {int(sum(sequence.loss_weights))} aligned target records"
    )
    rows = []
    for position, (token_id, scientific_position_id) in enumerate(
        zip(sequence.token_ids, sequence.scientific_position_ids, strict=True)
    ):
        target_id = sequence.target_ids[position]
        scientific_position = (
            codec.scientific_position_name(scientific_position_id) if scientific_position_id >= 0 else "-"
        )
        if target_id >= 0:
            component = "target record"
            treatment = "query token + position policy; target value is a label, not an input"
            target = codec.token_label(target_id)
        else:
            component = "context record"
            treatment = "value token + position policy; no direct loss"
            target = "-"
        rows.append(
            (
                str(position),
                str(sequence.rotary_position_ids[position]),
                component,
                scientific_position,
                f"{token_id} {codec.token_label(token_id)}",
                treatment,
                target,
                f"{sequence.loss_weights[position]:g}",
            )
        )
    table = _markdown_table(
        (
            "Physical position",
            "Rotary position",
            "Component",
            "Scientific position embedding",
            "Content token",
            "Model treatment",
            "Predicts",
            "Loss",
        ),
        rows,
    )
    return "\n\n".join((metadata, table))


def _predicted_semantics(sequence: ExecutionSequenceIR, codec: TokenCodec) -> tuple[str, ...]:
    return tuple(
        codec.scientific_position_name(scientific_position_id)
        for scientific_position_id, target_id in zip(
            sequence.scientific_position_ids,
            sequence.target_ids,
            strict=True,
        )
        if target_id >= 0
    )


def _packed_record_label(batch: PackedBatch, codec: TokenCodec, row: int, position: int) -> str:
    scientific_position_id = int(batch.scientific_position_ids[row, position])
    if scientific_position_id < 0:
        return "<pad>"
    scientific_position = codec.scientific_position_name(scientific_position_id)
    token = codec.token_label(int(batch.token_ids[row, position]))
    return f"{scientific_position} <= {token}"


def _segment_spans(segment_ids: Iterable[int]) -> tuple[tuple[int, int, int], ...]:
    values = [int(value) for value in segment_ids]
    spans = []
    start = 0
    while start < len(values):
        segment_id = values[start]
        if segment_id < 0:
            break
        end = start + 1
        while end < len(values) and values[end] == segment_id:
            end += 1
        spans.append((segment_id, start, end))
        start = end
    return tuple(spans)


def _selected_row_indices(num_rows: int, max_rows: int) -> tuple[int, ...]:
    if num_rows <= max_rows:
        return tuple(range(num_rows))
    first_count = max_rows // 2
    last_count = max_rows - first_count
    return (*range(first_count), *range(num_rows - last_count, num_rows))


def _field_type(value_type: FieldType) -> str:
    axes = ", ".join(_axis(axis) for axis in value_type.axes) or "scalar"
    return f"{value_type.name}[{axes}] bins={value_type.bins} tokens={value_type.token_count}"


def _axis(axis: Axis) -> str:
    return f"{axis.name}:{axis.kind}={axis.size}"


def _flow(flow: FlowInfo) -> str:
    return (
        f"provenance={_set(flow.provenance)}<br>"
        f"split_keys={_set(flow.split_keys)}<br>"
        f"random_ancestors={_set(flow.random_ancestors)}"
    )


def _node_names(program: InferenceProgram, node_ids: Iterable[int]) -> str:
    names = [f"%{node_id} {program.node(node_id).name}" for node_id in node_ids]
    return "<br>".join(names) or "-"


def _set(values: Iterable[str]) -> str:
    rendered = ", ".join(sorted(values))
    return "{" + rendered + "}" if rendered else "{}"


def _markdown_table(headers: tuple[str, ...], rows: Iterable[tuple[str, ...]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_escape_cell(value) for value in row) + " |" for row in rows]
    return "\n".join((header, separator, *body))


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _report(title: str, sections: tuple[str, ...]) -> str:
    return "\n\n".join((f"# {title}", *sections)) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render probabilistic dataflow IRs and generated documents")
    parser.add_argument("--output-dir", type=Path, default=DEBUG_OUTPUT_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        stale = check_example_outputs(args.output_dir)
        if stale:
            raise SystemExit(f"Stale debug outputs: {', '.join(stale)}")
        print(f"Debug outputs are current: {args.output_dir}")
        return
    for path in write_example_outputs(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
