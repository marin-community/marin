# Conversation input and delivery constraints

**Status:** Proposed TaskCompendium schema extension. The current implementation
uses one `instructions` string and does not implement this contract.

`TaskSpec` needs to preserve the input a model actually receives and any
source requirement that its answer be delivered directly. A submission
convention chooses how an answer is delivered and extracted when the task
allows a choice. The verifier checks the answer's content and format. A target
adapter chooses the model API and execution environment.

This proposal covers one assistant decision after a fixed input history. It
does not specify a reactive user, live tool execution, or a multi-step rollout.
The history produced during a rollout remains the harness's responsibility.

## TaskSpec shape

Replace the single `instructions` string with `input`. Keep `answer_type` as the
coarse result kind and add a `delivery_requirement`:

```text
TaskSpec
  id
  input: ConversationInput
  answer_type: text | number | file | workspace_state | native_action
  delivery_requirement: any_compatible | direct_assistant_message
  requirements
  verifier                         # private
  source
  schema_version

ConversationInput
  events: ordered tuple of TextMessage | AssistantToolCalls | ToolResult
  available_functions: tuple of FunctionDefinition
  next_action: unspecified | none | auto | required | named(name)
  call_cardinality: unspecified | single | multiple

```

The names are proposed API names. The required distinction is between the
coarse answer type, the source conversation, the delivery requirement used by
lowering, and the verifier's correctness rules. A plain arithmetic task has
one user `TextMessage`, `answer_type=number`, and the default
`delivery_requirement=any_compatible`.

## Conversation input

`ConversationInput.events` is the immutable, model-visible prefix before the
model produces the answer being graded. It preserves event order, role, text,
tool-call identifiers, function names and arguments, and tool results. The
event variants are:

| Variant | Required data | Meaning |
| --- | --- | --- |
| `TextMessage` | `role` (`system`, `developer`, `user`, or `assistant`) and nonempty `content` | A source message. Consecutive messages of the same role remain separate. |
| `AssistantToolCalls` | One or more ordered calls, each with `call_id`, `name`, and source arguments; optional assistant text | Calls already present in the prefix. These are historical observations, not calls to execute again. |
| `ToolResult` | `call_id` and model-visible result content | The result of a preceding call. |

`available_functions` contains the function names, descriptions, and parameter
schemas advertised for the *next* assistant decision. `next_action` preserves
whether the source request permits a message or call (`auto`), forbids a call
(`none`), requires a call, or requires one named function. `unspecified` means
the source imposed no tool-choice policy; a submission convention may then
introduce its own answer tool. `call_cardinality` preserves an explicit source
limit of one or multiple calls; `unspecified` means the source did not set that
limit. A named choice must refer to an advertised function; a required choice
needs at least one. These policies are part of the source request. A target
adapter translates them to its API or rejects an unsupported combination. A
convention cannot override an explicit source policy.

The function declarations are independent of `answer_type`: a model may
answer in text or call a function after the same prefix. They do not imply an
executable tool backend. A task that permits live calls also needs a
compatible action interface and target binding under `requirements`. Lowering
preserves the advertised function names, order, and parameter schemas; it
cannot silently change their argument contracts.

Every historical call must have exactly one later result before the next
assistant turn or the decision being graded. While calls in a batch are
pending, only their results may follow; results may arrive in any order. Call
IDs must be unique within the prefix, and each result must reference one of
those pending calls. An importer may assign deterministic IDs if the source
omits them and the call/result pairing is unambiguous; otherwise it rejects
the row. Importers preserve source message boundaries and visible content.
They must take the actual model request as input, not source metadata
containing reference completions, hidden reasoning, labels, or grader rubrics.

A target adapter must deliver the prefix with the same roles and event order.
It may translate function-call wire syntax, but it cannot flatten a system
message into a user instruction, merge consecutive messages, drop a tool
result, or silently repair source arguments. An adapter that cannot represent
the prefix is incompatible with the task. Submission-convention instructions
may be added after the prefix only where the target can do so without changing
the source events or separating a call from its result.

For example, a [MultiTurnChat row](https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1/viewer/default/train?row=0)
contains a system message and alternating user/assistant turns before the next
text answer. [Agentic pivot rows](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1/blob/9643c8103d7bfbc2d7fc4d15991d6739c612ff58/train.jsonl)
also contain prior function calls and results. Neither can be represented
faithfully by concatenating their text into `instructions`.

## Delivery and verification

`answer_type` continues to describe what kind of result is requested. A
submission convention first checks `supports(spec.answer_type)`. It then
checks whether it preserves `spec.delivery_requirement`. Neither check
requires the spec to list permitted convention IDs.

`any_compatible` allows a convention to wrap and extract the answer. For
example, a number may be delivered as plain text or in a JSON `answer` field.
`direct_assistant_message` requires the answer itself to be the final assistant
message content. A JSON `answer` wrapper, answer function call, or file
submission is incompatible. This expresses a source requirement such as
“return only the C++ code” without binding the task to a named convention.
Lowering must reject an incompatible convention before execution; grading a
contradictory prompt after the fact would not repair the task.

The verifier owns content checks, including required formats. A JSON Schema
grader parses the entire extracted answer as one JSON value and validates it
against a pinned schema and dialect. It may be the whole verifier for a
structured-output task or one check in a verifier that also grades content.
The schema does not need a second top-level `TaskSpec` field. The model-visible
input supplies the public schema; the verifier keeps the canonical schema used
for grading. Importers must check that the two agree and reject known
contradictions. The verifier's reference answers, hidden tests, and rubrics
remain private.

For reproducible JSON Schema grading, the verifier config declares the dialect
and contains a self-contained schema. Local fragment `$ref`s may resolve within
it; external references are rejected. An embedded `$schema` URI must agree
with the declared dialect. Validation uses no network access, rejects duplicate
object keys, and does not strip prose or Markdown fences from the answer.
These are verifier rules, not submission-convention compatibility rules.

After a convention extracts a submission, the verifier checks its content. A
schema-invalid answer is a graded task failure. An unparseable convention
envelope or missing submission is an extraction failure. A verifier may define
partial credit for other content criteria.

For example, [structured-output rows](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs/tree/4ed3d0e404c03dd47453217ce498c3511aa90a44)
carry a separate `schema_str`; some place the schema in a second user message.
A row that demands raw JSON has `answer_type=text` and
`delivery_requirement=direct_assistant_message`. Its verifier checks the JSON
schema. A JSON `answer` wrapper is excluded even though that convention
supports text answers generally.

| Task | Delivery requirement | Compatible example | Excluded example | Content check |
| --- | --- | --- | --- | --- |
| Arithmetic answer | `any_compatible` | Plain text or JSON `answer` field | A convention for workspace state | Numeric verifier |
| Raw C++ response | `direct_assistant_message` | Final message containing code | JSON `answer` field or answer call | Code verifier |
| Raw structured output | `direct_assistant_message` | Final message containing a JSON document | JSON `answer` field or answer call | JSON Schema verifier |
| Structured value with flexible delivery | `any_compatible` | Plain JSON document or a convention that extracts that document | A convention that changes the extracted value | JSON Schema verifier |

## Validation and migration

- Version the serialized schema when these fields land. Update importers,
  exports, and readers together; do not infer a conversation from a serialized
  `instructions` field at read time.
- Convert a current single-instruction task to one user `TextMessage` during
  the importer migration. Move source messages and advertised functions from
  native-action-specific request data into `ConversationInput`.
- Validate the verifier's JSON Schema and declared dialect at import time.
  Reject unsupported dialects, non-object schemas, external references, and
  disagreement with the public schema in the input. Reject ambiguous or
  incomplete call/result histories.
- Reject a lowering when it cannot preserve message roles, call/result
  pairing, or direct-output delivery. Candidate
  enumeration and export must apply the same compatibility rule.
- Include the ordered input, delivery requirement, and verifier configuration in
  the canonical spec hash.
  A trace records the selected convention, target adapter, and source revision.

An implementation is complete when a plain arithmetic task still has both
plain and wrapped candidates; a raw-output text task excludes wrappers; a
structured-output task preserves two separate user messages and its verifier
rejects an invalid schema answer; and a pivot task retains prior calls and
results without executing them during prefix construction. Hidden reference
data must remain absent from all model-visible events.
