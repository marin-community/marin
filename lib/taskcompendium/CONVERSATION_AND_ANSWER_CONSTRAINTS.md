# Conversation input and answer constraints

**Status:** Proposed TaskCompendium schema extension. The current implementation
uses one `instructions` string and does not implement this contract.

`TaskSpec` needs to preserve the input a model actually receives and the
intrinsic form of a valid answer. These are task semantics. A submission
convention still chooses how an answer is delivered and extracted, and a target
adapter still chooses the model API and execution environment.

This proposal covers one assistant decision after a fixed input history. It
does not specify a reactive user, live tool execution, or a multi-step rollout.
The history produced during a rollout remains the harness's responsibility.

## TaskSpec shape

Replace the single `instructions` string with `input`. Keep `answer_type` as the
coarse result kind and add optional `answer_constraints`:

```text
TaskSpec
  id
  input: ConversationInput
  answer_type: text | number | file | workspace_state | native_action
  answer_constraints: AnswerConstraints
  requirements
  verifier                         # private
  source
  schema_version

ConversationInput
  events: ordered tuple of TextMessage | AssistantToolCalls | ToolResult
  available_functions: tuple of FunctionDefinition
  next_action: unspecified | none | auto | required | named(name)
  call_cardinality: unspecified | single | multiple

AnswerConstraints
  delivery: any_compatible | direct_assistant_message
  content: unconstrained | JsonSchemaConstraint

JsonSchemaConstraint
  dialect: explicit JSON Schema dialect URI
  schema: JSON object
```

The names are proposed API names. The required distinction is between the
coarse answer type, the source conversation, and constraints on the answer's
form. A plain arithmetic task has one user `TextMessage`, `answer_type=number`,
and the default `AnswerConstraints(any_compatible, unconstrained)`.

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

## Intrinsic answer constraints

`answer_type` continues to describe what kind of result is requested. A
submission convention first checks `supports(spec.answer_type)`. It then
checks whether it preserves `spec.answer_constraints`. Neither check requires
the spec to list permitted convention IDs.

`delivery=any_compatible` allows a convention to wrap and extract the answer
if the extracted value still satisfies the content constraint. For example, a
number may be delivered as plain text or in a JSON `answer` field.
`delivery=direct_assistant_message` requires the answer itself to be the final
assistant message content. A JSON `answer` wrapper, answer function call, or
file submission is incompatible. This expresses a source requirement such as
“return only the C++ code” without binding the task to a named convention.

`content=unconstrained` adds no machine-readable format requirement. A
`JsonSchemaConstraint` requires the answer content to be a JSON document that
validates against its pinned schema and dialect. The constraint can apply to
the extracted answer under `any_compatible`, or to the raw final assistant
message under `direct_assistant_message`. It does not turn `answer_type=text`
into a new result kind. Other intrinsic formats can add tagged constraint
variants when a source requires them; this proposal defines JSON Schema only.
Validation parses the entire answer as one JSON value, allowing surrounding
whitespace but no prose or Markdown fences. Duplicate object keys are invalid.
The first implementation accepts self-contained schemas: local fragment
`$ref`s may resolve within the schema, and external references are rejected.
An embedded `$schema` URI must agree with the declared dialect. Validation
uses the declared dialect with no network access.

The model-visible input must communicate every intrinsic constraint. The
canonical constraint in `TaskSpec` gives lowering and grading a machine-readable
version of that rule. If a source prompt already supplies a schema, the
importer stores the same schema in `answer_constraints` and does not append a
second, conflicting schema. A known contradiction between source text and
schema is an import rejection. The schema itself is public task material;
reference answers, hidden tests, and grading rubrics remain in the verifier.

After a convention extracts a submission, the format constraint is checked
before the source verifier. An answer that was extracted but violates the
task's JSON Schema is a graded task failure, not an extraction failure. An
unparseable convention envelope or a missing submission remains an extraction
failure. Verifier-specific criteria may then grade the content, including
partial credit where the verifier defines it.

For example, [structured-output rows](https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following-structured_outputs/tree/4ed3d0e404c03dd47453217ce498c3511aa90a44)
carry a separate `schema_str`; some place the schema in a second user message.
A row that demands raw JSON has `answer_type=text`, a JSON Schema content
constraint, and `delivery=direct_assistant_message`. A JSON `answer` wrapper
must be excluded even though that convention supports text answers generally.

| Task | Answer constraints | Compatible example | Excluded example |
| --- | --- | --- | --- |
| Arithmetic answer | `any_compatible`, unconstrained | Plain text or JSON `answer` field | A convention for workspace state |
| Raw C++ response | `direct_assistant_message`, unconstrained | Final message containing code | JSON `answer` field or answer call |
| Raw structured output | `direct_assistant_message`, JSON Schema | Final message containing one valid JSON document | JSON `answer` field or answer call |
| Structured value with flexible delivery | `any_compatible`, JSON Schema | Plain JSON document or a convention that extracts that document | A convention that changes the extracted value |

## Validation and migration

- Version the serialized schema when these fields land. Update importers,
  exports, and readers together; do not infer a conversation from a serialized
  `instructions` field at read time.
- Convert a current single-instruction task to one user `TextMessage` during
  the importer migration. Move source messages and advertised functions from
  native-action-specific request data into `ConversationInput`.
- Validate the JSON Schema and its declared dialect at import time. Reject
  unsupported dialects, non-object schemas, external references, ambiguous or
  incomplete call/result histories, and known prompt/constraint contradictions.
- Reject a lowering when it cannot preserve message roles, call/result
  pairing, direct-output delivery, or the intrinsic format. Candidate
  enumeration and export must apply the same compatibility rule.
- Include the ordered input and answer constraints in the canonical spec hash.
  A trace records the selected convention, target adapter, and source revision.

An implementation is complete when a plain arithmetic task still has both
plain and wrapped candidates; a raw-output text task excludes wrappers; a
structured-output task preserves two separate user messages and rejects an
invalid schema answer; and a pivot task retains prior calls and results without
executing them during prefix construction. Hidden reference data must remain
absent from all model-visible events.
