# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conservative weight-operand provenance through optimized HLO control flow."""

import re
from dataclasses import dataclass

# Array roles are True (a parameter/layout transform), False (computed activation)
# or "constant". Tuples preserve roles for while carries and get-tuple-element.
type Role = bool | str | tuple[Role, ...]


@dataclass(frozen=True)
class Operation:
    name: str
    opcode: str
    operands: tuple[str, ...]
    line: str


def _intersect(left: Role, right: Role) -> Role:
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
        return tuple(_intersect(a, b) for a, b in zip(left, right, strict=True))
    return left if left == right else False


def weight_operand_operations(
    text: str, weight_parameter_prefixes: tuple[str, ...]
) -> tuple[list[tuple[Operation, tuple[Role, ...]]], bool]:
    """Evaluate parameter/layout provenance, preserving per-call and loop contexts.

    Dot results are activations even when their inputs depend on weights. Thus a
    Wgrad contraction of two activations is distinct from a forward or activation
    VJP contraction with a sliced/transposed weight. Unknown constructs remain
    unverified; they cannot certify an operand as a weight.
    """
    functions: dict[str, list[Operation]] = {}
    entry = None
    current = None
    for line in text.splitlines():
        header = re.match(r"^(ENTRY )?%([^ (]+) .*\{$", line)
        if header:
            current = header.group(2)
            functions[current] = []
            if header.group(1):
                entry = current
            continue
        if current is None or " = " not in line:
            continue
        instruction = re.match(r"\s*(?:ROOT )?%([^ ]+) = (.*)", line)
        if instruction is None:
            continue
        operation = re.search(r"\b([a-z][a-z0-9-]*)\(([^)]*)\)", instruction.group(2))
        if operation is None:
            continue
        operands = tuple(re.findall(r"%([a-zA-Z_][\w.-]*)", operation.group(2)))
        functions[current].append(Operation(instruction.group(1), operation.group(1), operands, line.strip()))
    if entry is None:
        return [], False
    observations: list[tuple[Operation, tuple[Role, ...]]] = []
    parsed = True
    unary_layout = {
        "bitcast",
        "reshape",
        "transpose",
        "slice",
        "dynamic-slice",
        "convert",
        "copy",
        "broadcast",
        "reverse",
        "all-gather",
        "all-to-all",
        "collective-permute",
        "optimization-barrier",
    }
    binary_layout = {"concatenate", "multiply", "divide", "add", "subtract", "pad"}

    def evaluate(function: str, parameters: tuple[Role, ...], depth: int = 0) -> Role:
        nonlocal parsed
        if function not in functions or depth > 20:
            parsed = False
            return False
        values: dict[str, Role] = {}
        root: Role = False
        for operation in functions[function]:
            inputs = tuple(values.get(name, False) for name in operation.operands)
            opcode = operation.opcode
            role: Role = False
            if opcode == "parameter":
                number = int(re.search(r"parameter\((\d+)\)", operation.line).group(1))
                if function == entry:
                    # Certification comes from the compiled training entry's
                    # explicit parameter boundary, never from floating dtype.
                    # Unknown floating inputs may be activation cotangents.
                    metadata = re.search(r'op_name="([^" ]+)"', operation.line)
                    path = metadata.group(1) if metadata else operation.name
                    role = any(
                        path.startswith(prefix) or operation.name.startswith(prefix.replace(".", "_"))
                        for prefix in weight_parameter_prefixes
                    )
                elif number < len(parameters):
                    role = parameters[number]
            elif opcode in ("constant", "iota"):
                role = "constant"
            elif opcode == "compare" and inputs and all(value == "constant" for value in inputs):
                # Fixed token-ID/vocabulary-column equality maps are the sparse
                # output-delta selector, not learned projection weights.
                role = "selector"
            elif opcode == "tuple":
                role = inputs
            elif opcode == "get-tuple-element":
                index = int(re.search(r"index=(\d+)", operation.line).group(1))
                if inputs and isinstance(inputs[0], tuple) and index < len(inputs[0]):
                    role = inputs[0][index]
            elif opcode in unary_layout and inputs:
                role = inputs[0]
            elif opcode in binary_layout and inputs:
                if all(value == "constant" for value in inputs):
                    role = "constant"
                elif all(value is True or value == "constant" for value in inputs) and True in inputs:
                    role = True
            elif opcode == "select" and len(inputs) == 3:
                role = _intersect(inputs[1], inputs[2])
            elif opcode in ("fusion", "call"):
                callee = re.search(r"(?:calls|to_apply)=%([^, }]+)", operation.line)
                if callee:
                    role = evaluate(callee.group(1), inputs, depth + 1)
                else:
                    parsed = False
            elif opcode == "conditional":
                branches = re.search(r"branch_computations=\{([^}]+)\}", operation.line)
                names = (
                    re.findall(r"%([^, }]+)", branches.group(1))
                    if branches
                    else re.findall(r"(?:true|false)_computation=%([^, }]+)", operation.line)
                )
                if len(names) != len(inputs) - 1:
                    parsed = False
                else:
                    results = [
                        evaluate(name, (value,), depth + 1) for name, value in zip(names, inputs[1:], strict=True)
                    ]
                    role = results[0]
                    for result in results[1:]:
                        role = _intersect(role, result)
            elif opcode == "while" and inputs:
                body = re.search(r"body=%([^, }]+)", operation.line)
                if body:
                    role = inputs[0]
                    condition = re.search(r"condition=%([^, }]+)", operation.line)
                    if condition:
                        evaluate(condition.group(1), (role,), depth + 1)
                    # Frozen parameters remain unchanged in layer/vocab scans.
                    # Intersect with the body result so modified carries lose certification.
                    for _ in range(3):
                        after = evaluate(body.group(1), (role,), depth + 1)
                        stable = _intersect(role, after)
                        if stable == role:
                            break
                        role = stable
                    else:
                        parsed = False
                else:
                    parsed = False
            if opcode in ("dot", "custom-call", "scatter"):
                observations.append((operation, inputs))
            values[operation.name] = role
            if operation.line.startswith("ROOT "):
                root = role
        return root

    evaluate(entry, ())
    return list(dict.fromkeys(observations)), parsed


@dataclass(frozen=True)
class GradientAudit:
    parsed_weight_operand_dataflow: bool
    inspected_operations: int
    full_weight_gradient_dot_results: list[dict]
    certified_activation_dots: list[dict]
    unknown_custom_call_outputs: list[str]

    @property
    def avoids_full_base_gradients(self) -> bool:
        return (
            self.parsed_weight_operand_dataflow
            and self.inspected_operations > 0
            and not self.full_weight_gradient_dot_results
            and not self.unknown_custom_call_outputs
        )


def audit_gradient_operations(
    text: str,
    shapes: dict[tuple[int, ...], set[str]],
    *,
    weight_parameter_prefixes: tuple[str, ...] = (),
) -> GradientAudit:
    """Check every declared optimized matrix result against frozen parameter shapes."""
    forbidden = []
    unknown_calls = []
    certified_activation_dots = []
    observations, parsed_dataflow = weight_operand_operations(text, weight_parameter_prefixes)
    for operation, operands in observations:
        line = operation.line
        output_type = line.split(" = ", 1)[1].split(" dot(", 1)[0].split("custom-call(", 1)[0].split("scatter(", 1)[0]
        results = [
            tuple(int(dim) for dim in match.split(","))
            for match in re.findall(r"(?:bf16|f16|f32|f64)\[([0-9,]+)\]", output_type)
        ]
        if operation.opcode == "custom-call" and not results:
            unknown_calls.append(line)
        for result in results:
            if result not in shapes:
                continue
            match = {"shape": list(result), "matching_frozen_paths": sorted(shapes[result]), "operation": line}
            if operation.opcode == "dot" and any(role is True or role == "selector" for role in operands):
                match["certification"] = (
                    "known parameter/layout operand"
                    if any(role is True for role in operands)
                    else "fixed token-ID/vocabulary-column selector operand"
                )
                certified_activation_dots.append(match)
            else:
                forbidden.append(match)
    return GradientAudit(parsed_dataflow, len(observations), forbidden, certified_activation_dots, unknown_calls)
