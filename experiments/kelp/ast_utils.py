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

"""
AST parsing and tensor conversion utilities for tree diffusion.

This module provides utilities for:
- Parsing Python code into AST
- Converting AST to padded tensor representation
- Converting tensors back to AST/code
"""

import ast
from dataclasses import dataclass

import numpy as np

from experiments.kelp.python_grammar import PythonNodeVocab, PythonValueVocab


@dataclass
class TreeTensors:
    """Padded tensor representation of an AST.

    This representation is designed for efficient batching in JAX:
    - All arrays have fixed shape (max_nodes, ...)
    - Invalid nodes are masked out with node_mask

    Attributes:
        node_types: (max_nodes,) - Node type IDs from PythonNodeVocab
        node_values: (max_nodes, max_value_len) - Token IDs for node values
        parent_indices: (max_nodes,) - Parent node index (-1 for root)
        child_indices: (max_nodes, max_children) - Child node indices (-1 for none)
        num_children: (max_nodes,) - Number of children for each node
        node_mask: (max_nodes,) - 1 for valid nodes, 0 for padding
        depth: (max_nodes,) - Depth in tree (root=0)
    """

    node_types: np.ndarray  # (max_nodes,) int32
    node_values: np.ndarray  # (max_nodes, max_value_len) int32
    parent_indices: np.ndarray  # (max_nodes,) int32
    child_indices: np.ndarray  # (max_nodes, max_children) int32
    num_children: np.ndarray  # (max_nodes,) int32
    node_mask: np.ndarray  # (max_nodes,) bool or int32
    depth: np.ndarray  # (max_nodes,) int32

    @property
    def max_nodes(self) -> int:
        return self.node_types.shape[0]

    @property
    def num_valid_nodes(self) -> int:
        return int(self.node_mask.sum())


def get_node_value(node: ast.AST) -> str | None:
    """Extract the value from an AST node if applicable.

    Different node types store their values in different attributes.
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return node.name
    elif isinstance(node, ast.ClassDef):
        return node.name
    elif isinstance(node, ast.arg):
        return node.arg
    elif isinstance(node, ast.alias):
        return node.name
    elif isinstance(node, ast.keyword):
        return node.arg if node.arg else None
    elif isinstance(node, ast.Attribute):
        return node.attr
    return None


def set_node_value(node: ast.AST, value: str) -> None:
    """Set the value on an AST node if applicable."""
    if isinstance(node, ast.Name):
        node.id = value
    elif isinstance(node, ast.Constant):
        # Try to parse the repr'd value
        try:
            node.value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            node.value = value
    elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        node.name = value
    elif isinstance(node, ast.ClassDef):
        node.name = value
    elif isinstance(node, ast.arg):
        node.arg = value
    elif isinstance(node, ast.alias):
        node.name = value
    elif isinstance(node, ast.keyword):
        node.arg = value if value else None
    elif isinstance(node, ast.Attribute):
        node.attr = value


def ast_to_tensors(
    tree: ast.AST,
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
    max_nodes: int = 256,
    max_children: int = 16,
    max_value_len: int = 32,
) -> TreeTensors:
    """Convert an AST to padded tensor representation.

    Traverses the AST in pre-order (parent before children) and assigns
    indices to nodes. Builds the parent/child relationship arrays.

    Args:
        tree: The AST to convert
        node_vocab: Vocabulary for node types
        value_vocab: Vocabulary for node values
        max_nodes: Maximum number of nodes (padding size)
        max_children: Maximum number of children per node
        max_value_len: Maximum length of value encoding

    Returns:
        TreeTensors with the tensor representation
    """
    # Initialize arrays
    node_types = np.full(max_nodes, node_vocab.pad_id, dtype=np.int32)
    node_values = np.full((max_nodes, max_value_len), value_vocab.pad_id, dtype=np.int32)
    parent_indices = np.full(max_nodes, -1, dtype=np.int32)
    child_indices = np.full((max_nodes, max_children), -1, dtype=np.int32)
    num_children = np.zeros(max_nodes, dtype=np.int32)
    node_mask = np.zeros(max_nodes, dtype=np.int32)
    depth = np.zeros(max_nodes, dtype=np.int32)

    # Pre-order traversal with parent tracking
    node_to_idx: dict[int, int] = {}  # id(node) -> index
    stack: list[tuple[ast.AST, int, int]] = [(tree, -1, 0)]  # (node, parent_idx, depth)
    current_idx = 0

    while stack and current_idx < max_nodes:
        node, parent_idx, node_depth = stack.pop()
        idx = current_idx
        current_idx += 1

        # Record node info
        node_to_idx[id(node)] = idx
        node_types[idx] = node_vocab.encode_node(node)
        node_mask[idx] = 1
        depth[idx] = node_depth
        parent_indices[idx] = parent_idx

        # Update parent's child list
        if parent_idx >= 0:
            child_count = num_children[parent_idx]
            if child_count < max_children:
                child_indices[parent_idx, child_count] = idx
                num_children[parent_idx] = child_count + 1

        # Encode node value if present
        value = get_node_value(node)
        if value is not None:
            value_ids = value_vocab.encode_value(value, max_value_len)
            node_values[idx] = np.array(value_ids, dtype=np.int32)

        # Add children to stack (reverse order for pre-order traversal)
        children = list(ast.iter_child_nodes(node))
        for child in reversed(children):
            stack.append((child, idx, node_depth + 1))

    return TreeTensors(
        node_types=node_types,
        node_values=node_values,
        parent_indices=parent_indices,
        child_indices=child_indices,
        num_children=num_children,
        node_mask=node_mask,
        depth=depth,
    )


def tensors_to_ast(
    tensors: TreeTensors,
    node_vocab: PythonNodeVocab,
    value_vocab: PythonValueVocab,
) -> ast.AST:
    """Convert tensor representation back to AST.

    Reconstructs the AST from the tensor representation by creating nodes
    and linking them according to the parent/child arrays.

    Args:
        tensors: The tensor representation
        node_vocab: Vocabulary for node types
        value_vocab: Vocabulary for node values

    Returns:
        The reconstructed AST
    """
    # Map node type names to AST classes
    ast_classes: dict[str, type[ast.AST]] = {}
    for name in dir(ast):
        obj = getattr(ast, name)
        if isinstance(obj, type) and issubclass(obj, ast.AST):
            ast_classes[name] = obj

    # Create all nodes first (without linking)
    nodes: list[ast.AST | None] = [None] * tensors.max_nodes
    node_children: dict[int, list[tuple[int, ast.AST]]] = {}  # parent_idx -> [(child_idx, child_node)]

    for idx in range(tensors.num_valid_nodes):
        if tensors.node_mask[idx] == 0:
            continue

        node_type = node_vocab.decode(int(tensors.node_types[idx]))
        if node_type in ("PAD", "UNK"):
            continue

        if node_type not in ast_classes:
            continue

        # Create node with minimal required fields
        node_class = ast_classes[node_type]
        node = _create_ast_node(node_class)
        if node is None:
            continue

        # Set value if present
        value_ids = tensors.node_values[idx].tolist()
        value = value_vocab.decode_value(value_ids)
        if value:
            set_node_value(node, value)

        nodes[idx] = node

        # Track parent-child relationships
        parent_idx = int(tensors.parent_indices[idx])
        if parent_idx >= 0:
            if parent_idx not in node_children:
                node_children[parent_idx] = []
            node_children[parent_idx].append((idx, node))

    # Link children to parents
    for parent_idx, children in node_children.items():
        parent = nodes[parent_idx]
        if parent is None:
            continue

        # Sort children by their index to maintain order
        children.sort(key=lambda x: x[0])
        child_nodes = [c[1] for c in children]

        # Assign children to appropriate fields based on parent type
        _assign_children(parent, child_nodes)

    # Return root (index 0)
    root = nodes[0]
    if root is None:
        return ast.Module(body=[], type_ignores=[])

    # Fix missing locations
    ast.fix_missing_locations(root)
    return root


def _create_ast_node(node_class: type[ast.AST]) -> ast.AST | None:
    """Create an AST node with minimal required fields."""
    try:
        # Handle different node types with their required fields
        if node_class == ast.Module:
            return ast.Module(body=[], type_ignores=[])
        elif node_class == ast.FunctionDef:
            args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
            return ast.FunctionDef(name="f", args=args, body=[], decorator_list=[])
        elif node_class == ast.AsyncFunctionDef:
            args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
            return ast.AsyncFunctionDef(name="f", args=args, body=[], decorator_list=[])
        elif node_class == ast.ClassDef:
            return ast.ClassDef(name="C", bases=[], keywords=[], body=[], decorator_list=[])
        elif node_class == ast.Return:
            return ast.Return(value=None)
        elif node_class == ast.Assign:
            return ast.Assign(targets=[], value=ast.Constant(value=None))
        elif node_class == ast.AugAssign:
            return ast.AugAssign(target=ast.Name(id="x", ctx=ast.Store()), op=ast.Add(), value=ast.Constant(value=0))
        elif node_class == ast.For:
            return ast.For(
                target=ast.Name(id="i", ctx=ast.Store()), iter=ast.Name(id="iter", ctx=ast.Load()), body=[], orelse=[]
            )
        elif node_class == ast.While:
            return ast.While(test=ast.Constant(value=True), body=[], orelse=[])
        elif node_class == ast.If:
            return ast.If(test=ast.Constant(value=True), body=[], orelse=[])
        elif node_class == ast.With:
            return ast.With(items=[], body=[])
        elif node_class == ast.Raise:
            return ast.Raise(exc=None, cause=None)
        elif node_class == ast.Try:
            return ast.Try(body=[], handlers=[], orelse=[], finalbody=[])
        elif node_class == ast.Assert:
            return ast.Assert(test=ast.Constant(value=True), msg=None)
        elif node_class == ast.Import:
            return ast.Import(names=[])
        elif node_class == ast.ImportFrom:
            return ast.ImportFrom(module=None, names=[], level=0)
        elif node_class == ast.Expr:
            return ast.Expr(value=ast.Constant(value=None))
        elif node_class == ast.Pass:
            return ast.Pass()
        elif node_class == ast.Break:
            return ast.Break()
        elif node_class == ast.Continue:
            return ast.Continue()
        elif node_class == ast.BoolOp:
            return ast.BoolOp(op=ast.And(), values=[])
        elif node_class == ast.BinOp:
            return ast.BinOp(left=ast.Constant(value=0), op=ast.Add(), right=ast.Constant(value=0))
        elif node_class == ast.UnaryOp:
            return ast.UnaryOp(op=ast.Not(), operand=ast.Constant(value=True))
        elif node_class == ast.Lambda:
            return ast.Lambda(
                args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=ast.Constant(value=None),
            )
        elif node_class == ast.IfExp:
            return ast.IfExp(test=ast.Constant(value=True), body=ast.Constant(value=1), orelse=ast.Constant(value=0))
        elif node_class == ast.Dict:
            return ast.Dict(keys=[], values=[])
        elif node_class == ast.Set:
            return ast.Set(elts=[])
        elif node_class == ast.ListComp:
            return ast.ListComp(elt=ast.Constant(value=None), generators=[])
        elif node_class == ast.SetComp:
            return ast.SetComp(elt=ast.Constant(value=None), generators=[])
        elif node_class == ast.DictComp:
            return ast.DictComp(key=ast.Constant(value=None), value=ast.Constant(value=None), generators=[])
        elif node_class == ast.GeneratorExp:
            return ast.GeneratorExp(elt=ast.Constant(value=None), generators=[])
        elif node_class == ast.Await:
            return ast.Await(value=ast.Constant(value=None))
        elif node_class == ast.Yield:
            return ast.Yield(value=None)
        elif node_class == ast.YieldFrom:
            return ast.YieldFrom(value=ast.Constant(value=None))
        elif node_class == ast.Compare:
            return ast.Compare(left=ast.Constant(value=0), ops=[], comparators=[])
        elif node_class == ast.Call:
            return ast.Call(func=ast.Name(id="f", ctx=ast.Load()), args=[], keywords=[])
        elif node_class == ast.Constant:
            return ast.Constant(value=None)
        elif node_class == ast.Attribute:
            return ast.Attribute(value=ast.Name(id="x", ctx=ast.Load()), attr="attr", ctx=ast.Load())
        elif node_class == ast.Subscript:
            return ast.Subscript(value=ast.Name(id="x", ctx=ast.Load()), slice=ast.Constant(value=0), ctx=ast.Load())
        elif node_class == ast.Starred:
            return ast.Starred(value=ast.Name(id="x", ctx=ast.Load()), ctx=ast.Load())
        elif node_class == ast.Name:
            return ast.Name(id="x", ctx=ast.Load())
        elif node_class == ast.List:
            return ast.List(elts=[], ctx=ast.Load())
        elif node_class == ast.Tuple:
            return ast.Tuple(elts=[], ctx=ast.Load())
        elif node_class == ast.Slice:
            return ast.Slice(lower=None, upper=None, step=None)
        elif node_class == ast.arguments:
            return ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
        elif node_class == ast.arg:
            return ast.arg(arg="x", annotation=None)
        elif node_class == ast.keyword:
            return ast.keyword(arg=None, value=ast.Constant(value=None))
        elif node_class == ast.alias:
            return ast.alias(name="module", asname=None)
        elif node_class == ast.withitem:
            return ast.withitem(context_expr=ast.Name(id="x", ctx=ast.Load()), optional_vars=None)
        elif node_class == ast.comprehension:
            return ast.comprehension(
                target=ast.Name(id="x", ctx=ast.Store()),
                iter=ast.Name(id="iter", ctx=ast.Load()),
                ifs=[],
                is_async=0,
            )
        elif node_class == ast.ExceptHandler:
            return ast.ExceptHandler(type=None, name=None, body=[])
        elif node_class in (ast.Load, ast.Store, ast.Del):
            return node_class()
        elif node_class in (ast.And, ast.Or):
            return node_class()
        elif node_class in (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.MatMult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.LShift,
            ast.RShift,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
            ast.FloorDiv,
        ):
            return node_class()
        elif node_class in (ast.Invert, ast.Not, ast.UAdd, ast.USub):
            return node_class()
        elif node_class in (
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Is,
            ast.IsNot,
            ast.In,
            ast.NotIn,
        ):
            return node_class()
        else:
            # Try to instantiate with no arguments
            return node_class()
    except TypeError:
        return None


def _assign_children(parent: ast.AST, children: list[ast.AST]) -> None:
    """Assign children to appropriate fields of the parent node."""
    if not children:
        return

    # Map parent types to their child-accepting fields
    if isinstance(parent, ast.Module):
        parent.body = [c for c in children if isinstance(c, ast.stmt)]
    elif isinstance(parent, ast.FunctionDef | ast.AsyncFunctionDef):
        for child in children:
            if isinstance(child, ast.arguments):
                parent.args = child
            elif isinstance(child, ast.stmt):
                parent.body.append(child)
            elif isinstance(child, ast.expr):
                parent.decorator_list.append(child)
    elif isinstance(parent, ast.ClassDef):
        for child in children:
            if isinstance(child, ast.stmt):
                parent.body.append(child)
            elif isinstance(child, ast.expr):
                parent.bases.append(child)
    elif isinstance(parent, ast.Return):
        for child in children:
            if isinstance(child, ast.expr):
                parent.value = child
                break
    elif isinstance(parent, ast.Assign):
        exprs = [c for c in children if isinstance(c, ast.expr)]
        if len(exprs) >= 2:
            parent.targets = exprs[:-1]
            parent.value = exprs[-1]
        elif len(exprs) == 1:
            parent.value = exprs[0]
    elif isinstance(parent, ast.For | ast.AsyncFor):
        for child in children:
            if isinstance(child, ast.expr):
                if parent.target == ast.Name(id="i", ctx=ast.Store()):
                    parent.target = child
                else:
                    parent.iter = child
            elif isinstance(child, ast.stmt):
                parent.body.append(child)
    elif isinstance(parent, ast.While):
        for child in children:
            if isinstance(child, ast.expr):
                parent.test = child
            elif isinstance(child, ast.stmt):
                parent.body.append(child)
    elif isinstance(parent, ast.If):
        for child in children:
            if isinstance(child, ast.expr):
                parent.test = child
            elif isinstance(child, ast.stmt):
                parent.body.append(child)
    elif isinstance(parent, ast.BinOp):
        exprs = [c for c in children if isinstance(c, ast.expr)]
        ops = [c for c in children if isinstance(c, ast.operator)]
        if len(exprs) >= 2:
            parent.left = exprs[0]
            parent.right = exprs[1]
        if ops:
            parent.op = ops[0]
    elif isinstance(parent, ast.UnaryOp):
        for child in children:
            if isinstance(child, ast.expr):
                parent.operand = child
            elif isinstance(child, ast.unaryop):
                parent.op = child
    elif isinstance(parent, ast.BoolOp):
        parent.values = [c for c in children if isinstance(c, ast.expr)]
        for child in children:
            if isinstance(child, ast.boolop):
                parent.op = child
    elif isinstance(parent, ast.Compare):
        exprs = [c for c in children if isinstance(c, ast.expr)]
        ops = [c for c in children if isinstance(c, ast.cmpop)]
        if exprs:
            parent.left = exprs[0]
            parent.comparators = exprs[1:]
        parent.ops = ops
    elif isinstance(parent, ast.Call):
        for child in children:
            if isinstance(child, ast.expr) and parent.func == ast.Name(id="f", ctx=ast.Load()):
                parent.func = child
            elif isinstance(child, ast.expr):
                parent.args.append(child)
            elif isinstance(child, ast.keyword):
                parent.keywords.append(child)
    elif isinstance(parent, ast.List | ast.Tuple | ast.Set):
        parent.elts = [c for c in children if isinstance(c, ast.expr)]
    elif isinstance(parent, ast.Dict):
        exprs = [c for c in children if isinstance(c, ast.expr)]
        # Alternate between keys and values
        parent.keys = exprs[::2]
        parent.values = exprs[1::2]
    elif isinstance(parent, ast.Subscript):
        for child in children:
            if isinstance(child, ast.expr):
                if hasattr(parent, "slice") and parent.slice == ast.Constant(value=0):
                    parent.slice = child
                else:
                    parent.value = child
    elif isinstance(parent, ast.Attribute):
        for child in children:
            if isinstance(child, ast.expr):
                parent.value = child
    elif isinstance(parent, ast.arguments):
        parent.args = [c for c in children if isinstance(c, ast.arg)]
    elif isinstance(parent, ast.Lambda):
        for child in children:
            if isinstance(child, ast.arguments):
                parent.args = child
            elif isinstance(child, ast.expr):
                parent.body = child
    elif isinstance(parent, ast.IfExp):
        exprs = [c for c in children if isinstance(c, ast.expr)]
        if len(exprs) >= 3:
            parent.test = exprs[0]
            parent.body = exprs[1]
            parent.orelse = exprs[2]
    elif isinstance(parent, ast.Expr):
        for child in children:
            if isinstance(child, ast.expr):
                parent.value = child


def parse_python_to_tensors(
    code: str,
    node_vocab: PythonNodeVocab | None = None,
    value_vocab: PythonValueVocab | None = None,
    max_nodes: int = 256,
    max_children: int = 16,
    max_value_len: int = 32,
) -> TreeTensors:
    """Parse Python code to padded tensor representation.

    Args:
        code: Python source code string
        node_vocab: Vocabulary for node types (default: create new)
        value_vocab: Vocabulary for node values (default: create new)
        max_nodes: Maximum number of nodes
        max_children: Maximum children per node
        max_value_len: Maximum value encoding length

    Returns:
        TreeTensors representing the parsed AST
    """
    if node_vocab is None:
        node_vocab = PythonNodeVocab()
    if value_vocab is None:
        value_vocab = PythonValueVocab()

    tree = ast.parse(code)
    return ast_to_tensors(tree, node_vocab, value_vocab, max_nodes, max_children, max_value_len)


def tensors_to_code(
    tensors: TreeTensors,
    node_vocab: PythonNodeVocab | None = None,
    value_vocab: PythonValueVocab | None = None,
) -> str:
    """Convert tensor representation back to Python code.

    Args:
        tensors: The tensor representation
        node_vocab: Vocabulary for node types
        value_vocab: Vocabulary for node values

    Returns:
        Python source code string
    """
    if node_vocab is None:
        node_vocab = PythonNodeVocab()
    if value_vocab is None:
        value_vocab = PythonValueVocab()

    tree = tensors_to_ast(tensors, node_vocab, value_vocab)
    return ast.unparse(tree)


def count_nodes(node: ast.AST) -> int:
    """Count the number of nodes in an AST (size function sigma)."""
    count = 1
    for child in ast.iter_child_nodes(node):
        count += count_nodes(child)
    return count


def get_subtree_size(tensors: TreeTensors, node_idx: int) -> int:
    """Get the size (number of nodes) of a subtree.

    Args:
        tensors: The tree tensor representation
        node_idx: Index of the root of the subtree

    Returns:
        Number of nodes in the subtree
    """
    if node_idx < 0 or tensors.node_mask[node_idx] == 0:
        return 0

    size = 1
    num_ch = int(tensors.num_children[node_idx])
    for i in range(num_ch):
        child_idx = int(tensors.child_indices[node_idx, i])
        if child_idx >= 0:
            size += get_subtree_size(tensors, child_idx)

    return size
