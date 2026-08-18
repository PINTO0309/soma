#!/usr/bin/env python3
"""Rewrite PersonViT QKV rank-5 transposes into WebGPU-friendly rank-4 ops.

The transformation is performed directly on a TFLite FlatBuffer:

  RESHAPE [1,T,3,H,D] -> TRANSPOSE [3,1,H,T,D]
    -> 3 x (GATHER [1,1,H,T,D] -> RESHAPE [1,H,T,D])

becomes:

  RESHAPE [T,3,H,D] -> TRANSPOSE [3,H,T,D]
    -> 3 x GATHER [1,H,T,D]

This transformation is valid only for a static batch size of one. Gather
indices remain one-dimensional ([0], [1], [2]), as required by LiteRT's GPU
delegate.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import flatbuffers
import numpy as np

try:
    from ai_edge_litert import schema_py_generated as schema
except ImportError as error:
    raise SystemExit(
        "ai-edge-litert is required: python -m pip install ai-edge-litert"
    ) from error


RESHAPE = schema.BuiltinOperator.RESHAPE
TRANSPOSE = schema.BuiltinOperator.TRANSPOSE
GATHER = schema.BuiltinOperator.GATHER


@dataclass(frozen=True)
class QkvMatch:
    start_op_index: int
    reshape_op_index: int
    transpose_op_index: int
    gather_op_indices: tuple[int, int, int]
    trailing_reshape_op_indices: tuple[int, int, int]
    tokens: int
    heads: int
    head_dim: int


def _shape(tensor: schema.TensorT) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _set_shape(tensor: schema.TensorT, shape: Iterable[int]) -> None:
    values = np.asarray(tuple(shape), dtype=np.int32)
    tensor.shape = values
    if tensor.shapeSignature is not None:
        tensor.shapeSignature = values.copy()


def _operator_code(model: schema.ModelT, operator: schema.OperatorT) -> int:
    return int(model.operatorCodes[int(operator.opcodeIndex)].builtinCode)


def _read_int32_buffer(model: schema.ModelT, tensor: schema.TensorT) -> tuple[int, ...]:
    buffer = model.buffers[int(tensor.buffer)]
    if buffer.data is None:
        raise ValueError(f"constant tensor {tensor.name!r} has no inline buffer")
    data = bytes(buffer.data)
    if len(data) % 4 != 0:
        raise ValueError(f"constant tensor {tensor.name!r} is not int32-aligned")
    return struct.unpack(f"<{len(data) // 4}i", data)


def _write_int32_buffer(
    model: schema.ModelT, tensor: schema.TensorT, values: Iterable[int]
) -> None:
    value_tuple = tuple(int(value) for value in values)
    data = struct.pack(f"<{len(value_tuple)}i", *value_tuple)
    model.buffers[int(tensor.buffer)].data = np.frombuffer(data, dtype=np.uint8).copy()
    _set_shape(tensor, (len(value_tuple),))


def _replace_tensor_references(
    subgraph: schema.SubGraphT, replacements: dict[int, int]
) -> None:
    def replace(values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        return np.asarray(
            [replacements.get(int(value), int(value)) for value in values],
            dtype=np.int32,
        )

    subgraph.inputs = replace(subgraph.inputs)
    subgraph.outputs = replace(subgraph.outputs)
    for operator in subgraph.operators:
        operator.inputs = replace(operator.inputs)
        operator.outputs = replace(operator.outputs)
        operator.intermediates = replace(operator.intermediates)


def _find_qkv_matches(model: schema.ModelT, subgraph: schema.SubGraphT) -> list[QkvMatch]:
    operators = subgraph.operators
    tensors = subgraph.tensors
    matches: list[QkvMatch] = []

    expected_codes = (
        RESHAPE,
        TRANSPOSE,
        GATHER,
        RESHAPE,
        GATHER,
        RESHAPE,
        GATHER,
        RESHAPE,
    )

    for start in range(len(operators) - len(expected_codes) + 1):
        window = operators[start : start + len(expected_codes)]
        if tuple(_operator_code(model, op) for op in window) != expected_codes:
            continue

        reshape, transpose = window[0], window[1]
        gathers = (window[2], window[4], window[6])
        trailing_reshapes = (window[3], window[5], window[7])

        reshape_output = tensors[int(reshape.outputs[0])]
        transpose_output = tensors[int(transpose.outputs[0])]
        if len(_shape(reshape_output)) != 5:
            continue

        batch, tokens, qkv, heads, head_dim = _shape(reshape_output)
        if batch != 1 or qkv != 3:
            continue
        # Validate via the reshape TARGET constant instead of the input's
        # recorded shape: dynamic-batch exports record stale batch-1 input
        # shapes, and the qkv projection output arrives in several layouts
        # ([1, T, 3HD], [T, 3HD], [B*T, 3HD]). Accept [1|-1, T, 3, H, D];
        # the rewrite specializes a dynamic batch dim to 1 (the script's
        # stated batch-size-one premise).
        target = _read_int32_buffer(model, tensors[int(reshape.inputs[1])])
        if (
            len(target) != 5
            or target[0] not in (1, -1)
            or tuple(target[1:]) != (tokens, qkv, heads, head_dim)
        ):
            continue

        transpose_perm = _read_int32_buffer(
            model, tensors[int(transpose.inputs[1])]
        )
        if transpose_perm != (2, 0, 3, 1, 4):
            continue
        if _shape(transpose_output) != (3, 1, heads, tokens, head_dim):
            continue
        if int(transpose.inputs[0]) != int(reshape.outputs[0]):
            continue

        valid_branches = True
        for branch, (gather, branch_reshape) in enumerate(
            zip(gathers, trailing_reshapes)
        ):
            gather_options = gather.builtinOptions
            if (
                int(gather.inputs[0]) != int(transpose.outputs[0])
                or int(getattr(gather_options, "axis", -1)) != 0
                or int(getattr(gather_options, "batchDims", -1)) != 0
                or _read_int32_buffer(model, tensors[int(gather.inputs[1])])
                != (branch,)
                or _shape(tensors[int(gather.outputs[0])])
                != (1, 1, heads, tokens, head_dim)
                or int(branch_reshape.inputs[0]) != int(gather.outputs[0])
                or _shape(tensors[int(branch_reshape.outputs[0])])
                != (1, heads, tokens, head_dim)
            ):
                valid_branches = False
                break
        if not valid_branches:
            continue

        matches.append(
            QkvMatch(
                start_op_index=start,
                reshape_op_index=start,
                transpose_op_index=start + 1,
                gather_op_indices=(start + 2, start + 4, start + 6),
                trailing_reshape_op_indices=(start + 3, start + 5, start + 7),
                tokens=tokens,
                heads=heads,
                head_dim=head_dim,
            )
        )

    return matches


def optimize_model(model: schema.ModelT, expected_blocks: int) -> int:
    if len(model.subgraphs) != 1:
        raise ValueError(f"expected one subgraph, found {len(model.subgraphs)}")

    subgraph = model.subgraphs[0]
    matches = _find_qkv_matches(model, subgraph)
    if len(matches) != expected_blocks:
        raise ValueError(
            f"expected {expected_blocks} QKV patterns, found {len(matches)}; "
            "input model was not modified"
        )

    replacements: dict[int, int] = {}
    remove_operator_indices: set[int] = set()

    for match in matches:
        reshape = subgraph.operators[match.reshape_op_index]
        transpose = subgraph.operators[match.transpose_op_index]
        reshape_output = subgraph.tensors[int(reshape.outputs[0])]
        transpose_output = subgraph.tensors[int(transpose.outputs[0])]

        rank4_qkv_shape = (match.tokens, 3, match.heads, match.head_dim)
        _write_int32_buffer(
            model, subgraph.tensors[int(reshape.inputs[1])], rank4_qkv_shape
        )
        reshape.builtinOptions.newShape = np.asarray(rank4_qkv_shape, dtype=np.int32)
        _set_shape(reshape_output, rank4_qkv_shape)

        rank4_perm = (1, 2, 0, 3)
        _write_int32_buffer(
            model, subgraph.tensors[int(transpose.inputs[1])], rank4_perm
        )
        _set_shape(
            transpose_output,
            (3, match.heads, match.tokens, match.head_dim),
        )

        branch_shape = (1, match.heads, match.tokens, match.head_dim)
        for gather_index, trailing_reshape_index in zip(
            match.gather_op_indices, match.trailing_reshape_op_indices
        ):
            gather = subgraph.operators[gather_index]
            trailing_reshape = subgraph.operators[trailing_reshape_index]
            gather_output_index = int(gather.outputs[0])
            old_branch_output_index = int(trailing_reshape.outputs[0])
            _set_shape(subgraph.tensors[gather_output_index], branch_shape)
            replacements[old_branch_output_index] = gather_output_index
            remove_operator_indices.add(trailing_reshape_index)

    _replace_tensor_references(subgraph, replacements)
    subgraph.operators = [
        operator
        for index, operator in enumerate(subgraph.operators)
        if index not in remove_operator_indices
    ]
    return len(matches)


def _serialize_model(model: schema.ModelT) -> bytes:
    builder = flatbuffers.Builder(0)
    model_offset = model.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    return bytes(builder.Output())


def _write_atomic(output_path: Path, data: bytes, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "wb") as output_file:
            output_file.write(data)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="rank-5 PersonViT .tflite")
    parser.add_argument("output", type=Path, help="optimized output .tflite")
    parser.add_argument(
        "--expected-blocks",
        type=int,
        default=12,
        help="abort unless this many QKV patterns are found (default: 12)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing output file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise SystemExit("input and output must differ; validate before replacing input")
    if not input_path.is_file():
        raise SystemExit(f"input does not exist: {input_path}")
    if args.expected_blocks <= 0:
        raise SystemExit("--expected-blocks must be positive")

    model_bytes = input_path.read_bytes()
    packed_model = schema.Model.GetRootAsModel(model_bytes, 0)
    model = schema.ModelT.InitFromObj(packed_model)
    optimized_blocks = optimize_model(model, args.expected_blocks)
    optimized_bytes = _serialize_model(model)
    _write_atomic(output_path, optimized_bytes, args.overwrite)

    print(
        f"optimized {optimized_blocks} QKV blocks: "
        f"{input_path} ({len(model_bytes)} bytes) -> "
        f"{output_path} ({len(optimized_bytes)} bytes)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
