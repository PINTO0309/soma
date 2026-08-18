#!/usr/bin/env python3
"""Make the rank-4 PersonViT QKV rewrite dynamic-batch-correct.

The rank-4 optimization (optimize_personvit_tflite.py) specializes the QKV
unpack to batch 1: RESHAPE with a FIXED [129, 3, 6, 64] target on a dynamic
[-1, 1152] input (e.g. Netron identifier 119) — for batch N >= 2 the element
counts mismatch and the graph is inconsistent.

This tool rewrites each block's

  RESHAPE [B*129,1152]->[129,3,6,64] -> TRANSPOSE(1,2,0,3) -> 3x GATHER

into the batch-preserving, still rank-<=4 form

  3x ( SLICE [B*129, b*384:(b+1)*384] -> RESHAPE [-1,129,6,64]
       -> TRANSPOSE(0,2,1,3) -> [B,6,129,64] )

The qkv projection packs the 1152 axis as (3, 6, 64) with q|k|v contiguous
384-blocks, so slicing that axis is exactly the axis-0 gather of the rank-5
layout. Outputs feed the original downstream tensors, whose shapes stay
dynamic ([-1, 6, 129, 64]).

Requires: python -m pip install ai-edge-litert flatbuffers numpy
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import tempfile
from pathlib import Path

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
SLICE = schema.BuiltinOperator.SLICE
FLOAT32 = schema.TensorType.FLOAT32
INT32 = schema.TensorType.INT32


def _read_i32(model: schema.ModelT, sg: schema.SubGraphT, ti: int) -> tuple[int, ...] | None:
    buf = model.buffers[int(sg.tensors[ti].buffer)]
    if buf.data is None:
        return None
    d = bytes(buf.data)
    if len(d) % 4:
        return None
    return struct.unpack(f"<{len(d) // 4}i", d)


def _op_code(model: schema.ModelT, op: schema.OperatorT) -> int:
    return int(model.operatorCodes[int(op.opcodeIndex)].builtinCode)


def _opcode_index(model: schema.ModelT, builtin: int) -> int:
    for i, oc in enumerate(model.operatorCodes):
        if int(oc.builtinCode) == builtin:
            return i
    oc = schema.OperatorCodeT()
    oc.deprecatedBuiltinCode = builtin if builtin <= 127 else 127
    oc.builtinCode = builtin
    oc.version = 1
    model.operatorCodes.append(oc)
    return len(model.operatorCodes) - 1


def _add_const(model: schema.ModelT, sg: schema.SubGraphT, name: str,
               values: list[int]) -> int:
    buf = schema.BufferT()
    buf.data = np.frombuffer(struct.pack(f"<{len(values)}i", *values),
                             dtype=np.uint8).copy()
    model.buffers.append(buf)
    t = schema.TensorT()
    t.shape = np.asarray([len(values)], dtype=np.int32)
    t.type = INT32
    t.buffer = len(model.buffers) - 1
    t.name = name
    sg.tensors.append(t)
    return len(sg.tensors) - 1


def _add_activation(sg: schema.SubGraphT, name: str, shape: list[int],
                    signature: list[int], dtype: int = FLOAT32) -> int:
    t = schema.TensorT()
    t.shape = np.asarray(shape, dtype=np.int32)
    t.shapeSignature = np.asarray(signature, dtype=np.int32)
    t.type = dtype
    t.buffer = 0
    t.name = name
    sg.tensors.append(t)
    return len(sg.tensors) - 1


def _make_op(opcode_index: int, inputs: list[int], outputs: list[int],
             options_type: int = 0, options: object | None = None) -> schema.OperatorT:
    op = schema.OperatorT()
    op.opcodeIndex = opcode_index
    op.inputs = np.asarray(inputs, dtype=np.int32)
    op.outputs = np.asarray(outputs, dtype=np.int32)
    op.builtinOptionsType = options_type
    op.builtinOptions = options
    return op


def fix(model: schema.ModelT) -> int:
    if len(model.subgraphs) != 1:
        raise ValueError(f"expected one subgraph, found {len(model.subgraphs)}")
    sg = model.subgraphs[0]
    ops = sg.operators

    # locate the batch-1 rank-4 blocks: RESHAPE(target [T,3,H,D]) ->
    # TRANSPOSE(perm (1,2,0,3)) -> 3x GATHER(indices 0/1/2)
    blocks: list[tuple[int, int, int, int, tuple[int, int, int, int]]] = []
    for i in range(len(ops) - 4):
        if _op_code(model, ops[i]) != RESHAPE:
            continue
        target = _read_i32(model, sg, int(ops[i].inputs[1]))
        if target is None or len(target) != 4 or target[1] != 3:
            continue
        tokens, _three, heads, head_dim = target
        if _op_code(model, ops[i + 1]) != TRANSPOSE:
            continue
        if _read_i32(model, sg, int(ops[i + 1].inputs[1])) != (1, 2, 0, 3):
            continue
        gathers = ops[i + 2 : i + 5]
        if any(_op_code(model, g) != GATHER for g in gathers):
            continue
        if any(_read_i32(model, sg, int(g.inputs[1])) != (b,)
               for b, g in enumerate(gathers)):
            continue
        x = int(ops[i].inputs[0])
        outs = tuple(int(g.outputs[0]) for g in gathers)
        blocks.append((i, x, tokens, heads * head_dim,
                       (heads, head_dim, outs[0], outs[1])))
        # stash v output too
        blocks[-1] = (i, x, tokens, heads * head_dim, (heads, head_dim, *outs))

    if not blocks:
        raise ValueError("no batch-1 rank-4 QKV blocks found; input model was not modified")

    slice_op = _opcode_index(model, SLICE)
    reshape_op = _opcode_index(model, RESHAPE)
    transpose_op = _opcode_index(model, TRANSPOSE)

    # shared constants
    tokens = blocks[0][2]
    span = blocks[0][3]
    heads, head_dim = blocks[0][4][0], blocks[0][4][1]
    size_ti = _add_const(model, sg, "qkv_slice_size", [-1, span])
    begin_tis = [_add_const(model, sg, f"qkv_slice_begin_{b}", [0, b * span])
                 for b in range(3)]
    rs_target_ti = _add_const(model, sg, "qkv_reshape_shape_nbatch",
                              [-1, tokens, heads, head_dim])
    perm_ti = _add_const(model, sg, "qkv_transpose_perm_nbatch", [0, 2, 1, 3])

    for block_index, (i, x, tk, sp, meta) in enumerate(reversed(blocks)):
        hd, dd, out_q, out_k, out_v = meta
        new_ops: list[schema.OperatorT] = []
        # intermediates inherit the activation dtype of the sliced source
        # (float16 exports carry FLOAT16 activations)
        act_dtype = int(sg.tensors[x].type)
        for b, out_ti in enumerate((out_q, out_k, out_v)):
            base = sg.tensors[out_ti].name
            base = (bytes(base).decode() if isinstance(base, (bytes, bytearray)) else str(base))
            slice_ti = _add_activation(sg, f"{base}_qkv_slice", [tk, sp], [-1, sp],
                                       act_dtype)
            rs_ti = _add_activation(sg, f"{base}_qkv_bthd", [1, tk, hd, dd],
                                    [-1, tk, hd, dd], act_dtype)
            new_ops.append(_make_op(slice_op, [x, begin_tis[b], size_ti], [slice_ti],
                                    schema.BuiltinOptions.SliceOptions,
                                    schema.SliceOptionsT()))
            rs_opts = schema.ReshapeOptionsT()
            rs_opts.newShape = np.asarray([-1, tk, hd, dd], dtype=np.int32)
            new_ops.append(_make_op(reshape_op, [slice_ti, rs_target_ti],
                                    [rs_ti], schema.BuiltinOptions.ReshapeOptions,
                                    rs_opts))
            new_ops.append(_make_op(transpose_op, [rs_ti, perm_ti], [out_ti],
                                    schema.BuiltinOptions.TransposeOptions,
                                    schema.TransposeOptionsT()))
            # downstream tensor becomes dynamic-batch again
            t = sg.tensors[out_ti]
            t.shape = np.asarray([1, hd, tk, dd], dtype=np.int32)
            t.shapeSignature = np.asarray([-1, hd, tk, dd], dtype=np.int32)
        sg.operators = sg.operators[:i] + new_ops + sg.operators[i + 5:]

    return len(blocks)


def _serialize(model: schema.ModelT) -> bytes:
    builder = flatbuffers.Builder(0)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())


def _write_atomic(output_path: Path, data: bytes, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp",
                               dir=output_path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, output_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("input and output must differ; validate before replacing input")

    model_bytes = args.input.read_bytes()
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(model_bytes, 0))
    n = fix(model)
    out_bytes = _serialize(model)
    _write_atomic(args.output, out_bytes, args.overwrite)
    print(f"rewrote {n} QKV blocks to dynamic-batch SLICE form: "
          f"{args.input} ({len(model_bytes)} bytes) -> {args.output} ({len(out_bytes)} bytes)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
