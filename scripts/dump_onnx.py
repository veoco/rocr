#!/usr/bin/env python
"""Dump the operator graph of a PP-OCRv6 ONNX model with conv attributes.

Usage: python scripts/dump_onnx.py <path/to/inference.onnx> [start] [count]
"""
import sys

import onnx
import onnx.helper


def main(path, start, count):
    m = onnx.load(path)
    print(f"# initializers={len(m.graph.initializer)} nodes={len(m.graph.node)}")
    print("# inputs:", [
        (i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
        for i in m.graph.input
    ])
    print("# outputs:", [o.name for o in m.graph.output])
    # initializer name -> shape
    inits = {}
    for it in m.graph.initializer:
        dims = list(it.dims)
        inits[it.name] = dims
    for i, n in enumerate(m.graph.node):
        if i < start or i >= start + count:
            continue
        attrs = {
            a.name: onnx.helper.get_attribute_value(a)
            for a in n.attribute
        }
        keep = {k: v for k, v in attrs.items()
                if k in ("strides", "pads", "kernel_shape", "group", "dilations",
                         "axis", "keepdims", "perm", "transpose_b")}
        outs = ",".join(n.output)
        print(f"{i:4d} {n.op_type:18s} {keep}  {n.input[0] if n.input else '':42s} -> {outs}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dev-models/PP-OCRv6_small_rec_onnx/inference.onnx"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    main(path, start, count)
