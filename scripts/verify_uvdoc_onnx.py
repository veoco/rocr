#!/usr/bin/env python3
"""Generate numeric reference fixtures for the UVDoc (document unwarping) model.

The official `PaddlePaddle/UVDoc_onnx` export is the numeric oracle. Its graph
is self-contained end-to-end: it resizes the input to 712×488, runs the UVDoc
network to predict a 2-channel sampling grid, resizes the grid to the original
input size, and applies `grid_sample` (bilinear, zeros padding, align_corners).
The output is a rectified image with the same size as the input (pixel range
0..255).

In addition to the end-to-end input/output, intermediate tensors (the network
input after resize, the predicted grid, and the grid_sample inputs/output) are
extracted so rocr can be verified per-module.

Outputs to dev-models/reference/:
  paddle_unwarp_{name}_input.npy       original image [1,3,H,W] float32, 0..255
  paddle_unwarp_{name}_output.npy      rectified image [1,3,H,W] float32, 0..255
  paddle_unwarp_{name}_network_input.npy  resized [1,3,712,488] float32, 0..255
  paddle_unwarp_{name}_grid.npy           predicted grid [1,2,45,31] float32
  uvdoc_gs_input.npy / _grid.npy / _output.npy  grid_sample IO (doc image)
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

import onnx

ROOT = Path(__file__).resolve().parent.parent
ONNX = ROOT / "dev-models" / "UVDoc_onnx" / "inference.onnx"
OUT = ROOT / "dev-models" / "reference"
OUT.mkdir(parents=True, exist_ok=True)

INTERMEDIATE = {
    "p2o.pd_op.bilinear_interp.0.0": "network_input",  # input resized to 712×488
    "p2o.pd_op.add.41.0": "grid",  # predicted grid
    "auto.cast.39": "gs_input",  # grid_sample data (original image)
    "auto.cast.40": "gs_grid",  # grid_sample grid [1,H,W,2]
    "GridSample.1": "gs_output",  # grid_sample output
}


def main():
    if not ONNX.exists():
        print(f"skip: {ONNX} not present")
        return 0

    # Build a session that also exposes the intermediate tensors.
    m = onnx.load(ONNX)
    for name, _ in INTERMEDIATE.items():
        m.graph.output.append(
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        )
    tmp = OUT / "_uvdoc_intermediate.onnx"
    onnx.save(m, tmp)
    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])

    names = list(INTERMEDIATE.keys())
    cases = {
        "doc": ROOT / "assets" / "doc.png",
        "line_long": ROOT / "assets" / "line_long.png",
    }
    for name, path in cases.items():
        if not path.exists():
            print(f"skip: {path} not present")
            continue
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype(np.float32)  # H,W,3 in 0..255
        x = arr.transpose(2, 0, 1)[None]
        outs = sess.run(names, {"image": x})
        data = dict(zip(names, outs))
        np.save(OUT / f"paddle_unwarp_{name}_input.npy", x)
        np.save(OUT / f"paddle_unwarp_{name}_output.npy", data["GridSample.1"])
        np.save(OUT / f"paddle_unwarp_{name}_network_input.npy", data["p2o.pd_op.bilinear_interp.0.0"])
        np.save(OUT / f"paddle_unwarp_{name}_grid.npy", data["p2o.pd_op.add.41.0"])
        if name == "doc":
            np.save(OUT / "uvdoc_gs_input.npy", data["auto.cast.39"])
            np.save(OUT / "uvdoc_gs_grid.npy", data["auto.cast.40"])
            np.save(OUT / "uvdoc_gs_output.npy", data["GridSample.1"])
        print(f"{name}: {img.size} -> output {data['GridSample.1'].shape} "
              f"grid {data['p2o.pd_op.add.41.0'].shape}")

    tmp.unlink(missing_ok=True)
    print(f"fixtures written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
