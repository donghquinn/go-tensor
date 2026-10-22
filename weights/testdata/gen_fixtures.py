"""Regenerates the weight-file fixtures used by go-tensor's tests.

Every file is written by the reference library for its format, and the expected
values are read back with those same libraries:

    uv venv --python 3.12 .venv
    uv pip install --python .venv torch safetensors onnx gguf numpy
    .venv/bin/python weights/testdata/gen_fixtures.py
"""

import json
import os

import gguf
import numpy as np
import onnx
import torch
import torch.nn as nn
from gguf import GGMLQuantizationType as Q
from onnx import TensorProto, helper, numpy_helper
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))


def path(name):
    return os.path.join(HERE, name)


def entry(a):
    a = np.asarray(a)
    return {"shape": list(a.shape), "values": a.astype(np.float64).ravel().tolist()}


def write_gguf(name, tensors, metadata=None):
    w = gguf.GGUFWriter(path(name), "go-tensor-test")
    for k, v in (metadata or {}).items():
        if isinstance(v, str):
            w.add_string(k, v)
        elif isinstance(v, list):
            w.add_array(k, v)
        else:
            w.add_uint32(k, v)
    for k, (data, raw_dtype) in tensors.items():
        if raw_dtype is None:
            w.add_tensor(k, data)
        elif data.dtype == np.uint8:  # already-quantized blocks
            w.add_tensor(k, data, raw_dtype=raw_dtype)
        else:
            w.add_tensor(k, gguf.quants.quantize(data, raw_dtype), raw_dtype=raw_dtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


# ---------------------------------------------------------------------------
# 1. A small CNN using go-tensor's layer layout, exported to every format, plus
#    the PyTorch output for a fixed input so tests can compare forward passes.
# ---------------------------------------------------------------------------
torch.manual_seed(0)
model = nn.Sequential(
    nn.Conv2d(1, 2, 3, 1, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(2 * 4 * 4, 3),
).eval()
x = torch.randn(2, 1, 8, 8)
with torch.no_grad():
    y = model(x)
sd = model.state_dict()

torch.save(sd, path("cnn.pth"))
save_file(sd, path("cnn.safetensors"))
np.savez(path("cnn.npz"), **{k: v.numpy() for k, v in sd.items()})
torch.onnx.export(model, (x,), path("cnn.onnx"), dynamo=False, input_names=["x"], output_names=["y"])
onnx.save_model(
    onnx.load(path("cnn.onnx")),
    path("cnn_external.onnx"),
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="cnn_external.onnx.data",
    size_threshold=0,
)
write_gguf("cnn.gguf", {k: (v.numpy(), None) for k, v in sd.items()})
with open(path("cnn_io.json"), "w") as f:
    json.dump({"input": entry(x), "output": entry(y)}, f)

# ---------------------------------------------------------------------------
# 2. A "zoo" per format covering dtypes, layouts and container structure.
# ---------------------------------------------------------------------------
expected = {}

zoo = {
    "f32": torch.randn(2, 3),
    "f64": torch.randn(4, dtype=torch.float64),
    "f16": torch.randn(3, 2).half(),
    "bf16": torch.randn(5).bfloat16(),
    "i64": torch.arange(-3, 3),
    "i32": torch.tensor([[1, -2], [3, 4]], dtype=torch.int32),
    "i8": torch.tensor([-128, 0, 127], dtype=torch.int8),
    "u8": torch.tensor([0, 7, 255], dtype=torch.uint8),
    "bool": torch.tensor([True, False, True]),
    "scalar": torch.tensor(3.5),
}

# PyTorch: nested checkpoint, Parameters, non-contiguous views sharing storage.
base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
ckpt = {
    "model": {
        **zoo,
        "base": base,
        "transposed": base.t(),
        "view": base[1:, 2:],
        "param": nn.Parameter(torch.randn(2, 2)),
    },
    "epoch": 7,
    "lr": 0.01,
    "history": [torch.ones(2), torch.zeros(1)],
    "name": "zoo",
}
torch.save(ckpt, path("zoo.pth"))
pth = {f"model.{k}": entry(v.detach().double()) for k, v in ckpt["model"].items()}
pth.update({f"history.{i}": entry(v) for i, v in enumerate(ckpt["history"])})
expected["pth"] = pth

# TorchScript archive: parameters live in object state rather than a dict.
torch.jit.save(torch.jit.script(nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))), path("scripted.pt"))
scripted = torch.jit.load(path("scripted.pt"))
expected["torchscript"] = {k: entry(v.detach()) for k, v in scripted.state_dict().items()}

# safetensors
st = {k: v.contiguous() for k, v in zoo.items()}
save_file(st, path("zoo.safetensors"), metadata={"format": "pt", "note": "zoo"})
expected["safetensors"] = {k: entry(v.double()) for k, v in st.items()}

# NumPy: many dtypes, Fortran order, big-endian, compressed archive, bare .npy.
rng = np.random.default_rng(0)
npz = {
    "f32": rng.standard_normal((2, 3)).astype(np.float32),
    "f64": rng.standard_normal(4),
    "f16": rng.standard_normal((3, 2)).astype(np.float16),
    "i64": np.arange(-3, 3),
    "u16": np.array([1, 65535], dtype=np.uint16),
    "bool": np.array([True, False]),
    "fortran": np.asfortranarray(np.arange(24, dtype=np.float32).reshape(2, 3, 4)),
    "big_endian": np.arange(6, dtype=">f4").reshape(2, 3),
    "scalar": np.array(2.5),
}
np.savez_compressed(path("zoo.npz"), **npz)
np.save(path("fortran.npy"), npz["fortran"])
expected["npz"] = {k: entry(v) for k, v in npz.items()}
expected["npy"] = {"fortran": entry(npz["fortran"])}

# ONNX: raw_data and typed fields (float_data, int32_data, ...) for each dtype.
onnx_vals = {
    "raw_f32": (TensorProto.FLOAT, rng.standard_normal((2, 3)).astype(np.float32), True),
    "float_data": (TensorProto.FLOAT, rng.standard_normal((2, 2)).astype(np.float32), False),
    "double_data": (TensorProto.DOUBLE, rng.standard_normal(3), False),
    "raw_f64": (TensorProto.DOUBLE, rng.standard_normal(3), True),
    "int64_data": (TensorProto.INT64, np.array([-5, 0, 9]), False),
    "int32_data": (TensorProto.INT32, np.array([[-1, 2], [3, -4]], dtype=np.int32), False),
    "raw_f16": (TensorProto.FLOAT16, rng.standard_normal(4).astype(np.float16), True),
    "f16_bits": (TensorProto.FLOAT16, rng.standard_normal(4).astype(np.float16), False),
    "raw_u8": (TensorProto.UINT8, np.array([0, 128, 255], dtype=np.uint8), True),
    "bool_data": (TensorProto.BOOL, np.array([True, False, True]), False),
    "scalar": (TensorProto.FLOAT, np.array(1.25, dtype=np.float32), False),
}
inits = []
for name, (dt, arr, raw) in onnx_vals.items():
    if raw:
        inits.append(helper.make_tensor(name, dt, arr.shape, arr.tobytes(), raw=True))
    else:
        inits.append(numpy_helper.from_array(arr, name) if dt == TensorProto.FLOAT16 else
                     helper.make_tensor(name, dt, arr.shape, arr.ravel().tolist()))
zoo_model = helper.make_model(helper.make_graph([], "zoo", [], [], initializer=inits))
onnx.save_model(zoo_model, path("zoo.onnx"))
expected["onnx"] = {t.name: entry(numpy_helper.to_array(t)) for t in onnx.load(path("zoo.onnx")).graph.initializer}

# GGUF: float types and block-quantized types (values are the dequantized ones).
g = rng.standard_normal((4, 64)).astype(np.float32)
gguf_tensors = {
    "f32": (rng.standard_normal((2, 3)).astype(np.float32), None),
    "f16": (rng.standard_normal((3, 2)).astype(np.float16), None),
    "f64": (rng.standard_normal(4), None),
    "i32": (np.array([-7, 0, 7], dtype=np.int32), None),
    "bf16": (g, Q.BF16),
    "q8_0": (g, Q.Q8_0),
    "q4_0": (g, Q.Q4_0),
    "q4_1": (g, Q.Q4_1),
    "q5_0": (g, Q.Q5_0),
    "q5_1": (g, Q.Q5_1),
}


def random_kquant_blocks(qtype, rows, scale_fields):
    """Random K-quant blocks with sane float16 scales (gguf-py cannot quantize these)."""
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    blocks = rng.integers(0, 256, size=(rows, 256 // block, size), dtype=np.uint8)
    for offset in scale_fields:
        scales = rng.uniform(0.001, 0.05, size=(rows, 256 // block)).astype(np.float16)
        blocks[:, :, offset:offset + 2] = scales.view(np.uint8).reshape(rows, 256 // block, 2)
    return blocks.reshape(rows, -1)


# Block layouts: Q4_K/Q5_K start with f16 d and dmin; Q6_K ends with f16 d.
gguf_tensors["q4_k"] = (random_kquant_blocks(Q.Q4_K, 2, [0, 2]), Q.Q4_K)
gguf_tensors["q5_k"] = (random_kquant_blocks(Q.Q5_K, 2, [0, 2]), Q.Q5_K)
gguf_tensors["q6_k"] = (random_kquant_blocks(Q.Q6_K, 2, [208]), Q.Q6_K)
write_gguf("zoo.gguf", gguf_tensors, {"zoo.answer": 42, "zoo.name": "zoo", "zoo.list": [1, 2, 3]})
expected["gguf"] = {
    k: entry(a if q is None else gguf.quants.dequantize(a if a.dtype == np.uint8 else gguf.quants.quantize(a, q), q))
    for k, (a, q) in gguf_tensors.items()
}

with open(path("zoo_expected.json"), "w") as f:
    json.dump(expected, f, indent=1)

for name in sorted(os.listdir(HERE)):
    print(f"{os.path.getsize(path(name)):>8}  {name}")
