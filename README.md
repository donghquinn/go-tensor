# go-tensor

A PyTorch-style deep learning library for Go: n-dimensional tensors, reverse-mode
automatic differentiation, neural-network layers and optimizers — in pure Go, no cgo.

```go
model := nn.NewSequential(
	nn.NewLinear(2, 8),
	nn.NewTanh(),
	nn.NewLinear(8, 2),
)
opt := optim.NewAdam(model.Parameters(), optim.AdamConfig{LR: 0.05})

for epoch := 0; epoch < 500; epoch++ {
	loss := nn.CrossEntropyLoss(model.Forward(x), y)
	opt.ZeroGrad()
	loss.Backward()
	opt.Step()
}
```

## Examples

```sh
go run ./examples/xor                # 2-8-2 MLP learns XOR
go run ./examples/mnist -model cnn   # 2-layer CNN on MNIST
go run ./examples/mnist -model mlp   # 784-128-10 MLP on MNIST
```

The MNIST example downloads the dataset into `.cache/mnist` (checksum-verified),
trains for 3 epochs, then saves the model and reloads it. On an Apple M5:

| Model | Test accuracy (3 epochs) | Time per epoch |
|-------|--------------------------|----------------|
| CNN (conv 16 → conv 32 → fc 128, dropout) | 98.78% | ~26 s |
| MLP (784 → 128 → 10) | 96.75% | ~1.2 s |

## Loading pretrained weights

`weights.Load` reads PyTorch, safetensors, ONNX, NumPy and GGUF files into a
`state_dict`-style map, and `nn.LoadStateDict` copies it into a model. Parameter
names follow PyTorch (`0.weight`, `3.bias`, …), so a model saved in Python loads
straight into the same architecture built in Go:

```python
# Python
model = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                      nn.Flatten(), nn.Linear(16 * 14 * 14, 10))
torch.save(model.state_dict(), "model.pth")   # or save_file(...), torch.onnx.export(...)
```

```go
// Go
model := nn.NewSequential(nn.NewConv2d(1, 16, 3, 1, 1), nn.NewReLU(), nn.NewMaxPool2d(2, 0),
	nn.NewFlatten(), nn.NewLinear(16*14*14, 10))
sd, err := weights.Load("model.pth") // format detected from the file contents
if err != nil { ... }
if err := nn.LoadStateDict(model, sd, true); err != nil { ... } // strict, like PyTorch
```

| Format | Package | Read | Write | Notes |
|--------|---------|------|-------|-------|
| PyTorch `.pth` / `.pt` / `.bin` | `weights/pth` | ✓ | | `torch.save` zip format (PyTorch ≥ 1.6), nested checkpoints, strided views, TorchScript archives |
| safetensors | `weights/safetensors` | ✓ | ✓ | All integer/float dtypes incl. F16/BF16; metadata |
| ONNX | `weights/onnx` | ✓ | | Graph initializers, raw and typed fields, external data files |
| NumPy `.npy` / `.npz` | `weights/numpy` | ✓ | ✓ | Fortran order, big-endian, compressed archives |
| GGUF | `weights/gguf` | ✓ | | F32/F16/BF16/F64/ints, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K; metadata |

**Safe by construction.** `torch.load` without `weights_only=True` can run arbitrary
code from a file. The Go pickle reader never imports or calls anything: it
rebuilds tensors through a fixed allowlist and turns every other object into
inert data. All parsers bound allocations by the real file size and are fuzzed.

**Verified against the reference libraries.** `weights/testdata/gen_fixtures.py`
writes every fixture with torch, safetensors, onnx, numpy and gguf-py and records
their values. Every dtype and quantization type loads bit-exactly, and a PyTorch
CNN loaded from each of the six files reproduces PyTorch's outputs in Go.

## Packages

| Package  | PyTorch equivalent      | Contents |
|----------|-------------------------|----------|
| `tensor` | `torch`, `torch.autograd` | `Tensor`, creation (`New`, `Zeros`, `Randn`, …), broadcasting element-wise ops, reductions, `MatMul`, `Conv2d`, `MaxPool2d`, `Reshape`/`T`, `LogSoftmax`, `Backward`, `NoGrad`, `CustomOp` |
| `nn`     | `torch.nn`              | `Module`, `Linear`, `Conv2d`, `MaxPool2d`, `Dropout`, `Sequential`, `Flatten`, activations, `Train`/`Eval`, `StateDict`/`LoadStateDict`, `MSELoss`, `CrossEntropyLoss`, `BCELoss`, `Save`/`Load` |
| `optim`  | `torch.optim`           | `SGD` (momentum, weight decay), `Adam` |
| `data`   | `torch.utils.data`      | `Dataset`, `TensorDataset`, `DataLoader` (batching, shuffling) |
| `datasets/mnist` | `torchvision.datasets.MNIST` | Download, verify and load MNIST |
| `weights` | `torch.load`, `safetensors`, `onnx`, `gguf` | `Load` / `Detect` for every format below; one sub-package per format |

## PyTorch → go-tensor

| PyTorch | go-tensor |
|---------|-----------|
| `torch.tensor([[1, 2], [3, 4]])` | `tensor.New([]float64{1, 2, 3, 4}, 2, 2)` |
| `torch.randn(3, 4, requires_grad=True)` | `tensor.Randn(3, 4).SetRequiresGrad(true)` |
| `a + b`, `a * b`, `a @ b` | `a.Add(b)`, `a.Mul(b)`, `a.MatMul(b)` |
| `x.sum(dim=1, keepdim=True)` | `x.SumDim(1, true)` |
| `loss.backward()` | `loss.Backward()` |
| `x.grad` | `x.Grad()` |
| `with torch.no_grad():` | `tensor.NoGrad(func() { ... })` |
| `torch.manual_seed(0)` | `tensor.ManualSeed(0)` |
| `torch.load(p)` | `weights.Load(p)` |
| `model.load_state_dict(sd)` | `nn.LoadStateDict(model, sd, true)` |
| `model.state_dict()` | `nn.StateDict(model)` |
| `save_file(model.state_dict(), p)` | `safetensors.Save(p, nn.StateDict(model), safetensors.F32, nil)` |
| `model.train()` / `model.eval()` | `nn.Train(model)` / `nn.Eval(model)` |
| `for x, y in loader:` | `for x, y := range loader.All() { ... }` |

Custom differentiable ops work like `torch.autograd.Function`: compute the forward
result yourself and pass a backward closure to `tensor.CustomOp`.

## Design notes

- Data is `float64`, stored contiguously in row-major order.
- Requires Go 1.23+ (`DataLoader.All` is a range-over-func iterator).
- Shape errors panic, like out-of-range slice indexing.
- `MatMul` splits rows across goroutines for large matrices; `Conv2d` lowers to
  `MatMul` via im2col so convolutions benefit from it too.
- Every differentiable op is checked against finite differences in `tensor/tensor_test.go`.

## Known limitations

- **CPU utilization.** The CNN example keeps only ~3 of 10 cores busy: im2col,
  col2im and the gradient reordering in `Conv2d` are single-threaded, and with
  few output channels the matrix products are narrow.
- **Memory.** im2col materializes every receptive field; the second conv layer of
  the MNIST CNN allocates ~29 MB per batch of 64.
- **`float64` only.** Twice the memory and half the SIMD throughput of `float32`.
- **Copies instead of views.** `T` and broadcasting copy data, and there is no
  general slicing yet (only `Index` along the first dimension).
- **Weights are materialized as `float64`.** Loading reads every tensor into
  memory at once; there is no memory-mapping or lazy loading, and GGUF tensors
  grow 4× (F16) to ~14× (4-bit quants), so multi-billion-parameter LLMs are out of reach.
- **Weights, not graphs.** ONNX and TorchScript files contribute their tensors
  only; the Go model must be built to match. Keras `.h5`/`.keras` (HDF5),
  TensorFlow checkpoints, and legacy (pre-1.6) `torch.save` files are unsupported.
- **`nn.Save`/`nn.Load` are positional.** Prefer `safetensors.Save` with
  `nn.StateDict`, which stores names and opens in Python.
- **Process-wide `NoGrad`.** It is a global switch, not per goroutine.

## Roadmap

Suggested order — generics first, because the element type shapes every kernel
written afterwards:

1. **Generic element types** — `float32` (default, like PyTorch) and `float64`
   via Go generics.
2. **Performance**
   - [ ] Parallelize im2col/col2im and the `Conv2d` reorder loops across goroutines
   - [ ] SIMD kernels, or Apple Accelerate / OpenBLAS for `MatMul`
   - [ ] Strided views (zero-copy `Transpose`, `Slice`, `Expand`)
   - [ ] GPU backends (Metal, CUDA)
3. **More layers and ops**
   - [ ] `BatchNorm`, `Embedding`, `AvgPool2d`
   - [ ] Batched `MatMul`, `Concat`, slicing
   - [ ] Attention / transformer blocks
4. **Ecosystem**
   - [ ] Memory-mapped / lazy weight loading, keeping quantized tensors packed
   - [ ] Keras HDF5, other GGUF quantizations (Q2_K, Q3_K, IQ*)
   - [ ] More datasets (Fashion-MNIST, CIFAR-10)

Done:

- [x] Autograd engine, broadcasting, `Linear`, `SGD`, `Adam`
- [x] `DataLoader` with Go iterators, model save/load, MNIST example
- [x] `Conv2d`, `MaxPool2d`, `Dropout`, train/eval modes
- [x] Named state dicts; loading PyTorch, safetensors, ONNX, NumPy and GGUF weights
