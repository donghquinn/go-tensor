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

Run the full example with `go run ./examples/xor`.

## Packages

| Package  | PyTorch equivalent      | Contents |
|----------|-------------------------|----------|
| `tensor` | `torch`, `torch.autograd` | `Tensor`, creation (`New`, `Zeros`, `Randn`, …), broadcasting element-wise ops, reductions, `MatMul`, `Reshape`/`T`, `LogSoftmax`, `Backward`, `NoGrad`, `CustomOp` |
| `nn`     | `torch.nn`              | `Module`, `Linear`, `Sequential`, activations, `MSELoss`, `CrossEntropyLoss`, `BCELoss` |
| `optim`  | `torch.optim`           | `SGD` (momentum, weight decay), `Adam` |

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

Custom differentiable ops work like `torch.autograd.Function`: compute the forward
result yourself and pass a backward closure to `tensor.CustomOp`.

## Design notes

- Data is `float64`, stored contiguously in row-major order.
- Shape errors panic, like out-of-range slice indexing.
- `MatMul` splits rows across goroutines for large matrices.
- Every differentiable op is checked against finite differences in `tensor/tensor_test.go`.

## Roadmap

- [ ] Generic element types (`float32`, `float16`) via Go generics
- [ ] Strided views (zero-copy `Transpose`, `Slice`, `Expand`)
- [ ] Batched `MatMul`, `Concat`, `Stack`, indexing/slicing
- [ ] `Conv2d`, `MaxPool2d`, `BatchNorm`, `Dropout`, `Embedding`, train/eval modes
- [ ] SIMD / BLAS backend, then GPU (Metal, CUDA)
- [ ] `DataLoader`-style batching, model save/load
- [ ] MNIST example
