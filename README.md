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

The MNIST example downloads the dataset into `data/mnist` (checksum-verified),
trains for 3 epochs, then saves the model and reloads it. On an Apple M5:

| Model | Test accuracy (3 epochs) | Time per epoch |
|-------|--------------------------|----------------|
| CNN (conv 16 → conv 32 → fc 128, dropout) | 98.78% | ~26 s |
| MLP (784 → 128 → 10) | 96.75% | ~1.2 s |

## Packages

| Package  | PyTorch equivalent      | Contents |
|----------|-------------------------|----------|
| `tensor` | `torch`, `torch.autograd` | `Tensor`, creation (`New`, `Zeros`, `Randn`, …), broadcasting element-wise ops, reductions, `MatMul`, `Conv2d`, `MaxPool2d`, `Reshape`/`T`, `LogSoftmax`, `Backward`, `NoGrad`, `CustomOp` |
| `nn`     | `torch.nn`              | `Module`, `Linear`, `Conv2d`, `MaxPool2d`, `Dropout`, `Sequential`, `Flatten`, activations, `Train`/`Eval`, `MSELoss`, `CrossEntropyLoss`, `BCELoss`, `Save`/`Load` |
| `optim`  | `torch.optim`           | `SGD` (momentum, weight decay), `Adam` |
| `data`   | `torch.utils.data`      | `Dataset`, `TensorDataset`, `DataLoader` (batching, shuffling) |
| `datasets/mnist` | `torchvision.datasets.MNIST` | Download, verify and load MNIST |

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
| `torch.save(model.state_dict(), p)` | `nn.SaveFile(p, model)` |
| `model.train()` / `model.eval()` | `nn.Train(model)` / `nn.Eval(model)` |
| `for x, y in loader:` | `for loader.Next() { x, y := loader.Batch() }` |

Custom differentiable ops work like `torch.autograd.Function`: compute the forward
result yourself and pass a backward closure to `tensor.CustomOp`.

## Design notes

- Data is `float64`, stored contiguously in row-major order.
- Shape errors panic, like out-of-range slice indexing.
- `MatMul` splits rows across goroutines for large matrices; `Conv2d` lowers to
  `MatMul` via im2col, so convolutions use every core too.
- Every differentiable op is checked against finite differences in `tensor/tensor_test.go`.

## Roadmap

- [ ] Generic element types (`float32`, `float16`) via Go generics
- [ ] Strided views (zero-copy `Transpose`, `Slice`, `Expand`)
- [ ] Batched `MatMul`, `Concat`, slicing
- [x] `Conv2d`, `MaxPool2d`, `Dropout`, train/eval modes
- [ ] `BatchNorm`, `Embedding`, `AvgPool2d`
- [ ] SIMD / BLAS backend, then GPU (Metal, CUDA)
- [x] `DataLoader`, model save/load, MNIST example
