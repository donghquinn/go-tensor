# go-tensor

[![Go Reference](https://pkg.go.dev/badge/github.com/donghquinn/go-tensor.svg)](https://pkg.go.dev/github.com/donghquinn/go-tensor)
![Go version](https://img.shields.io/badge/go-1.23%2B-00ADD8)

**A PyTorch-style deep learning library for Go.** Build, train and run neural
networks with n-dimensional tensors, automatic differentiation, layers,
optimizers and data loaders, and load pretrained weights from PyTorch,
safetensors, ONNX, NumPy and GGUF files.

Pure Go: no cgo, no external dependencies, nothing to install besides the Go toolchain.

```go
model := nn.NewSequential(
	nn.NewLinear(784, 128),
	nn.NewReLU(),
	nn.NewLinear(128, 10),
)
opt := optim.NewAdam(model.Parameters(), optim.AdamConfig{LR: 1e-3})

for x, y := range loader.All() {
	loss := nn.CrossEntropyLoss(model.Forward(x), y)
	opt.ZeroGrad()
	loss.Backward()
	opt.Step()
}
```

## Contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage guide](#usage-guide)
  - [Tensors](#tensors)
  - [Automatic differentiation](#automatic-differentiation)
  - [Custom differentiable operations](#custom-differentiable-operations)
  - [Building models](#building-models)
  - [Training](#training)
  - [Saving and loading models](#saving-and-loading-models)
  - [Loading pretrained weights](#loading-pretrained-weights)
  - [Datasets](#datasets)
- [Examples](#examples)
- [Reference](#reference)
- [Design notes](#design-notes)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Development](#development)

## Features

- **Tensors**: n-dimensional arrays with NumPy/PyTorch broadcasting, element-wise
  math, reductions, reshaping, matrix multiplication and 2-D convolution.
- **Autograd**: reverse-mode automatic differentiation with `Backward()`,
  gradient accumulation, `NoGrad`, and custom differentiable ops.
- **Neural networks**: `Linear`, `Conv2d`, `MaxPool2d`, `Dropout`, `Flatten`,
  ReLU/Sigmoid/Tanh/Softmax, `Sequential`, train/eval modes, and MSE,
  cross-entropy and binary cross-entropy losses.
- **Optimizers**: SGD (momentum, weight decay) and Adam.
- **Data**: `Dataset`/`DataLoader` with batching, per-epoch shuffling and Go
  1.23 iterators; MNIST with checksum-verified download.
- **Pretrained weights**: load PyTorch (`.pth`/`.pt`, TorchScript), safetensors,
  ONNX, NumPy (`.npy`/`.npz`) and GGUF (including quantized llama.cpp models), all
  without executing any code from the file.
- **Multi-core**: large matrix multiplications and convolutions are split across
  goroutines automatically.
- **Tested**: every gradient is checked numerically, and every weight format is
  verified bit-exactly against files written by the reference Python libraries.

## Installation

Requires **Go 1.23 or newer**.

```sh
go get github.com/donghquinn/go-tensor@latest
```

Then import the packages you need:

```go
import (
	"github.com/donghquinn/go-tensor/data"     // datasets and data loaders
	"github.com/donghquinn/go-tensor/nn"       // layers, losses, state dicts
	"github.com/donghquinn/go-tensor/optim"    // optimizers
	"github.com/donghquinn/go-tensor/tensor"   // tensors and autograd
	"github.com/donghquinn/go-tensor/weights"  // pretrained weight loading
)
```

To try the bundled examples, clone the repository:

```sh
git clone https://github.com/donghquinn/go-tensor.git
cd go-tensor
go run ./examples/xor
```

## Quick start

A complete program that trains a small network to learn XOR:

```go
package main

import (
	"fmt"

	"github.com/donghquinn/go-tensor/nn"
	"github.com/donghquinn/go-tensor/optim"
	"github.com/donghquinn/go-tensor/tensor"
)

func main() {
	tensor.ManualSeed(42) // reproducible initialization

	// Four samples with two features each, and their class labels.
	x := tensor.New([]float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}, 4, 2)
	y := []int{0, 1, 1, 0}

	model := nn.NewSequential(
		nn.NewLinear(2, 8),
		nn.NewTanh(),
		nn.NewLinear(8, 2),
	)
	opt := optim.NewAdam(model.Parameters(), optim.AdamConfig{LR: 0.05})

	for epoch := 0; epoch <= 500; epoch++ {
		loss := nn.CrossEntropyLoss(model.Forward(x), y)

		opt.ZeroGrad()  // clear gradients from the previous step
		loss.Backward() // compute gradients
		opt.Step()      // update the parameters

		if epoch%100 == 0 {
			fmt.Printf("epoch %3d  loss %.6f\n", epoch, loss.Item())
		}
	}

	tensor.NoGrad(func() {
		fmt.Println("predictions:", model.Forward(x).Argmax()) // [0 1 1 0]
	})
}
```

## Usage guide

### Tensors

A `*tensor.Tensor` is an n-dimensional array of `float64` values stored in
row-major order.

```go
a := tensor.New([]float64{1, 2, 3, 4, 5, 6}, 2, 3) // 2×3 matrix from data
b := tensor.Ones(2, 3)                               // 2×3 of ones
z := tensor.Zeros(4)                                 // vector of 4 zeros
r := tensor.Randn(3, 3)                              // standard normal
u := tensor.Uniform(-1, 1, 2, 2)                     // uniform in [-1, 1)
s := tensor.Scalar(3.5)                              // 0-dimensional
i := tensor.Arange(0, 5, 1)                          // [0 1 2 3 4]

fmt.Println(a)
// tensor([[1, 2, 3],
//         [4, 5, 6]])

fmt.Println(a.Shape(), a.Dim(), a.Numel()) // [2 3] 2 6
fmt.Println(a.At(1, 2))                    // 6
fmt.Println(s.Item())                      // 3.5
```

Operations return new tensors and can be chained. Binary operations broadcast
like NumPy and PyTorch:

```go
c := a.Add(b).MulScalar(2)                     // (a + b) * 2
bias := tensor.New([]float64{10, 20, 30}, 3)   // shape [3]
d := a.Add(bias)                               // [2 3] + [3] → [2 3]
e := a.MatMul(a.T())                           // [2 3] @ [3 2] → [2 2]

rowSums := a.SumDim(1, false)                  // [2]: sum of each row
colMeans := a.MeanDim(0, true)                 // [1 3]: mean of each column
total := a.Sum()                               // scalar

flat := a.Reshape(-1)                          // [6]; -1 is inferred
probs := a.Softmax()                           // softmax over the last dimension
classes := probs.Argmax()                      // []int{2, 2}
```

Available operations include `Add`, `Sub`, `Mul`, `Div`, `AddScalar`,
`MulScalar`, `Neg`, `Pow`, `Exp`, `Log`, `Sqrt`, `Abs`, `ReLU`, `Sigmoid`, `Tanh`,
`Sum`, `Mean`, `SumDim`, `MeanDim`, `Argmax`, `Reshape`, `Flatten`, `T`,
`Index`, `Stack`, `MatMul`, `Conv2d`, `MaxPool2d`, `LogSoftmax`, `Softmax` and `Pick`.
Shape mismatches panic, like out-of-range slice indexing.

`Data()` returns the underlying slice without copying, like `tensor.numpy()` in
PyTorch:

```go
vals := a.Data() // []float64{1, 2, 3, 4, 5, 6}, shares memory with a
```

### Automatic differentiation

Mark the tensors you want gradients for with `SetRequiresGrad(true)`. Every
operation on them is recorded, and `Backward()` on a scalar result computes the
gradient of every one of them:

```go
x := tensor.New([]float64{1, 2, 3}, 3).SetRequiresGrad(true)
w := tensor.New([]float64{0.5, -1, 2}, 3).SetRequiresGrad(true)

y := x.Mul(w).Sum() // y = Σ xᵢ·wᵢ
y.Backward()

fmt.Println(x.Grad()) // tensor([0.5, -1, 2])  — dy/dx = w
fmt.Println(w.Grad()) // tensor([1, 2, 3])     — dy/dw = x
```

Gradients accumulate across calls to `Backward`, as in PyTorch. Clear them with
`ZeroGrad()` on each tensor, or `opt.ZeroGrad()` for all parameters of an
optimizer.

Skip graph recording during inference with `tensor.NoGrad`, the equivalent of
`with torch.no_grad():`:

```go
tensor.NoGrad(func() {
	out := model.Forward(x) // no graph is built, so this is faster and lighter
	fmt.Println(out.RequiresGrad()) // false
})
```

`Detach()` returns a tensor that shares data but is cut off from the graph, and
`Clone()` returns an independent copy.

### Custom differentiable operations

Like subclassing `torch.autograd.Function`: compute the forward result yourself
and supply the backward rule. The backward function receives the gradient of the
output and returns one gradient per input:

```go
// Softplus computes log(1 + eˣ); its derivative is sigmoid(x).
func Softplus(x *tensor.Tensor) *tensor.Tensor {
	in := x.Data()
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = math.Log1p(math.Exp(v))
	}
	return tensor.CustomOp("Softplus", out, x.Shape(), []*tensor.Tensor{x},
		func(grad *tensor.Tensor) []*tensor.Tensor {
			return []*tensor.Tensor{grad.Mul(x.Sigmoid())}
		})
}
```

### Building models

Any type with `Forward` and `Parameters` methods is an `nn.Module`. For simple
stacks, use `nn.Sequential`:

```go
cnn := nn.NewSequential(
	nn.NewConv2d(1, 16, 3, 1, 1), // in channels, out channels, kernel, stride, padding
	nn.NewReLU(),
	nn.NewMaxPool2d(2, 0),        // 2×2 windows; stride 0 means "same as kernel"
	nn.NewConv2d(16, 32, 3, 1, 1),
	nn.NewReLU(),
	nn.NewMaxPool2d(2, 0),
	nn.NewFlatten(),              // (N, 32, 7, 7) → (N, 1568)
	nn.NewDropout(0.25),
	nn.NewLinear(32*7*7, 128),
	nn.NewReLU(),
	nn.NewLinear(128, 10),
)
logits := cnn.Forward(tensor.Randn(8, 1, 28, 28)) // batch of 8 grayscale 28×28 images
fmt.Println(logits.Shape())                       // [8 10]
```

For custom architectures, write your own module. Implementing `NamedParameters`
is optional, but gives parameters PyTorch-style names in state dicts:

```go
type MLP struct {
	fc1, fc2 *nn.Linear
}

func NewMLP(in, hidden, out int) *MLP {
	return &MLP{fc1: nn.NewLinear(in, hidden), fc2: nn.NewLinear(hidden, out)}
}

func (m *MLP) Forward(x *tensor.Tensor) *tensor.Tensor {
	return m.fc2.Forward(m.fc1.Forward(x).ReLU())
}

func (m *MLP) Parameters() []*tensor.Tensor {
	return append(m.fc1.Parameters(), m.fc2.Parameters()...)
}

// NamedParameters names the parameters "fc1.weight", "fc1.bias", "fc2.weight", ...
func (m *MLP) NamedParameters() []nn.NamedParameter {
	var out []nn.NamedParameter
	for _, p := range m.fc1.NamedParameters() {
		out = append(out, nn.NamedParameter{Name: "fc1." + p.Name, Value: p.Value})
	}
	for _, p := range m.fc2.NamedParameters() {
		out = append(out, nn.NamedParameter{Name: "fc2." + p.Name, Value: p.Value})
	}
	return out
}
```

Layer weights use the same shapes and default initialization as PyTorch:
`Linear` weight is `(out, in)`, and `Conv2d` weight is `(out, in, k, k)`.

### Training

A full training loop with mini-batches, a held-out test set, and train/eval
modes (needed so `Dropout` is only active during training):

```go
func train(model nn.Module, trainSet, testSet data.Dataset, epochs int) {
	opt := optim.NewAdam(model.Parameters(), optim.AdamConfig{LR: 1e-3})
	loader := data.NewDataLoader(trainSet, 64, true) // batch size 64, shuffled

	for epoch := 1; epoch <= epochs; epoch++ {
		nn.Train(model) // like model.train()
		total, batches := 0.0, 0
		for x, y := range loader.All() { // one epoch; reshuffled each time
			loss := nn.CrossEntropyLoss(model.Forward(x), y)
			opt.ZeroGrad()
			loss.Backward()
			opt.Step()
			total += loss.Item()
			batches++
		}
		fmt.Printf("epoch %d  loss %.4f  accuracy %.2f%%\n",
			epoch, total/float64(batches), 100*accuracy(model, testSet))
	}
}

func accuracy(model nn.Module, ds data.Dataset) float64 {
	nn.Eval(model) // like model.eval()
	correct := 0
	tensor.NoGrad(func() {
		for x, y := range data.NewDataLoader(ds, 1000, false).All() {
			for i, pred := range model.Forward(x).Argmax() {
				if pred == y[i] {
					correct++
				}
			}
		}
	})
	return float64(correct) / float64(ds.Len())
}
```

Wrap in-memory data with `data.NewTensorDataset(x, labels)`, where sample `i`
is `x.Index(i)`. For other sources, implement the `data.Dataset` interface:

```go
type Dataset interface {
	Len() int
	Get(i int) (x *tensor.Tensor, y int)
}
```

Optimizers:

```go
sgd := optim.NewSGD(model.Parameters(), optim.SGDConfig{LR: 0.1, Momentum: 0.9, WeightDecay: 5e-4})
adam := optim.NewAdam(model.Parameters(), optim.AdamConfig{LR: 1e-3}) // zero fields use PyTorch defaults
```

Losses: `nn.CrossEntropyLoss(logits, labels)` (takes raw logits, like PyTorch),
`nn.MSELoss(pred, target)` and `nn.BCELoss(prob, target)`.

### Saving and loading models

`nn.StateDict` returns a model's parameters by name, like `model.state_dict()`.
Save it as safetensors to get a file that PyTorch and Hugging Face tools can also
read:

```go
// Save
err := safetensors.Save("model.safetensors", nn.StateDict(model), safetensors.F32, nil)

// Load into a model with the same architecture
sd, err := weights.Load("model.safetensors")
err = nn.LoadStateDict(model, sd, true)
```

Use `safetensors.F64` instead of `F32` to store values without rounding. The
safetensors package is `github.com/donghquinn/go-tensor/weights/safetensors`.

### Loading pretrained weights

`weights.Load` detects the file format from its contents and returns a
name → tensor map. Because parameter names follow PyTorch's conventions, a model
trained in Python loads directly into the same architecture built in Go.

In Python:

```python
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(16 * 14 * 14, 10),
)
# ... train ...
torch.save(model.state_dict(), "model.pth")
```

In Go:

```go
model := nn.NewSequential(
	nn.NewConv2d(1, 16, 3, 1, 1), nn.NewReLU(), nn.NewMaxPool2d(2, 0),
	nn.NewFlatten(), nn.NewLinear(16*14*14, 10),
)
sd, err := weights.Load("model.pth") // also .safetensors, .onnx, .npz, .gguf, ...
if err != nil {
	log.Fatal(err)
}
if err := nn.LoadStateDict(model, sd, true); err != nil { // strict, like PyTorch
	log.Fatal(err) // reports missing and unexpected keys
}
nn.Eval(model)
```

Pass `strict=false` to load only the keys that match, for example when
fine-tuning with a different final layer. A shape mismatch is always an error,
and a failed load leaves the model unchanged.

| Format | Package | Read | Write | Notes |
|--------|---------|:----:|:-----:|-------|
| PyTorch `.pth` / `.pt` / `.bin` | `weights/pth` | ✓ | | `torch.save` (PyTorch ≥ 1.6), nested checkpoints, strided views, TorchScript |
| safetensors | `weights/safetensors` | ✓ | ✓ | All integer and float dtypes including F16/BF16; metadata |
| ONNX | `weights/onnx` | ✓ | | Graph initializers, including external data files |
| NumPy `.npy` / `.npz` | `weights/numpy` | ✓ | ✓ | Fortran order, big-endian, compressed archives |
| GGUF | `weights/gguf` | ✓ | | F32/F16/BF16/F64/ints, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K; metadata |

All values are converted to `float64`. Each format's package can also be used
directly when you need more than the tensors:

```go
// A training checkpoint such as {"model": state_dict, "epoch": 7}.
obj, err := pth.LoadObject("checkpoint.pth")
ckpt := obj.(*pth.Dict)
epoch := ckpt.Values["epoch"].(int64)
modelWeights := pth.Tensors(ckpt.Values["model"]) // keys like "0.weight"

// GGUF metadata, e.g. a llama.cpp model's architecture.
tensors, meta, err := gguf.LoadWithMetadata("model.gguf")
fmt.Println(meta["general.architecture"])

// safetensors metadata.
tensors, info, err := safetensors.LoadWithMetadata("model.safetensors")
```

When a whole checkpoint is loaded with `weights.Load`, nested keys are joined
with dots: `{"model": {"0.weight": ...}}` becomes `"model.0.weight"`.

**Loading is safe.** Python's `torch.load` without `weights_only=True` can run
arbitrary code from a malicious file. go-tensor's pickle reader never imports or
calls anything: it rebuilds tensors through a fixed allowlist and turns every
other object into inert data. All parsers check sizes against the actual file
before allocating, and are fuzz-tested.

### Datasets

`datasets/mnist` downloads MNIST (verifying checksums) and returns datasets of
`(1, 28, 28)` images scaled to [0, 1]:

```go
train, err := mnist.Load(".cache/mnist", true, true) // dir, train split, download if missing
test, err := mnist.Load(".cache/mnist", false, true)
```

## Examples

Runnable programs live in [`examples/`](examples):

```sh
go run ./examples/xor                         # MLP learns XOR
go run ./examples/mnist -model cnn            # CNN on MNIST
go run ./examples/mnist -model mlp -epochs 5  # MLP on MNIST
```

`examples/mnist` downloads the dataset into `.cache/mnist`, trains, prints test
accuracy per epoch, then saves the model and checks a reloaded copy. Flags:
`-model`, `-epochs`, `-batch`, `-lr`, `-data`, `-save`. Results after 3 epochs
on an Apple M5:

| Model | Test accuracy | Time per epoch |
|-------|---------------|----------------|
| CNN (conv 16 → conv 32 → fc 128, dropout) | 98.78% | ~26 s |
| MLP (784 → 128 → 10) | 96.75% | ~1.2 s |

## Reference

Full API documentation is on [pkg.go.dev](https://pkg.go.dev/github.com/donghquinn/go-tensor).

### Packages

| Package | PyTorch equivalent | Contents |
|---------|--------------------|----------|
| `tensor` | `torch`, `torch.autograd` | `Tensor`, creation functions, broadcasting math, reductions, `MatMul`, `Conv2d`, `MaxPool2d`, `Backward`, `NoGrad`, `CustomOp` |
| `nn` | `torch.nn` | `Module`, `Linear`, `Conv2d`, `MaxPool2d`, `Dropout`, `Flatten`, `Sequential`, activations, losses, `Train`/`Eval`, `StateDict`/`LoadStateDict` |
| `optim` | `torch.optim` | `SGD`, `Adam` |
| `data` | `torch.utils.data` | `Dataset`, `TensorDataset`, `DataLoader` |
| `datasets/mnist` | `torchvision.datasets.MNIST` | MNIST download and loading |
| `weights` | `torch.load`, `safetensors`, `onnx`, `gguf` | `Load`, `Detect`; one sub-package per format |

### PyTorch → go-tensor

| PyTorch | go-tensor |
|---------|-----------|
| `torch.tensor([[1, 2], [3, 4]])` | `tensor.New([]float64{1, 2, 3, 4}, 2, 2)` |
| `torch.randn(3, 4, requires_grad=True)` | `tensor.Randn(3, 4).SetRequiresGrad(true)` |
| `a + b`, `a * b`, `a @ b` | `a.Add(b)`, `a.Mul(b)`, `a.MatMul(b)` |
| `x.sum(dim=1, keepdim=True)` | `x.SumDim(1, true)` |
| `x.view(-1, 4)` | `x.Reshape(-1, 4)` |
| `loss.backward()` | `loss.Backward()` |
| `x.grad` | `x.Grad()` |
| `with torch.no_grad():` | `tensor.NoGrad(func() { ... })` |
| `torch.manual_seed(0)` | `tensor.ManualSeed(0)` |
| `model.train()` / `model.eval()` | `nn.Train(model)` / `nn.Eval(model)` |
| `model.state_dict()` | `nn.StateDict(model)` |
| `model.load_state_dict(sd)` | `nn.LoadStateDict(model, sd, true)` |
| `torch.load(p)` | `weights.Load(p)` |
| `save_file(model.state_dict(), p)` | `safetensors.Save(p, nn.StateDict(model), safetensors.F32, nil)` |
| `for x, y in loader:` | `for x, y := range loader.All() { ... }` |

## Design notes

- Values are `float64`, stored contiguously in row-major order.
- Shape errors panic, like out-of-range slice indexing; I/O and file-format
  errors are returned as `error`.
- `MatMul` splits rows across goroutines for large matrices; `Conv2d` is lowered
  to `MatMul` via im2col, so convolutions use multiple cores too.
- Every differentiable operation is checked against finite differences in
  `tensor/tensor_test.go`.

## Known limitations

- **`float64` only.** Uses twice the memory of `float32`, PyTorch's default.
- **CPU only, no SIMD.** The MNIST CNN keeps about 3 of 10 cores busy; im2col and
  the `Conv2d` reordering loops are single-threaded.
- **Copies instead of views.** `T` and broadcasting copy data, and slicing is
  limited to `Index` along the first dimension.
- **Weights are loaded fully into memory as `float64`.** There is no
  memory-mapping, and GGUF tensors grow 4× (F16) to about 14× (4-bit), so
  multi-billion-parameter LLMs do not fit.
- **Weights, not graphs.** ONNX and TorchScript files supply tensors only; build
  the matching model in Go. Keras `.h5`/`.keras`, TensorFlow checkpoints and
  pre-1.6 `torch.save` files are not supported.
- **`NoGrad` is process-wide**, not per goroutine.
- `nn.Save`/`nn.Load` match parameters by position; prefer safetensors with
  `nn.StateDict`, which stores names.

## Roadmap

1. **Generic element types**: `float32` (default) and `float64` via Go generics.
2. **Performance**
   - [ ] Parallel im2col/col2im and `Conv2d` reordering
   - [ ] SIMD kernels, or Apple Accelerate / OpenBLAS for `MatMul`
   - [ ] Strided views (zero-copy `Transpose`, `Slice`, `Expand`)
   - [ ] GPU backends (Metal, CUDA)
3. **More layers and ops**
   - [ ] `BatchNorm`, `Embedding`, `AvgPool2d`
   - [ ] Batched `MatMul`, `Concat`, slicing
   - [ ] Attention and transformer blocks
4. **Ecosystem**
   - [ ] Memory-mapped, lazy weight loading that keeps quantized tensors packed
   - [ ] Keras HDF5; more GGUF quantizations (Q2_K, Q3_K, IQ*)
   - [ ] More datasets (Fashion-MNIST, CIFAR-10)

Done:

- [x] Autograd engine, broadcasting, `Linear`, `SGD`, `Adam`
- [x] `DataLoader` with Go iterators, MNIST example
- [x] `Conv2d`, `MaxPool2d`, `Dropout`, train/eval modes
- [x] Named state dicts; PyTorch, safetensors, ONNX, NumPy and GGUF weight loading

## Development

```sh
go test ./...                     # all tests
go test -race ./...               # with the race detector
go test ./tensor -bench Conv2d    # convolution benchmark
go test ./weights/pth -fuzz FuzzUnpickle -fuzztime 30s
```

The weight-format fixtures in `weights/testdata` are generated by the reference
Python libraries. To regenerate them:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv torch safetensors onnx gguf numpy
.venv/bin/python weights/testdata/gen_fixtures.py
```

Contributions are welcome: please open an issue to discuss larger changes, and
include tests (gradient checks for new operations) with pull requests.
