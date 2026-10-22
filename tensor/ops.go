package tensor

import (
	"fmt"

	"github.com/donghquinn/go-tensor/internal/backend"
)

// Backward functions run with gradient tracking disabled, so they are written
// with the same tensor operations as the forward passes and work on every
// device.

// ---- element-wise binary ops (with broadcasting) ----

func (t *Tensor) binary(name string, op backend.BinaryOp, o *Tensor, backward func(g *Tensor) []*Tensor) *Tensor {
	checkSame(name, t, o)
	shape := broadcastShapes(t.shape, o.shape)
	av := backend.View{Shape: shape, Strides: broadcastStrides(t.shape, shape)}
	bv := backend.View{Shape: shape, Strides: broadcastStrides(o.shape, shape)}
	return newResult(t.be().Binary(op, t.st, av, o.st, bv), shape, name, []*Tensor{t, o}, backward)
}

// Add returns t + o.
func (t *Tensor) Add(o *Tensor) *Tensor {
	return t.binary("Add", backend.Add, o, func(g *Tensor) []*Tensor {
		return []*Tensor{sumTo(g, t.shape), sumTo(g, o.shape)}
	})
}

// Sub returns t - o.
func (t *Tensor) Sub(o *Tensor) *Tensor {
	return t.binary("Sub", backend.Sub, o, func(g *Tensor) []*Tensor {
		return []*Tensor{sumTo(g, t.shape), sumTo(g.Neg(), o.shape)}
	})
}

// Mul returns the element-wise product t * o.
func (t *Tensor) Mul(o *Tensor) *Tensor {
	return t.binary("Mul", backend.Mul, o, func(g *Tensor) []*Tensor {
		return []*Tensor{sumTo(g.Mul(o), t.shape), sumTo(g.Mul(t), o.shape)}
	})
}

// Div returns the element-wise quotient t / o.
func (t *Tensor) Div(o *Tensor) *Tensor {
	return t.binary("Div", backend.Div, o, func(g *Tensor) []*Tensor {
		gt := g.Div(o)
		return []*Tensor{sumTo(gt, t.shape), sumTo(gt.Mul(t).Div(o).Neg(), o.shape)}
	})
}

// ---- element-wise unary ops ----

// unary applies op; backward receives the output gradient g and the output y.
func (t *Tensor) unary(name string, op backend.UnaryOp, s float64, backward func(g, y *Tensor) *Tensor) *Tensor {
	st := t.be().Unary(op, s, t.st)
	y := fromStorage(st, t.shape)
	return newResult(st, t.shape, name, []*Tensor{t}, func(g *Tensor) []*Tensor {
		return []*Tensor{backward(g, y)}
	})
}

// raw applies a unary op without recording it.
func (t *Tensor) raw(op backend.UnaryOp, s float64) *Tensor {
	return fromStorage(t.be().Unary(op, s, t.st), t.shape)
}

// Neg returns -t.
func (t *Tensor) Neg() *Tensor {
	return t.unary("Neg", backend.Neg, 0, func(g, _ *Tensor) *Tensor { return g.Neg() })
}

// AddScalar returns t + s.
func (t *Tensor) AddScalar(s float64) *Tensor {
	return t.unary("AddScalar", backend.AddScalar, s, func(g, _ *Tensor) *Tensor { return g })
}

// MulScalar returns t * s.
func (t *Tensor) MulScalar(s float64) *Tensor {
	return t.unary("MulScalar", backend.MulScalar, s, func(g, _ *Tensor) *Tensor { return g.MulScalar(s) })
}

// Pow returns t raised element-wise to the power p.
func (t *Tensor) Pow(p float64) *Tensor {
	return t.unary("Pow", backend.Pow, p, func(g, _ *Tensor) *Tensor { return g.Mul(t.Pow(p - 1).MulScalar(p)) })
}

// Exp returns e^t.
func (t *Tensor) Exp() *Tensor {
	return t.unary("Exp", backend.Exp, 0, func(g, y *Tensor) *Tensor { return g.Mul(y) })
}

// Log returns the natural logarithm of t.
func (t *Tensor) Log() *Tensor {
	return t.unary("Log", backend.Log, 0, func(g, _ *Tensor) *Tensor { return g.Div(t) })
}

// Sqrt returns the square root of t.
func (t *Tensor) Sqrt() *Tensor {
	return t.unary("Sqrt", backend.Sqrt, 0, func(g, y *Tensor) *Tensor { return g.Div(y).MulScalar(0.5) })
}

// Abs returns |t|.
func (t *Tensor) Abs() *Tensor {
	return t.unary("Abs", backend.Abs, 0, func(g, _ *Tensor) *Tensor { return g.Mul(t.raw(backend.Sign, 0)) })
}

// ReLU returns max(t, 0).
func (t *Tensor) ReLU() *Tensor {
	return t.unary("ReLU", backend.ReLU, 0, func(g, _ *Tensor) *Tensor { return g.Mul(t.raw(backend.Step, 0)) })
}

// Sigmoid returns 1 / (1 + e^-t).
func (t *Tensor) Sigmoid() *Tensor {
	return t.unary("Sigmoid", backend.Sigmoid, 0, func(g, y *Tensor) *Tensor {
		return g.Mul(y.Mul(y.Neg().AddScalar(1)))
	})
}

// Tanh returns the hyperbolic tangent of t.
func (t *Tensor) Tanh() *Tensor {
	return t.unary("Tanh", backend.Tanh, 0, func(g, y *Tensor) *Tensor {
		return g.Mul(y.Mul(y).Neg().AddScalar(1))
	})
}

// ---- reductions ----

// reduce applies op over dimension dim without recording it.
func (t *Tensor) reduce(op backend.ReduceOp, dim int, keepDim bool) *Tensor {
	dim = normDim(dim, len(t.shape))
	st := t.be().Reduce(op, t.st, numel(t.shape[:dim]), t.shape[dim], numel(t.shape[dim+1:]))
	return fromStorage(st, reducedShape(t.shape, dim, keepDim))
}

func reducedShape(shape []int, dim int, keepDim bool) []int {
	if keepDim {
		s := append([]int(nil), shape...)
		s[dim] = 1
		return s
	}
	return append(append([]int(nil), shape[:dim]...), shape[dim+1:]...)
}

// Sum returns the sum of all elements as a scalar tensor.
func (t *Tensor) Sum() *Tensor {
	st := t.be().Reduce(backend.Sum, t.st, 1, t.Numel(), 1)
	return newResult(st, nil, "Sum", []*Tensor{t}, func(g *Tensor) []*Tensor {
		return []*Tensor{expand(g, t.shape)}
	})
}

// Mean returns the mean of all elements as a scalar tensor.
func (t *Tensor) Mean() *Tensor { return t.Sum().MulScalar(1 / float64(t.Numel())) }

// SumDim sums over dimension dim. If keepDim is true the reduced dimension is
// kept with size 1.
func (t *Tensor) SumDim(dim int, keepDim bool) *Tensor {
	dim = normDim(dim, len(t.shape))
	r := t.reduce(backend.Sum, dim, keepDim)
	kept := reducedShape(t.shape, dim, true)
	return newResult(r.st, r.shape, "SumDim", []*Tensor{t}, func(g *Tensor) []*Tensor {
		return []*Tensor{expand(fromStorage(g.st, kept), t.shape)}
	})
}

// MeanDim averages over dimension dim.
func (t *Tensor) MeanDim(dim int, keepDim bool) *Tensor {
	n := t.shape[normDim(dim, len(t.shape))]
	return t.SumDim(dim, keepDim).MulScalar(1 / float64(n))
}

// MaxDim returns the maximum over dimension dim. It is not differentiable.
func (t *Tensor) MaxDim(dim int, keepDim bool) *Tensor { return t.reduce(backend.Max, dim, keepDim) }

// Argmax returns the index of the largest value along the last dimension for
// every row. It is not differentiable.
func (t *Tensor) Argmax() []int {
	vals := t.Float64s()
	n := t.shape[len(t.shape)-1]
	rows := len(vals) / n
	idx := make([]int, rows)
	for r := 0; r < rows; r++ {
		row := vals[r*n : (r+1)*n]
		for j, x := range row {
			if x > row[idx[r]] {
				idx[r] = j
			}
		}
	}
	return idx
}

// ---- shape ops ----

// Reshape returns a tensor with the same data and a new shape. One dimension
// may be -1, in which case it is inferred.
func (t *Tensor) Reshape(shape ...int) *Tensor {
	shape = append([]int(nil), shape...)
	infer, known := -1, 1
	for i, s := range shape {
		if s == -1 {
			if infer >= 0 {
				panic("tensor: Reshape allows only one -1 dimension")
			}
			infer = i
		} else {
			known *= s
		}
	}
	if infer >= 0 && known != 0 {
		shape[infer] = t.Numel() / known
	}
	if numel(shape) != t.Numel() {
		panic(fmt.Sprintf("tensor: cannot reshape %v into %v", t.shape, shape))
	}
	return newResult(t.st, shape, "Reshape", []*Tensor{t}, func(g *Tensor) []*Tensor {
		return []*Tensor{fromStorage(g.st, t.shape)}
	})
}

// Flatten reshapes t into a 2-D tensor (batch, features), keeping dim 0.
func (t *Tensor) Flatten() *Tensor { return t.Reshape(t.shape[0], -1) }

// Permute reorders the dimensions, like torch.permute: dimension i of the
// result is dimension dims[i] of t.
func (t *Tensor) Permute(dims ...int) *Tensor {
	if len(dims) != len(t.shape) {
		panic(fmt.Sprintf("tensor: Permute got %d dims for shape %v", len(dims), t.shape))
	}
	src := backend.ContiguousStrides(t.shape)
	shape := make([]int, len(dims))
	strides := make([]int, len(dims))
	inverse := make([]int, len(dims))
	seen := make([]bool, len(dims))
	for i, d := range dims {
		d = normDim(d, len(t.shape))
		if seen[d] {
			panic(fmt.Sprintf("tensor: Permute dims %v repeat dimension %d", dims, d))
		}
		seen[d] = true
		shape[i], strides[i], inverse[d] = t.shape[d], src[d], i
	}
	st := t.be().Copy(t.st, backend.View{Shape: shape, Strides: strides})
	return newResult(st, shape, "Permute", []*Tensor{t}, func(g *Tensor) []*Tensor {
		return []*Tensor{g.Permute(inverse...)}
	})
}

// T swaps the last two dimensions.
func (t *Tensor) T() *Tensor {
	nd := len(t.shape)
	if nd < 2 {
		panic(fmt.Sprintf("tensor: T requires at least 2 dimensions, got %v", t.shape))
	}
	dims := make([]int, nd)
	for i := range dims {
		dims[i] = i
	}
	dims[nd-2], dims[nd-1] = nd-1, nd-2
	return t.Permute(dims...)
}

// Index returns a copy of t[i], selecting along the first dimension.
func (t *Tensor) Index(i int) *Tensor {
	if len(t.shape) == 0 || i < 0 || i >= t.shape[0] {
		panic(fmt.Sprintf("tensor: Index %d out of range for shape %v", i, t.shape))
	}
	inner := t.shape[1:]
	sz := numel(inner)
	v := contiguousView(inner)
	v.Offset = i * sz
	return newResult(t.be().Copy(t.st, v), inner, "Index", []*Tensor{t}, func(g *Tensor) []*Tensor {
		d := ZerosLike(t)
		g.be().Write(d.st, i*sz, g.st)
		return []*Tensor{d}
	})
}

// Stack joins tensors of identical shape, dtype and device along a new first
// dimension.
func Stack(ts []*Tensor) *Tensor {
	if len(ts) == 0 {
		panic("tensor: Stack needs at least one tensor")
	}
	checkSame("Stack", ts...)
	inner := ts[0].shape
	sz := numel(inner)
	b := ts[0].be()
	st := b.Zeros(ts[0].DType(), len(ts)*sz)
	for i, t := range ts {
		if !equalShape(t.shape, inner) {
			panic(fmt.Sprintf("tensor: Stack shape mismatch %v vs %v", t.shape, inner))
		}
		b.Write(st, i*sz, t.st)
	}
	shape := append([]int{len(ts)}, inner...)
	return newResult(st, shape, "Stack", ts, func(g *Tensor) []*Tensor {
		gs := make([]*Tensor, len(ts))
		for i := range gs {
			v := contiguousView(inner)
			v.Offset = i * sz
			gs[i] = fromStorage(g.be().Copy(g.st, v), inner)
		}
		return gs
	})
}

// ---- linear algebra ----

// MatMul returns the matrix product of two 2-D tensors.
func (t *Tensor) MatMul(o *Tensor) *Tensor {
	checkSame("MatMul", t, o)
	if len(t.shape) != 2 || len(o.shape) != 2 || t.shape[1] != o.shape[0] {
		panic(fmt.Sprintf("tensor: MatMul shape mismatch %v @ %v", t.shape, o.shape))
	}
	m, k, n := t.shape[0], t.shape[1], o.shape[1]
	b := t.be()
	return newResult(b.MatMul(t.st, o.st, m, k, n, false, false), []int{m, n}, "MatMul", []*Tensor{t, o}, func(g *Tensor) []*Tensor {
		var gt, gotensor *Tensor
		if t.requiresGrad { // g (m,n) @ oᵀ (n,k)
			gt = fromStorage(b.MatMul(g.st, o.st, m, n, k, false, true), t.shape)
		}
		if o.requiresGrad { // tᵀ (k,m) @ g (m,n)
			gotensor = fromStorage(b.MatMul(t.st, g.st, k, m, n, true, false), o.shape)
		}
		return []*Tensor{gt, gotensor}
	})
}

// ---- neural-network helpers ----

// LogSoftmax computes log(softmax(t)) along the last dimension in a
// numerically stable way.
func (t *Tensor) LogSoftmax() *Tensor {
	m := t.MaxDim(-1, true) // a constant shift: not part of the graph
	shifted := t.Sub(m)
	return shifted.Sub(shifted.Exp().SumDim(-1, true).Log())
}

// Softmax computes softmax along the last dimension.
func (t *Tensor) Softmax() *Tensor { return t.LogSoftmax().Exp() }

// Pick selects t[i, index[i]] from a 2-D tensor, returning a 1-D tensor. It is
// the building block for classification losses.
func (t *Tensor) Pick(index []int) *Tensor {
	if len(t.shape) != 2 || len(index) != t.shape[0] {
		panic(fmt.Sprintf("tensor: Pick needs a 2-D tensor and one index per row, got shape %v and %d indices", t.shape, len(index)))
	}
	c := t.shape[1]
	onehot := make([]float64, t.Numel())
	for i, j := range index {
		if j < 0 || j >= c {
			panic(fmt.Sprintf("tensor: Pick index %d out of range for %d columns", j, c))
		}
		onehot[i*c+j] = 1
	}
	return t.Mul(fromHost(onehot, t.shape, t.DType(), t.Device())).SumDim(1, false)
}

// ---- broadcasting helpers ----

// broadcastShapes returns the shape that a and b broadcast to, following the
// NumPy/PyTorch rules.
func broadcastShapes(a, b []int) []int {
	n := max(len(a), len(b))
	out := make([]int, n)
	for i := 0; i < n; i++ {
		da, db := 1, 1
		if j := len(a) - n + i; j >= 0 {
			da = a[j]
		}
		if j := len(b) - n + i; j >= 0 {
			db = b[j]
		}
		switch {
		case da == db:
			out[i] = da
		case da == 1:
			out[i] = db
		case db == 1:
			out[i] = da
		default:
			panic(fmt.Sprintf("tensor: shapes %v and %v are not broadcastable", a, b))
		}
	}
	if n > backend.MaxDims {
		panic(fmt.Sprintf("tensor: at most %d dimensions are supported, got %d", backend.MaxDims, n))
	}
	return out
}

// broadcastStrides returns strides for reading a tensor of the given shape as
// if it had the target shape; broadcast dimensions get stride 0.
func broadcastStrides(shape, target []int) []int {
	st := make([]int, len(target))
	off := len(target) - len(shape)
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		if shape[i] != 1 {
			st[i+off] = s
		}
		s *= shape[i]
	}
	return st
}

// expand broadcasts t to shape. It is not recorded.
func expand(t *Tensor, shape []int) *Tensor {
	if equalShape(t.shape, shape) {
		return t
	}
	return fromStorage(t.be().Copy(t.st, backend.View{Shape: shape, Strides: broadcastStrides(t.shape, shape)}), shape)
}

// sumTo reduces a broadcast gradient back to shape by summing over the
// broadcast dimensions. It is not recorded.
func sumTo(g *Tensor, shape []int) *Tensor {
	if equalShape(g.shape, shape) {
		return g
	}
	b := g.be()
	if lead := len(g.shape) - len(shape); lead > 0 {
		rest := g.shape[lead:]
		g = fromStorage(b.Reduce(backend.Sum, g.st, 1, numel(g.shape[:lead]), numel(rest)), rest)
	}
	for d := range shape {
		if shape[d] == 1 && g.shape[d] != 1 {
			g = g.reduce(backend.Sum, d, true)
		}
	}
	return fromStorage(g.st, shape)
}
