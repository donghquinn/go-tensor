package tensor

import (
	"fmt"
	"math"
)

// ---- element-wise binary ops (with broadcasting) ----

// Add returns t + o.
func (t *Tensor) Add(o *Tensor) *Tensor {
	data, shape := broadcastBinary(t, o, func(x, y float64) float64 { return x + y })
	return newResult(data, shape, "Add", []*Tensor{t, o}, func(g *Tensor) []*Tensor {
		return []*Tensor{sumTo(g, t.shape), sumTo(g, o.shape)}
	})
}

// Sub returns t - o.
func (t *Tensor) Sub(o *Tensor) *Tensor {
	data, shape := broadcastBinary(t, o, func(x, y float64) float64 { return x - y })
	return newResult(data, shape, "Sub", []*Tensor{t, o}, func(g *Tensor) []*Tensor {
		return []*Tensor{sumTo(g, t.shape), sumTo(g.Neg(), o.shape)}
	})
}

// Mul returns the element-wise product t * o.
func (t *Tensor) Mul(o *Tensor) *Tensor {
	data, shape := broadcastBinary(t, o, func(x, y float64) float64 { return x * y })
	return newResult(data, shape, "Mul", []*Tensor{t, o}, func(g *Tensor) []*Tensor {
		return []*Tensor{sumTo(g.Mul(o), t.shape), sumTo(g.Mul(t), o.shape)}
	})
}

// Div returns the element-wise quotient t / o.
func (t *Tensor) Div(o *Tensor) *Tensor {
	data, shape := broadcastBinary(t, o, func(x, y float64) float64 { return x / y })
	return newResult(data, shape, "Div", []*Tensor{t, o}, func(g *Tensor) []*Tensor {
		gt := g.Div(o)
		go_ := gt.Mul(t).Div(o).Neg()
		return []*Tensor{sumTo(gt, t.shape), sumTo(go_, o.shape)}
	})
}

// ---- element-wise unary ops ----

// unary applies f element-wise. df returns dy/dx given the input x and output y.
func (t *Tensor) unary(op string, f func(x float64) float64, df func(x, y float64) float64) *Tensor {
	out := make([]float64, len(t.data))
	for i, x := range t.data {
		out[i] = f(x)
	}
	return newResult(out, t.shape, op, []*Tensor{t}, func(g *Tensor) []*Tensor {
		d := make([]float64, len(g.data))
		for i := range d {
			d[i] = g.data[i] * df(t.data[i], out[i])
		}
		return []*Tensor{fromData(d, t.shape)}
	})
}

// Neg returns -t.
func (t *Tensor) Neg() *Tensor {
	return t.unary("Neg", func(x float64) float64 { return -x }, func(_, _ float64) float64 { return -1 })
}

// AddScalar returns t + s.
func (t *Tensor) AddScalar(s float64) *Tensor {
	return t.unary("AddScalar", func(x float64) float64 { return x + s }, func(_, _ float64) float64 { return 1 })
}

// MulScalar returns t * s.
func (t *Tensor) MulScalar(s float64) *Tensor {
	return t.unary("MulScalar", func(x float64) float64 { return x * s }, func(_, _ float64) float64 { return s })
}

// Pow returns t raised element-wise to the power p.
func (t *Tensor) Pow(p float64) *Tensor {
	return t.unary("Pow", func(x float64) float64 { return math.Pow(x, p) },
		func(x, _ float64) float64 { return p * math.Pow(x, p-1) })
}

// Exp returns e^t.
func (t *Tensor) Exp() *Tensor {
	return t.unary("Exp", math.Exp, func(_, y float64) float64 { return y })
}

// Log returns the natural logarithm of t.
func (t *Tensor) Log() *Tensor {
	return t.unary("Log", math.Log, func(x, _ float64) float64 { return 1 / x })
}

// Sqrt returns the square root of t.
func (t *Tensor) Sqrt() *Tensor {
	return t.unary("Sqrt", math.Sqrt, func(_, y float64) float64 { return 0.5 / y })
}

// Abs returns |t|.
func (t *Tensor) Abs() *Tensor {
	return t.unary("Abs", math.Abs, func(x, _ float64) float64 {
		switch {
		case x > 0:
			return 1
		case x < 0:
			return -1
		}
		return 0
	})
}

// ReLU returns max(t, 0).
func (t *Tensor) ReLU() *Tensor {
	return t.unary("ReLU", func(x float64) float64 { return math.Max(x, 0) }, func(x, _ float64) float64 {
		if x > 0 {
			return 1
		}
		return 0
	})
}

// Sigmoid returns 1 / (1 + e^-t).
func (t *Tensor) Sigmoid() *Tensor {
	return t.unary("Sigmoid", func(x float64) float64 { return 1 / (1 + math.Exp(-x)) },
		func(_, y float64) float64 { return y * (1 - y) })
}

// Tanh returns the hyperbolic tangent of t.
func (t *Tensor) Tanh() *Tensor {
	return t.unary("Tanh", math.Tanh, func(_, y float64) float64 { return 1 - y*y })
}

// ---- reductions ----

// Sum returns the sum of all elements as a scalar tensor.
func (t *Tensor) Sum() *Tensor {
	s := 0.0
	for _, x := range t.data {
		s += x
	}
	return newResult([]float64{s}, nil, "Sum", []*Tensor{t}, func(g *Tensor) []*Tensor {
		return []*Tensor{Full(g.data[0], t.shape...)}
	})
}

// Mean returns the mean of all elements as a scalar tensor.
func (t *Tensor) Mean() *Tensor {
	return t.Sum().MulScalar(1 / float64(len(t.data)))
}

// SumDim sums over dimension dim. If keepDim is true the reduced dimension is
// kept with size 1.
func (t *Tensor) SumDim(dim int, keepDim bool) *Tensor {
	dim = normDim(dim, len(t.shape))
	outer, n, inner := numel(t.shape[:dim]), t.shape[dim], numel(t.shape[dim+1:])
	out := make([]float64, outer*inner)
	for o := 0; o < outer; o++ {
		for k := 0; k < n; k++ {
			src := t.data[(o*n+k)*inner : (o*n+k+1)*inner]
			dst := out[o*inner : (o+1)*inner]
			for i, x := range src {
				dst[i] += x
			}
		}
	}
	kept := append([]int(nil), t.shape...)
	kept[dim] = 1
	shape := kept
	if !keepDim {
		shape = append(append([]int(nil), t.shape[:dim]...), t.shape[dim+1:]...)
	}
	return newResult(out, shape, "SumDim", []*Tensor{t}, func(g *Tensor) []*Tensor {
		return []*Tensor{expand(fromData(g.data, kept), t.shape)}
	})
}

// MeanDim averages over dimension dim.
func (t *Tensor) MeanDim(dim int, keepDim bool) *Tensor {
	n := t.shape[normDim(dim, len(t.shape))]
	return t.SumDim(dim, keepDim).MulScalar(1 / float64(n))
}

// Argmax returns the index of the largest value along the last dimension for
// every row. It is not differentiable.
func (t *Tensor) Argmax() []int {
	n := t.shape[len(t.shape)-1]
	rows := len(t.data) / n
	idx := make([]int, rows)
	for r := 0; r < rows; r++ {
		row := t.data[r*n : (r+1)*n]
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
		shape[infer] = len(t.data) / known
	}
	if numel(shape) != len(t.data) {
		panic(fmt.Sprintf("tensor: cannot reshape %v into %v", t.shape, shape))
	}
	return newResult(t.data, shape, "Reshape", []*Tensor{t}, func(g *Tensor) []*Tensor {
		return []*Tensor{fromData(g.data, t.shape)}
	})
}

// Flatten reshapes t into a 2-D tensor (batch, features), keeping dim 0.
func (t *Tensor) Flatten() *Tensor { return t.Reshape(t.shape[0], -1) }

// T swaps the last two dimensions.
func (t *Tensor) T() *Tensor {
	nd := len(t.shape)
	if nd < 2 {
		panic(fmt.Sprintf("tensor: T requires at least 2 dimensions, got %v", t.shape))
	}
	r, c := t.shape[nd-2], t.shape[nd-1]
	batch := len(t.data) / (r * c)
	out := make([]float64, len(t.data))
	for b := 0; b < batch; b++ {
		src, dst := t.data[b*r*c:(b+1)*r*c], out[b*r*c:(b+1)*r*c]
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				dst[j*r+i] = src[i*c+j]
			}
		}
	}
	shape := append([]int(nil), t.shape...)
	shape[nd-2], shape[nd-1] = c, r
	return newResult(out, shape, "T", []*Tensor{t}, func(g *Tensor) []*Tensor {
		return []*Tensor{g.T()}
	})
}

// ---- linear algebra ----

// MatMul returns the matrix product of two 2-D tensors.
func (t *Tensor) MatMul(o *Tensor) *Tensor {
	if len(t.shape) != 2 || len(o.shape) != 2 || t.shape[1] != o.shape[0] {
		panic(fmt.Sprintf("tensor: MatMul shape mismatch %v @ %v", t.shape, o.shape))
	}
	m, k, n := t.shape[0], t.shape[1], o.shape[1]
	out := matmul(t.data, o.data, m, k, n)
	return newResult(out, []int{m, n}, "MatMul", []*Tensor{t, o}, func(g *Tensor) []*Tensor {
		var gt, go_ *Tensor
		if t.requiresGrad {
			gt = g.MatMul(o.T())
		}
		if o.requiresGrad {
			go_ = t.T().MatMul(g)
		}
		return []*Tensor{gt, go_}
	})
}

// ---- neural-network helpers ----

// LogSoftmax computes log(softmax(t)) along the last dimension in a
// numerically stable way.
func (t *Tensor) LogSoftmax() *Tensor {
	n := t.shape[len(t.shape)-1]
	rows := len(t.data) / n
	out := make([]float64, len(t.data))
	for r := 0; r < rows; r++ {
		row := t.data[r*n : (r+1)*n]
		m := math.Inf(-1)
		for _, x := range row {
			m = math.Max(m, x)
		}
		s := 0.0
		for _, x := range row {
			s += math.Exp(x - m)
		}
		lse := m + math.Log(s)
		for j, x := range row {
			out[r*n+j] = x - lse
		}
	}
	return newResult(out, t.shape, "LogSoftmax", []*Tensor{t}, func(g *Tensor) []*Tensor {
		d := make([]float64, len(g.data))
		for r := 0; r < rows; r++ {
			gs := 0.0
			for j := 0; j < n; j++ {
				gs += g.data[r*n+j]
			}
			for j := 0; j < n; j++ {
				d[r*n+j] = g.data[r*n+j] - math.Exp(out[r*n+j])*gs
			}
		}
		return []*Tensor{fromData(d, t.shape)}
	})
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
	out := make([]float64, len(index))
	for i, j := range index {
		if j < 0 || j >= c {
			panic(fmt.Sprintf("tensor: Pick index %d out of range for %d columns", j, c))
		}
		out[i] = t.data[i*c+j]
	}
	return newResult(out, []int{len(index)}, "Pick", []*Tensor{t}, func(g *Tensor) []*Tensor {
		d := make([]float64, len(t.data))
		for i, j := range index {
			d[i*c+j] = g.data[i]
		}
		return []*Tensor{fromData(d, t.shape)}
	})
}

// Index returns t[i], selecting along the first dimension. The result shares
// memory with t.
func (t *Tensor) Index(i int) *Tensor {
	if len(t.shape) == 0 || i < 0 || i >= t.shape[0] {
		panic(fmt.Sprintf("tensor: Index %d out of range for shape %v", i, t.shape))
	}
	sz := len(t.data) / t.shape[0]
	return newResult(t.data[i*sz:(i+1)*sz], t.shape[1:], "Index", []*Tensor{t}, func(g *Tensor) []*Tensor {
		d := make([]float64, len(t.data))
		copy(d[i*sz:], g.data)
		return []*Tensor{fromData(d, t.shape)}
	})
}

// Stack joins tensors of identical shape along a new first dimension.
func Stack(ts []*Tensor) *Tensor {
	if len(ts) == 0 {
		panic("tensor: Stack needs at least one tensor")
	}
	inner := ts[0].shape
	sz := numel(inner)
	out := make([]float64, 0, len(ts)*sz)
	for _, t := range ts {
		if !equalShape(t.shape, inner) {
			panic(fmt.Sprintf("tensor: Stack shape mismatch %v vs %v", t.shape, inner))
		}
		out = append(out, t.data...)
	}
	shape := append([]int{len(ts)}, inner...)
	return newResult(out, shape, "Stack", ts, func(g *Tensor) []*Tensor {
		gs := make([]*Tensor, len(ts))
		for i := range gs {
			gs[i] = fromData(g.data[i*sz:(i+1)*sz], inner)
		}
		return gs
	})
}
