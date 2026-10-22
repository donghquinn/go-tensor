// Package cpu is the pure-Go backend. Every primitive is written once,
// generically over float32 and float64.
package cpu

import (
	"fmt"
	"math"

	"github.com/donghquinn/go-tensor/internal/backend"
)

type float interface{ ~float32 | ~float64 }

type storage[T float] struct{ data []T }

func (s *storage[T]) Device() backend.Device { return backend.Device{Type: backend.CPU} }
func (s *storage[T]) Len() int               { return len(s.data) }
func (s *storage[T]) DType() backend.DType {
	var zero T
	if _, ok := any(zero).(float64); ok {
		return backend.Float64
	}
	return backend.Float32
}

// Backend is the CPU implementation of backend.Backend.
type Backend struct{}

func (Backend) Device() backend.Device { return backend.Device{Type: backend.CPU} }

func f32(s backend.Storage) []float32 { return s.(*storage[float32]).data }
func f64(s backend.Storage) []float64 { return s.(*storage[float64]).data }

func wrap[T float](data []T) backend.Storage { return &storage[T]{data} }

// apply runs the float32 or float64 instantiation of a generic kernel,
// depending on the dtype of s.
func apply(s backend.Storage, k32 func([]float32) backend.Storage, k64 func([]float64) backend.Storage) backend.Storage {
	switch s := s.(type) {
	case *storage[float32]:
		return k32(s.data)
	case *storage[float64]:
		return k64(s.data)
	}
	panic(fmt.Sprintf("cpu: foreign storage %T", s))
}

func (Backend) Zeros(dt backend.DType, n int) backend.Storage {
	if dt == backend.Float64 {
		return wrap(make([]float64, n))
	}
	return wrap(make([]float32, n))
}

func (Backend) FromHost(data any) backend.Storage {
	switch d := data.(type) {
	case []float32:
		return wrap(append([]float32(nil), d...))
	case []float64:
		return wrap(append([]float64(nil), d...))
	}
	panic(fmt.Sprintf("cpu: FromHost(%T)", data))
}

func (Backend) ToHost(s backend.Storage) any {
	switch s := s.(type) {
	case *storage[float32]:
		return append([]float32(nil), s.data...)
	case *storage[float64]:
		return append([]float64(nil), s.data...)
	}
	panic(fmt.Sprintf("cpu: foreign storage %T", s))
}

func (Backend) Cast(s backend.Storage, dt backend.DType) backend.Storage {
	if dt == backend.Float64 {
		return apply(s, func(x []float32) backend.Storage { return wrap(convert[float32, float64](x)) },
			func(x []float64) backend.Storage { return wrap(append([]float64(nil), x...)) })
	}
	return apply(s, func(x []float32) backend.Storage { return wrap(append([]float32(nil), x...)) },
		func(x []float64) backend.Storage { return wrap(convert[float64, float32](x)) })
}

func convert[S, D float](x []S) []D {
	out := make([]D, len(x))
	for i, v := range x {
		out[i] = D(v)
	}
	return out
}

func (Backend) Write(dst backend.Storage, offset int, src backend.Storage) {
	switch d := dst.(type) {
	case *storage[float32]:
		copy(d.data[offset:], f32(src))
	case *storage[float64]:
		copy(d.data[offset:], f64(src))
	}
}

func (Backend) Sync() {}

func (Backend) Unary(op backend.UnaryOp, s float64, x backend.Storage) backend.Storage {
	return apply(x, func(x []float32) backend.Storage { return wrap(unary(op, s, x)) },
		func(x []float64) backend.Storage { return wrap(unary(op, s, x)) })
}

func unary[T float](op backend.UnaryOp, s float64, x []T) []T {
	out := make([]T, len(x))
	mathf := func(f func(float64) float64) {
		for i, v := range x {
			out[i] = T(f(float64(v)))
		}
	}
	switch op {
	case backend.Neg:
		for i, v := range x {
			out[i] = -v
		}
	case backend.Exp:
		mathf(math.Exp)
	case backend.Log:
		mathf(math.Log)
	case backend.Sqrt:
		mathf(math.Sqrt)
	case backend.Abs:
		mathf(math.Abs)
	case backend.ReLU:
		for i, v := range x {
			out[i] = max(v, 0)
		}
	case backend.Sigmoid:
		mathf(func(v float64) float64 { return 1 / (1 + math.Exp(-v)) })
	case backend.Tanh:
		mathf(math.Tanh)
	case backend.Sign:
		for i, v := range x {
			switch {
			case v > 0:
				out[i] = 1
			case v < 0:
				out[i] = -1
			}
		}
	case backend.Step:
		for i, v := range x {
			if v > 0 {
				out[i] = 1
			}
		}
	case backend.AddScalar:
		c := T(s)
		for i, v := range x {
			out[i] = v + c
		}
	case backend.MulScalar:
		c := T(s)
		for i, v := range x {
			out[i] = v * c
		}
	case backend.Pow:
		mathf(func(v float64) float64 { return math.Pow(v, s) })
	default:
		panic(fmt.Sprintf("cpu: unknown unary op %d", op))
	}
	return out
}

func (Backend) Binary(op backend.BinaryOp, a backend.Storage, av backend.View, b backend.Storage, bv backend.View) backend.Storage {
	return apply(a, func(x []float32) backend.Storage { return wrap(binary(op, x, av, f32(b), bv)) },
		func(x []float64) backend.Storage { return wrap(binary(op, x, av, f64(b), bv)) })
}

func binaryFunc[T float](op backend.BinaryOp) func(x, y T) T {
	switch op {
	case backend.Add:
		return func(x, y T) T { return x + y }
	case backend.Sub:
		return func(x, y T) T { return x - y }
	case backend.Mul:
		return func(x, y T) T { return x * y }
	case backend.Div:
		return func(x, y T) T { return x / y }
	}
	panic(fmt.Sprintf("cpu: unknown binary op %d", op))
}

func binary[T float](op backend.BinaryOp, a []T, av backend.View, b []T, bv backend.View) []T {
	shape := av.Shape
	out := make([]T, backend.Numel(shape))
	if contiguous(av) && contiguous(bv) {
		a, b := a[:len(out)], b[:len(out)]
		switch op { // the common case, without a function call per element
		case backend.Add:
			for i := range out {
				out[i] = a[i] + b[i]
			}
		case backend.Sub:
			for i := range out {
				out[i] = a[i] - b[i]
			}
		case backend.Mul:
			for i := range out {
				out[i] = a[i] * b[i]
			}
		case backend.Div:
			for i := range out {
				out[i] = a[i] / b[i]
			}
		}
		return out
	}
	f := binaryFunc[T](op)
	idx := make([]int, len(shape))
	ia, ib := av.Offset, bv.Offset
	for k := range out {
		out[k] = f(a[ia], b[ib])
		for d := len(shape) - 1; d >= 0; d-- {
			idx[d]++
			ia += av.Strides[d]
			ib += bv.Strides[d]
			if idx[d] < shape[d] {
				break
			}
			ia -= av.Strides[d] * shape[d]
			ib -= bv.Strides[d] * shape[d]
			idx[d] = 0
		}
	}
	return out
}

func contiguous(v backend.View) bool {
	if v.Offset != 0 {
		return false
	}
	want := backend.ContiguousStrides(v.Shape)
	for d := range want {
		if v.Shape[d] != 1 && v.Strides[d] != want[d] {
			return false
		}
	}
	return true
}

func (Backend) Copy(src backend.Storage, v backend.View) backend.Storage {
	return apply(src, func(x []float32) backend.Storage { return wrap(gather(x, v)) },
		func(x []float64) backend.Storage { return wrap(gather(x, v)) })
}

func gather[T float](src []T, v backend.View) []T {
	out := make([]T, backend.Numel(v.Shape))
	idx := make([]int, len(v.Shape))
	off := v.Offset
	for k := range out {
		out[k] = src[off]
		for d := len(v.Shape) - 1; d >= 0; d-- {
			idx[d]++
			off += v.Strides[d]
			if idx[d] < v.Shape[d] {
				break
			}
			off -= v.Strides[d] * v.Shape[d]
			idx[d] = 0
		}
	}
	return out
}

func (Backend) Reduce(op backend.ReduceOp, x backend.Storage, outer, n, inner int) backend.Storage {
	return apply(x, func(x []float32) backend.Storage { return wrap(reduce(op, x, outer, n, inner)) },
		func(x []float64) backend.Storage { return wrap(reduce(op, x, outer, n, inner)) })
}

func reduce[T float](op backend.ReduceOp, x []T, outer, n, inner int) []T {
	out := make([]T, outer*inner)
	for o := 0; o < outer; o++ {
		dst := out[o*inner : (o+1)*inner]
		if op == backend.Max {
			copy(dst, x[o*n*inner:(o*n+1)*inner])
		}
		for k := 0; k < n; k++ {
			src := x[(o*n+k)*inner : (o*n+k+1)*inner]
			if op == backend.Max {
				for i, v := range src {
					dst[i] = max(dst[i], v)
				}
			} else {
				for i, v := range src {
					dst[i] += v
				}
			}
		}
	}
	return out
}
