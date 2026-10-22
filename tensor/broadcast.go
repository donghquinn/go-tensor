package tensor

import "fmt"

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

// walk visits every element of shape in row-major order, passing the flat
// index and the corresponding offset under strides.
func walk(shape, strides []int, fn func(k, off int)) {
	n := numel(shape)
	idx := make([]int, len(shape))
	off := 0
	for k := 0; k < n; k++ {
		fn(k, off)
		for d := len(shape) - 1; d >= 0; d-- {
			idx[d]++
			off += strides[d]
			if idx[d] < shape[d] {
				break
			}
			off -= strides[d] * shape[d]
			idx[d] = 0
		}
	}
}

// broadcastBinary applies f element-wise to a and b after broadcasting.
func broadcastBinary(a, b *Tensor, f func(x, y float64) float64) ([]float64, []int) {
	if equalShape(a.shape, b.shape) {
		out := make([]float64, len(a.data))
		for i := range out {
			out[i] = f(a.data[i], b.data[i])
		}
		return out, a.shape
	}
	shape := broadcastShapes(a.shape, b.shape)
	sa := broadcastStrides(a.shape, shape)
	sb := broadcastStrides(b.shape, shape)
	out := make([]float64, numel(shape))
	idx := make([]int, len(shape))
	ia, ib := 0, 0
	for k := range out {
		out[k] = f(a.data[ia], b.data[ib])
		for d := len(shape) - 1; d >= 0; d-- {
			idx[d]++
			ia += sa[d]
			ib += sb[d]
			if idx[d] < shape[d] {
				break
			}
			ia -= sa[d] * shape[d]
			ib -= sb[d] * shape[d]
			idx[d] = 0
		}
	}
	return out, shape
}

// sumTo reduces a broadcast gradient back to shape by summing over the
// broadcast dimensions.
func sumTo(g *Tensor, shape []int) *Tensor {
	if equalShape(g.shape, shape) {
		return g
	}
	out := make([]float64, numel(shape))
	walk(g.shape, broadcastStrides(shape, g.shape), func(k, off int) {
		out[off] += g.data[k]
	})
	return fromData(out, shape)
}

// expand broadcasts t to shape, copying the data.
func expand(t *Tensor, shape []int) *Tensor {
	if equalShape(t.shape, shape) {
		return t
	}
	out := make([]float64, numel(shape))
	walk(shape, broadcastStrides(t.shape, shape), func(k, off int) {
		out[k] = t.data[off]
	})
	return fromData(out, shape)
}
