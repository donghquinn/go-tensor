package cpu

import (
	"math"
	"runtime"
	"sync"

	"github.com/donghquinn/go-tensor/internal/backend"
)

// parallelThreshold is the number of multiply-adds above which matmul splits
// rows across goroutines.
const parallelThreshold = 1 << 16

func (Backend) MatMul(a, b backend.Storage, m, k, n int, transA, transB bool) backend.Storage {
	return apply(a, func(x []float32) backend.Storage { return wrap(matmulT(x, f32(b), m, k, n, transA, transB)) },
		func(x []float64) backend.Storage { return wrap(matmulT(x, f64(b), m, k, n, transA, transB)) })
}

func matmulT[T float](a, b []T, m, k, n int, transA, transB bool) []T {
	if transA {
		a = transpose2d(a, k, m)
	}
	if transB {
		b = transpose2d(b, n, k)
	}
	return matmul(a, b, m, k, n)
}

// matmul computes the (m, n) product of row-major matrices a (m, k) and b (k, n).
// The i-k-j loop order keeps the inner loop streaming over contiguous memory.
func matmul[T float](a, b []T, m, k, n int) []T {
	c := make([]T, m*n)
	rows := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			ci := c[i*n : (i+1)*n]
			ai := a[i*k : (i+1)*k]
			p := 0
			// Fold four rows of b into ci per pass to cut loads and stores of ci.
			for ; p+4 <= k; p += 4 {
				a0, a1, a2, a3 := ai[p], ai[p+1], ai[p+2], ai[p+3]
				b0 := b[p*n : (p+1)*n]
				b1 := b[(p+1)*n : (p+2)*n][:len(b0)]
				b2 := b[(p+2)*n : (p+3)*n][:len(b0)]
				b3 := b[(p+3)*n : (p+4)*n][:len(b0)]
				ci := ci[:len(b0)]
				for j, v := range b0 {
					ci[j] += a0*v + a1*b1[j] + a2*b2[j] + a3*b3[j]
				}
			}
			for ; p < k; p++ {
				aip := ai[p]
				bp := b[p*n : (p+1)*n]
				for j := range ci {
					ci[j] += aip * bp[j]
				}
			}
		}
	}
	parallelRows(m, m*k*n, rows)
	return c
}

// parallelRows runs fn over [0, m) in chunks on all CPUs when work is large.
func parallelRows(m, work int, fn func(lo, hi int)) {
	workers := min(runtime.GOMAXPROCS(0), m)
	if work < parallelThreshold || workers <= 1 {
		fn(0, m)
		return
	}
	chunk := (m + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < m; lo += chunk {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, min(lo+chunk, m))
	}
	wg.Wait()
}

// transpose2d returns the transpose of the row-major (r, c) matrix a.
func transpose2d[T float](a []T, r, c int) []T {
	out := make([]T, len(a))
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out[j*r+i] = a[i*c+j]
		}
	}
	return out
}

func (Backend) Im2col(x backend.Storage, g backend.ConvGeom) backend.Storage {
	return apply(x, func(x []float32) backend.Storage { return wrap(im2col(x, g)) },
		func(x []float64) backend.Storage { return wrap(im2col(x, g)) })
}

func (Backend) Col2im(cols backend.Storage, g backend.ConvGeom) backend.Storage {
	return apply(cols, func(c []float32) backend.Storage { return wrap(col2im(c, g)) },
		func(c []float64) backend.Storage { return wrap(col2im(c, g)) })
}

// im2col unrolls every receptive field of x into a row, giving a matrix of
// shape (N*OH*OW, C*KH*KW). Padded positions are zero.
func im2col[T float](x []T, g backend.ConvGeom) []T {
	k := g.C * g.KH * g.KW
	cols := make([]T, g.N*g.OH*g.OW*k)
	eachPatch(g, func(col, xi int) { cols[col] = x[xi] })
	return cols
}

// col2im sums each column entry back into the input position it came from.
func col2im[T float](cols []T, g backend.ConvGeom) []T {
	x := make([]T, g.N*g.C*g.H*g.W)
	eachPatch(g, func(col, xi int) { x[xi] += cols[col] })
	return x
}

// eachPatch calls fn with the flat im2col index and the flat input index for
// every in-bounds element of every receptive field.
func eachPatch(g backend.ConvGeom, fn func(colIdx, xIdx int)) {
	k := g.C * g.KH * g.KW
	row := 0
	for n := 0; n < g.N; n++ {
		for oh := 0; oh < g.OH; oh++ {
			for ow := 0; ow < g.OW; ow++ {
				idx := row * k
				for c := 0; c < g.C; c++ {
					plane := (n*g.C + c) * g.H
					for kh := 0; kh < g.KH; kh++ {
						ih := oh*g.Stride - g.Padding + kh
						for kw := 0; kw < g.KW; kw++ {
							iw := ow*g.Stride - g.Padding + kw
							if ih >= 0 && ih < g.H && iw >= 0 && iw < g.W {
								fn(idx, (plane+ih)*g.W+iw)
							}
							idx++
						}
					}
				}
				row++
			}
		}
	}
}

func (Backend) MaxPool2d(x backend.Storage, g backend.ConvGeom) backend.Storage {
	return apply(x, func(x []float32) backend.Storage { return wrap(maxPool(x, nil, g)) },
		func(x []float64) backend.Storage { return wrap(maxPool(x, nil, g)) })
}

func (Backend) MaxPool2dBackward(x, grad backend.Storage, g backend.ConvGeom) backend.Storage {
	return apply(x, func(x []float32) backend.Storage { return wrap(maxPool(x, f32(grad), g)) },
		func(x []float64) backend.Storage { return wrap(maxPool(x, f64(grad), g)) })
}

// maxPool computes the forward pass when grad is nil. Otherwise it routes each
// window's gradient to the first position holding the window's maximum.
func maxPool[T float](x, grad []T, g backend.ConvGeom) []T {
	planes := g.N * g.C
	var out []T
	if grad == nil {
		out = make([]T, planes*g.OH*g.OW)
	} else {
		out = make([]T, len(x))
	}
	i := 0
	for pl := 0; pl < planes; pl++ {
		base := pl * g.H * g.W
		for oh := 0; oh < g.OH; oh++ {
			for ow := 0; ow < g.OW; ow++ {
				best, bestIdx := T(math.Inf(-1)), -1
				for kh := 0; kh < g.KH; kh++ {
					for kw := 0; kw < g.KW; kw++ {
						idx := base + (oh*g.Stride+kh)*g.W + ow*g.Stride + kw
						if v := x[idx]; v > best || bestIdx < 0 {
							best, bestIdx = v, idx
						}
					}
				}
				if grad == nil {
					out[i] = best
				} else {
					out[bestIdx] += grad[i]
				}
				i++
			}
		}
	}
	return out
}
