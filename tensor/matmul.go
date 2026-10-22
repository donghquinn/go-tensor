package tensor

import (
	"runtime"
	"sync"
)

// parallelThreshold is the number of multiply-adds above which matmul splits
// rows across goroutines.
const parallelThreshold = 1 << 16

// matmul computes the (m, n) product of row-major matrices a (m, k) and b (k, n).
// The i-k-j loop order keeps the inner loop streaming over contiguous memory.
func matmul(a, b []float64, m, k, n int) []float64 {
	c := make([]float64, m*n)
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

	workers := min(runtime.GOMAXPROCS(0), m)
	if m*k*n < parallelThreshold || workers <= 1 {
		rows(0, m)
		return c
	}
	chunk := (m + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < m; lo += chunk {
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			rows(lo, hi)
		}(lo, min(lo+chunk, m))
	}
	wg.Wait()
	return c
}

// transpose2d returns the transpose of the row-major (r, c) matrix a.
func transpose2d(a []float64, r, c int) []float64 {
	out := make([]float64, len(a))
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out[j*r+i] = a[i*c+j]
		}
	}
	return out
}
