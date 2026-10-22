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
			for p := 0; p < k; p++ {
				aip := a[i*k+p]
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
