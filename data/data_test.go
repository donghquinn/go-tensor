package data_test

import (
	"sort"
	"testing"

	"github.com/donghquinn/go-tensor/data"
	"github.com/donghquinn/go-tensor/tensor"
)

func TestDataLoaderCoversEveryEpoch(t *testing.T) {
	// Sample i holds the values (i, -i), so batches can be checked against labels.
	const n = 10
	xs := make([]float64, 0, 2*n)
	ys := make([]int, n)
	for i := 0; i < n; i++ {
		xs = append(xs, float64(i), -float64(i))
		ys[i] = i
	}
	loader := data.NewDataLoader(data.NewTensorDataset(tensor.New(xs, n, 2), ys), 4, true)
	if got := loader.NumBatches(); got != 3 {
		t.Fatalf("NumBatches = %d, want 3", got)
	}

	for epoch := 0; epoch < 2; epoch++ {
		var seen []int
		var sizes []int
		for x, y := range loader.All() {
			sizes = append(sizes, len(y))
			for i, label := range y {
				if x.At(i, 0) != float64(label) || x.At(i, 1) != -float64(label) {
					t.Fatalf("epoch %d: sample %v does not match label %d", epoch, x.Index(i), label)
				}
			}
			seen = append(seen, y...)
		}
		sort.Ints(seen)
		for i, v := range seen {
			if v != i {
				t.Fatalf("epoch %d: saw labels %v, want each of 0..%d once", epoch, seen, n-1)
			}
		}
		if len(seen) != n || sizes[0] != 4 || sizes[2] != 2 {
			t.Fatalf("epoch %d: batch sizes %v", epoch, sizes)
		}
	}
}

func TestDataLoaderEarlyBreak(t *testing.T) {
	loader := data.NewDataLoader(data.NewTensorDataset(tensor.Zeros(10, 1), make([]int, 10)), 3, false)
	batches := 0
	for range loader.All() {
		batches++
		if batches == 2 {
			break
		}
	}
	if batches != 2 {
		t.Fatalf("got %d batches before break, want 2", batches)
	}
	// A fresh range starts a new epoch from the beginning.
	sizes := []int{}
	for _, y := range loader.All() {
		sizes = append(sizes, len(y))
	}
	if len(sizes) != 4 || sizes[3] != 1 {
		t.Fatalf("batch sizes %v, want [3 3 3 1]", sizes)
	}
}
