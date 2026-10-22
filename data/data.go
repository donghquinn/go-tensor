// Package data provides datasets and mini-batch loading, modeled after
// torch.utils.data.
package data

import (
	"fmt"
	"iter"

	"github.com/donghquinn/go-tensor/tensor"
)

// Dataset is an indexable collection of (sample, label) pairs.
type Dataset interface {
	Len() int
	Get(i int) (x *tensor.Tensor, y int)
}

// TensorDataset serves rows of X along the first dimension, paired with Y.
type TensorDataset struct {
	X *tensor.Tensor
	Y []int
}

// NewTensorDataset creates a dataset where sample i is x.Index(i) with label y[i].
func NewTensorDataset(x *tensor.Tensor, y []int) *TensorDataset {
	if x.Dim() == 0 || x.Shape()[0] != len(y) {
		panic(fmt.Sprintf("data: %d labels for tensor of shape %v", len(y), x.Shape()))
	}
	return &TensorDataset{X: x, Y: y}
}

func (d *TensorDataset) Len() int { return len(d.Y) }

func (d *TensorDataset) Get(i int) (*tensor.Tensor, int) { return d.X.Index(i), d.Y[i] }

// DataLoader iterates over a dataset in mini-batches, like
// torch.utils.data.DataLoader. Each range over All is one epoch:
//
//	for x, y := range loader.All() {
//		...
//	}
type DataLoader struct {
	ds        Dataset
	batchSize int
	shuffle   bool
}

// NewDataLoader creates a loader yielding batches of batchSize samples. When
// shuffle is true the order is reshuffled at the start of every epoch using
// the generator seeded by tensor.ManualSeed.
func NewDataLoader(ds Dataset, batchSize int, shuffle bool) *DataLoader {
	if batchSize <= 0 {
		panic("data: batch size must be positive")
	}
	return &DataLoader{ds: ds, batchSize: batchSize, shuffle: shuffle}
}

// All returns an iterator over one epoch of batches. Each batch holds the
// samples stacked along a new first dimension, and their labels.
func (l *DataLoader) All() iter.Seq2[*tensor.Tensor, []int] {
	return func(yield func(*tensor.Tensor, []int) bool) {
		n := l.ds.Len()
		var order []int
		if l.shuffle {
			order = tensor.RandPerm(n)
		} else {
			order = make([]int, n)
			for i := range order {
				order[i] = i
			}
		}
		for start := 0; start < n; start += l.batchSize {
			idx := order[start:min(start+l.batchSize, n)]
			xs := make([]*tensor.Tensor, len(idx))
			ys := make([]int, len(idx))
			for j, i := range idx {
				xs[j], ys[j] = l.ds.Get(i)
			}
			if !yield(tensor.Stack(xs), ys) {
				return
			}
		}
	}
}

// NumBatches returns the number of batches per epoch.
func (l *DataLoader) NumBatches() int {
	return (l.ds.Len() + l.batchSize - 1) / l.batchSize
}
