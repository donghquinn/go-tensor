package tensor

import (
	"math/rand"
	"sync"
)

var (
	rngMu sync.Mutex
	rng   = rand.New(rand.NewSource(rand.Int63()))
)

// ManualSeed seeds the generator used by Rand, Randn and Uniform, like
// torch.manual_seed.
func ManualSeed(seed int64) {
	rngMu.Lock()
	rng = rand.New(rand.NewSource(seed))
	rngMu.Unlock()
}

// Scalar creates a zero-dimensional tensor.
func Scalar(v float64) *Tensor { return fromData([]float64{v}, nil) }

// Zeros creates a tensor filled with 0.
func Zeros(shape ...int) *Tensor { return fromData(make([]float64, numel(shape)), shape) }

// Ones creates a tensor filled with 1.
func Ones(shape ...int) *Tensor { return Full(1, shape...) }

// Full creates a tensor filled with v.
func Full(v float64, shape ...int) *Tensor {
	data := make([]float64, numel(shape))
	for i := range data {
		data[i] = v
	}
	return fromData(data, shape)
}

// ZerosLike creates a zero tensor with the same shape as t.
func ZerosLike(t *Tensor) *Tensor { return Zeros(t.shape...) }

// OnesLike creates a tensor of ones with the same shape as t.
func OnesLike(t *Tensor) *Tensor { return Ones(t.shape...) }

// Arange creates a 1-D tensor with values from start up to (excluding) end.
func Arange(start, end, step float64) *Tensor {
	if step == 0 {
		panic("tensor: Arange step must be non-zero")
	}
	var data []float64
	for v := start; (step > 0 && v < end) || (step < 0 && v > end); v += step {
		data = append(data, v)
	}
	return fromData(data, []int{len(data)})
}

// Rand creates a tensor with values drawn uniformly from [0, 1).
func Rand(shape ...int) *Tensor { return Uniform(0, 1, shape...) }

// Uniform creates a tensor with values drawn uniformly from [low, high).
func Uniform(low, high float64, shape ...int) *Tensor {
	data := make([]float64, numel(shape))
	rngMu.Lock()
	for i := range data {
		data[i] = low + (high-low)*rng.Float64()
	}
	rngMu.Unlock()
	return fromData(data, shape)
}

// Randn creates a tensor with values drawn from the standard normal distribution.
func Randn(shape ...int) *Tensor {
	data := make([]float64, numel(shape))
	rngMu.Lock()
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	rngMu.Unlock()
	return fromData(data, shape)
}

// RandPerm returns a random permutation of 0..n-1, like torch.randperm.
func RandPerm(n int) []int {
	rngMu.Lock()
	defer rngMu.Unlock()
	return rng.Perm(n)
}
