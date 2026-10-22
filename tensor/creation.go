package tensor

import (
	"math/rand"
	"sync"
)

var (
	rngMu sync.Mutex
	rng   = rand.New(rand.NewSource(rand.Int63()))
)

// ManualSeed seeds the generator used by Rand, Randn, Uniform and RandPerm,
// like torch.manual_seed. Random values are generated on the CPU and copied to
// the default device, so a seed gives the same values on every device.
func ManualSeed(seed int64) {
	rngMu.Lock()
	rng = rand.New(rand.NewSource(seed))
	rngMu.Unlock()
}

// Scalar creates a zero-dimensional tensor.
func Scalar(v float64) *Tensor { return fromHost([]float64{v}, nil, DefaultDType(), DefaultDevice()) }

// Zeros creates a tensor filled with 0.
func Zeros(shape ...int) *Tensor {
	return zeros(shape, DefaultDType(), DefaultDevice())
}

func zeros(shape []int, dt DType, dev Device) *Tensor {
	b := mustBackend(dev)
	checkDTypeSupported(b, dt)
	return fromStorage(b.Zeros(dt, numel(shape)), shape)
}

// Ones creates a tensor filled with 1.
func Ones(shape ...int) *Tensor { return Full(1, shape...) }

// Full creates a tensor filled with v.
func Full(v float64, shape ...int) *Tensor {
	return full(v, shape, DefaultDType(), DefaultDevice())
}

func full(v float64, shape []int, dt DType, dev Device) *Tensor {
	data := make([]float64, numel(shape))
	for i := range data {
		data[i] = v
	}
	return fromHost(data, shape, dt, dev)
}

// ZerosLike creates a zero tensor with the shape, dtype and device of t.
func ZerosLike(t *Tensor) *Tensor { return zeros(t.shape, t.DType(), t.Device()) }

// OnesLike creates a tensor of ones with the shape, dtype and device of t.
func OnesLike(t *Tensor) *Tensor { return full(1, t.shape, t.DType(), t.Device()) }

// Arange creates a 1-D tensor with values from start up to (excluding) end.
func Arange(start, end, step float64) *Tensor {
	if step == 0 {
		panic("tensor: Arange step must be non-zero")
	}
	var data []float64
	for v := start; (step > 0 && v < end) || (step < 0 && v > end); v += step {
		data = append(data, v)
	}
	return New(data, len(data))
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
	return fromHost(data, shape, DefaultDType(), DefaultDevice())
}

// Randn creates a tensor with values drawn from the standard normal distribution.
func Randn(shape ...int) *Tensor {
	data := make([]float64, numel(shape))
	rngMu.Lock()
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	rngMu.Unlock()
	return fromHost(data, shape, DefaultDType(), DefaultDevice())
}

// RandPerm returns a random permutation of 0..n-1, like torch.randperm.
func RandPerm(n int) []int {
	rngMu.Lock()
	defer rngMu.Unlock()
	return rng.Perm(n)
}
