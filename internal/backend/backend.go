// Package backend defines the contract between the tensor package and the
// compute devices (CPU, MPS, CUDA). The tensor package expresses every
// operation, including gradients, in terms of these primitives, so a device
// only has to implement this interface to support training and inference.
package backend

import "fmt"

// DType is the element type of a tensor.
type DType int

const (
	Float32 DType = iota
	Float64
)

func (d DType) String() string {
	switch d {
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	}
	return fmt.Sprintf("DType(%d)", int(d))
}

// Size returns the number of bytes per element.
func (d DType) Size() int {
	if d == Float64 {
		return 8
	}
	return 4
}

// DeviceType identifies a kind of compute device.
type DeviceType int

const (
	CPU DeviceType = iota
	CUDA
	MPS
)

// Device is a compute device, like torch.device.
type Device struct {
	Type  DeviceType
	Index int // device ordinal for CUDA; always 0 otherwise
}

func (d Device) String() string {
	switch d.Type {
	case CPU:
		return "cpu"
	case MPS:
		return "mps"
	case CUDA:
		return fmt.Sprintf("cuda:%d", d.Index)
	}
	return fmt.Sprintf("device(%d:%d)", int(d.Type), d.Index)
}

// Storage is a flat, contiguous buffer of Len elements of one dtype living
// on one device. Storages are immutable once returned by an operation, except
// through Backend.Write.
type Storage interface {
	Device() Device
	DType() DType
	Len() int
}

// UnaryOp is an element-wise function of one input and an optional scalar.
type UnaryOp int

const (
	Neg UnaryOp = iota
	Exp
	Log
	Sqrt
	Abs
	ReLU
	Sigmoid
	Tanh
	Sign      // -1, 0 or 1
	Step      // 1 if x > 0, else 0
	AddScalar // x + s
	MulScalar // x * s
	Pow       // x ** s
)

// BinaryOp is an element-wise function of two inputs.
type BinaryOp int

const (
	Add BinaryOp = iota
	Sub
	Mul
	Div
)

// ReduceOp combines values along one dimension.
type ReduceOp int

const (
	Sum ReduceOp = iota
	Max
)

// MaxDims is the largest number of dimensions a strided view may have.
const MaxDims = 8

// View reads a Storage as a strided array: element idx (row-major over
// Shape) is at Offset + Σ idx[d]*Strides[d]. Stride 0 broadcasts.
type View struct {
	Shape   []int
	Strides []int
	Offset  int
}

// ConvGeom describes a 2-D sliding window over a (N, C, H, W) input.
type ConvGeom struct {
	N, C, H, W      int
	KH, KW          int
	OH, OW          int
	Stride, Padding int
}

// Backend implements the primitives for one device. All Storage arguments
// belong to this backend and share the same dtype; every operation returns a
// new contiguous Storage of that dtype unless stated otherwise.
type Backend interface {
	Device() Device

	// Memory.
	Zeros(dt DType, n int) Storage
	FromHost(data any) Storage // data is []float32 or []float64
	ToHost(s Storage) any      // []float32 or []float64, a copy
	Cast(s Storage, dt DType) Storage
	Write(dst Storage, offset int, src Storage) // dst[offset:offset+src.Len()] = src
	Sync()                                      // wait for queued work

	// Element-wise.
	Unary(op UnaryOp, scalar float64, x Storage) Storage
	Binary(op BinaryOp, a Storage, av View, b Storage, bv View) Storage // over av.Shape
	Copy(src Storage, v View) Storage                                   // gather a strided view

	// Reductions over the middle axis of an (outer, n, inner) array.
	Reduce(op ReduceOp, x Storage, outer, n, inner int) Storage

	// Linear algebra: (m, k) @ (k, n), with each input optionally stored transposed.
	MatMul(a, b Storage, m, k, n int, transA, transB bool) Storage

	// Convolution helpers.
	Im2col(x Storage, g ConvGeom) Storage                  // -> (N*OH*OW, C*KH*KW)
	Col2im(cols Storage, g ConvGeom) Storage               // adjoint of Im2col
	MaxPool2d(x Storage, g ConvGeom) Storage               // -> (N, C, OH, OW)
	MaxPool2dBackward(x, grad Storage, g ConvGeom) Storage // -> (N, C, H, W)
}

// ContiguousStrides returns row-major strides for shape.
func ContiguousStrides(shape []int) []int {
	st := make([]int, len(shape))
	s := 1
	for i := len(shape) - 1; i >= 0; i-- {
		st[i] = s
		s *= shape[i]
	}
	return st
}

// Numel returns the number of elements in shape.
func Numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}
