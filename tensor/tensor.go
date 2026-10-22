// Package tensor provides n-dimensional arrays with reverse-mode automatic
// differentiation that run on the CPU or a GPU, modeled after torch.Tensor.
//
// Every tensor has a DType (Float32 by default, or Float64) and a Device (CPU,
// MPS or CUDA). Operations run on the device of their inputs, and all inputs
// of an operation must share the same device and dtype. Operations that
// receive invalid shapes panic, the same way slices panic on bad indices.
package tensor

import (
	"fmt"
	"strings"

	"github.com/donghquinn/go-tensor/internal/backend"
)

// Tensor is an n-dimensional array that can record the operations applied to
// it so that gradients can be computed with Backward.
type Tensor struct {
	st           backend.Storage
	shape        []int
	requiresGrad bool
	grad         *Tensor
	ctx          *node // nil for leaf tensors
}

func fromStorage(st backend.Storage, shape []int) *Tensor {
	return &Tensor{st: st, shape: append([]int(nil), shape...)}
}

func (t *Tensor) be() backend.Backend { return mustBackend(t.st.Device()) }

// New creates a tensor from data with the given shape, using the default
// dtype and device. If no shape is given the tensor is one-dimensional.
func New(data []float64, shape ...int) *Tensor {
	return fromHost(data, checkShape(len(data), shape), DefaultDType(), DefaultDevice())
}

// FromFloat32 creates a float32 tensor on the CPU from a copy of data, like
// torch.from_numpy. If no shape is given the tensor is one-dimensional.
func FromFloat32(data []float32, shape ...int) *Tensor {
	shape = checkShape(len(data), shape)
	return fromStorage(mustBackend(CPU).FromHost(data), shape)
}

// FromFloat64 creates a float64 tensor on the CPU from a copy of data. If no
// shape is given the tensor is one-dimensional.
func FromFloat64(data []float64, shape ...int) *Tensor {
	shape = checkShape(len(data), shape)
	return fromStorage(mustBackend(CPU).FromHost(data), shape)
}

func checkShape(n int, shape []int) []int {
	if len(shape) == 0 {
		return []int{n}
	}
	if numel(shape) != n {
		panic(fmt.Sprintf("tensor: data length %d does not match shape %v (%d elements)", n, shape, numel(shape)))
	}
	return shape
}

// fromHost uploads float64 host data as a tensor of the given dtype and device.
func fromHost(data []float64, shape []int, dt DType, dev Device) *Tensor {
	b := mustBackend(dev)
	checkDTypeSupported(b, dt)
	var host any = data
	if dt == Float32 {
		f := make([]float32, len(data))
		for i, v := range data {
			f[i] = float32(v)
		}
		host = f
	}
	return fromStorage(b.FromHost(host), shape)
}

func checkDTypeSupported(b backend.Backend, dt DType) {
	if dt == Float64 && b.Device().Type == backend.MPS {
		panic("tensor: the MPS device does not support float64; convert with ToDType(tensor.Float32) first")
	}
}

// Shape returns a copy of the tensor's dimensions.
func (t *Tensor) Shape() []int { return append([]int(nil), t.shape...) }

// Dim returns the number of dimensions.
func (t *Tensor) Dim() int { return len(t.shape) }

// Numel returns the total number of elements.
func (t *Tensor) Numel() int { return t.st.Len() }

// DType returns the element type.
func (t *Tensor) DType() DType { return t.st.DType() }

// Device returns the device holding the tensor's data.
func (t *Tensor) Device() Device { return t.st.Device() }

// Float64s returns a copy of the values as float64, copying from the device
// if necessary.
func (t *Tensor) Float64s() []float64 {
	switch h := t.be().ToHost(t.st).(type) {
	case []float64:
		return h
	case []float32:
		out := make([]float64, len(h))
		for i, v := range h {
			out[i] = float64(v)
		}
		return out
	}
	panic("tensor: unexpected host data")
}

// Float32s returns a copy of the values as float32, copying from the device
// if necessary.
func (t *Tensor) Float32s() []float32 {
	switch h := t.be().ToHost(t.st).(type) {
	case []float32:
		return h
	case []float64:
		out := make([]float32, len(h))
		for i, v := range h {
			out[i] = float32(v)
		}
		return out
	}
	panic("tensor: unexpected host data")
}

// Item returns the value of a tensor holding exactly one element.
func (t *Tensor) Item() float64 {
	if t.Numel() != 1 {
		panic(fmt.Sprintf("tensor: Item requires a single element, got shape %v", t.shape))
	}
	return t.Float64s()[0]
}

// At returns the element at the given multi-dimensional index.
func (t *Tensor) At(idx ...int) float64 {
	if len(idx) != len(t.shape) {
		panic(fmt.Sprintf("tensor: At got %d indices for shape %v", len(idx), t.shape))
	}
	off := 0
	for d, i := range idx {
		if i < 0 || i >= t.shape[d] {
			panic(fmt.Sprintf("tensor: index %d out of range for dim %d of size %d", i, d, t.shape[d]))
		}
		off = off*t.shape[d] + i
	}
	one := t.be().Copy(t.st, backend.View{Shape: []int{1}, Strides: []int{1}, Offset: off})
	return fromStorage(one, nil).Float64s()[0]
}

// To returns the tensor on device d, copying the data if it lives elsewhere,
// like tensor.to(device). The copy is differentiable.
func (t *Tensor) To(d Device) *Tensor {
	if t.Device() == d {
		return t
	}
	dst := mustBackend(d)
	checkDTypeSupported(dst, t.DType())
	src := t.Device()
	return newResult(dst.FromHost(t.be().ToHost(t.st)), t.shape, "To", []*Tensor{t}, func(g *Tensor) []*Tensor {
		return []*Tensor{g.To(src)}
	})
}

// ToDType returns the tensor converted to dtype dt, like tensor.to(dtype).
// The conversion is differentiable.
func (t *Tensor) ToDType(dt DType) *Tensor {
	if t.DType() == dt {
		return t
	}
	checkDTypeSupported(t.be(), dt)
	src := t.DType()
	return newResult(t.be().Cast(t.st, dt), t.shape, "ToDType", []*Tensor{t}, func(g *Tensor) []*Tensor {
		return []*Tensor{g.ToDType(src)}
	})
}

// MoveTo moves a leaf tensor's data (and gradient) to device d in place,
// keeping the same *Tensor. nn.To uses it to move model parameters.
func (t *Tensor) MoveTo(d Device) {
	if t.ctx != nil {
		panic("tensor: MoveTo can only be called on leaf tensors")
	}
	moved := t.Detach().To(d)
	t.st = moved.st
	if t.grad != nil {
		t.grad = t.grad.To(d)
	}
}

// CastTo converts a leaf tensor's data (and gradient) to dtype dt in place.
func (t *Tensor) CastTo(dt DType) {
	if t.ctx != nil {
		panic("tensor: CastTo can only be called on leaf tensors")
	}
	t.st = t.Detach().ToDType(dt).st
	if t.grad != nil {
		t.grad = t.grad.ToDType(dt)
	}
}

// CopyFrom overwrites t's values with src's, converting dtype and device as
// needed, like tensor.copy_(). The element counts must match. It is not
// recorded for autograd.
func (t *Tensor) CopyFrom(src *Tensor) {
	if src.Numel() != t.Numel() {
		panic(fmt.Sprintf("tensor: CopyFrom shape %v into %v", src.shape, t.shape))
	}
	// Convert before moving: the source dtype may not exist on t's device.
	src = src.Detach().ToDType(t.DType()).To(t.Device())
	t.be().Write(t.st, 0, src.st)
}

// RequiresGrad reports whether gradients are tracked for this tensor.
func (t *Tensor) RequiresGrad() bool { return t.requiresGrad }

// SetRequiresGrad turns gradient tracking on or off for a leaf tensor and
// returns the tensor for chaining.
func (t *Tensor) SetRequiresGrad(b bool) *Tensor {
	if t.ctx != nil {
		panic("tensor: SetRequiresGrad can only be called on leaf tensors")
	}
	t.requiresGrad = b
	return t
}

// IsLeaf reports whether the tensor was created directly rather than by an
// operation on tensors that require grad.
func (t *Tensor) IsLeaf() bool { return t.ctx == nil }

// Grad returns the accumulated gradient, or nil if none has been computed.
func (t *Tensor) Grad() *Tensor { return t.grad }

// ZeroGrad clears the accumulated gradient.
func (t *Tensor) ZeroGrad() { t.grad = nil }

// Detach returns a tensor sharing the same data but cut off from the graph.
func (t *Tensor) Detach() *Tensor { return fromStorage(t.st, t.shape) }

// Clone returns a detached copy of the tensor on the same device.
func (t *Tensor) Clone() *Tensor {
	return fromStorage(t.be().Copy(t.st, contiguousView(t.shape)), t.shape)
}

func (t *Tensor) String() string {
	vals := t.Float64s()
	var sb strings.Builder
	sb.WriteString("tensor(")
	if len(t.shape) == 0 {
		fmt.Fprintf(&sb, "%.4g", vals[0])
	} else {
		format(&sb, vals, t.shape, 0, 0)
	}
	if t.DType() != Float32 {
		fmt.Fprintf(&sb, ", dtype=%v", t.DType())
	}
	if t.Device() != CPU {
		fmt.Fprintf(&sb, ", device=%v", t.Device())
	}
	if t.requiresGrad {
		sb.WriteString(", requires_grad=true")
	}
	sb.WriteString(")")
	return sb.String()
}

func format(sb *strings.Builder, vals []float64, shape []int, dim, offset int) {
	sb.WriteByte('[')
	last := dim == len(shape)-1
	stride := numel(shape[dim+1:])
	for i := 0; i < shape[dim]; i++ {
		if i > 0 {
			sb.WriteString(",")
			if last {
				sb.WriteString(" ")
			} else {
				sb.WriteString("\n" + strings.Repeat(" ", len("tensor(")+dim+1))
			}
		}
		if last {
			fmt.Fprintf(sb, "%.4g", vals[offset+i])
		} else {
			format(sb, vals, shape, dim+1, offset+i*stride)
		}
	}
	sb.WriteByte(']')
}

func numel(shape []int) int { return backend.Numel(shape) }

func contiguousView(shape []int) backend.View {
	return backend.View{Shape: shape, Strides: backend.ContiguousStrides(shape)}
}

func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// normDim converts a possibly negative dimension index to a positive one.
func normDim(dim, ndim int) int {
	if dim < 0 {
		dim += ndim
	}
	if dim < 0 || dim >= ndim {
		panic(fmt.Sprintf("tensor: dimension %d out of range for %d-D tensor", dim, ndim))
	}
	return dim
}

// checkSame panics unless all tensors share a device and dtype.
func checkSame(op string, ts ...*Tensor) {
	for _, o := range ts[1:] {
		if o.Device() != ts[0].Device() {
			panic(fmt.Sprintf("tensor: %s: expected all tensors on the same device, got %v and %v", op, ts[0].Device(), o.Device()))
		}
		if o.DType() != ts[0].DType() {
			panic(fmt.Sprintf("tensor: %s: expected all tensors to have the same dtype, got %v and %v", op, ts[0].DType(), o.DType()))
		}
	}
}
