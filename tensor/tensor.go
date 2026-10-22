// Package tensor provides an n-dimensional array of float64 values with
// reverse-mode automatic differentiation, modeled after PyTorch's torch.Tensor.
//
// Tensors are stored contiguously in row-major order. Operations that receive
// invalid shapes panic, the same way slices panic on out-of-range indices.
package tensor

import (
	"fmt"
	"strings"
)

// Tensor is an n-dimensional array that can record the operations applied to
// it so that gradients can be computed with Backward.
type Tensor struct {
	data         []float64
	shape        []int
	requiresGrad bool
	grad         *Tensor
	ctx          *node // nil for leaf tensors
}

// New creates a tensor from a copy of data with the given shape. If no shape
// is given the tensor is one-dimensional.
func New(data []float64, shape ...int) *Tensor {
	if len(shape) == 0 {
		shape = []int{len(data)}
	}
	if n := numel(shape); n != len(data) {
		panic(fmt.Sprintf("tensor: data length %d does not match shape %v (%d elements)", len(data), shape, n))
	}
	return fromData(append([]float64(nil), data...), shape)
}

// fromData wraps data without copying it.
func fromData(data []float64, shape []int) *Tensor {
	return &Tensor{data: data, shape: append([]int(nil), shape...)}
}

// Shape returns a copy of the tensor's dimensions.
func (t *Tensor) Shape() []int { return append([]int(nil), t.shape...) }

// Dim returns the number of dimensions.
func (t *Tensor) Dim() int { return len(t.shape) }

// Numel returns the total number of elements.
func (t *Tensor) Numel() int { return len(t.data) }

// Data returns the underlying storage. It shares memory with the tensor, like
// torch.Tensor.numpy(); writing to it modifies the tensor in place.
func (t *Tensor) Data() []float64 { return t.data }

// Item returns the value of a tensor holding exactly one element.
func (t *Tensor) Item() float64 {
	if len(t.data) != 1 {
		panic(fmt.Sprintf("tensor: Item requires a single element, got shape %v", t.shape))
	}
	return t.data[0]
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
	return t.data[off]
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
func (t *Tensor) Detach() *Tensor { return fromData(t.data, t.shape) }

// Clone returns a detached deep copy of the tensor.
func (t *Tensor) Clone() *Tensor { return New(t.data, t.shape...) }

func (t *Tensor) String() string {
	var sb strings.Builder
	sb.WriteString("tensor(")
	if len(t.shape) == 0 {
		fmt.Fprintf(&sb, "%.4g", t.data[0])
	} else {
		t.format(&sb, 0, 0)
	}
	if t.requiresGrad {
		sb.WriteString(", requires_grad=true")
	}
	sb.WriteString(")")
	return sb.String()
}

func (t *Tensor) format(sb *strings.Builder, dim, offset int) {
	sb.WriteByte('[')
	last := dim == len(t.shape)-1
	stride := numel(t.shape[dim+1:])
	for i := 0; i < t.shape[dim]; i++ {
		if i > 0 {
			sb.WriteString(",")
			if last {
				sb.WriteString(" ")
			} else {
				sb.WriteString("\n" + strings.Repeat(" ", len("tensor(")+dim+1))
			}
		}
		if last {
			fmt.Fprintf(sb, "%.4g", t.data[offset+i])
		} else {
			t.format(sb, dim+1, offset+i*stride)
		}
	}
	sb.WriteByte(']')
}

func numel(shape []int) int {
	n := 1
	for _, s := range shape {
		n *= s
	}
	return n
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
