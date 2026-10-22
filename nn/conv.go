package nn

import (
	"math"

	"github.com/donghquinn/go-tensor/tensor"
)

// Conv2d is a 2-D convolution layer over inputs of shape (N, C, H, W), like
// torch.nn.Conv2d. Weight has shape (out, in, kernel, kernel).
type Conv2d struct {
	Weight  *tensor.Tensor
	Bias    *tensor.Tensor
	Stride  int
	Padding int
}

// NewConv2d creates a Conv2d layer with a square kernel, initialized like
// PyTorch with values drawn uniformly from ±1/sqrt(in*kernel*kernel).
func NewConv2d(in, out, kernel, stride, padding int) *Conv2d {
	bound := 1 / math.Sqrt(float64(in*kernel*kernel))
	return &Conv2d{
		Weight:  tensor.Uniform(-bound, bound, out, in, kernel, kernel).SetRequiresGrad(true),
		Bias:    tensor.Uniform(-bound, bound, out).SetRequiresGrad(true),
		Stride:  stride,
		Padding: padding,
	}
}

func (c *Conv2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.Conv2d(x, c.Weight, c.Bias, c.Stride, c.Padding)
}

func (c *Conv2d) Parameters() []*tensor.Tensor { return []*tensor.Tensor{c.Weight, c.Bias} }

// MaxPool2d takes the maximum over square windows, like torch.nn.MaxPool2d.
type MaxPool2d struct {
	Kernel int
	Stride int
}

// NewMaxPool2d creates a max-pooling layer. A stride of 0 defaults to the
// kernel size, giving non-overlapping windows.
func NewMaxPool2d(kernel, stride int) *MaxPool2d {
	if stride == 0 {
		stride = kernel
	}
	return &MaxPool2d{Kernel: kernel, Stride: stride}
}

func (p *MaxPool2d) Forward(x *tensor.Tensor) *tensor.Tensor {
	return tensor.MaxPool2d(x, p.Kernel, p.Stride)
}

func (p *MaxPool2d) Parameters() []*tensor.Tensor { return nil }
